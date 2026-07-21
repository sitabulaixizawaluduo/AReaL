# SPDX-License-Identifier: Apache-2.0

"""Distributed integration tests for the Megatron VLM path.

All tests in this file launch ``tests/torchrun/run_megatron_engine_vlm_distributed.py``
as a torchrun subprocess so the parent pytest process never allocates GPU
memory. The full suite runs on as few as 2 devices for the dense parametric
tests; Qwen3-VL-MoE tests need 8.

CPU-only unit tests for the converters / detection helpers live in
``tests/test_megatron_engine_vlm.py``.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import torch

from areal.api.alloc_mode import ModelAllocation
from areal.utils.network import find_free_ports
from areal.utils.testing_utils import DENSE_MODEL_PATHS, MOE_MODEL_PATHS

_TORCHRUN_SCRIPT = (
    pathlib.Path(__file__).parent
    / "torchrun"
    / "run_megatron_engine_vlm_distributed.py"
).resolve()

CUDA_AVAILABLE = torch.cuda.is_available()


def _run_vlm_test(
    test_type: str,
    output_path: str,
    *,
    backend: str = "megatron:d1p1t1",
    nproc: int | None = None,
    extra_args: list[str] | None = None,
    timeout: int = 1800,
    env_overrides: dict[str, str] | None = None,
):
    """Launch a VLM integration test via torchrun subprocess.

    ``nproc`` is inferred from ``ModelAllocation.from_str(backend).parallel.world_size``
    when not provided, so asymmetric ``(attn:...|ffn:...)`` allocations work
    without callers having to compute the rank count separately. Pass
    ``env_overrides={"VLM_MODEL_PATH": ...}`` to switch the model under test.

    Output is streamed to the parent's stdout/stderr (helpful for diagnosing
    long distributed runs); OOM detection happens via the runner's own
    ``write_result(output, "OOM")`` marker rather than stderr scraping.
    """
    if nproc is None:
        nproc = ModelAllocation.from_str(backend).parallel.world_size

    port = find_free_ports(1)[0]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(nproc))
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        str(_TORCHRUN_SCRIPT),
        f"--backend={backend}",
        f"--test_type={test_type}",
        f"--output={output_path}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    output_path_obj = pathlib.Path(output_path)
    try:
        subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        if output_path_obj.exists() and output_path_obj.read_text().strip() == "OOM":
            pytest.skip(f"OOM: VLM {test_type} requires more GPU memory")
        pytest.fail(f"VLM {test_type} test failed (exit {e.returncode})")
    except subprocess.TimeoutExpired:
        pytest.fail(f"VLM {test_type} test timed out ({timeout}s)")

    result = output_path_obj.read_text().strip()
    if result == "OOM":
        pytest.skip(f"OOM: VLM {test_type} requires more GPU memory")
    assert result == "Passed", f"VLM {test_type} test failed: {result}"


# ──────────────────────────────────────────────────────────────────────
# Per-model env fixtures.
#
# Each parametric scenario below runs once for every entry in _VLM_MODELS.
# To add a new VLM, register the path in ``areal.utils.testing_utils`` and
# append one tuple here — no test-body changes needed.
# ──────────────────────────────────────────────────────────────────────

_VLM_MODELS = [
    pytest.param(
        {"VLM_MODEL_PATH": DENSE_MODEL_PATHS["qwen2_5_vl"]},
        id="qwen25_vl",
    ),
    pytest.param(
        {"VLM_MODEL_PATH": DENSE_MODEL_PATHS["qwen3_vl"]},
        id="qwen3_vl",
    ),
    pytest.param(
        {"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
        id="qwen3_vl_moe",
        marks=pytest.mark.skipif(
            torch.cuda.device_count() < 8,
            reason="Qwen3-VL-MoE-30B-A3B requires at least 8 GPUs",
        ),
    ),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_engine_initializes(model_env, tmp_path_factory):
    """Verify VLM engine detects vision model and loads processor."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "init.out")
    _run_vlm_test("init", output, env_overrides=model_env)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_simple_forward(model_env, tmp_path_factory):
    """Verify forward pass with VLM inputs completes."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "forward.out")
    _run_vlm_test("forward", output, env_overrides=model_env)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_hf_save_load_weights(model_env, tmp_path_factory):
    """Verify save/load preserves VLM weights and saves processor."""
    save_dir = str(tmp_path_factory.mktemp("vlm_save"))
    output = str(tmp_path_factory.mktemp("vlm_test") / "save_load.out")
    _run_vlm_test(
        "save_load",
        output,
        extra_args=[f"--save_dir={save_dir}"],
        env_overrides=model_env,
    )


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_train_tensor_parallel(model_env, tmp_path_factory):
    """VLM training with TP=2 to avoid single-device OOM."""
    if torch.cuda.device_count() < 2:
        pytest.skip("VLM TP training requires at least 2 GPUs")
    output = str(tmp_path_factory.mktemp("vlm_test") / "train_tp2.out")
    _run_vlm_test(
        "train",
        output,
        backend="megatron:d1p1t2",
        env_overrides=model_env,
    )


# ──────────────────────────────────────────────────────────────────────
# Qwen3-VL-MoE: 30B-A3B-Instruct under hybrid (attn|ffn) allocation.
# CP > 1 is forbidden for VLMs (megatron_engine.py:347).
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_qwen3vl_moe_expert_parallel(tmp_path_factory):
    """Forward smoke test for Qwen3-VL-MoE under ``(attn:d2t4|ffn:d2e4)``.

    Allocation: attn DP=2 TP=4 (8 GPUs); ffn DP=2 EP=4 (8 GPUs). World sizes
    must match — earlier ``(attn:d2t2|ffn:d2e4)`` raised
    ``InvalidAllocationModeError`` (attn=4 vs ffn=8). Validates the hybrid
    attn/ffn parser, EP-aware weight init, and VLM forward path.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("Qwen3-VL-MoE expert parallel requires 8 GPUs to run")
    output = str(
        tmp_path_factory.mktemp("test_output") / "qwen3vl_moe_expert_parallel.out"
    )
    _run_vlm_test(
        "forward",
        output,
        backend="megatron:(attn:d2t4|ffn:d2e4)",
        env_overrides={"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_qwen3vl_moe_dcp_save_load(tmp_path_factory):
    """DCP save/load round-trip for Qwen3-VL-MoE under ``(attn:d2p1t4|ffn:d1p1t2e4)``.

    Allocation: attn DP=2 PP=1 TP=4 (8 GPUs); ffn DP=1 PP=1 TP=2 EP=4 (8 GPUs).
    Drops the ``cp=2`` segment from the dense Qwen3-MoE analog because VLM
    forbids CP>1 (megatron_engine.py:347).
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("Qwen3-VL-MoE DCP save load requires 8 GPUs to run")
    output = str(tmp_path_factory.mktemp("test_output") / "qwen3vl_moe_save_load.out")
    _run_vlm_test(
        "dcp_save_load",
        output,
        backend="megatron:(attn:d2p1t4|ffn:d1p1t2e4)",
        env_overrides={"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
    )


# ──────────────────────────────────────────────────────────────────────
# Qwen3.5-MoE (tiny fixture) on the mbridge registry path: the ONLY VLM
# family where CP > 1 is allowed (packed THD + GDN a2a CP + per-segment
# mRoPE; the megatron_engine VLM CP guard exempts mbridge qwen3_5).
#
# GPU runbook:
#   python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny
#   pytest tests/test_megatron_engine_vlm_distributed.py -k qwen3_5_moe_vl -x -s
# ──────────────────────────────────────────────────────────────────────

_TINY_QWEN35_PATH = os.environ.get(
    "AREAL_TINY_QWEN35_MOE_PATH", "/tmp/qwen3_5_moe_tiny"
)
_TINY_QWEN35_ENV = {"VLM_MODEL_PATH": _TINY_QWEN35_PATH}
_tiny_qwen35_skip = pytest.mark.skipif(
    not pathlib.Path(_TINY_QWEN35_PATH).exists(),
    reason=(
        "tiny Qwen3.5-MoE fixture missing — run "
        "`python tests/make_tiny_qwen3_5_moe.py` first "
        f"(looked at {_TINY_QWEN35_PATH})"
    ),
)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@_tiny_qwen35_skip
def test_qwen3_5_moe_vl_forward_single_gpu(tmp_path_factory):
    """Multimodal forward smoke on the mbridge packed-THD path (1 GPU)."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "qwen3_5_moe_vl_forward.out")
    _run_vlm_test(
        "forward",
        output,
        backend="megatron:d1p1t1",
        env_overrides=_TINY_QWEN35_ENV,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@_tiny_qwen35_skip
def test_qwen3_5_moe_vl_forward_context_parallel(tmp_path_factory):
    """Multimodal forward with CP=2 (2 GPUs): GDN a2a + mRoPE zigzag + vision."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Qwen3.5-MoE VL context parallel requires 2 GPUs to run")
    output = str(tmp_path_factory.mktemp("vlm_test") / "qwen3_5_moe_vl_cp2.out")
    _run_vlm_test(
        "forward",
        output,
        backend="megatron:d1p1t1c2",
        env_overrides=_TINY_QWEN35_ENV,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@_tiny_qwen35_skip
def test_qwen3_5_moe_vl_logprob_cp_equivalence(tmp_path_factory):
    """Multimodal logprob CP=1 vs CP=2 must agree within 2% relative diff.

    Catches broken CP routing for images: wrong per-segment mRoPE positions,
    mis-selected CP-local vision embeddings, or zigzag layout drift. Healthy
    band established by radixark/miles PR #1308: abs logprob diff ~0.011-0.016
    for real Qwen3-VL runs.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Qwen3.5-MoE VL logprob equivalence requires 2 GPUs to run")
    out_dir = tmp_path_factory.mktemp("vlm_test")
    out_c1 = out_dir / "qwen3_5_moe_vl_logprob_c1.out"
    out_c2 = out_dir / "qwen3_5_moe_vl_logprob_c2.out"
    _run_vlm_test(
        "logprob_value",
        str(out_c1),
        backend="megatron:d1p1t1",
        env_overrides=_TINY_QWEN35_ENV,
    )
    _run_vlm_test(
        "logprob_value",
        str(out_c2),
        backend="megatron:d1p1t1c2",
        env_overrides=_TINY_QWEN35_ENV,
    )
    logprob_c1 = float(pathlib.Path(f"{out_c1}.logprob").read_text())
    logprob_c2 = float(pathlib.Path(f"{out_c2}.logprob").read_text())
    rel_diff = abs(logprob_c1 - logprob_c2) / max(abs(logprob_c1), 1e-12)
    assert rel_diff <= 0.02, (
        f"multimodal logprob differs between CP=1 ({logprob_c1}) and CP=2 "
        f"({logprob_c2}): rel_diff={rel_diff:.4f} > 0.02. This indicates broken "
        "CP data routing (mRoPE positions, vision-embed selection, or zigzag)."
    )
