# SPDX-License-Identifier: Apache-2.0
"""End-to-end awex weight-update tests for Qwen3.5-MoE (Megatron bridge -> SGLang).

Requires GPUs and the tiny Qwen3.5-MoE fixture:

    python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny
    export TINY_QWEN35_MOE_PATH=/tmp/qwen3_5_moe_tiny
    uv run pytest tests/v2/weight_update/test_awex_qwen3_5_e2e.py -v -s

Adapted from ``test_nccl_integration._run_megatron_awex_e2e`` with two
qwen3.5-specific extensions the shared harness does not support:

- inference-side TP > 1 (validation slices the train-side full tensor with
  :class:`Qwen3_5MoeShardingStrategy` before comparing against the inference
  rank-0 dump);
- ``megatron-bridge`` engine config (``use_bridge_for_update_weights=True``),
  which the awex adapter's bridge export path requires for this model family.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import httpx
import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.slow,
    pytest.mark.sglang,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

# Representative sample across every layout class the name protocol handles.
VALIDATE_PARAM_NAMES = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.0.linear_attn.in_proj_q.weight",
    "model.layers.0.linear_attn.in_proj_z.weight",
    "model.layers.0.linear_attn.in_proj_b.weight",
    "model.layers.0.linear_attn.conv1d_q.weight",
    "model.layers.0.linear_attn.A_log",
    "model.layers.0.linear_attn.dt_bias",
    "model.layers.0.linear_attn.norm.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.1.down_proj.weight",
    "model.layers.0.mlp.shared_expert.gate_proj.weight",
    "model.layers.0.mlp.shared_expert_gate.weight",
]


def _tiny_model_path() -> str:
    path = os.environ.get("TINY_QWEN35_MOE_PATH", "/tmp/qwen3_5_moe_tiny")
    if not os.path.isdir(path):
        pytest.skip(
            f"Tiny Qwen3.5-MoE fixture not found at {path}; generate with "
            "tests/make_tiny_qwen3_5_moe.py"
        )
    return path


def _attention_sample_names(model_path: str) -> list[str]:
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    names: list[str] = []
    # Cover the LAST expert too: under EP it lives on the highest ep_rank,
    # so ep_rank>0 ownership/offset bugs cannot hide behind expert-0 samples.
    num_experts = text_cfg.get("num_experts")
    if num_experts:
        names += [
            f"model.layers.0.mlp.experts.{num_experts - 1}.gate_proj.weight",
            f"model.layers.0.mlp.experts.{num_experts - 1}.down_proj.weight",
        ]
    block_types = text_cfg.get("layer_types") or text_cfg.get(
        "layers_block_type", []
    )
    for idx, kind in enumerate(block_types):
        if kind == "attention" or kind == "full_attention":
            names += [
                f"model.layers.{idx}.self_attn.q_proj.weight",
                f"model.layers.{idx}.self_attn.k_proj.weight",
                f"model.layers.{idx}.self_attn.o_proj.weight",
                f"model.layers.{idx}.self_attn.q_norm.weight",
            ]
            break
    return names


def _validate_qwen3_5(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
    infer_tp: int,
    names: list[str],
) -> None:
    """Union train dumps, slice per the sharding table, compare bitwise."""
    from awex.sharding.param_sharding import ShardingType

    from areal.v2.weight_update.awex.qwen3_5 import Qwen3_5MoeShardingStrategy

    # get_parameters triggers bridge.export_hf_weights on the train side,
    # which is a COLLECTIVE across all train ranks: every worker must enter
    # it concurrently or the first one blocks forever waiting for peers.
    import concurrent.futures

    train_paths = [
        str(param_dir / f"qwen35_train_rank{i}.pt")
        for i in range(len(train_worker_urls))
    ]

    def _fetch_train(args):
        i, url, p = args
        resp = httpx.post(
            f"{url}/awex/debug/get_parameters",
            json={"save_path": p, "names": names},
            timeout=300.0,
        )
        assert resp.status_code == 200, f"train get_parameters[{i}]: {resp.text}"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(train_worker_urls)
    ) as pool:
        list(
            pool.map(
                _fetch_train,
                [
                    (i, url, p)
                    for i, (url, p) in enumerate(zip(train_worker_urls, train_paths))
                ],
            )
        )

    inf_path = str(param_dir / "qwen35_infer_rank0.pt")
    resp = httpx.post(
        f"{inf_worker_url}/awex/debug/get_parameters",
        json={"save_path": inf_path, "names": names},
        timeout=300.0,
    )
    assert resp.status_code == 200, f"infer get_parameters: {resp.text}"

    train_full: dict[str, torch.Tensor] = {}
    for p in train_paths:
        for name, tensor in torch.load(
            p, map_location="cpu", weights_only=True
        ).items():
            train_full.setdefault(name, tensor)
    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)

    strategy = Qwen3_5MoeShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
        tp_size=infer_tp,
        ep_size=1,
        ep_tp_size=1,
        rank_info=SimpleNamespace(tp_size=infer_tp),
    )

    checked = 0
    for name in names:
        assert name in train_full, f"train dump missing {name}"
        assert name in infer_params, f"infer dump missing {name}"
        full = train_full[name]
        local = infer_params[name]
        stype, dim, _ = strategy.get_sharding_strategy(name)
        expected = (
            full
            if stype == ShardingType.NO_SHARDING
            else full.narrow(dim, 0, local.shape[dim])
        )
        torch.testing.assert_close(
            local.float(),
            expected.float(),
            rtol=0,
            atol=0,
            msg=f"mismatch after weight update: {name}",
        )
        checked += 1
        print(
            f"[qwen35-validation] {name}: OK "
            f"(shape={list(local.shape)}, dtype={local.dtype})"
        )
    print(f"[qwen35-validation] all {checked} sampled parameters match")


def _run_qwen3_5_awex_e2e(
    *,
    n_gpus: int,
    train_backend: str,
    infer_gpus: int,
    infer_tp: int,
    tag: str,
    tmp_path_factory,
) -> None:
    from tests.v2.weight_update.test_nccl_integration import _make_local_scheduler

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        MegatronEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import RolloutControllerV2
    from areal.v2.training_service.controller.controller import GatewayTrainController
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    model_path = _tiny_model_path()
    tmp = tmp_path_factory.mktemp(tag)
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    n_engines = infer_gpus // infer_tp
    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_engines}t{infer_tp}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=infer_tp,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=600.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=train_backend,
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=600.0,
        megatron=MegatronEngineConfig(
            bridge_type="megatron-bridge",
            use_bridge_for_update_weights=True,
        ),
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    names = VALIDATE_PARAM_NAMES + _attention_sample_names(model_path)
    wu_ctrl: WeightUpdateController | None = None
    try:
        inf_ctrl.initialize(
            role="rollout",
            server_args={
                "model_path": model_path,
                "mem_fraction_static": 0.6,
                "tp_size": infer_tp,
            },
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=300.0)
            assert resp.status_code == 200, f"randomize failed: {resp.text}"

        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
            wait=True,
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=600.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check()

        master_port = find_free_ports(1)[0]
        wu_ctrl.connect(
            pair_name=f"qwen35-{tag}",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            nccl_master_addr="127.0.0.1",
            nccl_master_port=master_port,
        )
        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok", f"update_weights failed: {result.error}"
        wu_ctrl.disconnect()

        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=60.0,
        )
        assert gen_resp.status_code == 200, f"generation failed: {gen_resp.text}"

        _validate_qwen3_5(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            infer_tp=infer_tp,
            names=names,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.parametrize(
    "n_gpus,train_backend,infer_gpus,infer_tp",
    [
        (2, "megatron:d1", 1, 1),
        (4, "megatron:d1t2", 2, 2),
        (6, "megatron:d1t2p2", 2, 2),
        (8, "megatron:d1t2p2", 4, 4),
    ],
    ids=["1t1i-tp1", "2t2i-tp2", "4t2i-tp2-pp2", "4t4i-tp4-pp2"],
)
def test_awex_qwen3_5_moe_e2e_weight_update(
    n_gpus, train_backend, infer_gpus, infer_tp, tmp_path_factory
):
    """Randomize SGLang weights, transfer from Megatron via awex, compare."""
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_qwen3_5_awex_e2e(
        n_gpus=n_gpus,
        train_backend=train_backend,
        infer_gpus=infer_gpus,
        infer_tp=infer_tp,
        tag=f"qwen35_{train_backend.replace(':', '_')}_itp{infer_tp}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.parametrize(
    "n_gpus,train_backend,infer_gpus,infer_tp",
    [
        (8, "megatron:d1t2e2", 2, 2),
    ],
    ids=["4t-tp2ep2-2i-tp2"],
)
def test_awex_qwen3_5_moe_e2e_weight_update_ep(
    n_gpus, train_backend, infer_gpus, infer_tp, tmp_path_factory
):
    """EP>1 on the train side: expert ownership must tile across EP ranks."""
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_qwen3_5_awex_e2e(
        n_gpus=n_gpus,
        train_backend=train_backend,
        infer_gpus=infer_gpus,
        infer_tp=infer_tp,
        tag="qwen35_ep2",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.parametrize(
    "n_gpus,train_backend,infer_gpus,infer_tp",
    [
        (3, "megatron:d1c2", 1, 1),
        (6, "megatron:d1t2c2", 2, 2),
    ],
    ids=["2t-cp2-1i-tp1", "4t-tp2cp2-2i-tp2"],
)
def test_awex_qwen3_5_moe_e2e_weight_update_cp(
    n_gpus, train_backend, infer_gpus, infer_tp, tmp_path_factory
):
    """CP>1 on the train side: CP ranks hold weight replicas, so they all
    report identical full tensors (dp_replicated) and awex must still build
    a plan that updates every inference shard exactly once."""
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_qwen3_5_awex_e2e(
        n_gpus=n_gpus,
        train_backend=train_backend,
        infer_gpus=infer_gpus,
        infer_tp=infer_tp,
        tag=f"qwen35_cp_{train_backend.split(':')[1]}_itp{infer_tp}",
        tmp_path_factory=tmp_path_factory,
    )
