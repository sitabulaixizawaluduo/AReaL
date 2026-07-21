# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TOKEN_ID = 90
VIDEO_TOKEN_ID = 91
VISION_START_TOKEN_ID = 89
SPATIAL_MERGE_SIZE = 2


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module(monkeypatch, name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qwen3_5_vl_modules(monkeypatch):
    class _MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    parallel_state = _stub_module(
        monkeypatch,
        "megatron.core.parallel_state",
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
    )
    _stub_module(monkeypatch, "megatron")
    _stub_module(monkeypatch, "megatron.core", parallel_state=parallel_state)
    _stub_module(monkeypatch, "megatron.core.models")
    _stub_module(monkeypatch, "megatron.core.models.gpt")
    _stub_module(
        monkeypatch,
        "megatron.core.models.gpt.gpt_model",
        GPTModel=type("GPTModel", (nn.Module,), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.packed_seq_params",
        PackedSeqParams=type("PackedSeqParams", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.tensor_parallel",
        scatter_to_sequence_parallel_region=lambda tensor: tensor,
    )
    _stub_module(monkeypatch, "megatron.core.transformer")
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.module",
        MegatronModule=_MegatronModule,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.spec_utils",
        ModuleSpec=type("ModuleSpec", (), {}),
    )

    vl = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5_vl_model_test",
        "areal/models/mcore/qwen3_5_vl_model.py",
    )
    packed_cp = _load_module(
        monkeypatch,
        "areal.engine.megatron_utils.packed_context_parallel_test",
        "areal/engine/megatron_utils/packed_context_parallel.py",
    )
    return SimpleNamespace(vl=vl, packed_cp=packed_cp, mpu=parallel_state)


def _packed_segments():
    segments = [
        torch.tensor([11, 12, 13, 14]),
        torch.tensor(
            [
                20,
                VISION_START_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                21,
                22,
            ]
        ),
        torch.tensor([31, 32, 33, 34]),
    ]
    packed = torch.cat(segments)
    cu_seqlens = torch.tensor([0, 4, 12, 16], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    return segments, packed, cu_seqlens, image_grid_thw


def _build_positions(vl, packed, cu_seqlens, image_grid_thw):
    return vl.build_qwen3_5_segment_position_ids(
        packed_input_ids=packed,
        cu_seqlens=cu_seqlens,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
    )


def _hf_positions(input_ids, image_grid_thw=None):
    oracle = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
        )
    )
    oracle.get_vision_position_ids = MethodType(
        Qwen3_5MoeModel.get_vision_position_ids, oracle
    )
    token_types = torch.where(input_ids == IMAGE_TOKEN_ID, 1, 0).to(torch.int32)
    positions, _ = Qwen3_5MoeModel.get_rope_index(
        oracle,
        input_ids=input_ids.unsqueeze(0),
        mm_token_type_ids=token_types.unsqueeze(0),
        image_grid_thw=image_grid_thw,
    )
    return positions[:, 0].transpose(0, 1).contiguous()


def test_segment_position_ids_restart_and_text_segments_are_linear(qwen3_5_vl_modules):
    vl = qwen3_5_vl_modules.vl
    segments, packed, cu_seqlens, image_grid_thw = _packed_segments()

    actual = _build_positions(vl, packed, cu_seqlens, image_grid_thw)

    assert actual.shape == (packed.numel(), 3)
    for segment_id, (start, end) in enumerate(
        zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)
    ):
        standalone = _build_positions(
            vl,
            segments[segment_id],
            torch.tensor([0, end - start]),
            image_grid_thw if segment_id == 1 else None,
        )
        torch.testing.assert_close(actual[start:end], standalone, rtol=0, atol=0)

    for segment_id in (0, 2):
        expected = torch.arange(segments[segment_id].numel()).view(-1, 1).expand(-1, 3)
        start = int(cu_seqlens[segment_id])
        end = int(cu_seqlens[segment_id + 1])
        torch.testing.assert_close(actual[start:end], expected, rtol=0, atol=0)


def test_segment_position_ids_match_hf_get_rope_index_per_segment(qwen3_5_vl_modules):
    vl = qwen3_5_vl_modules.vl
    segments, packed, cu_seqlens, image_grid_thw = _packed_segments()
    actual = _build_positions(vl, packed, cu_seqlens, image_grid_thw)

    expected = torch.cat(
        [
            _hf_positions(segment, image_grid_thw if segment_id == 1 else None)
            for segment_id, segment in enumerate(segments)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_segment_position_ids_zigzag_split_and_reassembly_roundtrip(
    qwen3_5_vl_modules, monkeypatch
):
    modules = qwen3_5_vl_modules
    _, packed, cu_seqlens, image_grid_thw = _packed_segments()
    positions = _build_positions(modules.vl, packed, cu_seqlens, image_grid_thw)
    cp_size = 2
    local_positions = []

    monkeypatch.setattr(modules.mpu, "get_context_parallel_world_size", lambda: cp_size)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            modules.mpu,
            "get_context_parallel_rank",
            lambda cp_rank=cp_rank: cp_rank,
        )
        local_positions.append(
            modules.packed_cp.split_packed_seqs_for_context_parallel(
                positions, cu_seqlens
            )
        )

    restored = torch.empty_like(positions)
    for segment_id, (start, end) in enumerate(
        zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)
    ):
        chunk_len = (end - start) // (2 * cp_size)
        local_start = start // cp_size
        for cp_rank in range(cp_size):
            local_segment = local_positions[cp_rank][
                local_start : local_start + 2 * chunk_len
            ]
            restored[
                start + cp_rank * chunk_len : start + (cp_rank + 1) * chunk_len
            ] = local_segment[:chunk_len]
            mirror_start = end - (cp_rank + 1) * chunk_len
            restored[mirror_start : mirror_start + chunk_len] = local_segment[
                chunk_len:
            ]

    torch.testing.assert_close(restored, positions, rtol=0, atol=0)


def test_cp_vision_indices_select_global_rows_in_local_token_order(
    qwen3_5_vl_modules, monkeypatch
):
    modules = qwen3_5_vl_modules
    cp_size = 2
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.long)
    full_input_ids = torch.tensor(
        [
            10,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            11,
            12,
            13,
            20,
            21,
            22,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            23,
        ]
    )
    monkeypatch.setattr(modules.mpu, "get_context_parallel_world_size", lambda: cp_size)

    local_ids = []
    local_counts = []
    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            modules.mpu,
            "get_context_parallel_rank",
            lambda cp_rank=cp_rank: cp_rank,
        )
        shard = modules.packed_cp.split_packed_seqs_for_context_parallel(
            full_input_ids, cu_seqlens
        )
        local_ids.append(shard)
        local_counts.append(
            modules.vl.compute_local_vision_chunk_counts(
                flat_input_ids=shard,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                image_token_id=IMAGE_TOKEN_ID,
                video_token_id=VIDEO_TOKEN_ID,
            )
        )

    global_vision_rows = torch.arange(int((full_input_ids == IMAGE_TOKEN_ID).sum()))
    vision_row_by_position = torch.full_like(full_input_ids, -1)
    vision_row_by_position[full_input_ids == IMAGE_TOKEN_ID] = global_vision_rows

    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            modules.mpu,
            "get_context_parallel_rank",
            lambda cp_rank=cp_rank: cp_rank,
        )
        indices = modules.vl.build_cp_local_vision_embed_indices(
            local_chunk_counts=local_counts[cp_rank],
            gathered_chunk_counts=local_counts,
            cp_rank=cp_rank,
            cp_size=cp_size,
            total_vision_tokens=global_vision_rows.numel(),
            device=torch.device("cpu"),
        )
        expected = vision_row_by_position[
            modules.packed_cp.split_packed_seqs_for_context_parallel(
                torch.arange(full_input_ids.numel()), cu_seqlens
            )
        ]
        expected = expected[local_ids[cp_rank] == IMAGE_TOKEN_ID]
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)


def test_scatter_vision_embeddings_replaces_only_vision_tokens(qwen3_5_vl_modules):
    vl = qwen3_5_vl_modules.vl
    text_embeddings = torch.arange(6 * 3).reshape(6, 3).float()
    vision_token_mask = torch.tensor([False, True, False, True, True, False])
    vision_embeddings = torch.tensor(
        [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0], [301.0, 302.0, 303.0]]
    )

    actual = vl.scatter_vision_embeddings_into_text_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings,
        vision_token_mask=vision_token_mask,
    )

    torch.testing.assert_close(
        actual[vision_token_mask], vision_embeddings, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual[~vision_token_mask], text_embeddings[~vision_token_mask], rtol=0, atol=0
    )
    torch.testing.assert_close(
        text_embeddings, torch.arange(6 * 3).reshape(6, 3).float(), rtol=0, atol=0
    )


def test_no_image_batch_builds_pure_linear_positions(qwen3_5_vl_modules):
    vl = qwen3_5_vl_modules.vl
    packed = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cu_seqlens = torch.tensor([0, 3, 7])

    actual = _build_positions(vl, packed, cu_seqlens, None)
    expected = torch.cat(
        [
            torch.arange(3).view(-1, 1).expand(-1, 3),
            torch.arange(4).view(-1, 1).expand(-1, 3),
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_single_image_spanning_segment_matches_hf(qwen3_5_vl_modules):
    vl = qwen3_5_vl_modules.vl
    segment = torch.tensor(
        [
            VISION_START_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
        ]
    )
    image_grid_thw = torch.tensor([[1, 4, 4]])

    actual = _build_positions(
        vl, segment, torch.tensor([0, segment.numel()]), image_grid_thw
    )
    expected = _hf_positions(segment, image_grid_thw)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def _mcore_thd_exact_freq_indices(
    cp_rank: int, cp_size: int, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Vendored selection arithmetic of mcore 0.17 rope_utils.py:
    _apply_rotary_pos_emb_thd exact path (freqs.size(0)==cu_seqlens[-1]) +
    _get_thd_freqs_on_this_cp_rank(offset=cu_seqlens[i])."""
    token_indices = torch.arange(int(cu_seqlens[-1]))
    seqlens_local = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
    slices = []
    for i, local_len in enumerate(seqlens_local):
        offset = int(cu_seqlens[i])
        full_len = local_len * cp_size
        seg = local_len // 2
        slices.append(
            token_indices[offset + cp_rank * seg : offset + (cp_rank + 1) * seg]
        )
        slices.append(
            token_indices[
                offset + full_len - (cp_rank + 1) * seg : offset
                + full_len
                - cp_rank * seg
            ]
        )
    return torch.cat(slices)


def test_mcore_thd_rope_selection_matches_areal_zigzag_split(
    qwen3_5_vl_modules, monkeypatch
):
    """Pins the mrope/CP contract that crashed on cluster (128 vs 79).

    With FULL-length per-token freqs, mcore's THD exact path must select for
    each CP rank exactly the token positions AReaL's data-side per-sequence
    zigzag split assigns to that rank. Pre-splitting positions (the old
    behavior) breaks this: freqs enter the non-exact fallback and get mangled.
    """
    modules = qwen3_5_vl_modules
    cp_size = 2
    cu_seqlens = torch.tensor([0, 24, 60, 256], dtype=torch.long)
    token_positions = torch.arange(256, dtype=torch.long)

    monkeypatch.setattr(modules.mpu, "get_context_parallel_world_size", lambda: cp_size)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(
            modules.mpu,
            "get_context_parallel_rank",
            lambda cp_rank=cp_rank: cp_rank,
        )
        data_side = modules.packed_cp.split_packed_seqs_for_context_parallel(
            token_positions, cu_seqlens
        )
        rope_side = _mcore_thd_exact_freq_indices(cp_rank, cp_size, cu_seqlens)

        assert rope_side.shape == data_side.shape, (
            f"rank {cp_rank}: mcore rope selects {rope_side.numel()} freq rows, "
            f"data split holds {data_side.numel()} tokens — the 128-vs-79 class "
            "of mismatch"
        )
        torch.testing.assert_close(rope_side, data_side, rtol=0, atol=0)
