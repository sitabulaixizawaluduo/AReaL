# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec


def _slice_grid(
    grid: torch.Tensor | None,
    offset: int,
    count: int,
) -> torch.Tensor | None:
    if grid is None or count <= 0:
        return None
    return grid[offset : offset + count]


def qwen3_5_get_rope_index(
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None,
    video_grid_thw: torch.LongTensor | None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3.5 VL 3D position-id builder (ported from mbridge rope_utils).

    Returns:
        tuple(position_ids, mrope_position_deltas)
        - position_ids: [3, batch, seq]
        - mrope_position_deltas: [batch, 1]
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    image_grid_thw_list = (
        image_grid_thw.tolist() if image_grid_thw is not None else None
    )
    video_grid_thw_list = (
        video_grid_thw.tolist() if video_grid_thw is not None else None
    )

    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                (input_ids.shape[0], 1),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)

    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    mrope_position_deltas: list[torch.Tensor] = []
    image_index = 0
    video_index = 0
    attention_mask = attention_mask.to(total_input_ids.device)

    for i, row_ids in enumerate(total_input_ids):
        row_ids = row_ids[attention_mask[i] == 1]

        vision_start_indices = torch.argwhere(row_ids == vision_start_token_id).squeeze(
            1
        )
        vision_tokens = row_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int((vision_tokens == video_token_id).sum().item())

        input_tokens = row_ids.tolist()
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images = image_nums
        remain_videos = video_nums

        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                assert image_grid_thw_list is not None
                t, h, w = image_grid_thw_list[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                assert video_grid_thw_list is not None
                t, h, w = video_grid_thw_list[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas_tensor = torch.tensor(
        mrope_position_deltas,
        device=input_ids.device,
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas_tensor


def build_qwen3_5_segment_position_ids(
    *,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.LongTensor,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    image_grid_thw: torch.LongTensor | None,
    video_grid_thw: torch.LongTensor | None,
) -> torch.LongTensor:
    """Build per-token [T,H,W] position ids over a packed row (shape [N, 3])."""
    if packed_input_ids.ndim != 1:
        raise ValueError(
            "build_qwen3_5_segment_position_ids expects packed_input_ids shape [total_tokens]."
        )
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must be 1D, non-empty, and start with 0.")

    num_segments = cu_seqlens.numel() - 1
    starts = torch.nonzero(
        packed_input_ids == vision_start_token_id, as_tuple=False
    ).flatten()
    starts = starts[starts + 1 < packed_input_ids.numel()]

    image_counts = torch.zeros(
        num_segments, dtype=torch.long, device=packed_input_ids.device
    )
    video_counts = torch.zeros(
        num_segments, dtype=torch.long, device=packed_input_ids.device
    )

    if starts.numel() > 0:
        vision_types = packed_input_ids[starts + 1]
        seg_idx = torch.bucketize(starts, cu_seqlens, right=True) - 1
        image_counts = torch.bincount(
            seg_idx[vision_types == image_token_id], minlength=num_segments
        )
        video_counts = torch.bincount(
            seg_idx[vision_types == video_token_id], minlength=num_segments
        )

    image_offset = 0
    video_offset = 0
    segments: list[torch.Tensor] = []

    for segment_id, (start, end) in enumerate(
        zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ):
        if end <= start:
            continue
        seg = packed_input_ids[start:end]
        image_count = int(image_counts[segment_id].item())
        video_count = int(video_counts[segment_id].item())

        if image_count == 0 and video_count == 0:
            pos = (
                torch.arange(seg.numel(), dtype=torch.long, device=seg.device)
                .view(-1, 1)
                .expand(-1, 3)
            )
        else:
            seg_pos, _ = qwen3_5_get_rope_index(
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                input_ids=seg.unsqueeze(0),
                image_grid_thw=_slice_grid(image_grid_thw, image_offset, image_count),
                video_grid_thw=_slice_grid(video_grid_thw, video_offset, video_count),
                attention_mask=None,
            )
            pos = seg_pos[:, 0, : seg.numel()].transpose(0, 1).contiguous().long()

        image_offset += image_count
        video_offset += video_count
        segments.append(pos)

    if not segments:
        return torch.zeros((0, 3), dtype=torch.long, device=packed_input_ids.device)
    return torch.cat(segments, dim=0).contiguous()


def compute_local_vision_chunk_counts(
    *,
    flat_input_ids: torch.LongTensor,
    cu_seqlens: torch.LongTensor,
    cp_size: int,
    image_token_id: int,
    video_token_id: int,
) -> torch.LongTensor | None:
    """Count local zigzag-chunk vision tokens per segment for packed CP rows.

    Returns tensor shape [num_segments, 2] or None when layout does not match
    packed zigzag THD (caller should keep stock full-sequence behavior).
    """
    if cp_size <= 1 or cu_seqlens.numel() < 2:
        return None

    local_len = int(flat_input_ids.numel())
    cu_list = cu_seqlens.tolist()
    if cu_list[0] != 0 or cu_list[-1] != cp_size * local_len:
        return None

    is_vis = (flat_input_ids == image_token_id) | (flat_input_ids == video_token_id)
    num_segments = len(cu_list) - 1
    num_chunks = 2 * cp_size
    counts_local = torch.zeros(
        num_segments, 2, dtype=torch.long, device=flat_input_ids.device
    )

    for i in range(num_segments):
        seg_full = cu_list[i + 1] - cu_list[i]
        if seg_full % num_chunks != 0:
            return None
        chunk = seg_full // num_chunks
        local_off = cu_list[i] // cp_size
        counts_local[i, 0] = is_vis[local_off : local_off + chunk].sum()
        counts_local[i, 1] = is_vis[local_off + chunk : local_off + 2 * chunk].sum()

    return counts_local


def build_cp_local_vision_embed_indices(
    *,
    local_chunk_counts: torch.LongTensor,
    gathered_chunk_counts: list[torch.LongTensor],
    cp_rank: int,
    cp_size: int,
    total_vision_tokens: int,
    device: torch.device,
) -> torch.LongTensor | None:
    """Build row indices selecting this CP rank's vision embeddings from full order."""
    if cp_size <= 1:
        return None
    if int(local_chunk_counts.sum().item()) == total_vision_tokens:
        return None

    num_segments = local_chunk_counts.shape[0]
    num_chunks = 2 * cp_size
    full_counts = torch.zeros(num_segments, num_chunks, dtype=torch.long)

    for r in range(cp_size):
        gathered_cpu = gathered_chunk_counts[r].cpu()
        full_counts[:, r] = gathered_cpu[:, 0]
        full_counts[:, num_chunks - 1 - r] = gathered_cpu[:, 1]

    flat_counts = full_counts.reshape(-1)
    offsets = (torch.cumsum(flat_counts, dim=0) - flat_counts).reshape(
        num_segments, num_chunks
    )

    local_counts_cpu = local_chunk_counts.cpu()
    idx_parts: list[torch.Tensor] = []
    for i in range(num_segments):
        for local_chunk_id, chunk_id in enumerate((cp_rank, num_chunks - 1 - cp_rank)):
            n = int(local_counts_cpu[i, local_chunk_id])
            if n <= 0:
                continue
            start = int(offsets[i, chunk_id])
            idx_parts.append(torch.arange(start, start + n, device=device))

    if not idx_parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(idx_parts)


def scatter_vision_embeddings_into_text_embeddings(
    *,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    vision_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter vision embeddings into image/video token positions in text embeddings."""
    expected = int(vision_token_mask.sum().item())
    actual = int(vision_embeddings.shape[0])
    if expected != actual:
        raise ValueError(
            "Mismatch between local vision token count and local vision embedding rows: "
            f"expected={expected}, got={actual}."
        )
    fused = text_embeddings.clone()
    fused[vision_token_mask] = vision_embeddings.to(dtype=fused.dtype)
    return fused


class Qwen3_5MoeVLModel(MegatronModule):
    """Qwen3.5-MoE VLM wrapper for packed THD + CP training on mbridge path."""

    def __init__(
        self,
        *,
        language_transformer_config,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        hf_config,
        hf_vision_cls: type | None,
        parallel_output: bool = True,
        language_rotary_percent: float = 1.0,
        language_rotary_base: int = 10000,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        freeze_vision_model: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.hf_config = hf_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = hf_config.vision_config.spatial_merge_size

        self.vision_model = None
        if self.pre_process and hf_vision_cls is not None:
            self.vision_model = hf_vision_cls._from_config(hf_config.vision_config)
            self._hook_fp32_rotary_emb(self.vision_model)
            self._hook_vision_params_avg_grad_across_tp(self.vision_model)
            if freeze_vision_model:
                for param in self.vision_model.parameters():
                    param.requires_grad = False

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            position_embedding_type="mrope",
            rotary_percent=language_rotary_percent,
            rotary_base=language_rotary_base,
            scatter_embedding_sequence_parallel=False,
        )

    @staticmethod
    def _hook_fp32_rotary_emb(module: torch.nn.Module):
        for submodule in module.modules():
            if hasattr(submodule, "inv_freq") and submodule.inv_freq is not None:
                submodule._inv_freq_fp32_original = (
                    submodule.inv_freq.detach().clone().float()
                )

                def _hook(mod, args):
                    del args
                    if hasattr(mod, "_inv_freq_fp32_original"):
                        mod.inv_freq = mod._inv_freq_fp32_original.to(
                            device=mod.inv_freq.device
                        )

                submodule.register_forward_pre_hook(_hook)

    @staticmethod
    def _hook_vision_params_avg_grad_across_tp(module: torch.nn.Module) -> None:
        for param in module.parameters(recurse=True):
            setattr(param, "average_gradients_across_tp_domain", True)

    @property
    def share_embeddings_and_output_weights(self):
        return self.language_model.share_embeddings_and_output_weights

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.language_model.set_input_tensor(input_tensor[0])

    def _cp_local_vision_embed_indices(
        self,
        *,
        vision_embeds: torch.Tensor | None,
        input_ids: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor | None:
        if vision_embeds is None or packed_seq_params is None:
            return None
        if getattr(packed_seq_params, "qkv_format", None) != "thd":
            return None

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size <= 1:
            return None

        cu = packed_seq_params.cu_seqlens_q
        if cu is None or cu.numel() < 2:
            return None

        flat = input_ids.reshape(-1)
        local_counts = compute_local_vision_chunk_counts(
            flat_input_ids=flat,
            cu_seqlens=cu,
            cp_size=cp_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )
        if local_counts is None:
            return None

        cp_group = mpu.get_context_parallel_group()
        gathered_counts = [torch.empty_like(local_counts) for _ in range(cp_size)]
        dist.all_gather(gathered_counts, local_counts, group=cp_group)
        cp_rank = mpu.get_context_parallel_rank()

        return build_cp_local_vision_embed_indices(
            local_chunk_counts=local_counts,
            gathered_chunk_counts=gathered_counts,
            cp_rank=cp_rank,
            cp_size=cp_size,
            total_vision_tokens=vision_embeds.shape[0],
            device=vision_embeds.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict[str, Any] | None = None,
        runtime_gather_output: bool | None = None,
        inference_params=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del pixel_values_videos
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "Qwen3_5MoeVLModel does not support inference mode."
            )

        combined_embeddings = None
        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            if (
                self.vision_model is not None
                and pixel_values is not None
                and image_grid_thw is not None
                and image_grid_thw.numel() > 0
            ):
                vision_outputs = self.vision_model(
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                )
                vision_embeds = vision_outputs.pooler_output
                split_sizes = (
                    image_grid_thw.prod(-1) // self.spatial_merge_size**2
                ).tolist()
                vision_embeds = torch.cat(
                    torch.split(vision_embeds, split_sizes), dim=0
                )

                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                vision_mask = (input_ids == self.image_token_id) | (
                    input_ids == self.video_token_id
                )
                cp_local_idx = self._cp_local_vision_embed_indices(
                    vision_embeds=vision_embeds,
                    input_ids=input_ids,
                    packed_seq_params=packed_seq_params,
                )
                if cp_local_idx is not None:
                    vision_embeds = vision_embeds.index_select(0, cp_local_idx)

                combined_embeddings = scatter_vision_embeddings_into_text_embeddings(
                    text_embeddings=combined_embeddings,
                    vision_embeddings=vision_embeds,
                    vision_token_mask=vision_mask,
                )
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = scatter_to_sequence_parallel_region(
                    combined_embeddings
                ).contiguous()

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
