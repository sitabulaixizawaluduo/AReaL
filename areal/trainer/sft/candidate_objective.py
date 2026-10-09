# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist

from areal.utils import stats_tracker


class _SumCandidateLogits(torch.autograd.Function):
    """Gather disjoint vocabulary slices; each TP rank owns its local gradient."""

    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        result = logits.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class CandidateSoftSFTObjective:
    """Four-option soft CE, opt-in through token-aligned teacher targets.

    The appended carrier token is only an alignment aid. Supervision is the
    complete teacher distribution, never the carrier's token identity.
    """

    size = 4
    keys = (
        "candidate_count",
        *(f"candidate_id_{i}" for i in range(4)),
        *(f"candidate_target_{i}" for i in range(4)),
    )

    @classmethod
    def enabled(cls, data: dict[str, Any]) -> bool:
        return any(f"candidate_target_{i}" in data for i in range(cls.size))

    @classmethod
    def validate(cls, data: dict[str, Any]) -> None:
        """Validate once on the CPU batch, before accelerator staging."""
        if any(key not in data for key in cls.keys):
            raise ValueError("Candidate soft SFT requires all four IDs and targets")
        shape = data["input_ids"].shape
        if any(data[key].shape != shape for key in cls.keys):
            raise ValueError("Candidate channels must match input_ids shape")
        mask = data["loss_mask"].bool()
        count = data["candidate_count"]
        if not torch.equal(count == cls.size, mask) or ((count != 0) & ~mask).any():
            raise ValueError("Candidate count must be four exactly at loss positions")
        targets = torch.stack(
            [data[f"candidate_target_{i}"] for i in range(cls.size)], -1
        )
        ids = torch.stack([data[f"candidate_id_{i}"] for i in range(cls.size)], -1)
        if not torch.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("Candidate targets must be finite and nonnegative")
        if (targets[~mask] != 0).any() or not torch.allclose(
            targets[mask].sum(-1),
            torch.ones_like(targets[mask][:, 0]),
            atol=2.1e-4,
            rtol=0,
        ):
            raise ValueError("Candidate targets must sum to one at loss positions")
        active_ids = ids[mask]
        if ids.dtype != torch.int64 or (active_ids < 0).any():
            raise ValueError("Candidate IDs must be nonnegative int64 token IDs")
        if (active_ids.sort(-1).values.diff(dim=-1) == 0).any():
            raise ValueError("Candidate token IDs must be distinct")
        if (ids[~mask] != 0).any():
            raise ValueError("Inactive candidate IDs must be zero")
        if mask.ndim != 2 or (mask.sum(-1) > 1).any() or mask[:, 0].any():
            raise ValueError(
                "Each decision has one non-initial loss token; dummy rows have zero"
            )

    @classmethod
    def readout(
        cls,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        tp_group: dist.ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-expected log p and entropy, in packed token order."""
        ids = torch.stack([inputs[f"candidate_id_{i}"] for i in range(cls.size)], -1)
        targets = torch.stack(
            [inputs[f"candidate_target_{i}"] for i in range(cls.size)], -1
        )
        ids = torch.roll(ids, -1, dims=0)
        targets = torch.roll(targets, -1, dims=0)
        # Never roll the first target of the next sequence across a packed boundary.
        targets[inputs["cu_seqlens"][1:].long() - 1] = 0
        ids = ids.reshape(*logits.shape[:-1], cls.size)
        targets = targets.reshape_as(ids).float()
        partition_size = logits.shape[-1]
        rank = dist.get_rank(tp_group) if tp_group is not None else 0
        start = rank * partition_size
        local_ids = ids - start
        owned = (local_ids >= 0) & (local_ids < partition_size)
        selected = logits.gather(-1, local_ids.clamp(0, partition_size - 1)).float()
        selected = torch.where(owned, selected, 0.0)
        if tp_group is not None:
            selected = _SumCandidateLogits.apply(selected, tp_group)
        logp = selected.log_softmax(-1)
        expected_logp = (targets * logp).sum(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return expected_logp.reshape(-1), entropy.reshape(-1)

    def __call__(
        self,
        logprobs: torch.Tensor,
        entropy: torch.Tensor,
        input_: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        mask = input_["loss_mask"].bool()
        ce = torch.where(mask, -logprobs, 0.0)
        stats_tracker.denominator(candidate_decisions=mask)
        stats_tracker.stat(soft_ce=ce.detach(), denominator="candidate_decisions")
        stats_tracker.stat(
            candidate_entropy=torch.where(mask, entropy.detach(), 0.0),
            denominator="candidate_decisions",
        )
        return ce.sum() / mask.count_nonzero().clamp_min(1)
