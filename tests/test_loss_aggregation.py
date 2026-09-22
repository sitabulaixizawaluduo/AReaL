# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import _build_rollout_mean_weights, grpo_loss_fn
from areal.utils.data import (
    RolloutGroup,
    TrajBatchMeta,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional.loss_aggregation import PolicyGradientReduction

LOSS = torch.tensor(
    [
        [1.0, 3.0, 0.0, 0.0],
        [2.0, 4.0, 6.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ]
)
MASK = torch.tensor(
    [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ],
    dtype=torch.bool,
)
PROMPT_GROUP_SIZES = [2, 1]


def _reduction(mode: str) -> PolicyGradientReduction:
    return PolicyGradientReduction(
        mode=mode,
        divisor=4.0 if mode == "constant" else None,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token_mean", 13 / 3),
        ("seq_mean", 16 / 3),
        ("prompt_mean", 33 / 5),
        ("constant", 13 / 6),
    ],
)
def test_policy_gradient_reduction_matches_definition(mode, expected):
    group_sizes = PROMPT_GROUP_SIZES if mode == "prompt_mean" else None

    actual = _reduction(mode).aggregate(
        LOSS,
        MASK,
        group_sizes=group_sizes,
    )

    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=1e-6)


def test_rollout_mean_averages_rows_then_logical_rollouts():
    meta = TrajBatchMeta(
        n_trajs=1,
        traj_group_sizes=[3],
        traj_seqlens=[4],
        rollout_groups=[RolloutGroup((2, 1))],
    )
    weights = _build_rollout_mean_weights(MASK, meta)

    actual = _reduction("rollout_mean").aggregate(
        LOSS,
        MASK,
        weights=weights,
    )

    expected_weights = torch.tensor(
        [
            [0.25, 0.25, 0.0, 0.0],
            [1 / 6, 1 / 6, 1 / 6, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, torch.tensor(6.5), rtol=1e-6, atol=1e-6)


def test_rollout_mean_combines_partial_rollout_microbatches_exactly():
    meta = TrajBatchMeta(
        n_trajs=1,
        traj_group_sizes=[3],
        traj_seqlens=[4],
        rollout_groups=[RolloutGroup((2, 1))],
    )
    reduction = _reduction("rollout_mean")
    weights = _build_rollout_mean_weights(MASK, meta)
    weighted_losses = []
    normalizers = []

    for row_slice in (slice(0, 1), slice(1, 3)):
        mb = {
            "loss_mask": MASK[row_slice],
            "loss_aggregation_weights": weights[row_slice],
        }
        normalizer = reduction.normalizer_fn(mb)
        local_loss = reduction.aggregate(
            LOSS[row_slice],
            MASK[row_slice],
            weights=weights[row_slice],
        )
        weighted_losses.append(local_loss * normalizer)
        normalizers.append(normalizer)

    combined = torch.stack(weighted_losses).sum() / torch.stack(normalizers).sum()
    full = reduction.aggregate(LOSS, MASK, weights=weights)
    torch.testing.assert_close(combined, full, rtol=1e-6, atol=1e-6)


def test_rollout_mean_requires_explicit_rollout_metadata():
    meta = TrajBatchMeta(
        n_trajs=1,
        traj_group_sizes=[1],
        traj_seqlens=[2],
        rollout_groups=[None],
    )

    with pytest.raises(ValueError, match="requires RolloutGroup metadata"):
        _build_rollout_mean_weights(torch.ones(1, 2, dtype=torch.bool), meta)


def test_prompt_mean_microbatch_split_keeps_groups_atomic():
    data = {
        "attention_mask": MASK,
        "input_ids": torch.arange(12).view(3, 4),
        "loss_mask": MASK,
        "group_sizes": PROMPT_GROUP_SIZES,
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=2, granularity=1),
    )

    actual_group_sizes = [tuple(mb["group_sizes"]) for mb in mb_list.mbs]
    assert sorted(actual_group_sizes) == [(1,), (2,)]
    assert sorted(mb["attention_mask"].shape[0] for mb in mb_list.mbs) == [1, 2]


def test_token_mean_preserves_existing_dtype_and_formula():
    loss = LOSS.to(torch.bfloat16)
    expected = torch.where(MASK, loss, 0).sum() / MASK.count_nonzero()

    actual = _reduction("token_mean").aggregate(loss, MASK)

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_token_mean_m2_mask_controls_numerator_and_denominator():
    input_data = {
        "input_ids": torch.tensor([[11, 12]]),
        "logprobs": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "prox_logp": torch.zeros(1, 2),
    }
    filtered_mask = torch.tensor([[1, 0]], dtype=torch.bool)

    with (
        patch(
            "areal.trainer.ppo.actor._apply_m2po_masking",
            return_value=filtered_mask,
        ),
        patch("areal.trainer.ppo.actor.stats_tracker"),
    ):
        loss = grpo_loss_fn(
            logprobs=torch.zeros(1, 2),
            entropy=torch.zeros(1, 2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            m2_threshold=0.1,
        )

    torch.testing.assert_close(loss, torch.tensor(-1.0), rtol=0, atol=0)


def test_loss_aggregation_config_defaults_and_validation():
    assert PPOActorConfig().loss_aggregation == "token_mean"
    assert OmegaConf.structured(PPOActorConfig).loss_aggregation == "token_mean"
    PPOActorConfig(loss_aggregation="rollout_mean")

    with pytest.raises(ValueError, match="loss_aggregation must be"):
        PPOActorConfig(loss_aggregation="bogus")
    with pytest.raises(ValueError, match="loss_aggregation_divisor"):
        PPOActorConfig(loss_aggregation="constant")
    with pytest.raises(ValueError, match="only used"):
        PPOActorConfig(loss_aggregation="rollout_mean", loss_aggregation_divisor=10)
