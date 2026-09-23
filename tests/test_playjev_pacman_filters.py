# SPDX-License-Identifier: Apache-2.0

"""Tests for prompt-group filtering with concat-exported tool trajectories."""

from __future__ import annotations

import torch

from examples.vlm.playjev_pacman_h1.filters import reward_has_variance

from areal.utils.data import RolloutGroup


def test_filter_compares_logical_rollout_rewards_not_tool_rows():
    trajectory = {
        "rollout_group": RolloutGroup((2, 2)),
        "original_rewards": torch.tensor([0.0, 0.8, 0.0, 0.8]),
    }
    assert not reward_has_variance(trajectory)

    trajectory["original_rewards"][3] = 0.5
    assert reward_has_variance(trajectory)


def test_filter_prefers_explicit_rollout_rewards():
    trajectory = {
        "rollout_group": RolloutGroup((2, 2), (0.2, 0.9)),
        "original_rewards": torch.tensor([0.0, 0.4, 0.0, 0.4]),
    }
    assert reward_has_variance(trajectory)
