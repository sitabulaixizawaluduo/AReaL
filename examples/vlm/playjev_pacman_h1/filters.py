# SPDX-License-Identifier: Apache-2.0

"""Accept only GRPO prompt groups with distinct logical rollout rewards."""

from __future__ import annotations

import math
from typing import Any


def reward_has_variance(trajectory: dict[str, Any]) -> bool:
    """Filter equal-reward groups before actor-side group normalization.

    A concat-exported rollout can occupy several physical rows. Compare one
    terminal reward per logical rollout, never every row independently.
    """
    group = trajectory.get("rollout_group")
    if group is None:
        return False
    rewards = list(group.rewards)
    if any(reward is None for reward in rewards):
        row_rewards = trajectory.get("original_rewards")
        if row_rewards is None:
            return False
        row_rewards = row_rewards.float().reshape(-1).tolist()
        rewards = []
        offset = 0
        for count in group.row_counts:
            offset += count
            if offset > len(row_rewards):
                return False
            rewards.append(row_rewards[offset - 1])
        if offset != len(row_rewards):
            return False
    return (
        len(rewards) >= 2
        and all(math.isfinite(value) for value in rewards)
        and (max(rewards) - min(rewards) > 1e-6)
    )
