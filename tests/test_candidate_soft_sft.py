# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest
import torch

from areal.trainer.sft.candidate_objective import CandidateSoftSFTObjective


def _channels(length: int = 7) -> dict[str, torch.Tensor]:
    data = {
        "input_ids": torch.zeros(1, length, dtype=torch.int64),
        "loss_mask": torch.zeros(1, length, dtype=torch.bool),
        "candidate_count": torch.zeros(1, length, dtype=torch.int64),
    }
    data["loss_mask"][0, 2] = data["loss_mask"][0, -1] = True
    data["candidate_count"][data["loss_mask"]] = 4
    for i, (token, prob) in enumerate(zip([1, 3, 6, 8], [0.05, 0.1, 0.25, 0.6])):
        data[f"candidate_id_{i}"] = torch.zeros(1, length, dtype=torch.int64)
        data[f"candidate_target_{i}"] = torch.zeros(1, length)
        data[f"candidate_id_{i}"][data["loss_mask"]] = token
        data[f"candidate_target_{i}"][data["loss_mask"]] = prob
    return data


def test_soft_readout_packed_sequences_matches_dense_loss_and_gradient():
    # Two decisions of different lengths, each with a carrier at its end.
    data = {key: value[0] for key, value in _channels().items()}
    data["cu_seqlens"] = torch.tensor([0, 3, 7], dtype=torch.int32)
    torch.manual_seed(0)
    logits = torch.randn(7, 10, requires_grad=True)
    expected_logp, entropy = CandidateSoftSFTObjective.readout(logits, data)
    positions = torch.tensor([1, 5])
    target = torch.tensor([0.05, 0.1, 0.25, 0.6])
    reference = (
        -(logits[positions][:, [1, 3, 6, 8]].log_softmax(-1) * target).sum(-1).mean()
    )
    actual = -expected_logp[positions].mean()
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(expected_logp).all() and torch.isfinite(entropy).all()
    assert expected_logp[[2, 6]].eq(0).all()
    assert actual_grad[:, [0, 2, 4, 5, 7, 9]].eq(0).all()
    # Carrier identity is never used as the hard target.
    data["input_ids"].fill_(9)
    changed, _ = CandidateSoftSFTObjective.readout(logits, data)
    torch.testing.assert_close(changed, expected_logp, rtol=0, atol=0)


def test_validation_all_zero_eval_dummy_is_accepted():
    data = _channels()
    for key in data:
        data[key][:, :3] = 0
        data[key] = torch.cat((data[key], torch.zeros_like(data[key])), dim=0)
    CandidateSoftSFTObjective.validate(data)


@pytest.mark.parametrize("invalid", ["negative", "mass", "duplicate", "partial"])
def test_validation_invalid_teacher_channels_are_rejected(invalid):
    data = _channels()
    for key in data:
        data[key][:, :3] = 0
    if invalid == "negative":
        data["candidate_target_0"][0, -1] = -0.1
    elif invalid == "mass":
        data["candidate_target_0"][0, -1] = 1.0
    elif invalid == "duplicate":
        data["candidate_id_0"][0, -1] = data["candidate_id_1"][0, -1]
    else:
        del data["candidate_target_3"]
    with pytest.raises(ValueError):
        CandidateSoftSFTObjective.validate(data)


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_tp_readout_disjoint_vocabulary_matches_dense_loss_and_gradient():
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "tests/torchrun/run_candidate_soft_sft.py",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=120,
    )
