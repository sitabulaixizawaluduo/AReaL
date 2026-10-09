# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist

from areal.trainer.sft.candidate_objective import CandidateSoftSFTObjective


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    try:
        assert dist.get_world_size(group) == 2
        device = torch.device("cuda", local_rank)
        torch.manual_seed(7)
        dense = torch.randn(7, 10, device=device, requires_grad=True)
        rank = dist.get_rank(group)
        local = (
            dense.detach()[:, rank * 5 : (rank + 1) * 5].clone().requires_grad_(True)
        )
        inputs = {
            "cu_seqlens": torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
        }
        target = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
        tokens = [1, 3, 6, 8]
        for i, token in enumerate(tokens):
            inputs[f"candidate_id_{i}"] = torch.zeros(
                7, dtype=torch.int64, device=device
            )
            inputs[f"candidate_target_{i}"] = torch.zeros(7, device=device)
            inputs[f"candidate_id_{i}"][[2, 6]] = token
            inputs[f"candidate_target_{i}"][[2, 6]] = target[i]
        expected_logp, entropy = CandidateSoftSFTObjective.readout(local, inputs, group)
        positions = torch.tensor([1, 5], device=device)
        reference_logp = dense[positions][:, tokens].log_softmax(-1)
        reference_loss = -(reference_logp * target).sum(-1).mean()
        actual_loss = -expected_logp[positions].mean()
        torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            entropy[positions],
            -(reference_logp.exp() * reference_logp).sum(-1),
            rtol=1e-6,
            atol=1e-6,
        )
        actual_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(
            local.grad, dense.grad[:, rank * 5 : (rank + 1) * 5], rtol=1e-6, atol=1e-6
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
