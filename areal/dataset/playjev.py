# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from areal.trainer.sft.candidate_objective import CandidateSoftSFTObjective


class PlayJevPacmanDataset(Dataset):
    """Adapt upstream single-frame Pacman SFT samples to AReaL token channels.

    Keep PlayJev's seed split, option permutation and plain prompt verbatim.
    The upstream checkout is an explicit external input, not a package dependency.
    """

    def __init__(
        self,
        path: str,
        split: str,
        processor: Any,
        max_length: int | None = None,
        *,
        playjev_root: str,
        shards: list[str] | None = None,
        seed: int = 0,
        limit: int = 0,
        boost_last: Sequence[int] | None = None,
    ) -> None:
        if boost_last is not None:
            if (
                not isinstance(boost_last, Sequence)
                or len(boost_last) != 2
                or any(type(value) is not int or value <= 0 for value in boost_last)
            ):
                raise ValueError(
                    "boost_last must contain two positive integers: [last_k, times]"
                )
            boost_last = (boost_last[0], boost_last[1])
        root = Path(playjev_root).expanduser().resolve()
        if not (root / "playjev" / "data.py").is_file():
            raise FileNotFoundError(f"Not a PlayJev checkout: {root}")
        sys.path.insert(0, str(root))
        upstream = importlib.import_module("playjev.data")
        model = importlib.import_module("playjev.model")
        if Path(upstream.__file__).resolve().parents[1] != root:
            raise ValueError("A different PlayJev checkout is already imported")
        if split not in ("train", "validation"):
            raise ValueError("Pacman split must be train or validation")
        records = upstream.load_records(
            ["pacman"],
            data_root=Path(path).expanduser(),
            shards=shards or [],
            boost_last=boost_last,
        )
        train, val = upstream.split_records(records)
        selected = train if split == "train" else val
        if limit < 0:
            raise ValueError("Dataset limit must be nonnegative")
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError(
                f"Empty {split} split; collect multiple seeds (val: seed % 10 == 0)"
            )
        if any(
            len(r.names) != 4 or set(r.names) != {"up", "down", "left", "right"}
            for r in selected
        ):
            raise ValueError("Pacman SFT expects all four directional actions")
        self.samples = upstream.SFTDataset(selected, train=split == "train", seed=seed)
        self.processor = processor
        self.build_prompt = model.build_plain_prompt
        self.max_length = max_length
        ids = [
            processor.tokenizer.encode(f" {letter}", add_special_tokens=False)
            for letter in "ABCD"
        ]
        if (
            any(len(tokens) != 1 for tokens in ids)
            or len({tokens[0] for tokens in ids}) != 4
        ):
            raise ValueError(
                "Tokenizer must encode space-prefixed A/B/C/D as distinct single tokens"
            )
        self.slot_ids = torch.tensor([tokens[0] for tokens in ids], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        prompt = self.build_prompt(item["options"], item["instructions"])
        enc = self.processor(
            text=[prompt], images=[item["image"]], padding=False, return_tensors="pt"
        )
        input_ids = torch.cat((enc["input_ids"][0], self.slot_ids[:1]))
        length = input_ids.numel()
        if self.max_length is not None and length > self.max_length:
            raise ValueError(
                f"Pacman sample {index} has {length} tokens, exceeds max_length={self.max_length}; "
                "increase the token budget rather than truncating the image or answer position"
            )
        target = torch.tensor(item["target"], dtype=torch.float32)
        if target.shape != (4,):
            raise ValueError(f"Expected four teacher probabilities at sample {index}")
        if not torch.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
            raise ValueError(f"Invalid teacher distribution at sample {index}")
        # Preserve the collector's rounded soft targets exactly, as upstream does.
        loss_mask = torch.zeros(length, dtype=torch.bool)
        loss_mask[-1] = True
        sample = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "multi_modal_input": [
                {
                    key: value
                    for key, value in enc.items()
                    if key in ("pixel_values", "image_grid_thw")
                }
            ],
            "candidate_count": torch.zeros(length, dtype=torch.int64),
        }
        sample["candidate_count"][-1] = 4
        for i in range(4):
            sample[f"candidate_id_{i}"] = torch.zeros(length, dtype=torch.int64)
            sample[f"candidate_id_{i}"][-1] = self.slot_ids[i]
            sample[f"candidate_target_{i}"] = torch.zeros(length, dtype=torch.float32)
            sample[f"candidate_target_{i}"][-1] = target[i]
        token_types = enc.get("mm_token_type_ids", enc.get("token_type_ids"))
        if token_types is not None:
            sample["mm_token_type_ids"] = torch.cat(
                (token_types[0], token_types.new_zeros(1))
            )
        CandidateSoftSFTObjective.validate(
            {
                key: value.unsqueeze(0)
                for key, value in sample.items()
                if key != "multi_modal_input"
            }
        )
        return sample
