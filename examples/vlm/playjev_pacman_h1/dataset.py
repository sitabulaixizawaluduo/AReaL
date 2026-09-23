# SPDX-License-Identifier: Apache-2.0

"""Lazy image loader for source-labeled Pacman decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class PacmanH1Dataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        if split not in {"train", "valid"}:
            raise ValueError("split must be train or valid")
        self.split = split
        root = Path(root).resolve()
        # Accept either a dedicated split directory or the original collection
        # root, so old source manifests remain readable for inspection.
        self.root = root if (root / "records.jsonl").is_file() else root / split
        manifest = self.root / "records.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        split_metadata = self.root / "split.json"
        if (
            split_metadata.is_file()
            and json.loads(split_metadata.read_bytes())["split"] != split
        ):
            raise ValueError(f"expected {split} data, found a different snapshot split")
        self.manifest = manifest
        self.offsets = []
        with manifest.open("rb") as reader:
            while line := reader.readline():
                if line.strip():
                    self.offsets.append(reader.tell() - len(line))
        if not self.offsets:
            raise ValueError(f"No Pacman H1 samples in {manifest}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.manifest.open("rb") as reader:
            reader.seek(self.offsets[index])
            record = json.loads(reader.readline())
        image_path = (self.root / record["image"]).resolve()
        if not image_path.is_relative_to(self.root):
            raise ValueError("image path escapes the dataset split")
        return {**record, "image": image_path.read_bytes()}
