# SPDX-License-Identifier: Apache-2.0

"""Tests for reproducible, non-overlapping Pacman experiment snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.vlm.playjev_pacman_h1.dataset import PacmanH1Dataset
from examples.vlm.playjev_pacman_h1.snapshot import build_snapshot, verify_snapshot


def _source_record(root: Path, split: str, index: int, state: int) -> None:
    split_dir = root / split
    (split_dir / "frames").mkdir(parents=True, exist_ok=True)
    image = f"frames/{index:03d}.jpg"
    (split_dir / image).write_bytes(f"image-{state}".encode())
    record = {
        "id": f"{split}-{index}-0-0",
        "seed": index + (1000 if split == "valid" else 0),
        "image": image,
        "oracle_info": {"state": state},
    }
    with (split_dir / "records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def test_snapshot_separates_and_deduplicates_train_valid(tmp_path: Path):
    source = tmp_path / "source"
    _source_record(source, "valid", 0, 0)
    _source_record(source, "valid", 1, 1)
    _source_record(source, "train", 0, 0)
    _source_record(source, "train", 1, 2)
    _source_record(source, "train", 2, 3)
    output = tmp_path / "experiment"

    metadata = build_snapshot(source, output, train_count=2, valid_count=1, seed=7)

    assert metadata["splits"]["train"]["count"] == 2
    assert metadata["splits"]["valid"]["count"] == 1
    assert verify_snapshot(output)["snapshot_id"] == metadata["snapshot_id"]
    assert len(PacmanH1Dataset(output / "train", "train")) == 2
    assert len(PacmanH1Dataset(output / "valid", "valid")) == 1
    with pytest.raises(ValueError, match="different snapshot split"):
        PacmanH1Dataset(output / "train", "valid")
    with pytest.raises(FileExistsError, match="already exists"):
        build_snapshot(source, output, train_count=2, valid_count=1, seed=7)


def test_snapshot_detects_changed_image(tmp_path: Path):
    source = tmp_path / "source"
    _source_record(source, "valid", 0, 0)
    _source_record(source, "train", 1, 1)
    output = tmp_path / "experiment"
    build_snapshot(source, output, train_count=1, valid_count=1)
    record = json.loads(
        (output / "train" / "records.jsonl").read_text().splitlines()[0]
    )
    (output / "train" / record["image"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="sample checksum mismatch"):
        verify_snapshot(output)
