# SPDX-License-Identifier: Apache-2.0

"""Tests for loading locally collected Pacman decision records."""

from __future__ import annotations

import json
from pathlib import Path

from examples.vlm.playjev_pacman_h1.dataset import PacmanH1Dataset


def test_dataset_loads_image_and_private_oracle(tmp_path: Path):
    split = tmp_path / "train"
    (split / "frames").mkdir(parents=True)
    (split / "frames" / "000.jpg").write_bytes(b"jpeg")
    record = {
        "image": "frames/000.jpg",
        "oracle_info": {"pac": {"x": 9.0}},
        "teacher_action": "left",
    }
    (split / "records.jsonl").write_text(json.dumps(record) + "\n")

    dataset = PacmanH1Dataset(tmp_path, "train")
    assert len(dataset) == 1
    assert dataset[0]["image"] == b"jpeg"
    assert dataset[0]["oracle_info"]["pac"]["x"] == 9.0
