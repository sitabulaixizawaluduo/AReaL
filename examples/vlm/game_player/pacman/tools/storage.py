# SPDX-License-Identifier: Apache-2.0

"""Strict JSON evidence storage, called from episode worker threads."""

import json
from pathlib import Path
from typing import Any


class JsonEpisodeStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def __call__(self, summary: dict[str, Any]) -> None:
        attempt_id = str(summary["attempt_id"])
        if (
            not attempt_id
            or Path(attempt_id).name != attempt_id
            or attempt_id in {".", ".."}
        ):
            raise ValueError("attempt_id must be a single nonempty path component")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{attempt_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, allow_nan=False) + "\n")
        temporary.replace(path)
