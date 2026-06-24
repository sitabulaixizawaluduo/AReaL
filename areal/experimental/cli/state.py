# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def areal_home() -> Path:
    """Return the AReaL CLI home directory.

    Resolves ``$AREAL_HOME`` if set, otherwise ``~/.areal``. The directory
    is created on first access so callers can mkdir-then-write subdirs
    without an explicit setup step.
    """

    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Writes to ``path + .tmp`` first then renames into place, so partial
    writes never leave a half-formed file on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(data, indent=indent) + "\n")
    os.replace(tmp, path)
