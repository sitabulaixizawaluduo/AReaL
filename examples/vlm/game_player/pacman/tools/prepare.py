"""Prepare only public game sources and game dependencies; no trainer imports."""

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from examples.vlm.game_player.pacman.tools.datasets import (
    ArtifactIdentity,
    SplitManifest,
)


class GamePreparation:
    REPOSITORIES = {
        "areal-pacman": (
            "https://github.com/luzai/areal-pacman.git",
            "a5e54e3d0edae4a760518753c41bf6f844793e5f",
        ),
        "pacman-python": (
            "https://github.com/luzai/pacman-python.git",
            "cbb97115e407abc86a44adc82a1b8f360b3e8da0",
        ),
    }

    @classmethod
    def prepare(cls, root: Path, install_game: bool = False):
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        versions = {}
        for name, (url, revision) in cls.REPOSITORIES.items():
            target = root / "sources" / name
            if not target.exists():
                target.parent.mkdir(exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--no-checkout", url, str(target)], check=True
                )
                subprocess.run(
                    ["git", "-C", str(target), "checkout", "--detach", revision],
                    check=True,
                )
            head = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(target), "status", "--porcelain"], text=True
            ).strip()
            if head != revision or dirty:
                raise ValueError(
                    f"Expected clean {name}@{revision}; preserve your checkout and use a new root"
                )
            versions[name] = {"url": url, "revision": head, "path": str(target)}
        if install_game:
            # Isolated recipe dependencies only; never resolve/upgrade CUDA or torch here.
            installer = (
                ["uv", "pip", "install", "--python", sys.executable]
                if shutil.which("uv")
                else [sys.executable, "-m", "pip", "install"]
            )
            subprocess.run(
                [
                    *installer,
                    "--no-deps",
                    "pygame==2.6.1",
                    "-e",
                    str(root / "sources/areal-pacman"),
                ],
                check=True,
            )
        SplitManifest.create().write(root / "splits.json")
        source_root = Path(__file__).resolve().parents[5]
        env = {
            "PACMAN_ARTIFACT_ROOT": str(root),
            "PACMAN_SPLIT_MANIFEST": str(root / "splits.json"),
            "AREAL_PACMAN_ROOT": str(root / "sources/areal-pacman"),
            "MAAPACMAN_PACMAN_ROOT": str(root / "sources/pacman-python"),
            "PACMAN_WORKER_BASE_DIR": str(root / "workers"),
            "GAME_PLAYER_SOURCE_ROOT": str(source_root),
        }
        (root / "environment.sh").write_text(
            "# Source in the controller and inside every worker image.\n"
            + "\n".join(
                f"export {key}={shlex.quote(value)}" for key, value in env.items()
            )
            + '\nexport PYTHONPATH="$GAME_PLAYER_SOURCE_ROOT:$AREAL_PACMAN_ROOT${PYTHONPATH:+:$PYTHONPATH}"\n'
        )
        (root / "source-lock.json").write_text(json.dumps(versions, indent=2) + "\n")
        return root / "environment.sh"

    @classmethod
    def verify_environment(cls):
        required = [
            "MAAPACMAN_PACMAN_ROOT",
            "AREAL_PACMAN_ROOT",
            "PACMAN_ARTIFACT_ROOT",
        ]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                f"Source the prepared environment.sh first; missing {missing}"
            )
        if sys.version_info < (3, 12):
            raise RuntimeError("The game tools require Python >= 3.12")
        failures = {}
        for name in ("numpy", "pygame", "PIL", "maapacman"):
            try:
                importlib.import_module(name)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
        for name, (_, revision) in cls.REPOSITORIES.items():
            root = Path(
                os.environ[
                    "AREAL_PACMAN_ROOT"
                    if name == "areal-pacman"
                    else "MAAPACMAN_PACMAN_ROOT"
                ]
            )
            identity = cls.describe_source(root)
            if identity["revision"] != revision or identity["dirty"]:
                failures[name] = f"Expected clean source at {revision}"
        module = sys.modules.get("maapacman")
        if module and not Path(module.__file__).resolve().is_relative_to(
            Path(os.environ["AREAL_PACMAN_ROOT"]).resolve()
        ):
            failures["maapacman"] = "Import is shadowed by another installation"
        if failures:
            raise RuntimeError(
                "Game dependency preflight failed:\n" + json.dumps(failures, indent=2)
            )
        return {"python": platform.python_version(), "sources": cls.source_identity()}

    @staticmethod
    def describe_source(root: Path):
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        )
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        untracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]
        ).split(b"\0")
        return {
            "revision": revision,
            "dirty": bool(status.strip()),
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files_sha256": {
                os.fsdecode(relative): ArtifactIdentity.sha256(
                    root / os.fsdecode(relative)
                )
                for relative in untracked
                if relative
                and (root / os.fsdecode(relative)).is_file()
                and (root / os.fsdecode(relative)).suffix
                in {".py", ".sh", ".yaml", ".yml", ".md", ".toml"}
            },
        }

    @classmethod
    def source_identity(cls):
        sources = {
            name: cls.describe_source(Path(os.environ[variable]))
            for name, variable in (
                ("areal-pacman", "AREAL_PACMAN_ROOT"),
                ("pacman-python", "MAAPACMAN_PACMAN_ROOT"),
            )
        }
        player_root = Path(__file__).resolve().parents[2]
        sources["player_sources"] = {
            str(path.relative_to(player_root)): ArtifactIdentity.sha256(path)
            for path in sorted(player_root.rglob("*.py"))
            if "train" not in path.relative_to(player_root).parts
        }
        return sources
