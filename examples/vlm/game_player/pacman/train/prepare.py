"""AReaL training preflight; game preparation remains independently usable."""

import importlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.vlm.game_player.pacman.tools.prepare import GamePreparation


class RuntimePreparation(GamePreparation):
    @classmethod
    def verify_environment(cls):
        report = GamePreparation.verify_environment()
        if os.getenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "0") != "0":
            raise RuntimeError(
                "SGLANG_RETURN_ORIGINAL_LOGPROB must be 0 for temperature-aligned PPO"
            )
        failures = {}
        modules = (
            "areal",
            "torch",
            "transformers",
            "datasets",
            "openai",
            "sglang",
            "megatron.core",
            "megatron.bridge",
            "awex",
        )
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
        if failures:
            raise RuntimeError(
                "Training dependency preflight failed before allocation:\n"
                + json.dumps(failures, indent=2)
            )
        return {
            **report,
            "modules": modules,
            "gpu_validation": "not performed",
            "sources": cls.source_identity(),
        }

    @classmethod
    def source_identity(cls):
        import areal

        sources = GamePreparation.source_identity()
        sources["AReaL"] = cls.describe_source(
            Path(areal.__file__).resolve().parents[1]
        )
        sources["training_packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sglang")
        }
        return sources

    @classmethod
    def verify_worker_image(cls, config):
        """Run dependency checks inside the actual image before allocating workers."""
        if config.scheduler.type != "slurm":
            return None
        import inspect

        import areal
        from areal.infra.scheduler.slurm import SlurmScheduler

        mounts = (
            inspect.signature(SlurmScheduler.__init__)
            .parameters["container_mounts"]
            .default
        )
        bindings = [item.split(":") for item in (mounts or "").split(",") if item]
        if not bindings or any(
            len(item) != 2 or item[0] != item[1] for item in bindings
        ):
            raise ValueError(
                "Preflight requires existing scheduler identity mounts; inspect updated Slurm defaults"
            )
        shared_roots = [Path(item[0]).resolve() for item in bindings]
        paths = (
            Path(__file__).resolve().parents[5],
            Path(areal.__file__).resolve().parents[1],
            Path(config.actor.path),
            Path(config.artifact_root),
            Path(config.run_artifact_root),
            Path(config.cluster.fileroot),
            Path(config.split_manifest),
            Path(config.worker_base_dir),
            Path(os.environ["AREAL_PACMAN_ROOT"]),
            Path(config.pacman_python_root),
        )
        if any(
            not path.is_absolute()
            or not any(path.resolve().is_relative_to(shared) for shared in shared_roots)
            for path in paths
        ):
            raise ValueError(
                f"Current Slurm scheduler mounts {mounts}; place code, sources, model and artifacts inside these shared paths"
            )
        image = os.environ.get("PACMAN_WORKER_IMAGE")
        python = os.environ.get("PACMAN_WORKER_PYTHON")
        if not image or not python or not Path(python).is_absolute():
            raise ValueError(
                "Set PACMAN_WORKER_IMAGE and the absolute image interpreter PACMAN_WORKER_PYTHON"
            )
        for spec in config.actor.scheduling_spec:
            if spec.image != image or shlex.split(spec.cmd)[0] != python:
                raise ValueError(
                    "Actor worker image/interpreter differs from the environment selected for preflight"
                )
        import shutil

        active_image = os.getenv("SINGULARITY_CONTAINER") or os.getenv(
            "APPTAINER_CONTAINER"
        )
        if (
            active_image
            and Path(active_image).resolve() == Path(image).resolve()
            and Path(sys.executable).resolve() == Path(python).resolve()
        ):
            report = {
                "image": image,
                "python": python,
                "same_running_image": True,
                "environment": cls.verify_environment(),
            }
            (Path(config.artifact_root) / "worker-image-preflight.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            return report
        executable = shutil.which("singularity")
        if executable is None:
            raise RuntimeError(
                "Slurm controller needs singularity for pre-allocation image preflight"
            )
        command = (
            "source "
            + shlex.quote(str(Path(config.artifact_root) / "environment.sh"))
            + " && exec "
            + shlex.quote(python)
            + " "
            + shlex.quote(str(Path(__file__).resolve()))
        )
        completed = subprocess.run(
            [
                executable,
                "exec",
                "--no-home",
                "--writable-tmpfs",
                "--nv",
                "--bind",
                mounts,
                image,
                "bash",
                "-c",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        report = {
            "image": image,
            "python": python,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "gpu_validation": "not performed",
        }
        destination = Path(config.artifact_root) / "worker-image-preflight.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        return report


if __name__ == "__main__":
    print(json.dumps(RuntimePreparation.verify_environment(), indent=2))
