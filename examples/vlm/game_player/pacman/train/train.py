"""Single-controller entrypoint; use the existing local or Slurm scheduler."""

import json
import sys
from datetime import UTC
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.vlm.game_player.pacman.tools.datasets import SplitManifest
from examples.vlm.game_player.pacman.train.config import PacmanConfig
from examples.vlm.game_player.pacman.train.prepare import RuntimePreparation

from areal.api.cli_args import load_expr_config
from areal.trainer import PPOTrainer
from areal.utils import checkpoint_pointer
from areal.utils.saver import Saver


def main(args: list[str]) -> None:
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = ["--config", str(Path(__file__).with_name("pacman_rl.yaml")), *args]
    config, _ = load_expr_config(args, PacmanConfig)
    preparation = RuntimePreparation.verify_environment()
    RuntimePreparation.verify_worker_image(config)
    provenance = Path(config.run_artifact_root) / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    (provenance / f"{stamp}.json").write_text(json.dumps(preparation, indent=2) + "\n")
    splits = SplitManifest.read(config.split_manifest)
    if config.require_recovery:
        if config.recover.mode in ("off", "disabled"):
            raise ValueError("recover command requires recover.mode=auto")
        root = Saver.get_save_root(
            config.experiment_name, config.trial_name, config.cluster.fileroot
        )
        source = checkpoint_pointer.resolve_checkpoint(root, ["default"])
        if source is None:
            raise FileNotFoundError(f"No complete recovery generation under {root}")
        info = json.loads((Path(source.manifest) / "step_info.json").read_text())
        if info["global_step"] < 0:
            raise ValueError("Recovery checkpoint has no completed optimizer update")
    from datasets import Dataset

    train_data = Dataset.from_list(splits.rows("train"))
    dev_data = Dataset.from_list(splits.rows("dev"))
    kwargs = dict(
        gconfig=config.gconfig,
        model=config.actor.path,
        options=config.workflow_options(),
    )
    eval_kwargs = dict(
        gconfig=config.eval_gconfig,
        model=config.actor.path,
        options={**config.workflow_options(), "is_eval": True},
    )
    with PPOTrainer(
        config, train_dataset=train_data, valid_dataset=dev_data
    ) as trainer:
        trainer.train(
            workflow="examples.vlm.game_player.pacman.train.agent.PacmanAgent",
            workflow_kwargs=kwargs,
            eval_workflow="examples.vlm.game_player.pacman.train.agent.PacmanAgent",
            eval_workflow_kwargs=eval_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
