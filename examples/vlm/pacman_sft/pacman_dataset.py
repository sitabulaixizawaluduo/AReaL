# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from areal.dataset.tokenization import get_multimodal_sft_loss_mask
from areal.utils import logging

logger = logging.getLogger("PacmanSFTDataset")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
QUESTION = "Which move should the player make next?"
ACTION_DESCRIPTIONS = {
    "up": "head up at the next opening",
    "down": "head down at the next opening",
    "left": "head left at the next opening",
    "right": "head right at the next opening",
}


@dataclass(frozen=True, slots=True)
class PacmanDecision:
    frame_path: Path
    actions: tuple[str, ...]
    teacher_action: int
    seed: int
    sample_id: int


def _image_token(processor) -> str:
    image_processor = getattr(processor, "image_processor", None)
    processor_type = getattr(image_processor, "image_processor_type", "").lower()
    if "qwen" in processor_type:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    if "gemma3" in processor_type:
        return processor.boi_token
    token = getattr(processor, "image_token", None)
    if token is None:
        raise ValueError(
            f"Cannot determine the image token for processor {type(processor).__name__}"
        )
    return token


def render_question(actions: Sequence[str]) -> str:
    """Render the action-choice portion shared by training and service evaluation."""
    options = []
    for letter, action in zip(LETTERS[: len(actions)], actions, strict=True):
        description = ACTION_DESCRIPTIONS.get(action)
        option = f"{letter}. {action}"
        if description:
            option += f": {description}"
        options.append(option)
    allowed_letters = ", ".join(LETTERS[: len(actions)])
    options_text = "\n".join(options)
    return (
        f"Question: {QUESTION}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with one letter: {allowed_letters}.\n"
        "Answer:"
    )


def _render_prompt(actions: Sequence[str], image_token: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"<state>\n{image_token}\n</state>\n\n"
        f"{render_question(actions)}"
    )


def _permuted_actions_and_answer(
    decision: PacmanDecision,
    *,
    shuffle_options: bool,
    shuffle_seed: int,
) -> tuple[list[str], str]:
    permutation = list(range(len(decision.actions)))
    if shuffle_options:
        random.Random(shuffle_seed + decision.sample_id).shuffle(permutation)
    actions = [decision.actions[i] for i in permutation]
    answer_position = permutation.index(decision.teacher_action)
    return actions, LETTERS[answer_position]


def _load_rgb_image(frame_path: Path) -> Image.Image:
    try:
        with Image.open(frame_path) as source_image:
            return source_image.convert("RGB")
    except OSError as exc:
        raise OSError(f"Cannot load Pacman frame {frame_path}") from exc


def _is_selected_split(
    seed: int,
    split: str,
    split_modulus: int,
    validation_remainder: int,
) -> bool:
    is_validation = seed % split_modulus == validation_remainder
    if split == "train":
        return not is_validation
    if split in {"validation", "valid", "test"}:
        return is_validation
    raise ValueError(
        f"Unsupported Pacman split {split!r}; use train, validation, valid, or test"
    )


def load_pacman_decisions(
    root: Path,
    split: str,
    split_modulus: int,
    validation_remainder: int,
    shards: Sequence[str] | None,
) -> list[PacmanDecision]:
    """Load validated Pacman decisions while preserving their global sample IDs."""
    if split_modulus <= 1:
        raise ValueError("split_modulus must be greater than 1")
    if not 0 <= validation_remainder < split_modulus:
        raise ValueError(
            "validation_remainder must be in [0, split_modulus), got "
            f"{validation_remainder} for modulus {split_modulus}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Pacman data root does not exist: {root}")

    requested_shards = set(shards) if shards is not None else None
    record_files = sorted(root.glob("*/records.jsonl"))
    if requested_shards is not None:
        record_files = [p for p in record_files if p.parent.name in requested_shards]
    if not record_files:
        raise FileNotFoundError(f"No Pacman records.jsonl files found under {root}")

    decisions: list[PacmanDecision] = []
    sample_id = 0
    for record_file in record_files:
        with record_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if row.get("game") != "pacman":
                    raise ValueError(
                        f"{record_file}:{line_number} is not a Pacman record"
                    )
                seed = int(row["seed"])
                if not _is_selected_split(
                    seed,
                    split=split,
                    split_modulus=split_modulus,
                    validation_remainder=validation_remainder,
                ):
                    sample_id += 1
                    continue

                actions = tuple(str(action) for action in row["actions"])
                if not 1 <= len(actions) <= len(LETTERS):
                    raise ValueError(
                        f"{record_file}:{line_number} has {len(actions)} actions; "
                        f"expected between 1 and {len(LETTERS)}"
                    )
                teacher_action = int(row["teacher_action"])
                if not 0 <= teacher_action < len(actions):
                    raise ValueError(
                        f"{record_file}:{line_number} has teacher_action="
                        f"{teacher_action} for {len(actions)} actions"
                    )
                frame_path = record_file.parent / row["frame"]
                decisions.append(
                    PacmanDecision(
                        frame_path=frame_path,
                        actions=actions,
                        teacher_action=teacher_action,
                        seed=seed,
                        sample_id=sample_id,
                    )
                )
                sample_id += 1
    return decisions


class PacmanSFTDataset(Dataset[dict[str, Any]]):
    """Lazy Pacman frame preprocessing for AReaL's standard VLM SFT trainer."""

    def __init__(
        self,
        decisions: Sequence[PacmanDecision],
        processor,
        max_length: int | None,
        shuffle_options: bool,
        shuffle_seed: int,
    ) -> None:
        self.decisions = list(decisions)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.shuffle_seed = shuffle_seed
        self.image_token = _image_token(processor)
        if self.tokenizer.eos_token is None:
            raise ValueError("The tokenizer must define eos_token for SFT")

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        decision = self.decisions[index]
        actions, answer = _permuted_actions_and_answer(
            decision,
            shuffle_options=self.shuffle_options,
            shuffle_seed=self.shuffle_seed,
        )
        prompt = _render_prompt(actions, self.image_token)
        sequence = f"{prompt} {answer}{self.tokenizer.eos_token}"
        image = _load_rgb_image(decision.frame_path)

        processed = self.processor(
            text=[sequence],
            images=[image],
            padding=False,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=False,
        )
        input_ids = processed["input_ids"].squeeze(0)
        if self.max_length is not None and len(input_ids) > self.max_length:
            raise ValueError(
                f"Pacman sample {decision.frame_path} has {len(input_ids)} tokens, "
                f"exceeding max_length={self.max_length}"
            )

        multi_modal_input = {"pixel_values": processed["pixel_values"]}
        if "image_grid_thw" in processed:
            multi_modal_input["image_grid_thw"] = processed["image_grid_thw"]

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "loss_mask": get_multimodal_sft_loss_mask(
                input_ids=input_ids,
                prompt=prompt,
                sequence=sequence,
                tokenizer=self.tokenizer,
            ),
            "multi_modal_input": [multi_modal_input],
        }
        token_type_ids = processed.get("token_type_ids")
        if token_type_ids is not None:
            sample["mm_token_type_ids"] = token_type_ids.squeeze(0)
        return sample


class PacmanRLDataset(Dataset[dict[str, Any]]):
    """Lazy single-frame Pacman prompts for VisionRLVRWorkflow."""

    def __init__(
        self,
        decisions: Sequence[PacmanDecision],
        processor,
        shuffle_options: bool,
        shuffle_seed: int,
    ) -> None:
        self.decisions = list(decisions)
        self.image_token = _image_token(processor)
        self.shuffle_options = shuffle_options
        self.shuffle_seed = shuffle_seed

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        decision = self.decisions[index]
        actions, expected_letter = _permuted_actions_and_answer(
            decision,
            shuffle_options=self.shuffle_options,
            shuffle_seed=self.shuffle_seed,
        )
        return {
            "messages": _render_prompt(actions, self.image_token),
            "images": [_load_rgb_image(decision.frame_path)],
            "expected_letter": expected_letter,
            "actions": actions,
            "seed": decision.seed,
            "sample_id": decision.sample_id,
            "frame": str(decision.frame_path),
        }


def get_pacman_sft_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
    split_modulus: int = 10,
    validation_remainder: int = 0,
    shards: Sequence[str] | None = None,
    max_samples: int | None = None,
    subset_seed: int = 1,
    shuffle_options: bool = True,
    shuffle_seed: int = 1,
) -> PacmanSFTDataset:
    root = Path(path).expanduser().resolve()
    decisions = load_pacman_decisions(
        root=root,
        split=split,
        split_modulus=int(split_modulus),
        validation_remainder=int(validation_remainder),
        shards=shards,
    )
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        rng = random.Random(int(subset_seed))
        rng.shuffle(decisions)
        decisions = decisions[:max_samples]
    if not decisions:
        raise ValueError(f"Pacman split {split!r} selected no records from {root}")

    logger.info(
        f"Loaded {len(decisions)} Pacman {split} decisions from {root} "
        f"(seed % {split_modulus} {'!=' if split == 'train' else '=='} "
        f"{validation_remainder})"
    )
    return PacmanSFTDataset(
        decisions=decisions,
        processor=processor,
        max_length=max_length,
        shuffle_options=bool(shuffle_options),
        shuffle_seed=int(shuffle_seed),
    )


def get_pacman_rl_dataset(
    path: str,
    split: str,
    processor,
    split_modulus: int = 10,
    validation_remainder: int = 0,
    shards: Sequence[str] | None = None,
    max_samples: int | None = None,
    subset_seed: int = 1,
    shuffle_options: bool = True,
    shuffle_seed: int = 1,
) -> PacmanRLDataset:
    """Build the offline single-frame dataset used by Pacman GRPO."""
    root = Path(path).expanduser().resolve()
    decisions = load_pacman_decisions(
        root=root,
        split=split,
        split_modulus=int(split_modulus),
        validation_remainder=int(validation_remainder),
        shards=shards,
    )
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        rng = random.Random(int(subset_seed))
        rng.shuffle(decisions)
        decisions = decisions[:max_samples]
    if not decisions:
        raise ValueError(f"Pacman split {split!r} selected no records from {root}")

    logger.info(
        f"Loaded {len(decisions)} Pacman {split} RL decisions from {root} "
        f"(seed % {split_modulus} {'!=' if split == 'train' else '=='} "
        f"{validation_remainder})"
    )
    return PacmanRLDataset(
        decisions=decisions,
        processor=processor,
        shuffle_options=bool(shuffle_options),
        shuffle_seed=int(shuffle_seed),
    )
