"""Deterministic game task splits and file identity, independent of training."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[int, ...]
    dev: tuple[int, ...]
    test: tuple[int, ...]
    generation_seed: int = 20260916

    def __post_init__(self):
        seeds = self.train + self.dev + self.test
        if tuple(map(len, (self.train, self.dev, self.test))) != (80, 24, 96):
            raise ValueError("Expected exactly 80 train, 24 dev, 96 test seeds")
        if len(seeds) != len(set(seeds)):
            raise ValueError("Seed splits must be disjoint, without duplicates")

    @classmethod
    def create(cls):
        # A fresh seed pool; historical 28..111 and showcase seeds are excluded.
        seeds = list(range(10000, 10200))
        random.Random(20260916).shuffle(seeds)
        return cls(tuple(seeds[:80]), tuple(seeds[80:104]), tuple(seeds[104:]))

    @classmethod
    def read(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        return cls(
            **{k: tuple(data[k]) for k in ("train", "dev", "test")},
            generation_seed=data["generation_seed"],
        )

    def write(self, path: str | Path):
        data = {
            "schema": "pacman-splits-v1",
            "generation_seed": self.generation_seed,
            "train": self.train,
            "dev": self.dev,
            "test": self.test,
        }
        path = Path(path)
        encoded = json.dumps(data, indent=2) + "\n"
        if path.exists():
            if self.read(path) != self:
                raise ValueError(
                    f"Refusing to replace different split manifest: {path}"
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(encoded)

    def rows(self, split: str):
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split: {split}")
        return [
            {"seed": seed, "split": split, "episode_id": f"{split}-{seed}"}
            for seed in getattr(self, split)
        ]


class ArtifactIdentity:
    @staticmethod
    def sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
