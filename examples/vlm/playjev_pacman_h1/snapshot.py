# SPDX-License-Identifier: Apache-2.0

"""Build and verify an immutable-by-convention Pacman H1 experiment snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

SPLITS = ("train", "valid")
SCHEMA_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_image(split_dir: Path, record: dict[str, Any]) -> Path:
    path = (split_dir / record["image"]).resolve()
    if not path.is_relative_to(split_dir.resolve()):
        raise ValueError(f"image path escapes {split_dir}: {record['image']}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _offsets(manifest: Path) -> list[int]:
    offsets = []
    with manifest.open("rb") as stream:
        while line := stream.readline():
            if line.strip():
                offsets.append(stream.tell() - len(line))
    return offsets


def _select_split(
    source: Path,
    output: Path,
    split: str,
    count: int,
    rng: random.Random,
    seen_ids: set[str],
    seen_states: set[str],
    seen_images: set[str],
) -> dict[str, Any]:
    source_dir = source / split
    source_manifest = source_dir / "records.jsonl"
    indices = _offsets(source_manifest)
    if len(indices) < count:
        raise ValueError(f"{split} source has {len(indices)} rows, need {count}")
    rng.shuffle(indices)
    target_dir = output / split
    frames = target_dir / "frames"
    frames.mkdir(parents=True)
    selected_seeds: set[int] = set()
    manifest_digest = hashlib.sha256()
    selected = 0
    with (
        source_manifest.open("rb") as reader,
        (target_dir / "records.jsonl").open("wb") as writer,
    ):
        for offset in indices:
            reader.seek(offset)
            record = json.loads(reader.readline())
            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id.startswith(f"{split}-"):
                raise ValueError(f"invalid {split} sample ID: {record_id}")
            state_hash = _sha256(_canonical_bytes(record["oracle_info"]))
            if record_id in seen_ids or state_hash in seen_states:
                continue
            image_bytes = _record_image(source_dir, record).read_bytes()
            image_hash = _sha256(image_bytes)
            if image_hash in seen_images:
                continue

            image_name = f"frames/{selected:07d}.jpg"
            (target_dir / image_name).write_bytes(image_bytes)
            record = {
                **record,
                "image": image_name,
                "snapshot_split": split,
                "image_sha256": image_hash,
                "oracle_sha256": state_hash,
            }
            line = _canonical_bytes(record) + b"\n"
            writer.write(line)
            manifest_digest.update(line)
            seen_ids.add(record_id)
            seen_states.add(state_hash)
            seen_images.add(image_hash)
            selected_seeds.add(record["seed"])
            selected += 1
            if selected == count:
                break
    if selected != count:
        raise ValueError(
            f"{split} has only {selected} unique samples after ID/state/image deduplication; need {count}"
        )
    return {
        "count": selected,
        "source_count": len(indices),
        "source_manifest_sha256": _file_sha256(source_manifest),
        "manifest_sha256": manifest_digest.hexdigest(),
        "seed_count": len(selected_seeds),
        "seeds": sorted(selected_seeds),
    }


def build_snapshot(
    source: str | Path,
    output: str | Path,
    *,
    train_count: int = 4_000,
    valid_count: int = 256,
    seed: int = 1,
) -> dict[str, Any]:
    """Sample without replacement and materialize separate train/valid roots."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if train_count < 1 or valid_count < 1:
        raise ValueError("train_count and valid_count must be positive")
    if output.exists():
        raise FileExistsError(f"snapshot already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("snapshot output must not be inside the source dataset")
    if not all((source / split / "records.jsonl").is_file() for split in SPLITS):
        raise FileNotFoundError("source requires separate train/ and valid/ manifests")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-", dir=output.parent
    ) as temporary:
        target = Path(temporary)
        seen_ids: set[str] = set()
        seen_states: set[str] = set()
        seen_images: set[str] = set()
        rng = random.Random(seed)
        # Reserve the smaller holdout first so repeated initial screens in the
        # larger train pool cannot consume all validation candidates.
        splits = {
            "valid": _select_split(
                source,
                target,
                "valid",
                valid_count,
                rng,
                seen_ids,
                seen_states,
                seen_images,
            ),
            "train": _select_split(
                source,
                target,
                "train",
                train_count,
                rng,
                seen_ids,
                seen_states,
                seen_images,
            ),
        }
        if set(splits["train"]["seeds"]) & set(splits["valid"]["seeds"]):
            raise ValueError("source train/valid seeds overlap")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "sampling_seed": seed,
            "deduplicated_by": ["id", "oracle_info_sha256", "image_sha256"],
            "splits": splits,
        }
        metadata["snapshot_id"] = _sha256(_canonical_bytes(metadata))[:16]
        (target / "snapshot.json").write_bytes(_canonical_bytes(metadata) + b"\n")
        for split in SPLITS:
            (target / split / "split.json").write_bytes(
                _canonical_bytes(
                    {
                        "split": split,
                        "snapshot_id": metadata["snapshot_id"],
                        "manifest_sha256": splits[split]["manifest_sha256"],
                    }
                )
                + b"\n"
            )
        verify_snapshot(target)
        if output.exists():
            raise FileExistsError(f"snapshot appeared during creation: {output}")
        target.rename(output)
    return metadata


def verify_snapshot(root: str | Path) -> dict[str, Any]:
    """Check fixed split identities, content hashes, and zero train/valid overlap."""
    root = Path(root).resolve()
    metadata = json.loads((root / "snapshot.json").read_bytes())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Pacman H1 snapshot schema")
    expected_id = _sha256(
        _canonical_bytes({k: v for k, v in metadata.items() if k != "snapshot_id"})
    )[:16]
    if metadata.get("snapshot_id") != expected_id:
        raise ValueError("snapshot metadata fingerprint mismatch")
    ids: set[str] = set()
    states: set[str] = set()
    images: set[str] = set()
    split_seeds: dict[str, set[int]] = {}
    for split in SPLITS:
        split_dir = root / split
        split_metadata = json.loads((split_dir / "split.json").read_bytes())
        summary = metadata["splits"][split]
        if split_metadata != {
            "split": split,
            "snapshot_id": expected_id,
            "manifest_sha256": summary["manifest_sha256"],
        }:
            raise ValueError(f"{split} split identity mismatch")
        manifest = split_dir / "records.jsonl"
        if _file_sha256(manifest) != summary["manifest_sha256"]:
            raise ValueError(f"{split} manifest checksum mismatch")
        count = 0
        seeds: set[int] = set()
        with manifest.open("rb") as reader:
            for line in reader:
                record = json.loads(line)
                record_id = record["id"]
                state_hash = _sha256(_canonical_bytes(record["oracle_info"]))
                image_hash = _file_sha256(_record_image(split_dir, record))
                if record["snapshot_split"] != split:
                    raise ValueError(f"{split} contains a foreign sample")
                if record_id in ids or state_hash in states or image_hash in images:
                    raise ValueError(
                        "duplicate ID, oracle state, or image across snapshot"
                    )
                if (
                    state_hash != record["oracle_sha256"]
                    or image_hash != record["image_sha256"]
                ):
                    raise ValueError(f"{split} sample checksum mismatch: {record_id}")
                ids.add(record_id)
                states.add(state_hash)
                images.add(image_hash)
                seeds.add(record["seed"])
                count += 1
        if (
            count != summary["count"]
            or seeds != set(summary["seeds"])
            or len(seeds) != summary["seed_count"]
        ):
            raise ValueError(f"{split} count or seed set mismatch")
        split_seeds[split] = seeds
    if split_seeds["train"] & split_seeds["valid"]:
        raise ValueError("train and valid seeds overlap")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a unique Pacman H1 split")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=4_000)
    parser.add_argument("--valid-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_snapshot(args.output)
    else:
        if args.source is None:
            parser.error("--source is required when building a snapshot")
        result = build_snapshot(
            args.source,
            args.output,
            train_count=args.train_count,
            valid_count=args.valid_count,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "snapshot_id": result["snapshot_id"],
                "train_count": result["splits"]["train"]["count"],
                "valid_count": result["splits"]["valid"]["count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
