"""Episode-level game metrics; collection is not optimizer acceptance."""

import json
from collections import Counter, defaultdict
from pathlib import Path


class EpisodeMetrics:
    """Measure game outcomes; collected artifacts do not prove PPO acceptance."""

    @staticmethod
    def summarize(plan: list[dict], records: list[dict]) -> dict:
        expected = {row["episode_id"] for row in plan}
        ids = [row["episode_id"] for row in records]
        if len(expected) != len(plan) or len(ids) != len(set(ids)):
            raise ValueError("Duplicate planned or recorded episode IDs")
        if set(ids) - expected:
            raise ValueError("Records contain unplanned episode IDs")
        complete = [row for row in records if row.get("status") == "complete"]
        known = [row for row in complete if row.get("death_count") is not None]
        n = len(plan)
        wins = sum(bool(row.get("win")) for row in complete)
        zero = sum(bool(row.get("win")) and row["death_count"] == 0 for row in known)
        reasons = Counter(row.get("terminal_reason", "unknown") for row in records)
        reasons["pending"] += n - len(records)
        means = {}
        for name in (
            "game_score",
            "normal_pellet_clear_rate",
            "special_pellet_clear_rate",
            "total_shaped_reward",
            "env_steps",
            "decisions",
            "elapsed_seconds",
            "death_count",
            "ghosts_eaten",
            "planner_hint_rate",
            "planner_recommendation_match_rate",
            "wall_collision_count",
            "wall_collision_rate",
        ):
            values = [row[name] for row in complete if row.get(name) is not None]
            means[f"mean_{name}"] = sum(values) / len(values) if values else None
            means[f"{name}_observations"] = len(values)
        decision_rows = [
            decision for row in complete for decision in row.get("decision_records", [])
        ]
        action_metrics = {}
        for name in (
            "format_valid",
            "action_legal",
            "planner_recommendation_match",
            "wall_collision",
            "safe_advice_match",
        ):
            values = [
                decision[name]
                for decision in decision_rows
                if isinstance(decision.get(name), bool)
            ]
            action_metrics[f"{name}_decision_pct"] = (
                100 * sum(values) / len(values) if values else None
            )
            action_metrics[f"{name}_decisions_observed"] = len(values)
        components = sorted(
            {key for row in complete for key in row.get("reward_components", {})}
        )
        return {
            "planned_episodes": n,
            "recorded_episodes": len(records),
            "completed_episodes": len(complete),
            "pending_episodes": n - len(records),
            "wins": wins,
            "zero_death_wins": zero,
            "win_pct": 100 * wins / n if n else None,
            "zero_death_win_pct": 100 * zero / n if n else None,
            "death_count_unknown_episodes": len(complete) - len(known),
            "terminal_reasons": dict(reasons),
            "terminal_reason_pct": {
                key: 100 * count / n if n else None for key, count in reasons.items()
            },
            "invalid_format_episode_pct": 100
            * sum(bool(row.get("invalid_format")) for row in complete)
            / n
            if n
            else None,
            "invalid_action_episode_pct": 100
            * sum(bool(row.get("invalid_action")) for row in complete)
            / n
            if n
            else None,
            "death_limit_failure_pct": 100
            * sum(row["death_count"] >= 3 and not row.get("win") for row in known)
            / n
            if n
            else None,
            "policy_decisions": sum(row.get("decisions", 0) for row in complete),
            **means,
            **action_metrics,
            "mean_reward_components": {
                key: sum(
                    row.get("reward_components", {}).get(key, 0.0) for row in complete
                )
                / len(complete)
                for key in components
            },
            "denominator": "all planned episodes, including errors and pending attempts",
            "mean_denominator": "completed episodes with the named field; coverage reported",
            "reward_contract": "raw clipped [0,1] episode reward; not group-normalized training reward",
        }

    @staticmethod
    def read(directory: str | Path, scope: str = "all_attempts") -> dict:
        directory = Path(directory)
        if not (directory / "plan.json").exists():
            return EpisodeMetrics.training(directory, scope)
        if scope != "all_attempts":
            raise ValueError("Standalone evaluation uses all planned attempts")
        plan = json.loads((directory / "plan.json").read_text())["episodes"]
        records = [
            json.loads(path.read_text())
            for path in sorted((directory / "episodes").glob("*.json"))
        ]
        return EpisodeMetrics.summarize(plan, records)

    @staticmethod
    def training(directory: Path, scope: str) -> dict:
        if scope not in ("all_attempts", "completed"):
            raise ValueError(
                "Only all_attempts/completed scopes exist; artifacts do not confirm optimizer acceptance"
            )
        groups = defaultdict(list)
        seen = set()
        for split in ("train", "dev", "test", "validation"):
            for path in sorted((directory / split).glob("*.json")):
                row = json.loads(path.read_text())
                identity = row.get("attempt_id", row["episode_id"])
                if identity in seen:
                    raise ValueError(f"Duplicate attempt ID: {identity}")
                seen.add(identity)
                if scope == "completed" and row.get("status") != "complete":
                    continue
                versions = tuple(
                    value
                    for value in row.get("model_versions", [])
                    if value is not None
                )
                version_source = "verified" if versions else "unknown"
                if not versions and row.get("requested_model_version") is not None:
                    versions = (row["requested_model_version"],)
                    version_source = "requested"
                version = (
                    str(versions[0])
                    if len(versions) == 1
                    else "mixed:" + ",".join(map(str, versions))
                    if versions
                    else "unknown"
                )
                groups[(split, version, version_source)].append(
                    {**row, "episode_id": identity}
                )
        if not groups:
            raise FileNotFoundError(
                f"No episode summaries under {directory}/{{train,dev,test,validation}}"
            )
        return {
            "scope": scope,
            "groups": [
                {
                    "split": split,
                    "source_version": version,
                    "version_source": source,
                    **EpisodeMetrics.summarize(rows, rows),
                }
                for (split, version, source), rows in sorted(
                    groups.items(),
                    key=lambda item: (
                        item[0][0],
                        int(item[0][1]) if item[0][1].isdigit() else float("inf"),
                        item[0][1],
                        item[0][2],
                    ),
                )
            ],
            "grouping": "split + version + version_source; requested means the version observed at episode start, not verified output-token versions",
            "acceptance": "collection only; completed games and agent-submitted rewards do not prove PPO acceptance or optimizer completion",
            "coverage": "persisted attempts; in-flight or process-killed attempts may have no summary",
        }
