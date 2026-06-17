# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from areal.experimental.cli.agent.state import DEFAULT_SERVICE, agent_root

DEFAULT_ADMIN_API_KEY = "areal-agent-admin"


def config_path() -> Path:
    return agent_root() / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_config(extra: Path | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (config_path(), extra):
        if path is None:
            continue
        for section, values in _read_toml(path).items():
            if isinstance(values, dict):
                target = merged.setdefault(section, {})
                if isinstance(target, dict):
                    target.update(values)
            else:
                merged[section] = values
    return merged


def cfg_get(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    values = config.get(section)
    if isinstance(values, dict) and key in values:
        return values[key]
    return default


def resolve_default_service(config: dict[str, Any], explicit: str | None) -> str:
    return explicit or cfg_get(config, "default", "service", DEFAULT_SERVICE)


def resolve_admin_api_key(config: dict[str, Any], explicit: str | None) -> str:
    return explicit or cfg_get(
        config, "default", "admin_api_key", DEFAULT_ADMIN_API_KEY
    )


def resolve_inf_addr(config: dict[str, Any], explicit: str | None) -> str:
    return (
        explicit
        or os.environ.get("AREAL_INF_ADDR", "")
        or cfg_get(config, "inference", "addr", "")
    )


def resolve_inf_api_key(config: dict[str, Any], explicit: str | None) -> str:
    return (
        explicit
        or os.environ.get("AREAL_INF_API_KEY", "")
        or cfg_get(config, "inference", "api_key", "")
    )


def resolve_inf_model(config: dict[str, Any], explicit: str | None) -> str:
    return (
        explicit
        or os.environ.get("AREAL_INF_MODEL", "")
        or cfg_get(config, "inference", "model", "")
    )
