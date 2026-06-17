# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from areal.experimental.cli.agent.config import (
    cfg_get,
    load_config,
    resolve_admin_api_key,
    resolve_default_service,
    resolve_inf_addr,
    resolve_inf_api_key,
    resolve_inf_model,
)
from areal.experimental.cli.agent.http import AgentCLIHTTPError, AgentCLIUnreachable
from areal.experimental.cli.agent.interactive import run_shell
from areal.experimental.cli.agent.launcher import launch_agent_stack
from areal.experimental.cli.agent.process import kill_pids, pid_alive
from areal.experimental.cli.agent.session_ops import create_session
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    service_state_path,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Launch an agent service")
    parser.add_argument("--agent", default=None, help="Agent import path")
    parser.add_argument("--service", default=None, help="Service instance name")
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--admin-api-key", default=None)
    parser.add_argument("--setup-timeout", type=float, default=None)
    parser.add_argument("--health-poll-interval", type=float, default=None)
    parser.add_argument("--drain-timeout", type=float, default=None)
    parser.add_argument("--session-timeout", type=float, default=None)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=None,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inf-addr", default=None)
    parser.add_argument("--inf-api-key", default=None)
    parser.add_argument("--inf-model", default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--stop-on-exit", action="store_true")
    parser.add_argument("--history-file", type=Path, default=None)
    parser.add_argument("--session-key", default=None)
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    service = resolve_default_service(config, args.service)
    agent = args.agent or cfg_get(config, "run", "agent", "")
    if not agent:
        print("error: --agent is required", file=sys.stderr)
        return 2

    admin_api_key = resolve_admin_api_key(config, args.admin_api_key)
    num_pairs = int(args.num_pairs or cfg_get(config, "run", "num_pairs", 1))
    setup_timeout = float(
        args.setup_timeout or cfg_get(config, "run", "setup_timeout", 120.0)
    )
    health_poll_interval = float(
        args.health_poll_interval or cfg_get(config, "run", "health_poll_interval", 5.0)
    )
    drain_timeout = float(
        args.drain_timeout or cfg_get(config, "run", "drain_timeout", 30.0)
    )
    session_timeout = float(
        args.session_timeout or cfg_get(config, "run", "session_timeout", 1800.0)
    )
    log_level = args.log_level or cfg_get(config, "default", "log_level", "info")
    inf_addr = resolve_inf_addr(config, args.inf_addr)
    inf_api_key = resolve_inf_api_key(config, args.inf_api_key)
    inf_model = resolve_inf_model(config, args.inf_model)

    existing = _load_existing(service)
    if existing is not None and any(pid_alive(pid) for pid in existing.all_pids()):
        if not args.force:
            print(
                f"error: service {service!r} already has live processes; "
                "run `areal agent stop` first or use --force",
                file=sys.stderr,
            )
            return 1
        kill_pids(existing.all_pids(), grace_s=5.0)
    elif service_state_path(service).exists() and not args.force:
        print(
            f"error: stale state exists for {service!r}; use `areal agent run --force`",
            file=sys.stderr,
        )
        return 1

    launched: ServiceState | None = None
    try:
        launched = launch_agent_stack(
            service=service,
            agent=agent,
            admin_api_key=admin_api_key,
            num_pairs=num_pairs,
            setup_timeout=setup_timeout,
            session_timeout=session_timeout,
            health_poll_interval=health_poll_interval,
            drain_timeout=drain_timeout,
            log_level=log_level,
            inf_addr=inf_addr,
            inf_api_key=inf_api_key,
            inf_model=inf_model,
            interactive=args.interactive,
        )
        launched.save()
        sessions = SessionsState(service=service)
        session = create_session(
            launched,
            sessions,
            session_key=args.session_key,
            switch=True,
        )
    except (AgentCLIHTTPError, AgentCLIUnreachable, RuntimeError, ValueError) as exc:
        if launched is not None:
            kill_pids(launched.all_pids(), grace_s=5.0)
        print(f"error: failed to launch agent service: {exc}", file=sys.stderr)
        return 1

    print(f"service={service} gateway={launched.gateway.url}")
    print(f"session={session.key} rl={'yes' if session.rl_negotiated else 'no'}")
    if args.interactive:
        return run_shell(
            launched,
            stop_on_exit=args.stop_on_exit,
            history_file=args.history_file,
        )
    return 0


def _load_existing(service: str) -> ServiceState | None:
    if not service_state_path(service).exists():
        return None
    try:
        return ServiceState.load(service)
    except Exception:
        return None
