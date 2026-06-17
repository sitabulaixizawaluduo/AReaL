# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from areal.experimental.cli.agent.http import (
    AgentCLIHTTPError,
    AgentCLIUnreachable,
    AgentRouterClient,
    InferenceClient,
)
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    SessionState,
    generate_session_key,
)


def create_session(
    service_state: ServiceState,
    sessions_state: SessionsState,
    *,
    session_key: str | None = None,
    switch: bool = True,
    warn: bool = True,
) -> SessionState:
    key = session_key or generate_session_key()
    if (
        key in sessions_state.sessions
        and sessions_state.sessions[key].status == "active"
    ):
        raise ValueError(f"session {key!r} already exists")

    session = SessionState.create(
        key=key, session_timeout=service_state.session_timeout
    )
    router = AgentRouterClient(
        service_state.router.url,
        service_state.admin_api_key,
    )
    router.route(key)

    warning = _negotiate_inference_session(service_state, session)
    if warning and warn:
        print(f"Warning: {warning}", file=sys.stderr)
    session.warning = warning

    sessions_state.sessions[key] = session
    if switch or not sessions_state.current_session:
        sessions_state.current_session = key
    sessions_state.save()
    return session


def _negotiate_inference_session(
    service_state: ServiceState,
    session: SessionState,
) -> str:
    if not service_state.inf_addr:
        return ""
    if not service_state.inf_api_key:
        return "inference addr is configured but api key is missing"

    client = InferenceClient(service_state.inf_addr, service_state.inf_api_key)
    try:
        response = client.start_session(
            task_id=f"agent-{service_state.service}-{session.key}",
        )
    except (AgentCLIHTTPError, AgentCLIUnreachable):
        return (
            "inference service did not accept /rl/start_session; using original API key"
        )

    session_id = response.get("session_id", "")
    session_api_key = response.get("api_key", "") or response.get("session_api_key", "")
    sessions = response.get("sessions")
    if isinstance(sessions, list) and sessions:
        first = sessions[0]
        if isinstance(first, dict):
            session_id = first.get("session_id", session_id)
            session_api_key = first.get("session_api_key", session_api_key)

    if not session_api_key:
        return "inference /rl/start_session returned no session API key"

    session.rl_session_id = session_id
    session.rl_session_api_key = session_api_key
    session.rl_negotiated = True
    return ""
