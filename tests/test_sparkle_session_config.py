"""Sparkle CLI 配置变换通过 Tau CodingSession 注入 Tool Profile 与只读工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tau_ai.fake import FakeProvider
from tau_agent.session import InMemorySessionStorage
from tau_coding import CodingSession, CodingSessionConfig

from sparkle_navigator.presentation.tui.session_config import (
    SparkleSessionConfigTransform,
)


def test_sparkle_session_transform_adds_snapshot_tools_under_profile_policy(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _sparkle_session_transform_adds_snapshot_tools_under_profile_policy(tmp_path)
    )


async def _sparkle_session_transform_adds_snapshot_tools_under_profile_policy(
    tmp_path: Path,
) -> None:
    transform = SparkleSessionConfigTransform(
        database_path=tmp_path / "sparkle.sqlite3",
        snapshot_id="a" * 64,
        repository_trusted=False,
        approval_handler=None,
    )
    base_config = CodingSessionConfig(
        provider=FakeProvider([]),
        model="sparkle-session-test",
        storage=InMemorySessionStorage(),
        cwd=tmp_path,
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="deny",
    )
    session = await CodingSession.load(transform(base_config))

    try:
        assert session.tool_access_policy is transform.tool_profile_runtime
        assert {tool.name for tool in session.tools} == {
            "read",
            "overview_repository",
            "ask_codebase",
            "explain_code",
            "trace_call",
            "analyze_impact",
        }
        assert session.tool_access_policy is not None
    finally:
        await session.aclose()
