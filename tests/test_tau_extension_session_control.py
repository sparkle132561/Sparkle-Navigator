"""扩展主视图通过 Tau Session Control 切换已注册工具并持久化状态。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tau_agent.session.entries import CustomEntry
from tau_ai.fake import FakeProvider
from tau_agent.session import InMemorySessionStorage
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.tui import TauTuiApp, TuiSettings
from textual.widgets import Button, Static

from sparkle_navigator.runtime.tool_profile import (
    TOOL_PROFILE_NAMESPACE,
    ToolProfileRuntime,
)


def test_extension_session_control_updates_tools_and_tau_session(
    tmp_path: Path,
) -> None:
    asyncio.run(_extension_session_control_updates_tools_and_tau_session(tmp_path))


async def _extension_session_control_updates_tools_and_tau_session(
    tmp_path: Path,
) -> None:
    extension_path = (
        Path(__file__).parent / "fixtures" / "session_control_extension" / "extension.py"
    )
    profile_runtime = ToolProfileRuntime(repository_trusted=True)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="session-control-test-model",
            storage=InMemorySessionStorage(),
            cwd=tmp_path,
            extension_paths=(extension_path,),
            extensions_enabled=False,
            skills_enabled=False,
            tool_access_policy=profile_runtime,
        )
    )
    app = TauTuiApp(session, tui_settings=TuiSettings())

    try:
        async with app.run_test() as pilot:
            command_result = session.handle_command("/session-control-test")
            assert command_result.handled
            await pilot.pause()
            assert app.query_one("#session-control-view")

            assert await pilot.click(app.query_one("#toggle-profile", Button))
            await pilot.pause(0.2)

            assert app.query_one("#session-control-status", Static).render().plain == "Profile saved"
            assert [tool.name for tool in session.tools] == ["read", "write", "edit", "bash"]
            entries = await session.session_entries()
            assert any(
                isinstance(entry, CustomEntry)
                and entry.namespace == TOOL_PROFILE_NAMESPACE
                and entry.data == {"version": 1, "profile": "coding"}
                for entry in entries
            )

            await pilot.click("#close-session-control")
            await pilot.pause()
            assert app.query_one("#transcript").display
    finally:
        await session.aclose()
