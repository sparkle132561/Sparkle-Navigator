"""TUI 审批适配器展示完整材料并在缺少 UI 时默认拒绝。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sparkle_navigator.domain.tool_profile import ToolRisk
from sparkle_navigator.runtime.tool_profile import (
    ApprovalOutcome,
    ToolApprovalRequest,
)
from sparkle_navigator.presentation.tui.tool_approval import TuiApprovalPresenter


def test_tui_approval_shows_bash_command_and_working_directory() -> None:
    asyncio.run(_tui_approval_shows_bash_command_and_working_directory())


async def _tui_approval_shows_bash_command_and_working_directory() -> None:
    ui = _FakeUi(confirmed=True)
    presenter = TuiApprovalPresenter(redacted_values=("private-value",))
    presenter.bind(ui)
    request = ToolApprovalRequest(
        call_id="call-1",
        registration_id="tau.bash",
        tool_name="bash",
        source="tau.default",
        arguments={"command": "python -c 'print(private-value)'"},
        risk=ToolRisk.SIDE_EFFECT,
        working_directory=Path("C:/workspace/project"),
        command="python -c 'print(private-value)'",
        risk_level="high",
    )

    outcome = await presenter(request)

    assert outcome is ApprovalOutcome.APPROVED
    title, message, _timeout = ui.confirmation
    assert "bash" in title
    assert "workspace" in message and "project" in message
    assert "python -c" in message
    assert "high" in message
    assert "private-value" not in message
    assert "REDACTED" in message


def test_tui_approval_shows_write_diff_and_redacts_its_content() -> None:
    asyncio.run(_tui_approval_shows_write_diff_and_redacts_its_content())


async def _tui_approval_shows_write_diff_and_redacts_its_content() -> None:
    ui = _FakeUi(confirmed=True)
    presenter = TuiApprovalPresenter(redacted_values=("private-value",))
    presenter.bind(ui)

    outcome = await presenter(_approval_request())

    assert outcome is ApprovalOutcome.APPROVED
    _title, message, _timeout = ui.confirmation
    assert "src/example.py" in message
    assert "- old" in message
    assert "+" in message and "REDACTED" in message
    assert "private-value" not in message
    assert "a" * 64 in message


def test_tui_approval_cancel_is_not_approval() -> None:
    asyncio.run(_tui_approval_cancel_is_not_approval())


async def _tui_approval_cancel_is_not_approval() -> None:
    ui = _FakeUi(confirmed=False)
    presenter = TuiApprovalPresenter()
    presenter.bind(ui)

    outcome = await presenter(_approval_request())

    assert outcome is ApprovalOutcome.DENIED


def test_tui_approval_without_interactive_ui_cancels_closed() -> None:
    asyncio.run(_tui_approval_without_interactive_ui_cancels_closed())


async def _tui_approval_without_interactive_ui_cancels_closed() -> None:
    presenter = TuiApprovalPresenter()

    outcome = await presenter(_approval_request())

    assert outcome is ApprovalOutcome.CANCELLED


def _approval_request() -> ToolApprovalRequest:
    return ToolApprovalRequest(
        call_id="call-2",
        registration_id="tau.write",
        tool_name="write",
        source="tau.default",
        arguments={"path": "src/example.py", "content": "private-value"},
        risk=ToolRisk.SIDE_EFFECT,
        working_directory=Path("C:/workspace/project"),
        target_path=Path("C:/workspace/project/src/example.py"),
        diff_preview="- old\n+ private-value",
        original_sha256="a" * 64,
    )


class _FakeUi:
    has_ui = True

    def __init__(self, *, confirmed: bool) -> None:
        self._confirmed = confirmed
        self.confirmation: tuple[str, str, float | None] | None = None

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        self.confirmation = (title, message, timeout)
        return self._confirmed
