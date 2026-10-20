"""Tau ExtensionUi 的 session-control 兼容性测试扩展。"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from sparkle_navigator.domain.tool_profile import ToolProfile


class SessionControlView(Vertical):
    def __init__(self, handle, session_control) -> None:
        super().__init__(id="session-control-view")
        self._handle = handle
        self._session_control = session_control

    def compose(self) -> ComposeResult:
        yield Static("Session Control Probe", id="session-control-status")
        yield Button("选择只读工具并保存记录", id="toggle-profile")
        yield Button("返回聊天", id="close-session-control")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-profile":
            try:
                profile_runtime = self._session_control.tool_access_policy
                await profile_runtime.set_profile(
                    self._session_control,
                    ToolProfile.CODING,
                )
            except Exception as exc:
                self.query_one("#session-control-status", Static).update(
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                self.query_one("#session-control-status", Static).update("Profile saved")
        elif event.button.id == "close-session-control":
            self._handle.close()


def setup(tau) -> None:
    def open_view(_arguments, command_context) -> None:
        context = command_context.api.context
        context.ui.components.open_main_view(
            lambda handle, _theme: SessionControlView(
                handle,
                context.session_control,
            )
        )

    tau.register_command("session-control-test", open_view)
