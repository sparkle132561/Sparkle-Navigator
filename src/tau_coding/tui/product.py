"""Tau TUI 宿主承载一体化产品界面的窄接口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from textual.message import Message
from textual.widget import Widget

from tau_agent.messages import AssistantMessage
from tau_coding.session import CodingSession
from tau_coding.session_manager import CodingSessionRecord
from tau_coding.tui.config import TuiTheme


class ProductSessionChanged(Message):
    """通知常驻产品面板重新读取当前 Session。"""


class ConfirmDialog(Protocol):
    async def __call__(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class TuiProductContext:
    """产品页面使用的 Tau Session 与宿主交互能力。"""

    _get_session: Callable[[], CodingSession]
    theme: TuiTheme
    _confirm: ConfirmDialog
    _notify: Callable[[str, str], None]
    _replace_session: Callable[[CodingSession], Awaitable[None]]
    _is_busy: Callable[[], bool]

    @property
    def session(self) -> CodingSession:
        return self._get_session()

    @property
    def is_busy(self) -> bool:
        return self._is_busy()

    async def replace_session(self, session: CodingSession) -> None:
        """把已准备的 Session 交给宿主；运行中拒绝替换。"""
        await self._replace_session(session)

    @property
    def has_ui(self) -> bool:
        return True

    @property
    def cwd(self) -> Path:
        return self.session.cwd

    @property
    def provider_name(self) -> str:
        return self.session.provider_name

    @property
    def model(self) -> str:
        return self.session.model

    @property
    def session_control(self) -> CodingSession:
        return self.session

    @property
    def ui(self) -> TuiProductContext:
        return self

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        return await self._confirm(title, message, timeout=timeout)

    def notify(self, message: str, level: str = "info") -> None:
        self._notify(message, level)


@dataclass(frozen=True, slots=True)
class TuiProductPresentation:
    """向 Tau TUI 提供常驻 Agent 聊天两侧的产品面板。"""

    title: str
    workspace_panel_factory: Callable[[TuiProductContext], Widget]
    evaluation_panel_factory: Callable[[TuiProductContext], Widget]
    before_session_activate: (
        Callable[[CodingSession, CodingSessionRecord | None], Awaitable[None]] | None
    ) = None
    # Optional per-turn hook: given the assistant messages of one settled
    # agent turn plus the active session, return a one-line status summary
    # (or awaitable of one), or None for no line. The host owns accumulation;
    # the product owns presentation and may resolve session-bound state.
    run_status_formatter: (
        Callable[[Sequence[AssistantMessage], CodingSession], object] | None
    ) = None
    # Optional product-owned viewer: given an opaque key embedded in clickable
    # product markup (e.g. evidence locations), resolve it to a (title, body)
    # pair shown in the host's read-only output screen, or None to ignore.
    evidence_viewer: (
        Callable[[str], Awaitable[tuple[str, str] | None]] | None
    ) = None


__all__ = ["ProductSessionChanged", "TuiProductContext", "TuiProductPresentation"]
