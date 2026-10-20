"""Tau TUI 宿主承载一体化产品界面的窄接口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from textual.widget import Widget

from tau_coding.session import CodingSession
from tau_coding.tui.config import TuiTheme


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

    session: CodingSession
    theme: TuiTheme
    _confirm: ConfirmDialog
    _notify: Callable[[str, str], None]

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


__all__ = ["TuiProductContext", "TuiProductPresentation"]
