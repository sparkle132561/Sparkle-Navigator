"""把 Sparkle ToolApprovalRequest 显示为 Tau ExtensionUi 确认对话框。"""

from __future__ import annotations

import json
from typing import Protocol

from rich.markup import escape

from sparkle_navigator.runtime.tool_profile import (
    ApprovalOutcome,
    ToolApprovalRequest,
)
from sparkle_navigator.security.redaction import (
    sanitize_trace_payload,
    sanitize_trace_text,
)


class ApprovalUi(Protocol):
    @property
    def has_ui(self) -> bool: ...

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool: ...


class TuiApprovalPresenter:
    """Fail-closed 审批桥；没有交互式 UI 时永不返回批准。"""

    def __init__(
        self,
        *,
        redacted_values: tuple[str, ...] = (),
        timeout_seconds: float | None = None,
        max_message_chars: int = 12_000,
    ) -> None:
        if max_message_chars < 256:
            raise ValueError("max_message_chars 至少为 256。")
        self._redacted_values = tuple(value for value in redacted_values if value)
        self._timeout_seconds = timeout_seconds
        self._max_message_chars = max_message_chars
        self._ui: ApprovalUi | None = None

    def bind(self, ui: ApprovalUi) -> None:
        """绑定 Tau ExtensionUi；该对象会随 ExtensionRuntime 动态委托到宿主。"""

        self._ui = ui

    def unbind(self, ui: ApprovalUi) -> None:
        """只清除当前绑定，避免旧 Session 清掉后来绑定的 UI。"""

        if self._ui is ui:
            self._ui = None

    def clear(self) -> None:
        """在当前 Sparkle TUI Session 关闭时移除 UI 引用。"""

        self._ui = None

    async def __call__(self, request: ToolApprovalRequest) -> ApprovalOutcome:
        ui = self._ui
        if ui is None or not ui.has_ui:
            return ApprovalOutcome.CANCELLED
        title = escape(f"确认工具调用：{request.tool_name}")
        message = escape(
            _approval_message(
                request,
                redacted_values=self._redacted_values,
                max_chars=self._max_message_chars,
            )
        )
        approved = await ui.confirm(
            title,
            message,
            timeout=self._timeout_seconds,
        )
        return ApprovalOutcome.APPROVED if approved else ApprovalOutcome.DENIED


def _approval_message(
    request: ToolApprovalRequest,
    *,
    redacted_values: tuple[str, ...],
    max_chars: int,
) -> str:
    arguments = sanitize_trace_payload(
        dict(request.arguments),
        secret_values=redacted_values,
    )
    sections = [
        f"工具来源：{request.source}",
        f"风险：{request.risk.value}"
        + (f" ({request.risk_level})" if request.risk_level else ""),
        f"工作目录：{request.working_directory}",
    ]
    if request.target_path is not None:
        sections.append(f"目标路径：{request.target_path}")
    if request.command is not None:
        sections.append(f"命令：{request.command}")
    sections.append(
        "参数：\n"
        + json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True)
    )
    if request.diff_preview is not None:
        sections.append(f"文件差异预览：\n{request.diff_preview}")
        if request.original_sha256 is not None:
            sections.append(f"审批前 SHA-256：{request.original_sha256}")
    message = sanitize_trace_text(
        "\n\n".join(sections),
        secret_values=redacted_values,
    )
    if len(message) > max_chars:
        return message[: max_chars - 20] + "\n… 内容已截断。"
    return message


TUI_APPROVAL_PRESENTER = TuiApprovalPresenter()

__all__ = ["ApprovalUi", "TuiApprovalPresenter", "TUI_APPROVAL_PRESENTER"]
