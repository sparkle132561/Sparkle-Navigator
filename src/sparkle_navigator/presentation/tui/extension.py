"""把 Sparkle Workspace/Evaluation 视图挂载到 Tau Textual 宿主。"""

from __future__ import annotations

from tau_coding.extensions import ExtensionAPI, ExtensionCommandContext

from sparkle_navigator.presentation.tui.services import (
    SparklePresentationServices,
    default_sparkle_paths,
)
from sparkle_navigator.presentation.tui.tool_approval import TUI_APPROVAL_PRESENTER
from sparkle_navigator.presentation.tui.views import SparkleMainView


def setup(tau: ExtensionAPI) -> None:
    """注册 `/sparkle`，只在交互式 Tau TUI 中打开主视图。"""

    TUI_APPROVAL_PRESENTER.bind(tau.context.ui)

    def on_session_start(_event: object, context) -> None:
        TUI_APPROVAL_PRESENTER.bind(context.ui)

    def on_session_shutdown(_event: object, _context) -> None:
        TUI_APPROVAL_PRESENTER.clear()

    tau.on("session_start", on_session_start)
    tau.on("session_shutdown", on_session_shutdown)

    def open_sparkle_view(
        _arguments: str,
        command_context: ExtensionCommandContext,
    ) -> str | None:
        context = command_context.api.context
        components = context.ui.components
        if not components.supports_components:
            return "Sparkle 工作台需要交互式 Tau Textual TUI。"
        database_path, snapshots_root, working_copies_root, reports_root = (
            default_sparkle_paths(context.paths.home)
        )
        services = SparklePresentationServices(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
            reports_root=reports_root,
        )
        components.open_main_view(
            lambda handle, _theme: SparkleMainView(
                context=context,
                services=services,
                show_agent_chat=handle.close,
            )
        )
        return None

    tau.register_command(
        "sparkle",
        open_sparkle_view,
        description="Open the Sparkle Navigator Workspace and Evaluation views.",
        usage="/sparkle",
    )
