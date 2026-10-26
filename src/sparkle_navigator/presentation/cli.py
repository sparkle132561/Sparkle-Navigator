"""Sparkle Navigator 独立命令行入口；TUI 与 Session 仍由 Tau 托管。"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import anyio
import typer

from sparkle_navigator.domain.repository import Repository
from sparkle_navigator.presentation.tui.evidence_viewer import view_evidence_async
from sparkle_navigator.presentation.tui.services import (
    SparklePresentationServices,
    default_sparkle_paths,
)
from sparkle_navigator.presentation.tui.session_binding import WorkspaceSessions
from sparkle_navigator.presentation.tui.session_config import (
    SparkleSessionConfigTransform,
)
from sparkle_navigator.presentation.tui.tool_approval import TUI_APPROVAL_PRESENTER
from sparkle_navigator.presentation.tui.turn_status import build_turn_status_formatter
from sparkle_navigator.presentation.tui.views import EvaluationView, WorkspaceView
from sparkle_navigator.security.redaction import sanitize_trace_text
from tau_coding import CodingSession
from tau_coding.paths import TauPaths
from tau_coding.session_manager import CodingSessionRecord, SessionManager
from tau_coding.tui import TuiProductContext, TuiProductPresentation

app = typer.Typer(
    name="sparkle",
    help="启动 Sparkle Navigator 编码与代码理解工作台。",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main(
    context: typer.Context,
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="初始工作目录；默认为当前目录。",
    ),
    repository_id: str | None = typer.Option(
        None,
        "--repository-id",
        help="从指定 Repository 的 Snapshot 创建或恢复受管工作副本后启动。",
    ),
    provider_name: str | None = typer.Option(None, "--provider", help="Tau Provider ID。"),
    model: str | None = typer.Option(None, "--model", help="覆盖当前 Tau Model。"),
    session_id: str | None = typer.Option(None, "--session", help="恢复 Tau Session ID。"),
    new_session: bool = typer.Option(False, "--new-session", help="为所选工作目录创建新 Session。"),
    context_injection: bool = typer.Option(
        False,
        "--context",
        help="实验开关：为每轮提问注入固定 Snapshot 的证据与项目事实（默认关闭）。",
    ),
) -> None:
    """启动 Sparkle Navigator 产品 TUI。"""

    if context.invoked_subcommand is not None:
        return
    if repository_id is not None and cwd is not None:
        raise typer.BadParameter("--repository-id 与 --cwd 不能同时指定。")

    default_database, snapshots_root, working_copies_root, reports_root = (
        default_sparkle_paths(TauPaths().home)
    )
    database = default_database.expanduser().resolve()
    services = SparklePresentationServices(
        database_path=database,
        snapshots_root=snapshots_root,
        working_copies_root=working_copies_root,
        reports_root=reports_root,
    )

    try:
        resume_record = SessionManager().get_session(session_id) if session_id is not None else None
        if session_id is not None and resume_record is None:
            raise ValueError(f"Session 不存在：{session_id}")
        launch_cwd, repository = _resolve_workspace(
            services,
            cwd=cwd or (resume_record.cwd if resume_record is not None else Path.cwd()),
            repository_id=repository_id,
        )
        if resume_record is not None and launch_cwd.resolve() != resume_record.cwd.resolve():
            raise ValueError("Session 工作目录与所选目录不同，请仅使用 --session 恢复原会话。")
        snapshot_id = repository.current_snapshot_id if repository is not None else None
        if resume_record is not None:
            restored = anyio.run(partial(
                WorkspaceSessions.restore_binding, services, resume_record.path, resume_record.cwd,
            ))
            if restored is not None:
                repository, snapshot_id = restored
        repository_trusted = bool(
            repository is not None
            and repository.trusted
            and snapshot_id is not None
            and launch_cwd == (
                services.working_copies_root / repository.repository_id / snapshot_id
            ).resolve()
        )
        session_config_transform = SparkleSessionConfigTransform(
            database_path=database,
            snapshot_id=snapshot_id,
            repository_trusted=repository_trusted,
            approval_handler=TUI_APPROVAL_PRESENTER,
            context_repository_id=(
                repository.repository_id
                if context_injection and repository is not None and snapshot_id is not None
                else None
            ),
        )
        workspace_sessions: WorkspaceSessions | None = None

        def create_workspace_panel(product_context: TuiProductContext) -> WorkspaceView:
            nonlocal workspace_sessions
            TUI_APPROVAL_PRESENTER.bind(product_context.ui)
            base_config = session_config_transform.base_config
            if base_config is None:
                raise RuntimeError("Tau Session 基础配置尚未准备。")
            sessions = WorkspaceSessions(
                services=services,
                base_config=base_config,
                approval_handler=TUI_APPROVAL_PRESENTER,
                initial_binding=(repository.repository_id, snapshot_id)
                if repository is not None and snapshot_id is not None else None,
                context_wiring=context_injection,
            )
            workspace_sessions = sessions
            return WorkspaceView(
                context=product_context,
                services=services,
                sessions=sessions,
            )

        def create_evaluation_panel(_product_context: TuiProductContext) -> EvaluationView:
            return EvaluationView(services=services)

        async def before_session_activate(
            session: CodingSession, destination: CodingSessionRecord | None,
        ) -> None:
            if workspace_sessions is not None:
                await workspace_sessions.revalidate_trust(session, destination)

        product_presentation = TuiProductPresentation(
            title="Sparkle Navigator",
            workspace_panel_factory=create_workspace_panel,
            evaluation_panel_factory=create_evaluation_panel,
            before_session_activate=before_session_activate,
            run_status_formatter=build_turn_status_formatter(
                services,
                sessions_lookup=lambda: workspace_sessions,
            ),
            evidence_viewer=partial(view_evidence_async, database),
        )
        from tau_coding.cli import run_openai_tui

        typer.echo("Sparkle Navigator 正在启动；Workspace、Agent、Evaluation 将在同一屏显示。")
        try:
            anyio.run(
                partial(
                    run_openai_tui,
                    model=model,
                    cwd=launch_cwd,
                    session_id=session_id,
                    new_session=new_session,
                    provider_name=provider_name,
                    session_config_transform=session_config_transform,
                    product_presentation=product_presentation,
                )
            )
        finally:
            TUI_APPROVAL_PRESENTER.clear()
    except Exception as exc:
        message = sanitize_trace_text(str(exc) or type(exc).__name__)
        typer.echo(f"Sparkle 启动失败：{type(exc).__name__}: {message}", err=True)
        raise typer.Exit(code=1) from exc


def _resolve_workspace(
    services: SparklePresentationServices,
    *,
    cwd: Path,
    repository_id: str | None,
) -> tuple[Path, Repository | None]:
    if repository_id is not None:
        working_copy = anyio.run(
            partial(services.open_working_copy, repository_id)
        )
        repository = anyio.run(
            partial(services.get_repository, repository_id)
        )
        return working_copy.root, repository
    resolved_cwd = cwd.expanduser().resolve()
    repository = anyio.run(
        partial(services.repository_for_cwd, resolved_cwd)
    )
    return resolved_cwd, repository


__all__ = ["app"]
