"""Sparkle Navigator 独立命令行入口；TUI 与 Session 仍由 Tau 托管。"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import anyio
import typer

from tau_coding.paths import TauPaths
from tau_coding.tui import TuiProductContext, TuiProductPresentation

from sparkle_navigator.domain.repository import Repository
from sparkle_navigator.presentation.tui.session_config import (
    SparkleSessionConfigTransform,
)
from sparkle_navigator.presentation.tui.services import (
    SparklePresentationServices,
    default_sparkle_paths,
)
from sparkle_navigator.presentation.tui.tool_approval import TUI_APPROVAL_PRESENTER
from sparkle_navigator.presentation.tui.views import EvaluationView, WorkspaceView
from sparkle_navigator.security.redaction import sanitize_trace_text


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
        launch_cwd, repository = _resolve_workspace(
            services,
            cwd=cwd or Path.cwd(),
            repository_id=repository_id,
        )
        snapshot_id = repository.current_snapshot_id if repository is not None else None
        repository_trusted = bool(
            repository is not None
            and repository.trusted
            and services.is_managed_working_copy(repository, launch_cwd)
        )
        session_config_transform = SparkleSessionConfigTransform(
            database_path=database,
            snapshot_id=snapshot_id,
            repository_trusted=repository_trusted,
            approval_handler=TUI_APPROVAL_PRESENTER,
        )

        def create_workspace_panel(product_context: TuiProductContext) -> WorkspaceView:
            TUI_APPROVAL_PRESENTER.bind(product_context.ui)
            return WorkspaceView(
                context=product_context,
                services=services,
            )

        def create_evaluation_panel(_product_context: TuiProductContext) -> EvaluationView:
            return EvaluationView(services=services)

        product_presentation = TuiProductPresentation(
            title="Sparkle Navigator",
            workspace_panel_factory=create_workspace_panel,
            evaluation_panel_factory=create_evaluation_panel,
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
