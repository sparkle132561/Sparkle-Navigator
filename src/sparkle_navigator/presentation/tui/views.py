"""Tau Textual 宿主中的 Sparkle 主视图。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Select,
    Static,
    TabPane,
    TabbedContent,
)

from sparkle_navigator.domain.evidence import SnapshotOverview
from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    ExperimentComparison,
    ExperimentSnapshot,
)
from sparkle_navigator.domain.memory import ProjectFactStatus
from sparkle_navigator.domain.tool_profile import ToolProfile
from sparkle_navigator.evaluation.core import ExperimentHandle
from sparkle_navigator.evaluation.live_provider import (
    LIVE_MAX_AGENT_RUNS,
    LIVE_MAX_PROVIDER_REQUESTS_PER_RUN,
    LIVE_PROVIDER_TIMEOUT_SECONDS,
)
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.tool_approval import ApprovalUi
from sparkle_navigator.runtime.tool_profile import (
    ProfileSessionControl,
    ToolProfileRuntime,
)
from sparkle_navigator.security.redaction import sanitize_trace_payload, sanitize_trace_text


class SparkleTuiContext(Protocol):
    """Sparkle 页面从 TUI 宿主读取的最小会话能力。"""

    @property
    def cwd(self) -> Path: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def session_control(self) -> ProfileSessionControl: ...

    @property
    def ui(self) -> ApprovalUi: ...


class SparkleMainView(Vertical):
    """兼容 Tau Extension 的主视图；Sparkle CLI 使用独立的左右面板工厂。"""

    BINDINGS = [("escape", "show_agent_chat", "智能体")]
    DEFAULT_CSS = """
    SparkleMainView {
        width: 1fr;
        height: 1fr;
    }

    #sparkle-toolbar {
        width: 1fr;
        height: auto;
    }

    #sparkle-title,
    #sparkle-agent {
        width: auto;
    }

    #sparkle-model {
        width: 1fr;
        min-width: 0;
    }
    """

    def __init__(
        self,
        *,
        context: SparkleTuiContext,
        services: SparklePresentationServices,
        show_agent_chat: Callable[[], None],
    ) -> None:
        super().__init__(id="sparkle-main-view")
        self._product_context = context
        self._services = services
        self._show_agent_chat = show_agent_chat

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("Sparkle Navigator", id="sparkle-title"),
            Static(
                f"{self._product_context.provider_name} / {self._product_context.model}",
                id="sparkle-model",
            ),
            Button("智能体", id="sparkle-agent"),
            id="sparkle-toolbar",
        )
        with TabbedContent("工作区", "评测", "记忆 / MCP", "设置", id="sparkle-tabs"):
            yield TabPane(
                "工作区",
                WorkspaceView(
                    context=self._product_context,
                    services=self._services,
                ),
                id="sparkle-workspace-pane",
            )
            yield TabPane(
                "评测",
                EvaluationView(services=self._services),
                id="sparkle-evaluation-pane",
            )
            yield TabPane(
                "记忆 / MCP",
                Static(
                    "项目事实、学习者偏好和 MCP 工具状态显示在工作区摘要中。",
                    markup=False,
                ),
                id="sparkle-memory-mcp-pane",
            )
            yield TabPane(
                "设置",
                Static(
                    "Provider、Session 恢复与模型选择由 Tau 管理；可在聊天区使用 /model。",
                    markup=False,
                ),
                id="sparkle-settings-pane",
            )

    def action_show_agent_chat(self) -> None:
        self._show_agent_chat()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sparkle-agent":
            self._show_agent_chat()


class WorkspaceView(VerticalScroll):
    """浏览固定 Snapshot、展示代码证据并控制当前 Session Profile。"""

    def __init__(
        self,
        *,
        context: SparkleTuiContext,
        services: SparklePresentationServices,
    ) -> None:
        super().__init__(id="sparkle-workspace-view")
        self._product_context = context
        self._services = services
        self._repositories = {}
        self._selected_repository = None

    def compose(self) -> ComposeResult:
        yield Static("Workspace", classes="panel-title")
        yield Static("固定 Snapshot · 只读浏览", classes="panel-subtitle")
        yield Static("尚未选择 Snapshot。", id="workspace-status", markup=False)
        yield Static("Snapshot 文件", classes="section-title")
        yield Static("导入 Repository 后显示顶层路径。", id="workspace-file-tree", markup=False)
        yield Static("Snapshot 概览证据", classes="section-title")
        yield Static("尚未读取概览证据。", id="workspace-evidence", markup=False)
        yield Static("操作", classes="section-title")
        with Collapsible(
            title="导入与仓库操作",
            collapsed=True,
            id="workspace-repository-controls",
        ):
            yield Input(placeholder="本地 Git 路径或 Git URL", id="workspace-source")
            yield Button("导入并索引", id="workspace-import")
            yield Select(
                [],
                allow_blank=True,
                prompt="选择已导入 Repository",
                id="workspace-repository",
            )
            yield Button("刷新列表", id="workspace-refresh")
            yield Button("构建索引", id="workspace-index")
            yield Button("创建工作副本", id="workspace-copy")
            yield Button("信任 / 撤销信任", id="workspace-trust")
            yield Static("", id="workspace-repository-detail", markup=False)
        with Collapsible(
            title="工具配置档",
            collapsed=True,
            id="workspace-profile-controls",
        ):
            yield Static("", id="workspace-profile", markup=False)
            yield Button("Explain Profile", id="workspace-profile-explain")
            yield Button("Coding Profile", id="workspace-profile-coding")
            yield Static(
                "Coding Profile 只对受信任 Repository 的受管 Working Copy 开放；写入和 Shell 仍需逐次审批。",
                markup=False,
            )
            yield Static(
                "工具可见性会影响后续 Provider 请求；Tau Session 继续保存对话和 Profile 记录。",
                markup=False,
            )
        with Collapsible(
            title="Snapshot 代码分析",
            collapsed=True,
            id="workspace-query-controls",
        ):
            yield Static(
                "Overview / Ask / Explain / Trace / Impact",
                classes="section-title",
            )
            yield Input(placeholder="问题、文件路径或符号名", id="workspace-query")
            yield Button("概览", id="workspace-overview")
            yield Button("Ask", id="workspace-ask")
            yield Button("Explain", id="workspace-explain")
            yield Button("Trace path::symbol", id="workspace-trace")
            yield Button("Impact path::symbol", id="workspace-impact")
        with Collapsible(
            title="检索与分析详细结果",
            collapsed=True,
            id="workspace-result-details",
        ):
            yield Static("", id="workspace-result", markup=False)
        with Collapsible(
            title="Memory / MCP / Settings",
            collapsed=True,
            id="workspace-session-details",
        ):
            yield Static("", id="workspace-summary", markup=False)

    async def on_mount(self) -> None:
        await self.refresh_repositories()
        await self._refresh_session_summary()

    async def refresh_repositories(self) -> None:
        status = self.query_one("#workspace-status", Static)
        status.update("正在读取本机 Repository 目录…")
        try:
            repositories = await self._services.list_repositories()
            self._repositories = {
                repository.repository_id: repository for repository in repositories
            }
            selector = self.query_one("#workspace-repository", Select)
            selector.set_options(
                (
                    f"{repository.name} · {repository.current_snapshot_id[:10] if repository.current_snapshot_id else '无 Snapshot'}",
                    repository.repository_id,
                )
                for repository in repositories
            )
            current = await self._services.repository_for_cwd(self._product_context.cwd)
            selected = current or (repositories[0] if repositories else None)
            if selected is None:
                self._selected_repository = None
                status.update("尚无导入的 Repository。输入 Git 路径或 URL 后可创建固定 Snapshot。")
                return
            selector.value = selected.repository_id
            await self._select_repository(selected.repository_id)
        except Exception as exc:
            status.update(_safe_error(exc))

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "workspace-repository" and isinstance(event.value, str):
            await self._select_repository(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "workspace-refresh":
            await self.refresh_repositories()
        elif button_id == "workspace-import":
            await self._import_repository()
        elif button_id == "workspace-index":
            await self._build_indexes()
        elif button_id == "workspace-copy":
            await self._open_working_copy()
        elif button_id == "workspace-trust":
            # 确认结果回调会回到本控件的消息队列，不能在处理器中等待它。
            self.run_worker(self._toggle_trust(), group="repository-trust")
        elif button_id == "workspace-profile-explain":
            await self._set_profile(ToolProfile.EXPLAIN)
        elif button_id == "workspace-profile-coding":
            await self._set_profile(ToolProfile.CODING)
        elif button_id == "workspace-overview":
            await self._run_snapshot_query("overview")
        elif button_id == "workspace-ask":
            await self._run_snapshot_query("ask")
        elif button_id == "workspace-explain":
            await self._run_snapshot_query("explain")
        elif button_id == "workspace-trace":
            await self._run_snapshot_query("trace")
        elif button_id == "workspace-impact":
            await self._run_snapshot_query("impact")

    async def _select_repository(self, repository_id: str) -> None:
        self._selected_repository = self._repositories.get(repository_id)
        repository = self._selected_repository
        if repository is None or repository.current_snapshot_id is None:
            self.query_one("#workspace-status", Static).update(
                "所选 Repository 尚无当前 Snapshot。"
            )
            return
        cwd_repository = await self._services.repository_for_cwd(self._product_context.cwd)
        session_binding = (
            "当前 Tau Session 已绑定此 Repository。"
            if cwd_repository is not None
            and cwd_repository.repository_id == repository.repository_id
            else "Agent 工具仍绑定启动 Session 的 Snapshot；切换 Repository 后请用 --repository-id 重启。"
        )
        working_copy = self._services.is_managed_working_copy(repository, self._product_context.cwd)
        status = self.query_one("#workspace-status", Static)
        bound = cwd_repository is not None and cwd_repository.repository_id == repository.repository_id
        status.update(
            f"{repository.name}\n"
            f"Snapshot {repository.current_snapshot_id[:12]}\n"
            f"信任：{'已信任' if repository.trusted else '未信任'}；"
            f"{'受管副本' if working_copy else '非受管副本'}\n"
            f"{'Session 已绑定' if bound else 'Session 未绑定此仓库，见导入与仓库操作'}"
        )
        self.query_one("#workspace-repository-detail", Static).update(
            f"Repository {repository.repository_id}\n"
            f"Snapshot {repository.current_snapshot_id}\n"
            f"来源：{repository.source_locator}\n{session_binding}"
        )
        status.tooltip = f"来源：{repository.source_locator}\n{session_binding}"
        await self._show_profile()
        await self._run_snapshot_query("overview")
        await self._refresh_session_summary()

    async def _import_repository(self) -> None:
        raw_source = self.query_one("#workspace-source", Input).value.strip()
        if not raw_source:
            self.query_one("#workspace-status", Static).update("请输入本地路径或 Git URL。")
            return
        status = self.query_one("#workspace-status", Static)
        status.update("正在导入并冻结 Repository Snapshot…")
        try:
            if raw_source.startswith(("https://", "http://", "ssh://", "git@")):
                imported = await self._services.import_git_repository(raw_source)
            else:
                imported = await self._services.import_local_repository(Path(raw_source).expanduser())
            status.update(
                f"Snapshot 已冻结：{imported.snapshot.snapshot_id[:16]}… 正在构建索引。"
            )
            await self._services.build_snapshot_indexes(imported.snapshot.snapshot_id)
            await self.refresh_repositories()
            selector = self.query_one("#workspace-repository", Select)
            selector.value = imported.repository.repository_id
            await self._select_repository(imported.repository.repository_id)
            self.query_one("#workspace-source", Input).value = ""
        except Exception as exc:
            status.update(f"导入或索引失败：{_safe_error(exc)}")

    async def _build_indexes(self) -> None:
        repository = self._selected_repository
        if repository is None or repository.current_snapshot_id is None:
            return
        status = self.query_one("#workspace-status", Static)
        status.update("正在重建 FTS5、AST 符号和 Code Graph 索引…")
        try:
            await self._services.build_snapshot_indexes(repository.current_snapshot_id)
            status.update("Snapshot 索引已构建。")
            await self._run_snapshot_query("overview")
        except Exception as exc:
            status.update(f"索引构建失败：{_safe_error(exc)}")

    async def _open_working_copy(self) -> None:
        repository = self._selected_repository
        if repository is None:
            return
        try:
            working_copy = await self._services.open_working_copy(repository.repository_id)
            self.query_one("#workspace-status", Static).update(
                f"受管工作副本已就绪：{working_copy.root}\n"
                f"关闭 Sparkle 后运行：sparkle --repository-id {repository.repository_id}"
            )
        except Exception as exc:
            self.query_one("#workspace-status", Static).update(
                f"创建工作副本失败：{_safe_error(exc)}"
            )

    async def _toggle_trust(self) -> None:
        repository = self._selected_repository
        if repository is None:
            return
        trusted = not repository.trusted
        approved = await self._product_context.ui.confirm(
            "更改 Repository 信任状态",
            f"是否{'信任' if trusted else '撤销信任'} {repository.name}？"
            "信任不是操作系统沙箱，副作用工具仍需逐次审批。",
        )
        if not approved:
            return
        session_control = self._product_context.session_control
        policy = session_control.tool_access_policy
        if isinstance(policy, ToolProfileRuntime) and session_control.is_running:
            self.query_one("#workspace-status", Static).update(
                "Agent 正在运行，信任状态暂不能更改；请等待本轮结束后重试。"
            )
            return

        previous_trust = repository.trusted
        try:
            updated_repository = await self._services.set_repository_trust(
                repository.repository_id,
                trusted=trusted,
            )
            if isinstance(policy, ToolProfileRuntime):
                try:
                    policy.set_repository_trusted(
                        session_control,
                        trusted=trusted
                        and self._services.is_managed_working_copy(
                            updated_repository,
                            self._product_context.cwd,
                        ),
                    )
                except Exception as policy_error:
                    try:
                        await self._services.set_repository_trust(
                            repository.repository_id,
                            trusted=previous_trust,
                        )
                    except Exception as rollback_error:
                        raise RuntimeError(
                            "运行时策略更新失败，且 Repository 信任状态回滚失败："
                            f"{_safe_error(rollback_error)}"
                        ) from policy_error
                    raise
            repository = updated_repository
            self._repositories[repository.repository_id] = repository
            self._selected_repository = repository
            await self._select_repository(repository.repository_id)
        except Exception as exc:
            self.query_one("#workspace-status", Static).update(
                f"信任状态未更新：{_safe_error(exc)}"
            )

    async def _set_profile(self, profile: ToolProfile) -> None:
        policy = self._product_context.session_control.tool_access_policy
        if not isinstance(policy, ToolProfileRuntime):
            self.query_one("#workspace-profile", Static).update(
                "当前 Session 未配置 Sparkle Tool Profile。请通过 sparkle 命令启动。"
            )
            return
        try:
            await policy.set_profile(self._product_context.session_control, profile)
            await self._show_profile()
        except Exception as exc:
            self.query_one("#workspace-profile", Static).update(_safe_error(exc))

    async def _show_profile(self) -> None:
        policy = self._product_context.session_control.tool_access_policy
        if not isinstance(policy, ToolProfileRuntime):
            return
        profile = policy.current_profile(self._product_context.session_control)
        self.query_one("#workspace-profile", Static).update(
            f"当前 Profile：{profile.value}；模型可见工具："
            + ", ".join(self._product_context.session_control.visible_tool_names)
        )

    async def _run_snapshot_query(self, operation: str) -> None:
        repository = self._selected_repository
        if repository is None or repository.current_snapshot_id is None:
            return
        query = self.query_one("#workspace-query", Input).value.strip()
        snapshot_id = repository.current_snapshot_id
        result_widget = self.query_one("#workspace-result", Static)
        result_widget.update(f"正在运行 {operation}…")
        try:
            if operation == "overview":
                result = await self._services.overview(snapshot_id)
            elif operation == "ask":
                result = await self._services.ask(snapshot_id, query)
            elif operation == "explain":
                result = await self._services.explain(snapshot_id, query)
            elif operation == "trace":
                start_path, separator, start_symbol = query.partition("::")
                if not separator:
                    raise ValueError("Trace 格式为 文件路径::符号名。")
                result = await self._services.trace_call(
                    snapshot_id,
                    start_path.strip(),
                    start_symbol.strip(),
                )
            else:
                target_path, separator, target_symbol = query.partition("::")
                result = await self._services.impact(
                    snapshot_id,
                    target_path.strip(),
                    target_symbol=target_symbol.strip() if separator else None,
                )
            if operation == "overview" and isinstance(result, SnapshotOverview):
                self._render_snapshot_overview(result)
                result_widget.update(
                    f"Snapshot 概览已刷新：{result.file_count} 个文件，"
                    f"{result.python_file_count} 个 Python 文件。"
                )
                self.query_one("#workspace-result-details", Collapsible).collapsed = True
            else:
                result_widget.update(_format_snapshot_result(result))
                self.query_one("#workspace-result-details", Collapsible).collapsed = False
        except Exception as exc:
            result_widget.update(f"{type(exc).__name__}: {_safe_error(exc)}")
            self.query_one("#workspace-result-details", Collapsible).collapsed = False

    def _render_snapshot_overview(self, overview: SnapshotOverview) -> None:
        file_lines = [
            f"{overview.file_count} 个文件 · {overview.python_file_count} 个 Python 文件"
        ]
        file_lines.extend(f"• {path}" for path in overview.top_level_paths[:8])
        if overview.entry_point_candidates:
            file_lines.extend(
                f"入口：{path}" for path in overview.entry_point_candidates[:3]
            )
        self.query_one("#workspace-file-tree", Static).update("\n".join(file_lines))

        evidence_lines: list[str] = []
        for evidence in overview.evidence.items[:4]:
            location = evidence.path
            if evidence.line_start is not None:
                location += f":{evidence.line_start}"
                if evidence.line_end is not None and evidence.line_end != evidence.line_start:
                    location += f"–{evidence.line_end}"
            first_line = next(
                (line.strip() for line in evidence.snippet.splitlines() if line.strip()),
                "",
            )
            evidence_lines.append(
                f"{location}\n{sanitize_trace_text(first_line)[:120]}"
            )
        self.query_one("#workspace-evidence", Static).update(
            "\n\n".join(evidence_lines) if evidence_lines else "Snapshot 概览暂无文档证据。"
        )

    async def _refresh_session_summary(self) -> None:
        profile = self._product_context.session_control.tool_access_policy
        profile_name = (
            profile.current_profile(self._product_context.session_control).value
            if isinstance(profile, ToolProfileRuntime)
            else "未配置"
        )
        mcp_tools = tuple(
            name
            for name in self._product_context.session_control.visible_tool_names
            if name.startswith("mcp__")
        )
        repository = self._selected_repository
        facts = ()
        if repository is not None and repository.current_snapshot_id is not None:
            try:
                facts = await self._services.project_facts_for_snapshot(
                    repository.repository_id,
                    repository.current_snapshot_id,
                )
            except Exception:
                facts = ()
        learner = await self._services.learner_profile()
        confirmed_facts = sum(fact.status is ProjectFactStatus.CONFIRMED for fact in facts)
        preferences = learner.explanation_preferences if learner is not None else ()
        self.query_one("#workspace-summary", Static).update(
            f"Model：{self._product_context.provider_name} / {self._product_context.model}\n"
            f"Tool Profile：{profile_name}\n"
            f"当前 Snapshot 的 Project Fact：{confirmed_facts} 条已确认 / {len(facts)} 条总计\n"
            f"Learner Profile 讲解偏好：{', '.join(preferences) or '尚未设置'}\n"
            f"当前模型可见 MCP Tools：{', '.join(mcp_tools) or '无'}。"
            "这只表示工具可见性，不表示 MCP Server 当前连接健康。"
        )


def _format_snapshot_result(result: object) -> str:
    return json.dumps(
        sanitize_trace_payload(asdict(result)),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _safe_error(exc: Exception) -> str:
    return sanitize_trace_text(str(exc) or type(exc).__name__)


class EvaluationView(VerticalScroll):
    """浏览 Experiment/Run/Comparison 并导出已保存的报告。"""

    def __init__(self, *, services: SparklePresentationServices) -> None:
        super().__init__(id="sparkle-evaluation-view")
        self._services = services
        self._experiments: tuple = ()
        self._snapshot: ExperimentSnapshot | None = None
        self._selected_run_id: str | None = None
        self._comparisons: tuple[ExperimentComparison, ...] = ()
        self._comparison: ExperimentComparison | None = None
        self._active_handle: ExperimentHandle | None = None

    def compose(self) -> ComposeResult:
        yield Static("Evaluation", classes="panel-title")
        yield Static("独立 Experiment / AgentRun", classes="panel-subtitle")
        yield Static("当前运行", classes="section-title")
        yield Static("尚未启动本地评测。", id="evaluation-run-progress", markup=False)
        yield Static("近期实验", classes="section-title")
        yield Static("尚无持久化 Experiment。", id="evaluation-status", markup=False)
        yield Static("尚无 AgentRun。", id="evaluation-run-summary", markup=False)
        yield Static("评测操作", classes="section-title")
        with Collapsible(
            title="新建评测",
            collapsed=True,
            id="evaluation-run-controls",
        ):
            yield Static("本地证据评测（确定性执行，不调用模型）", classes="section-title")
            yield Select(
                [],
                allow_blank=True,
                prompt="选择 Repository Snapshot",
                id="evaluation-snapshot",
            )
            yield Input(placeholder="公开评测任务 / 检索问题", id="evaluation-input")
            yield Input(
                placeholder="仅评测器读取：预期证据路径，以逗号分隔",
                id="evaluation-expected-paths",
            )
            yield Button("开始评测", id="evaluation-run-case")
            yield Button("取消评测", id="evaluation-cancel", disabled=True)
            yield Static("真实模型评测（Agnes API）", classes="section-title")
            yield Static(
                f"Pilot 最多 {LIVE_MAX_AGENT_RUNS} 个 AgentRun；"
                f"每 Run 最多 {LIVE_MAX_PROVIDER_REQUESTS_PER_RUN} 次请求，"
                f"单次超时 {LIVE_PROVIDER_TIMEOUT_SECONDS} 秒；可能产生费用。",
                markup=False,
            )
            yield Button("开始真实模型评测", id="evaluation-live-run-case")

        with Collapsible(
            title="全部 Runs",
            collapsed=True,
            id="evaluation-run-list",
        ):
            yield DataTable(id="evaluation-runs")
        with Collapsible(
            title="Experiment 筛选与刷新",
            collapsed=True,
            id="evaluation-experiment-controls",
        ):
            yield Select([], allow_blank=True, prompt="选择 Experiment", id="evaluation-experiment")
            yield Button("刷新", id="evaluation-refresh")
        with Collapsible(
            title="Run 详情",
            collapsed=True,
            id="evaluation-run-details",
        ):
            yield Static(
                "选择一个 Run 查看输出、Trace、Evidence、Usage 和 Failure。",
                id="evaluation-run-detail",
                markup=False,
            )
        with Collapsible(
            title="比较与报告",
            collapsed=True,
            id="evaluation-comparison-controls",
        ):
            yield Static("Comparison / Regression", classes="section-title")
            yield Select([], allow_blank=True, prompt="Baseline", id="evaluation-baseline")
            yield Select([], allow_blank=True, prompt="Candidate", id="evaluation-candidate")
            yield Button("比较", id="evaluation-compare")
            yield Select(
                [],
                allow_blank=True,
                prompt="已保存 Comparison",
                id="evaluation-comparison",
            )
            yield Button("刷新比较", id="evaluation-refresh-comparisons")
            yield Static("尚无 Comparison。", id="evaluation-comparison-result", markup=False)
            yield Button("导出 Experiment HTML", id="export-experiment-html")
            yield Button("导出 Experiment JSON", id="export-experiment-json")
            yield Button("导出 Experiment CSV", id="export-experiment-csv")
            yield Button("导出 Comparison HTML", id="export-comparison-html")
            yield Button("导出 Comparison JSON", id="export-comparison-json")
            yield Button("导出 Comparison CSV", id="export-comparison-csv")
            yield Static("", id="evaluation-export-status", markup=False)

    async def on_mount(self) -> None:
        table = self.query_one("#evaluation-runs", DataTable)
        table.add_columns("Case", "状态", "指标")
        await self.refresh_snapshot_options()
        await self.refresh_experiments()

    async def refresh_snapshot_options(self) -> None:
        progress = self.query_one("#evaluation-run-progress", Static)
        try:
            repositories = await self._services.list_repositories()
            options = [
                (
                    f"{repository.name} · {repository.current_snapshot_id[:10]}",
                    repository.current_snapshot_id,
                )
                for repository in repositories
                if repository.current_snapshot_id is not None
            ]
            selector = self.query_one("#evaluation-snapshot", Select)
            selector.set_options(options)
            if options:
                selector.value = options[0][1]
            else:
                progress.update("请先在 Workspace 导入 Repository 并建立 Snapshot 索引。")
        except Exception as exc:
            progress.update(f"读取 Snapshot 失败：{_safe_error(exc)}")

    async def refresh_experiments(self) -> None:
        status = self.query_one("#evaluation-status", Static)
        status.update("正在读取持久化 Experiment…")
        try:
            self._experiments = await self._services.list_experiments(limit=50)
            options = [
                (
                    f"{record.suite_id} · {record.status.value} · {record.experiment_id[:8]}",
                    record.experiment_id,
                )
                for record in self._experiments
            ]
            self.query_one("#evaluation-experiment", Select).set_options(options)
            self.query_one("#evaluation-baseline", Select).set_options(options)
            self.query_one("#evaluation-candidate", Select).set_options(options)
            if not self._experiments:
                self._snapshot = None
                self.query_one("#evaluation-runs", DataTable).clear()
                status.update("尚无持久化 Experiment。")
                self.query_one("#evaluation-run-summary", Static).update(
                    "尚无 AgentRun。"
                )
            else:
                self.query_one("#evaluation-experiment", Select).value = (
                    self._experiments[0].experiment_id
                )
                await self._select_experiment(self._experiments[0].experiment_id)
                status.update(
                    f"已读取 {len(self._experiments)} 个 Experiment；"
                    "选择 Run 可查看只读详情。"
                )
            await self.refresh_comparisons()
        except Exception as exc:
            status.update(f"读取 Evaluation 失败：{_safe_error(exc)}")

    async def refresh_comparisons(self) -> None:
        try:
            self._comparisons = await self._services.list_comparisons(limit=50)
            self.query_one("#evaluation-comparison", Select).set_options(
                (
                    (
                        f"{item.baseline_experiment_id[:8]} → "
                        f"{item.candidate_experiment_id[:8]} · {item.comparison_id[:8]}",
                        item.comparison_id,
                    )
                    for item in self._comparisons
                )
            )
            if self._comparisons:
                comparison = self._comparisons[0]
                self.query_one("#evaluation-comparison", Select).value = (
                    comparison.comparison_id
                )
                self._comparison = comparison
                self._render_comparison(comparison)
        except Exception as exc:
            self.query_one("#evaluation-comparison-result", Static).update(
                f"读取 Comparison 失败：{_safe_error(exc)}"
            )

    async def on_select_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.select.id == "evaluation-experiment":
            await self._select_experiment(event.value)
        elif event.select.id == "evaluation-comparison":
            await self._select_comparison(event.value)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "evaluation-runs" or self._snapshot is None:
            return
        run_id = str(event.row_key.value)
        try:
            detail = await self._services.get_run(
                self._snapshot.experiment.experiment_id,
                run_id,
            )
            self._selected_run_id = run_id
            self.query_one("#evaluation-run-detail", Static).update(
                _format_run_detail(detail)
            )
            self.query_one("#evaluation-run-list", Collapsible).collapsed = False
            self.query_one("#evaluation-run-details", Collapsible).collapsed = False
        except Exception as exc:
            self.query_one("#evaluation-run-detail", Static).update(_safe_error(exc))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "evaluation-refresh":
            await self.refresh_experiments()
        elif button_id == "evaluation-run-case":
            await self._start_evidence_evaluation()
        elif button_id == "evaluation-live-run-case":
            await self._start_evidence_evaluation(live=True)
        elif button_id == "evaluation-cancel":
            await self._cancel_evidence_evaluation()
        elif button_id == "evaluation-refresh-comparisons":
            await self.refresh_comparisons()
        elif button_id == "evaluation-compare":
            await self._compare()
        elif button_id.startswith("export-experiment-"):
            await self._export_experiment(button_id.removeprefix("export-experiment-"))
        elif button_id.startswith("export-comparison-"):
            await self._export_comparison(button_id.removeprefix("export-comparison-"))

    async def _start_evidence_evaluation(self, *, live: bool = False) -> None:
        progress = self.query_one("#evaluation-run-progress", Static)
        snapshot_id = self.query_one("#evaluation-snapshot", Select).value
        user_input = self.query_one("#evaluation-input", Input).value.strip()
        expected_paths = tuple(
            path.strip()
            for path in self.query_one("#evaluation-expected-paths", Input).value.split(",")
        )
        if not isinstance(snapshot_id, str):
            progress.update("请选择一个已有 Snapshot。")
            return
        if not user_input:
            progress.update("请输入公开评测任务。")
            return
        if not expected_paths or any(not path for path in expected_paths):
            progress.update("请输入至少一个预期证据路径。")
            return
        if self._active_handle is not None:
            progress.update("当前已有评测运行中；请等待结束或先取消。")
            return

        try:
            if live:
                handle = await self._services.start_live_evidence_evaluation(
                    snapshot_id=snapshot_id,
                    user_input=user_input,
                    expected_paths=expected_paths,
                )
            else:
                handle = await self._services.start_evidence_evaluation(
                    snapshot_id=snapshot_id,
                    user_input=user_input,
                    expected_paths=expected_paths,
                )
            self._active_handle = handle
            self.query_one("#evaluation-run-case", Button).disabled = True
            self.query_one("#evaluation-live-run-case", Button).disabled = True
            self.query_one("#evaluation-cancel", Button).disabled = False
            mode = "真实模型" if live else "本地确定性"
            progress.update(f"{mode} Experiment {handle.experiment_id[:8]} · 已排队。")
            self.run_worker(
                self._follow_evidence_evaluation(handle),
                group="evaluation-progress",
                exit_on_error=False,
                exclusive=True,
            )
        except Exception as exc:
            progress.update(f"启动评测失败：{_safe_error(exc)}")

    async def _follow_evidence_evaluation(self, handle: ExperimentHandle) -> None:
        progress = self.query_one("#evaluation-run-progress", Static)
        try:
            async for event in handle.events():
                snapshot = await handle.snapshot()
                terminal_runs = sum(run.status.is_terminal for run in snapshot.runs)
                total_runs = len(snapshot.runs)
                progress.update(
                    f"{snapshot.experiment.status.value} · "
                    f"{terminal_runs}/{total_runs} Runs 进入终态 · "
                    f"Event {event.sequence}: {event.event_type}"
                )
                if snapshot.experiment.status.is_terminal:
                    await self.refresh_experiments()
                    self.query_one("#evaluation-experiment", Select).value = (
                        handle.experiment_id
                    )
                    await self._select_experiment(handle.experiment_id)
        except Exception as exc:
            progress.update(f"读取评测进度失败：{_safe_error(exc)}")
        finally:
            self._active_handle = None
            if self.is_mounted:
                self.query_one("#evaluation-run-case", Button).disabled = False
                self.query_one("#evaluation-live-run-case", Button).disabled = False
                self.query_one("#evaluation-cancel", Button).disabled = True

    async def _cancel_evidence_evaluation(self) -> None:
        handle = self._active_handle
        if handle is None:
            return
        try:
            await handle.cancel()
            self.query_one("#evaluation-run-progress", Static).update(
                "已请求取消；正在读取持久化终态。"
            )
        except Exception as exc:
            self.query_one("#evaluation-run-progress", Static).update(
                f"取消评测失败：{_safe_error(exc)}"
            )

    async def _select_experiment(self, experiment_id: str) -> None:
        self.query_one("#evaluation-status", Static).update("正在读取 Experiment Snapshot…")
        try:
            self._snapshot = await self._services.get_experiment(experiment_id)
            table = self.query_one("#evaluation-runs", DataTable)
            table.clear()
            for run in self._snapshot.runs:
                metric_summary = ", ".join(
                    f"{item.metric_name}={item.value}"
                    for item in run.results
                ) or "无"
                table.add_row(
                    run.case_id,
                    run.status.value,
                    metric_summary,
                    key=run.run_id,
                )
            recent_runs = self._snapshot.runs[-3:]
            self.query_one("#evaluation-run-summary", Static).update(
                "\n\n".join(
                    f"● {run.status.value} · {run.case_id}\n"
                    + (
                        ", ".join(
                            f"{item.metric_name}={item.value}"
                            for item in run.results
                        )
                        or "尚无指标结果"
                    )
                    for run in reversed(recent_runs)
                )
                or "该 Experiment 尚无 AgentRun。"
            )
            self.query_one("#evaluation-status", Static).update(
                f"{self._snapshot.experiment.suite_id} · "
                f"{self._snapshot.experiment.status.value} · "
                f"{len(self._snapshot.runs)} Runs · "
                f"Snapshot {self._snapshot.experiment.snapshot_id or '未绑定'}"
            )
        except Exception as exc:
            self._snapshot = None
            self.query_one("#evaluation-status", Static).update(_safe_error(exc))

    async def _select_comparison(self, comparison_id: str) -> None:
        try:
            self._comparison = await self._services.get_comparison(comparison_id)
            self._render_comparison(self._comparison)
        except Exception as exc:
            self._comparison = None
            self.query_one("#evaluation-comparison-result", Static).update(_safe_error(exc))

    async def _compare(self) -> None:
        baseline_id = self.query_one("#evaluation-baseline", Select).value
        candidate_id = self.query_one("#evaluation-candidate", Select).value
        if not isinstance(baseline_id, str) or not isinstance(candidate_id, str):
            self.query_one("#evaluation-comparison-result", Static).update(
                "请选择 Baseline 和 Candidate。"
            )
            return
        try:
            self._comparison = await self._services.compare_experiments(
                baseline_id,
                candidate_id,
            )
            await self.refresh_comparisons()
            self._render_comparison(self._comparison)
        except Exception as exc:
            self.query_one("#evaluation-comparison-result", Static).update(
                f"比较失败：{_safe_error(exc)}"
            )

    def _render_comparison(self, comparison: ExperimentComparison) -> None:
        lines = [
            f"Baseline：{comparison.baseline_experiment_id}",
            f"Candidate：{comparison.candidate_experiment_id}",
        ]
        lines.extend(
            f"{result.metric_name} ({result.direction.value})："
            f"{result.baseline_mean:.4f} → {result.candidate_mean:.4f}; "
            f"Δ={result.delta:+.4f}; 阈值={result.regression_threshold:.4f}; "
            f"{'回归' if result.is_regression else '未回归'}"
            for result in comparison.results
        )
        self.query_one("#evaluation-comparison-result", Static).update(
            sanitize_trace_text("\n".join(lines))
        )

    async def _export_experiment(self, format_name: str) -> None:
        if self._snapshot is None or format_name not in {"html", "json", "csv"}:
            return
        destination = _report_path(
            self._services.reports_root,
            "experiment",
            self._snapshot.experiment.experiment_id,
            format_name,
        )
        try:
            exported = await self._services.export_experiment(
                self._snapshot.experiment.experiment_id,
                destination,
                format=format_name,
            )
            self.query_one("#evaluation-export-status", Static).update(
                f"报告已导出：{exported}"
            )
        except Exception as exc:
            self.query_one("#evaluation-export-status", Static).update(
                f"导出失败：{_safe_error(exc)}"
            )

    async def _export_comparison(self, format_name: str) -> None:
        if self._comparison is None or format_name not in {"html", "json", "csv"}:
            return
        destination = _report_path(
            self._services.reports_root,
            "comparison",
            self._comparison.comparison_id,
            format_name,
        )
        try:
            exported = await self._services.export_comparison(
                self._comparison,
                destination,
                format=format_name,
            )
            self.query_one("#evaluation-export-status", Static).update(
                f"报告已导出：{exported}"
            )
        except Exception as exc:
            self.query_one("#evaluation-export-status", Static).update(
                f"导出失败：{_safe_error(exc)}"
            )


def _format_run_detail(detail: AgentRunDetail) -> str:
    run = detail.run
    data = {
        "run_id": run.run_id,
        "case_id": run.case_id,
        "status": run.status.value,
        "output": run.output,
        "failure": {
            "reason": run.failure_reason,
            "message": run.failure_message,
        },
        "usage": asdict(run.usage) if run.usage else None,
        "timing": asdict(run.timing) if run.timing else None,
        "results": [
            {
                "metric": result.metric_name,
                "value": result.value,
                "direction": result.direction.value,
                "details": dict(result.details),
            }
            for result in run.results
        ],
        "trace_events": [asdict(event) for event in detail.trace_events],
        "evidence": [asdict(item) for item in detail.evidence],
    }
    sanitized = sanitize_trace_payload(data)
    return json.dumps(sanitized, ensure_ascii=False, indent=2, default=str)


def _report_path(root: Path, kind: str, identity: str, format_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return root / f"{kind}-{identity}-{timestamp}-{suffix}.{format_name}"
