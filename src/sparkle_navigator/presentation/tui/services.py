"""为 Tau Textual 视图编排 Snapshot Workspace 与 Evaluation 读取。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock

from tau_coding import CodingSessionConfig

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.snapshot_explorer import SnapshotExplorer
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.evidence import (
    SnapshotAnswer,
    SnapshotExplanation,
    SnapshotImpact,
    SnapshotOverview,
    SnapshotTrace,
)
from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    AgentRunInput,
    EvaluationCase,
    EvaluationRequest,
    ExperimentComparison,
    ExperimentRecord,
    ExperimentSnapshot,
)
from sparkle_navigator.domain.memory import LearnerProfile, ProjectFact
from sparkle_navigator.domain.repository import (
    GitRepositorySource,
    LocalRepositorySource,
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    RepositoryWorkingCopy,
)
from sparkle_navigator.evaluation.core import EvaluationLab, ExperimentHandle
from sparkle_navigator.evaluation.live_provider import (
    LIVE_MAX_AGENT_RUNS,
    LIVE_MAX_PROVIDER_REQUESTS_PER_RUN,
    LIVE_PROVIDER_MAX_RETRIES,
    LIVE_PROVIDER_MAX_TOKENS,
    LIVE_PROVIDER_TIMEOUT_SECONDS,
    BoundedModelProvider,
    create_tau_live_session_config_factory,
)
from sparkle_navigator.evaluation.native_backend import (
    CodingSessionConfigFactory,
    NativeBackend,
)
from sparkle_navigator.evaluation.run_explorer import RunExplorer
from sparkle_navigator.evaluation.smoke_set import (
    EvidencePathEvaluator,
    SnapshotExplorerRunner,
)
from sparkle_navigator.memory.learner_profile import LearnerProfileStore
from sparkle_navigator.memory.project_memory import ProjectMemory
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.presentation.reports import (
    EvaluationReportExporter,
    ReportFormat,
)
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from sparkle_navigator.workspace.repository_import import RepositoryWorkspace


class SparklePresentationServices:
    """提供 Workspace/Evaluation 视图所需的本机数据操作。"""

    def __init__(
        self,
        *,
        database_path: Path,
        snapshots_root: Path,
        working_copies_root: Path,
        reports_root: Path,
        live_session_config_factory: CodingSessionConfigFactory | None = None,
    ) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.snapshots_root = snapshots_root.expanduser().resolve()
        self.working_copies_root = working_copies_root.expanduser().resolve()
        self.reports_root = reports_root.expanduser().resolve()
        self._service_lock = RLock()
        self._live_start_lock = asyncio.Lock()
        self._repository_store_instance: RepositoryStore | None = None
        self._repository_workspace = RepositoryWorkspace(
            database_path=self.database_path,
            snapshots_root=self.snapshots_root,
            working_copies_root=self.working_copies_root,
        )
        self._snapshot_explorer_instance: SnapshotExplorer | None = None
        self._text_index_instance: SnapshotTextIndex | None = None
        self._symbol_index_instance: SnapshotSymbolIndex | None = None
        self._code_graph_instance: SnapshotCodeGraph | None = None
        self._run_explorer_instance: RunExplorer | None = None
        self._evaluation_lab_instance: EvaluationLab | None = None
        self._live_session_config_factory = live_session_config_factory
        self._project_memory_instance: ProjectMemory | None = None
        self._learner_profiles_instance: LearnerProfileStore | None = None
        self._report_exporter = EvaluationReportExporter()

    async def list_repositories(self) -> tuple[Repository, ...]:
        """列出本机已导入的 Repository 与当前 Snapshot。"""

        store = await asyncio.to_thread(self._repository_store)
        return await asyncio.to_thread(store.list_repositories)

    async def import_local_repository(self, path: Path) -> RepositoryImportResult:
        """冻结本地 Git 工作树；索引由独立方法显式构建。"""

        workspace = await asyncio.to_thread(self._repository_importer)
        return await workspace.import_source(
            LocalRepositorySource(path=path)
        )

    async def import_git_repository(
        self,
        url: str,
        *,
        ref: str | None = None,
    ) -> RepositoryImportResult:
        """从公开 Git URL 导入仓库并创建不可变 Snapshot。"""

        workspace = await asyncio.to_thread(self._repository_importer)
        return await workspace.import_source(
            GitRepositorySource(url=url, ref=ref)
        )

    async def build_snapshot_indexes(self, snapshot_id: str) -> tuple[object, ...]:
        """构建 FTS5、AST 符号和 Code Graph 索引。"""

        store = await asyncio.to_thread(self._repository_store)
        snapshot = await asyncio.to_thread(store.get_snapshot, snapshot_id)
        text_index = await asyncio.to_thread(self._text_index)
        symbol_index = await asyncio.to_thread(self._symbol_index)
        code_graph = await asyncio.to_thread(self._code_graph)
        text_result = await text_index.build(snapshot)
        symbol_result = await symbol_index.build(snapshot)
        graph_result = await code_graph.build(snapshot)
        return text_result, symbol_result, graph_result

    async def get_repository(self, repository_id: str) -> Repository:
        store = await asyncio.to_thread(self._repository_store)
        return await asyncio.to_thread(store.get_repository, repository_id)

    async def get_repository_snapshot(
        self, repository_id: str, snapshot_id: str,
    ) -> RepositorySnapshot:
        store = await asyncio.to_thread(self._repository_store)
        return await asyncio.to_thread(store.get_repository_snapshot, repository_id, snapshot_id)

    async def repository_for_cwd(self, cwd: Path) -> Repository | None:
        """按来源目录或受管 Working Copy 根目录匹配当前 Repository。"""

        resolved_cwd = cwd.expanduser().resolve()
        repositories = await self.list_repositories()
        matches: list[Repository] = []
        for repository in repositories:
            if repository.current_snapshot_id is None:
                continue
            working_copy_root = (
                self.working_copies_root
                / repository.repository_id
                / repository.current_snapshot_id
            ).resolve()
            if resolved_cwd == working_copy_root:
                return repository
            if repository.source_type == "local":
                try:
                    if resolved_cwd == Path(repository.source_locator).resolve():
                        matches.append(repository)
                except (OSError, RuntimeError):
                    continue
        return matches[0] if len(matches) == 1 else None

    def is_managed_working_copy(self, repository: Repository, cwd: Path) -> bool:
        """判断当前 Tau cwd 是否正是该 Repository 当前 Snapshot 的工作副本。"""

        if repository.current_snapshot_id is None:
            return False
        expected = (
            self.working_copies_root
            / repository.repository_id
            / repository.current_snapshot_id
        ).resolve()
        try:
            return cwd.expanduser().resolve(strict=True) == expected
        except (OSError, RuntimeError):
            return False

    async def set_repository_trust(
        self,
        repository_id: str,
        *,
        trusted: bool,
    ) -> Repository:
        store = await asyncio.to_thread(self._repository_store)
        return await asyncio.to_thread(
            store.set_repository_trust,
            repository_id,
            trusted=trusted,
        )

    async def open_working_copy(self, repository_id: str) -> RepositoryWorkingCopy:
        workspace = await asyncio.to_thread(self._repository_importer)
        return await workspace.open_working_copy(repository_id)

    async def overview(self, snapshot_id: str) -> SnapshotOverview:
        explorer = await asyncio.to_thread(self._snapshot_explorer)
        return await explorer.overview(snapshot_id)

    async def ask(self, snapshot_id: str, question: str) -> SnapshotAnswer:
        explorer = await asyncio.to_thread(self._snapshot_explorer)
        return await explorer.ask(snapshot_id, question)

    async def explain(self, snapshot_id: str, target: str) -> SnapshotExplanation:
        explorer = await asyncio.to_thread(self._snapshot_explorer)
        return await explorer.explain(snapshot_id, target)

    async def trace_call(
        self,
        snapshot_id: str,
        start_path: str,
        start_symbol: str,
        *,
        max_depth: int = 3,
    ) -> SnapshotTrace:
        explorer = await asyncio.to_thread(self._snapshot_explorer)
        return await explorer.trace_call(
            snapshot_id,
            start_path,
            start_symbol,
            max_depth=max_depth,
        )

    async def impact(
        self,
        snapshot_id: str,
        target_path: str,
        *,
        target_symbol: str | None = None,
        max_depth: int = 3,
    ) -> SnapshotImpact:
        explorer = await asyncio.to_thread(self._snapshot_explorer)
        return await explorer.impact(
            snapshot_id,
            target_path,
            target_symbol=target_symbol,
            max_depth=max_depth,
        )

    async def project_facts_for_snapshot(
        self,
        repository_id: str,
        snapshot_id: str,
    ) -> tuple[ProjectFact, ...]:
        memory = await asyncio.to_thread(self._project_memory)
        return await asyncio.to_thread(
            memory.facts_for_snapshot,
            repository_id=repository_id,
            snapshot_id=snapshot_id,
        )

    async def learner_profile(self) -> LearnerProfile | None:
        profiles = await asyncio.to_thread(self._learner_profiles)
        return await asyncio.to_thread(profiles.get, "local")

    async def list_experiments(
        self,
        *,
        limit: int = 50,
    ) -> tuple[ExperimentRecord, ...]:
        explorer = await asyncio.to_thread(self._run_explorer)
        return await explorer.list_experiments(limit=limit)

    async def get_experiment(self, experiment_id: str) -> ExperimentSnapshot:
        explorer = await asyncio.to_thread(self._run_explorer)
        return await explorer.get_experiment(experiment_id)

    async def get_run(self, experiment_id: str, run_id: str) -> AgentRunDetail:
        explorer = await asyncio.to_thread(self._run_explorer)
        return await explorer.get_run(experiment_id, run_id)

    async def list_comparisons(
        self,
        *,
        limit: int = 50,
    ) -> tuple[ExperimentComparison, ...]:
        explorer = await asyncio.to_thread(self._run_explorer)
        return await explorer.list_comparisons(limit=limit)

    async def get_comparison(self, comparison_id: str) -> ExperimentComparison:
        explorer = await asyncio.to_thread(self._run_explorer)
        return await explorer.get_comparison(comparison_id)

    async def compare_experiments(
        self,
        baseline_experiment_id: str,
        candidate_experiment_id: str,
    ) -> ExperimentComparison:
        lab = await asyncio.to_thread(self._evaluation_lab)
        return await lab.compare(
            baseline_experiment_id,
            candidate_experiment_id,
        )

    async def start_evidence_evaluation(
        self,
        *,
        snapshot_id: str,
        user_input: str,
        expected_paths: Sequence[str],
    ) -> ExperimentHandle:
        """在固定 Snapshot 上运行一个不调用模型的证据检索评测。"""

        normalized_paths = tuple(
            path.strip().replace("\\", "/") for path in expected_paths
        )
        if not user_input.strip():
            raise ValueError("评测任务输入不能为空。")
        if not normalized_paths or any(not path for path in normalized_paths):
            raise ValueError("至少需要一个非空的预期证据路径。")
        canonical_paths = tuple(sorted(set(normalized_paths)))

        store = await asyncio.to_thread(self._repository_store)
        snapshot = await asyncio.to_thread(store.get_snapshot, snapshot_id)
        snapshot_paths = {item.path for item in snapshot.files}
        missing_paths = tuple(
            path for path in canonical_paths if path not in snapshot_paths
        )
        if missing_paths:
            raise ValueError(
                f"Expected paths not present in Snapshot: {missing_paths}"
            )

        case_input = user_input.strip()
        suite_identity = json.dumps(
            {
                "snapshot_id": snapshot_id,
                "user_input": case_input,
                "expected_paths": canonical_paths,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        suite_digest = hashlib.sha256(suite_identity.encode("utf-8")).hexdigest()[:16]
        request = EvaluationRequest(
            suite_id=f"workspace-evidence-v1-{suite_digest}",
            snapshot_id=snapshot_id,
            cases=(
                EvaluationCase(
                    case_id="manual-evidence",
                    user_input=case_input,
                    ground_truth={"expected_paths": canonical_paths},
                ),
            ),
            configuration={
                "runner": "snapshot_explorer",
                "evaluator": "sparkle.evidence-path@1",
            },
        )
        lab = await asyncio.to_thread(self._evaluation_lab)
        return await lab.evaluate(request)

    async def start_live_evidence_evaluation(
        self,
        *,
        snapshot_id: str,
        user_input: str,
        expected_paths: Sequence[str],
    ) -> ExperimentHandle:
        """用真实 Tau CodingSession 评测一个 Snapshot 证据任务。"""

        normalized_paths = tuple(
            path.strip().replace("\\", "/") for path in expected_paths
        )
        if not user_input.strip():
            raise ValueError("评测任务输入不能为空。")
        if not normalized_paths or any(not path for path in normalized_paths):
            raise ValueError("至少需要一个非空的预期证据路径。")
        canonical_paths = tuple(sorted(set(normalized_paths)))

        store = await asyncio.to_thread(self._repository_store)
        snapshot = await asyncio.to_thread(store.get_snapshot, snapshot_id)
        snapshot_paths = {item.path for item in snapshot.files}
        missing_paths = tuple(
            path for path in canonical_paths if path not in snapshot_paths
        )
        if missing_paths:
            raise ValueError(
                f"Expected paths not present in Snapshot: {missing_paths}"
            )
        snapshot_root = _require_snapshot_root(snapshot)

        provider_name = "注入的测试 Provider"
        model: str | None = None
        redacted_values: tuple[str, ...] = ()
        source_factory = self._live_session_config_factory
        if source_factory is None:
            source_factory, provider_name, model, api_key = (
                create_tau_live_session_config_factory(self.database_path)
            )
            redacted_values = (api_key,)

        def session_config_factory(
            agent_input: AgentRunInput,
        ) -> CodingSessionConfig:
            if agent_input.snapshot_id != snapshot_id:
                raise ValueError("AgentRun Snapshot 与本次真实评测不一致。")
            base_config = source_factory(agent_input)
            if base_config.provider is None:
                raise ValueError("真实评测需要可用的 Tau ModelProvider。")
            config = replace(
                base_config,
                provider=BoundedModelProvider(
                    base_config.provider,
                    max_requests=LIVE_MAX_PROVIDER_REQUESTS_PER_RUN,
                    timeout_seconds=LIVE_PROVIDER_TIMEOUT_SECONDS,
                ),
                cwd=snapshot_root,
                tool_access_policy=ToolProfileRuntime(
                    repository_trusted=False,
                    workspace_root=snapshot_root,
                ),
                owns_initial_provider=True,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
                trust_default="deny",
                trust_interactive=False,
            )
            return add_codebase_tools(
                config,
                database_path=self.database_path,
                snapshot_id=snapshot_id,
            )

        case_input = user_input.strip()
        suite_identity = json.dumps(
            {
                "snapshot_id": snapshot_id,
                "user_input": case_input,
                "expected_paths": canonical_paths,
                "runner": "native_coding_session",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        suite_digest = hashlib.sha256(suite_identity.encode("utf-8")).hexdigest()[:16]
        request = EvaluationRequest(
            suite_id=f"live-evidence-v1-{suite_digest}",
            snapshot_id=snapshot_id,
            cases=(
                EvaluationCase(
                    case_id="live-evidence",
                    user_input=case_input,
                    ground_truth={"expected_paths": canonical_paths},
                ),
            ),
            configuration={
                "runner": "tau_coding_session",
                "evaluator": "sparkle.evidence-path@1",
                "provider": provider_name,
                "model": model,
                "max_provider_requests_per_run": LIVE_MAX_PROVIDER_REQUESTS_PER_RUN,
                "provider_timeout_seconds": LIVE_PROVIDER_TIMEOUT_SECONDS,
                "provider_max_retries": LIVE_PROVIDER_MAX_RETRIES,
                "max_tokens": (
                    LIVE_PROVIDER_MAX_TOKENS if model is not None else None
                ),
            },
        )
        async with self._live_start_lock:
            used_agent_runs = await self._live_agent_runs_used()
            if used_agent_runs >= LIVE_MAX_AGENT_RUNS:
                raise ValueError(
                    f"真实模型 Pilot 已达到 {LIVE_MAX_AGENT_RUNS} 个 AgentRun 上限；"
                    "审阅结果前不能继续。"
                )
            lab = EvaluationLab(
                database_path=self.database_path,
                runner=NativeBackend(
                    session_config_factory=session_config_factory,
                    redacted_values=redacted_values,
                ),
                evaluator=EvidencePathEvaluator(),
            )
            return await lab.evaluate(request)

    async def _live_agent_runs_used(self) -> int:
        experiments = await self.list_experiments(limit=10_000)
        run_count = 0
        for experiment in experiments:
            if (
                experiment.configuration.get("runner") != "tau_coding_session"
                or experiment.configuration.get("provider") != "Agnes API"
            ):
                continue
            snapshot = await self.get_experiment(experiment.experiment_id)
            run_count += len(snapshot.runs)
        return run_count

    async def export_experiment(
        self,
        experiment_id: str,
        output_path: Path,
        *,
        format: ReportFormat,
        overwrite: bool = False,
        redacted_values: tuple[str, ...] = (),
    ) -> Path:
        explorer = await asyncio.to_thread(self._run_explorer)
        snapshot = await explorer.get_experiment(experiment_id)
        details = tuple(
            [
                await explorer.get_run(experiment_id, run.run_id)
                for run in snapshot.runs
            ]
        )
        return await asyncio.to_thread(
            self._report_exporter.export_experiment,
            snapshot,
            details,
            output_path,
            format=format,
            overwrite=overwrite,
            redacted_values=redacted_values,
        )

    async def export_comparison(
        self,
        comparison: ExperimentComparison,
        output_path: Path,
        *,
        format: ReportFormat,
        overwrite: bool = False,
        redacted_values: tuple[str, ...] = (),
    ) -> Path:
        return await asyncio.to_thread(
            self._report_exporter.export_comparison,
            comparison,
            output_path,
            format=format,
            overwrite=overwrite,
            redacted_values=redacted_values,
        )

    def _repository_store(self) -> RepositoryStore:
        with self._service_lock:
            if self._repository_store_instance is None:
                self._repository_store_instance = RepositoryStore(self.database_path)
            return self._repository_store_instance

    def _repository_importer(self) -> RepositoryWorkspace:
        return self._repository_workspace

    def _snapshot_explorer(self) -> SnapshotExplorer:
        with self._service_lock:
            if self._snapshot_explorer_instance is None:
                self._snapshot_explorer_instance = SnapshotExplorer(
                    database_path=self.database_path
                )
            return self._snapshot_explorer_instance

    def _text_index(self) -> SnapshotTextIndex:
        with self._service_lock:
            if self._text_index_instance is None:
                self._text_index_instance = SnapshotTextIndex(
                    database_path=self.database_path
                )
            return self._text_index_instance

    def _symbol_index(self) -> SnapshotSymbolIndex:
        with self._service_lock:
            if self._symbol_index_instance is None:
                self._symbol_index_instance = SnapshotSymbolIndex(
                    database_path=self.database_path
                )
            return self._symbol_index_instance

    def _code_graph(self) -> SnapshotCodeGraph:
        with self._service_lock:
            if self._code_graph_instance is None:
                self._code_graph_instance = SnapshotCodeGraph(
                    database_path=self.database_path
                )
            return self._code_graph_instance

    def _run_explorer(self) -> RunExplorer:
        with self._service_lock:
            if self._run_explorer_instance is None:
                self._run_explorer_instance = RunExplorer(
                    database_path=self.database_path
                )
            return self._run_explorer_instance

    def _evaluation_lab(self) -> EvaluationLab:
        with self._service_lock:
            if self._evaluation_lab_instance is None:
                self._evaluation_lab_instance = EvaluationLab(
                    database_path=self.database_path,
                    runner=SnapshotExplorerRunner(self._snapshot_explorer()),
                    evaluator=EvidencePathEvaluator(),
                )
            return self._evaluation_lab_instance

    def _project_memory(self) -> ProjectMemory:
        with self._service_lock:
            if self._project_memory_instance is None:
                self._project_memory_instance = ProjectMemory(self.database_path)
            return self._project_memory_instance

    def _learner_profiles(self) -> LearnerProfileStore:
        with self._service_lock:
            if self._learner_profiles_instance is None:
                self._learner_profiles_instance = LearnerProfileStore(self.database_path)
            return self._learner_profiles_instance


def default_sparkle_paths(tau_home: Path) -> tuple[Path, Path, Path, Path]:
    """返回 Tau 用户目录下 Sparkle 的数据库、Snapshot、工作副本和报告位置。"""

    root = tau_home / "sparkle-navigator"
    return (
        root / "navigator.sqlite3",
        root / "snapshots",
        root / "working-copies",
        root / "reports",
    )


def _require_snapshot_root(snapshot: RepositorySnapshot) -> Path:
    if snapshot.snapshot_root is None:
        raise ValueError("真实评测需要本机可访问的 Snapshot 文件目录。")
    root = snapshot.snapshot_root.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError("Snapshot 文件目录当前不可访问。")
    return root


__all__ = ["SparklePresentationServices", "default_sparkle_paths"]
