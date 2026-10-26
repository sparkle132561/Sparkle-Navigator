"""Repository/Snapshot 的 SQLAlchemy 持久化模型。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from sparkle_navigator.domain.code_index import RelationshipConfidence, RelationshipType
from sparkle_navigator.domain.semantic import SemanticDocumentKind


class Base(DeclarativeBase):
    """Sparkle SQLite schema root。"""


class RepositorySnapshotRow(Base):
    """一份不可变文件清单和快照副本位置。"""

    __tablename__ = "repository_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    file_manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    index_version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_root: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class RepositorySnapshotFileRow(Base):
    """固定 Snapshot 中的文件哈希及其文本索引状态。"""

    __tablename__ = "repository_snapshot_files"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"),
        primary_key=True,
    )
    path: Mapped[str] = mapped_column(Text, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    indexed_text: Mapped[bool] = mapped_column(Boolean, nullable=False)


class RepositorySnapshotSymbolRow(Base):
    """固定 Snapshot 中的类、函数和方法声明。"""

    __tablename__ = "repository_snapshot_symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    qualified_name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)


class RepositorySnapshotImportRow(Base):
    """固定 Snapshot 中逐 alias 保存的 Python import 记录。"""

    __tablename__ = "repository_snapshot_imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    module: Mapped[str | None] = mapped_column(Text, nullable=True)
    imported_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    alias: Mapped[str | None] = mapped_column(Text, nullable=True)
    relative_level: Mapped[int] = mapped_column(Integer, nullable=False)
    line: Mapped[int] = mapped_column(Integer, nullable=False)


class RepositorySnapshotRelationshipRow(Base):
    """固定 Snapshot 中带置信度和源码行证据的关系边。"""

    __tablename__ = "repository_snapshot_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[RelationshipType] = mapped_column(String(32), nullable=False)
    confidence: Mapped[RelationshipConfidence] = mapped_column(String(16), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_symbol: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_symbol: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_line: Mapped[int] = mapped_column(Integer, nullable=False)


class RepositorySnapshotSemanticVectorRow(Base):
    """固定 Snapshot 中一个文件或符号的模型向量和来源定位。"""

    __tablename__ = "repository_snapshot_semantic_vectors"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    document_kind: Mapped[SemanticDocumentKind] = mapped_column(String(16), nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class RepositoryRow(Base):
    """一个来源仓库及其当前信任状态。"""

    __tablename__ = "repositories"

    repository_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_locator: Mapped[str] = mapped_column(Text, nullable=False)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    current_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class RepositorySnapshotLinkRow(Base):
    """Repository 对 Snapshot 的导入关联。"""

    __tablename__ = "repository_snapshot_links"

    repository_id: Mapped[str] = mapped_column(
        ForeignKey("repositories.repository_id", ondelete="CASCADE"), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id", ondelete="CASCADE"), primary_key=True
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class ProjectFactRow(Base):
    """绑定 Repository、Snapshot 与源码证据的持久项目事实。"""

    __tablename__ = "project_facts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["repository_id", "snapshot_id"],
            [
                "repository_snapshot_links.repository_id",
                "repository_snapshot_links.snapshot_id",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "ix_project_facts_repository_snapshot_status",
            "repository_id",
            "snapshot_id",
            "status",
        ),
    )

    fact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    supersedes_fact_id: Mapped[str | None] = mapped_column(
        ForeignKey("project_facts.fact_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LearnerProfileRow(Base):
    """可由用户读取、修改和删除的本地学习者资料。"""

    __tablename__ = "learner_profiles"

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    learning_progress_json: Mapped[str] = mapped_column(Text, nullable=False)
    confusion_points_json: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_preferences_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class EvaluationExperimentRow(Base):
    """冻结套件、Snapshot 和配置的一次评测实验。"""

    __tablename__ = "evaluation_experiments"

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    suite_id: Mapped[str] = mapped_column(String(255), nullable=False)
    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("repository_snapshots.snapshot_id"),
        nullable=True,
    )
    repetitions: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    configuration_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class EvaluationCaseRow(Base):
    """Experiment 使用的公开输入和独立保存的 Ground Truth。"""

    __tablename__ = "evaluation_cases"
    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id"],
            ["evaluation_experiments.experiment_id"],
            ondelete="CASCADE",
        ),
    )

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_json: Mapped[str] = mapped_column(Text, nullable=False)


class EvaluationAgentRunRow(Base):
    """一个 Case 的运行状态、最终输出和 Tau Session 关联。"""

    __tablename__ = "evaluation_agent_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "case_id"],
            ["evaluation_cases.experiment_id", "evaluation_cases.case_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "experiment_id",
            "case_id",
            "repetition",
            "attempt",
            name="uq_evaluation_agent_run_repetition_attempt",
        ),
        Index("ix_evaluation_agent_runs_experiment_status", "experiment_id", "status"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False)
    case_id: Mapped[str] = mapped_column(String(255), nullable=False)
    repetition: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    tau_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    timing_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class EvaluationResultRow(Base):
    """一个 AgentRun 的已保存指标分项和评分明细。"""

    __tablename__ = "evaluation_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "metric_name",
            "evaluator_id",
            "evaluator_version",
            name="uq_evaluation_result_metric_version",
        ),
    )

    result_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluator_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)


class EvaluationComparisonRow(Base):
    """一次基线与候选 Experiment 的可重复比较。"""

    __tablename__ = "evaluation_comparisons"

    comparison_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    baseline_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_experiments.experiment_id"),
        nullable=False,
    )
    candidate_experiment_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_experiments.experiment_id"),
        nullable=False,
    )
    thresholds_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class EvaluationRegressionResultRow(Base):
    """Comparison 中按 metric/evaluator 版本保存的回归判定。"""

    __tablename__ = "evaluation_regression_results"
    __table_args__ = (
        ForeignKeyConstraint(
            ["comparison_id"],
            ["evaluation_comparisons.comparison_id"],
            ondelete="CASCADE",
        ),
    )

    comparison_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluator_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluator_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    baseline_mean: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_mean: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_variance: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_variance: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)
    regression_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_regression: Mapped[bool] = mapped_column(Boolean, nullable=False)


class EvaluationEventRow(Base):
    """Experiment 事件流；sequence 在单个 Experiment 内严格递增。"""

    __tablename__ = "evaluation_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id"],
            ["evaluation_experiments.experiment_id"],
            ondelete="CASCADE",
        ),
        Index("ix_evaluation_events_run", "run_id", "sequence"),
    )

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class EvaluationRunTraceEventRow(Base):
    """一个 AgentRun 内有序、脱敏后的 Tau CodingSession 事件。"""

    __tablename__ = "evaluation_run_trace_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        Index("ix_evaluation_run_trace_event_type", "run_id", "event_type"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False)


class EvaluationRunEvidenceRow(Base):
    """绑定 AgentRun 的 Snapshot 文件位置与 SHA-256 证据。"""

    __tablename__ = "evaluation_run_evidence"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        Index("ix_evaluation_run_evidence_path", "run_id", "path"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    relationship_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
