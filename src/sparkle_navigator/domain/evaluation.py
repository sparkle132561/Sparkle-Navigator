"""Sparkle Navigator 评测核心的领域记录。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from sparkle_navigator.domain.evidence import CodeEvidence


class ExperimentStatus(StrEnum):
    """一次评测批次的生命周期状态。"""

    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        }


class AgentRunStatus(StrEnum):
    """单个 Case 的被测智能体运行状态。"""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }


class MetricDirection(StrEnum):
    """指标分数变大或变小时的质量方向。"""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """公开任务输入与仅供评测器读取的隐藏 Ground Truth。"""

    case_id: str
    user_input: str
    ground_truth: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("EvaluationCase.case_id 不能为空。")
        if not self.user_input.strip():
            raise ValueError("EvaluationCase.user_input 不能为空。")
        object.__setattr__(self, "ground_truth", _freeze_json_object(self.ground_truth))


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """创建一次 Experiment 所需的套件、可选 Snapshot、Case 和冻结配置。"""

    suite_id: str
    snapshot_id: str | None
    cases: tuple[EvaluationCase, ...]
    repetitions: int = 1
    configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suite_id.strip():
            raise ValueError("EvaluationRequest.suite_id 不能为空。")
        if self.snapshot_id is not None and not self.snapshot_id.strip():
            raise ValueError("EvaluationRequest.snapshot_id 不能为空。")
        if not self.cases:
            raise ValueError("EvaluationRequest 至少需要一个 EvaluationCase。")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 1
        ):
            raise ValueError("EvaluationRequest.repetitions 必须是正整数。")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("同一个 EvaluationRequest 不能包含重复 case_id。")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "configuration", _freeze_json_object(self.configuration))


@dataclass(frozen=True, slots=True)
class AgentRunInput:
    """传给被测运行器的公开输入，不包含 Ground Truth。"""

    run_id: str
    case_id: str
    repetition: int
    snapshot_id: str | None
    user_input: str


@dataclass(frozen=True, slots=True)
class AgentRunOutput:
    """被测运行返回的可观察回答、证据路径和 Tau Session 关联。"""

    output: str
    evidence_paths: tuple[str, ...] = ()
    evidence: tuple[CodeEvidence, ...] = ()
    tau_session_id: str | None = None
    trace_events: tuple[RunTraceEvent, ...] = ()
    usage: AgentRunUsage | None = None
    timing: AgentRunTiming | None = None
    failure: AgentRunFailure | None = None
    # 最终非 toolUse 消息的 Provider stop_reason（如 stop/length/error/aborted）；
    # 截断（length）不是失败，但必须可与正常完成区分。
    final_stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunTraceEvent:
    """一条脱敏后的 Tau 原生事件，序号从单个 AgentRun 的 1 开始。"""

    sequence: int
    session_id: str | None
    event_type: str
    timestamp: float
    payload: Mapping[str, object]
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 1 or not self.event_type.strip():
            raise ValueError("RunTraceEvent 需要正序号和非空 event_type。")
        object.__setattr__(self, "payload", _freeze_json_object(self.payload))


@dataclass(frozen=True, slots=True)
class AgentRunUsage:
    """一次 AgentRun 中 Tau Provider Usage 的合计。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: float = 0.0
    cost_cache_write: float = 0.0
    cost_total: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentRunTiming:
    """一次 AgentRun 的实际耗时与 Tau Provider 响应耗时。"""

    run_duration_ms: int
    provider_duration_ms: int
    time_to_first_output_ms: int | None


@dataclass(frozen=True, slots=True)
class AgentRunFailure:
    """可在 Run Explorer 中展示的已脱敏失败摘要。"""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("AgentRunFailure.code 不能为空。")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """单个 AgentRun 的一项版本化分指标及其可复核数据。"""

    metric_name: str
    value: float | None
    details: Mapping[str, object] = field(default_factory=dict)
    evaluator_id: str = "sparkle.deterministic"
    evaluator_version: str = "1"
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER

    def __post_init__(self) -> None:
        if not self.metric_name.strip():
            raise ValueError("EvaluationResult.metric_name 不能为空。")
        if not self.evaluator_id.strip() or not self.evaluator_version.strip():
            raise ValueError("EvaluationResult 必须记录 evaluator_id 和 evaluator_version。")
        if self.value is not None and (
            not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
            or not math.isfinite(self.value)
        ):
            raise ValueError("EvaluationResult.value 必须是有限数值或 None。")
        object.__setattr__(self, "details", _freeze_json_object(self.details))


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """可持久化的 Experiment 当前状态。"""

    experiment_id: str
    suite_id: str
    snapshot_id: str | None
    repetitions: int
    configuration: Mapping[str, object]
    status: ExperimentStatus
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentRunRecord:
    """一个 EvaluationCase 对应的一次实际运行及已保存结果。"""

    run_id: str
    experiment_id: str
    case_id: str
    repetition: int
    attempt: int
    status: AgentRunStatus
    output: str | None
    tau_session_id: str | None
    failure_reason: str | None
    failure_message: str | None
    usage: AgentRunUsage | None
    timing: AgentRunTiming | None
    final_stop_reason: str | None
    results: tuple[EvaluationResult, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EventCursor:
    """指向某个 Experiment 事件流位置的续读游标。"""

    experiment_id: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.experiment_id.strip() or self.sequence < 0:
            raise ValueError("EventCursor 需要有效的 experiment_id 和非负 sequence。")


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    """已写入 SQLite、可按 Experiment 序号续读的状态变化。"""

    experiment_id: str
    sequence: int
    event_type: str
    run_id: str | None
    payload: Mapping[str, object]
    occurred_at: datetime

    @property
    def cursor(self) -> EventCursor:
        return EventCursor(experiment_id=self.experiment_id, sequence=self.sequence)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json_object(self.payload))


@dataclass(frozen=True, slots=True)
class ExperimentSnapshot:
    """一个 Experiment 与按创建顺序读取的 AgentRun 快照。"""

    experiment: ExperimentRecord
    runs: tuple[AgentRunRecord, ...]


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """两个 Experiment 在同一版本化指标上的统计比较。"""

    metric_name: str
    evaluator_id: str
    evaluator_version: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    baseline_variance: float
    candidate_variance: float
    baseline_sample_count: int
    candidate_sample_count: int
    direction: MetricDirection
    regression_threshold: float
    is_regression: bool


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    """可持久化的基线/候选 Experiment 比较。"""

    comparison_id: str
    baseline_experiment_id: str
    candidate_experiment_id: str
    thresholds: Mapping[str, float]
    results: tuple[RegressionResult, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.comparison_id.strip():
            raise ValueError("ExperimentComparison.comparison_id 不能为空。")
        object.__setattr__(self, "thresholds", _freeze_json_object(self.thresholds))


@dataclass(frozen=True, slots=True)
class AgentRunDetail:
    """Run Explorer 展示的一次 AgentRun、Trace 与源码 Evidence。"""

    run: AgentRunRecord
    trace_events: tuple[RunTraceEvent, ...]
    evidence: tuple[CodeEvidence, ...]


def _freeze_json_object(value: Mapping[str, object]) -> Mapping[str, object]:
    try:
        normalized = json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("评测数据只能包含 JSON 可序列化值。") from exc
    if not isinstance(normalized, dict):
        raise ValueError("评测对象必须是 JSON object。")
    return MappingProxyType(normalized)


__all__ = [
    "AgentRunInput",
    "AgentRunDetail",
    "AgentRunFailure",
    "AgentRunOutput",
    "AgentRunRecord",
    "AgentRunStatus",
    "AgentRunTiming",
    "AgentRunUsage",
    "EvaluationCase",
    "EvaluationEvent",
    "EvaluationRequest",
    "EvaluationResult",
    "EventCursor",
    "ExperimentComparison",
    "ExperimentRecord",
    "ExperimentSnapshot",
    "ExperimentStatus",
    "MetricDirection",
    "RegressionResult",
    "RunTraceEvent",
]
