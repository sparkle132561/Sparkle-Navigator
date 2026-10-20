"""以确定性文本锚点评估 Compaction 事实保留与过期事实泄漏。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactionMemoryCase:
    """评测侧的关键事实与禁止泄漏事实锚点，不传给被测 Agent。"""

    case_id: str
    critical_fact_anchors: tuple[str, ...]
    stale_fact_anchors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("CompactionMemoryCase.case_id 不能为空。")
        for field_name, anchors in (
            ("critical_fact_anchors", self.critical_fact_anchors),
            ("stale_fact_anchors", self.stale_fact_anchors),
        ):
            if any(not anchor.strip() for anchor in anchors):
                raise ValueError(f"{field_name} 不能包含空锚点。")
            normalized = tuple(_normalize(anchor) for anchor in anchors)
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{field_name} 不能包含重复锚点。")
        if {
            _normalize(anchor) for anchor in self.critical_fact_anchors
        } & {_normalize(anchor) for anchor in self.stale_fact_anchors}:
            raise ValueError("同一个锚点不能同时标记为关键事实和过期事实。")


@dataclass(frozen=True, slots=True)
class CompactionMemoryResult:
    """一次压缩/记忆评价的分项指标及可复核锚点。"""

    case_id: str
    critical_fact_retention: float | None
    stale_fact_leakage: float | None
    critical_fact_retention_numerator: int
    critical_fact_retention_denominator: int
    stale_fact_leakage_numerator: int
    stale_fact_leakage_denominator: int
    retained_critical_fact_anchors: tuple[str, ...]
    missing_critical_fact_anchors: tuple[str, ...]
    unobserved_critical_fact_anchors: tuple[str, ...]
    leaked_stale_fact_anchors: tuple[str, ...]


class CompactionMemoryEvaluator:
    """计算压缩前后关键事实锚点的保留率与过期锚点的泄漏率。"""

    def evaluate(
        self,
        case: CompactionMemoryCase,
        *,
        before_context: str,
        after_context: str,
    ) -> CompactionMemoryResult:
        """对模型可见文本作大小写/空白规范化后的字面锚点匹配。"""

        before = _normalize(before_context)
        after = _normalize(after_context)
        observed_critical = tuple(
            anchor
            for anchor in case.critical_fact_anchors
            if _normalize(anchor) in before
        )
        retained = tuple(
            anchor
            for anchor in observed_critical
            if _normalize(anchor) in after
        )
        missing = tuple(anchor for anchor in observed_critical if anchor not in retained)
        unobserved = tuple(
            anchor
            for anchor in case.critical_fact_anchors
            if anchor not in observed_critical
        )
        leaked_stale = tuple(
            anchor
            for anchor in case.stale_fact_anchors
            if _normalize(anchor) in after
        )
        critical_retention = (
            len(retained) / len(observed_critical) if observed_critical else None
        )
        stale_leakage = (
            len(leaked_stale) / len(case.stale_fact_anchors)
            if case.stale_fact_anchors
            else None
        )
        return CompactionMemoryResult(
            case_id=case.case_id,
            critical_fact_retention=critical_retention,
            stale_fact_leakage=stale_leakage,
            critical_fact_retention_numerator=len(retained),
            critical_fact_retention_denominator=len(observed_critical),
            stale_fact_leakage_numerator=len(leaked_stale),
            stale_fact_leakage_denominator=len(case.stale_fact_anchors),
            retained_critical_fact_anchors=retained,
            missing_critical_fact_anchors=missing,
            unobserved_critical_fact_anchors=unobserved,
            leaked_stale_fact_anchors=leaked_stale,
        )


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


__all__ = [
    "CompactionMemoryCase",
    "CompactionMemoryEvaluator",
    "CompactionMemoryResult",
]
