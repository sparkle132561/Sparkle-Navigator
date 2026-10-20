"""Compaction/Memory evaluator 的公开行为测试。"""

import pytest

from sparkle_navigator.evaluation.compaction_memory import (
    CompactionMemoryCase,
    CompactionMemoryEvaluator,
)


def test_evaluator_reports_fact_retention_and_stale_leakage_separately() -> None:
    case = CompactionMemoryCase(
        case_id="compaction-anchor-regression",
        critical_fact_anchors=("critical_fact_A", "critical_fact_B"),
        stale_fact_anchors=("STALE_FACT_A", "STALE_FACT_B"),
    )

    result = CompactionMemoryEvaluator().evaluate(
        case,
        before_context="critical_fact_A and critical_fact_B were present.",
        after_context="critical_fact_A remains. STALE_FACT_A leaked.",
    )

    assert result.critical_fact_retention == 0.5
    assert result.stale_fact_leakage == 0.5
    assert result.critical_fact_retention_numerator == 1
    assert result.critical_fact_retention_denominator == 2
    assert result.stale_fact_leakage_numerator == 1
    assert result.stale_fact_leakage_denominator == 2
    assert result.retained_critical_fact_anchors == ("critical_fact_A",)
    assert result.missing_critical_fact_anchors == ("critical_fact_B",)
    assert result.leaked_stale_fact_anchors == ("STALE_FACT_A",)

    clean_result = CompactionMemoryEvaluator().evaluate(
        case,
        before_context="critical_fact_A and critical_fact_B were present.",
        after_context="critical_fact_A remains.",
    )
    assert clean_result.critical_fact_retention == 0.5
    assert clean_result.stale_fact_leakage == 0.0


def test_evaluator_reports_anchors_absent_before_compaction_as_unobserved() -> None:
    case = CompactionMemoryCase(
        case_id="unobserved-critical-anchor",
        critical_fact_anchors=("KNOWN_FACT", "NOT_IN_SOURCE"),
        stale_fact_anchors=(),
    )

    result = CompactionMemoryEvaluator().evaluate(
        case,
        before_context="known_fact appeared before compaction",
        after_context="No required fact remains.",
    )

    assert result.critical_fact_retention == 0.0
    assert result.stale_fact_leakage is None
    assert result.missing_critical_fact_anchors == ("KNOWN_FACT",)
    assert result.unobserved_critical_fact_anchors == ("NOT_IN_SOURCE",)


def test_case_rejects_duplicate_anchors_after_normalization() -> None:
    with pytest.raises(ValueError, match="重复锚点"):
        CompactionMemoryCase(
            case_id="duplicate-anchor",
            critical_fact_anchors=("Important Fact", " important   fact "),
            stale_fact_anchors=(),
        )


def test_case_rejects_an_anchor_that_is_both_required_and_stale() -> None:
    with pytest.raises(ValueError, match="不能同时"):
        CompactionMemoryCase(
            case_id="conflicting-anchor",
            critical_fact_anchors=("same_fact",),
            stale_fact_anchors=("SAME_FACT",),
        )
