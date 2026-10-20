"""把 SnapshotExplorer 接入 Evaluation Core 的确定性 Smoke 能力。"""

from __future__ import annotations

from dataclasses import replace

from sparkle_navigator.context.snapshot_explorer import SnapshotExplorer
from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunOutput,
    EvaluationCase,
    EvaluationResult,
)
from sparkle_navigator.security.redaction import sanitize_trace_text


class SnapshotExplorerRunner:
    """用公开 Ask 接口执行单个检索 Smoke Case。"""

    def __init__(
        self,
        explorer: SnapshotExplorer,
        *,
        redacted_values: tuple[str, ...] = (),
    ) -> None:
        self._explorer = explorer
        self._redacted_values = tuple(value for value in redacted_values if value)

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        if agent_input.snapshot_id is None:
            raise ValueError("SnapshotExplorerRunner 需要绑定 Repository Snapshot。")
        answer = await self._explorer.ask(
            agent_input.snapshot_id,
            agent_input.user_input,
        )
        evidence_paths = tuple(
            dict.fromkeys(item.path for item in answer.evidence.items)
        )
        evidence = tuple(
            replace(
                item,
                snippet=sanitize_trace_text(
                    item.snippet,
                    secret_values=self._redacted_values,
                ),
            )
            for item in answer.evidence.items
        )
        return AgentRunOutput(
            output=sanitize_trace_text(
                answer.answer,
                secret_values=self._redacted_values,
            ),
            evidence_paths=evidence_paths,
            evidence=evidence,
        )


class EvidencePathEvaluator:
    """以 Case 中的预期路径评分检索证据的精确率和召回率。"""

    def evaluate(
        self,
        case: EvaluationCase,
        output: AgentRunOutput,
    ) -> tuple[EvaluationResult, ...]:
        raw_expected = case.ground_truth.get("expected_paths")
        if not isinstance(raw_expected, list) or any(
            not isinstance(path, str) or not path for path in raw_expected
        ):
            raise ValueError("EvidencePathEvaluator 需要非空字符串 expected_paths。")
        expected = set(raw_expected)
        retrieved = set(output.evidence_paths)
        overlap = expected & retrieved
        precision = len(overlap) / len(retrieved) if retrieved else 0.0
        recall = len(overlap) / len(expected) if expected else 1.0
        details = {
            "expected_paths": sorted(expected),
            "retrieved_paths": sorted(retrieved),
            "matched_paths": sorted(overlap),
        }
        return (
            EvaluationResult(
                metric_name="evidence_precision",
                value=precision,
                details=details,
                evaluator_id="sparkle.evidence-path",
                evaluator_version="1",
            ),
            EvaluationResult(
                metric_name="evidence_recall",
                value=recall,
                details=details,
                evaluator_id="sparkle.evidence-path",
                evaluator_version="1",
            ),
        )


__all__ = ["EvidencePathEvaluator", "SnapshotExplorerRunner"]
