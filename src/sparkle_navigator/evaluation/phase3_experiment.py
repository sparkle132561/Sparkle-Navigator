"""阶段三 A/B 对比实验的数据集加载、锚点核验、变体配置与确定性评分。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparkle_navigator.context.context_manager import ContextManager
from sparkle_navigator.runtime.citation_verification import extract_citations
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.context_injection import add_context_manager
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from tau_coding.session import CodingSessionConfig

VARIANT_A = "A"
VARIANT_B = "B"
VARIANT_C = "C"

_SYSTEM_PROMPT_A = (
    "你在一个固定的代码仓库工作副本中回答源码理解问题。"
    "可以使用只读工具查看文件；回答必须引用文件路径与行号；"
    "不确定或找不到时明确说明，不要编造。不得修改任何文件。"
)
_SYSTEM_PROMPT_B = (
    "使用 Sparkle Snapshot 只读工具查找并核实代码证据。"
    "回答必须引用文件路径与行号；证据不足时明确说明，不要编造。"
    "不得修改仓库文件；不得假设未检索到的实现。"
)
# C 在 B 之上叠加自动上下文注入；提示说明本轮附加上下文的作用，
# 仍以工具核实为主，不把注入内容当成结论来源。
_SYSTEM_PROMPT_C = _SYSTEM_PROMPT_B + (
    " 每轮可能附带自动检索的参考上下文；它是起点线索，不是结论，"
    "关键结论仍须用工具核实后作答。"
)

_NEGATION_PHRASES = (
    "没有",
    "不支持",
    "未提供",
    "未找到",
    "不存在",
    "无内置",
    "并无",
    "找不到",
    "not found",
    "no built-in",
    "does not",
    "doesn't",
)


@dataclass(frozen=True, slots=True)
class AnchorReport:
    """证据锚点对冻结 Snapshot 的确定性核验结果。"""

    case_id: str
    ok: bool
    problems: tuple[str, ...]


def load_phase3_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取冻结的实验数据集（manifest + cases）。"""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((dataset_dir / "cases.json").read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if len(cases) != manifest["case_count"]:
        raise ValueError("数据集 manifest 的 case_count 与 cases.json 不一致。")
    return manifest, cases


def verify_evidence_anchors(
    cases: list[dict[str, Any]],
    *,
    database_path: Path,
    repository_id: str,
    snapshot_id: str,
) -> tuple[AnchorReport, ...]:
    """核验每条 Case 的 expected_evidence 在冻结 Snapshot 中存在且行号在范围内。"""

    from sparkle_navigator.context._snapshot_files import (
        SnapshotFileScanError,
        read_verified_snapshot_files,
    )
    from sparkle_navigator.persistence.repository_store import RepositoryStore

    snapshot = RepositoryStore(database_path).get_repository_snapshot(
        repository_id, snapshot_id
    )
    files_by_path = {item.path: item for item in snapshot.files}
    anchored_paths = sorted(
        {
            anchor["path"]
            for case in cases
            for anchor in case["ground_truth"].get("expected_evidence", [])
            if anchor.get("path")
        }
    )
    line_counts: dict[str, int] = {}
    for path in anchored_paths:
        if path not in files_by_path:
            line_counts[path] = -1
            continue
        try:
            (_file, _size, content) = read_verified_snapshot_files(snapshot, (path,))[0]
        except SnapshotFileScanError:
            line_counts[path] = -1
            continue
        line_counts[path] = 0 if content is None else len(content.splitlines())

    reports: list[AnchorReport] = []
    for case in cases:
        problems: list[str] = []
        for anchor in case["ground_truth"].get("expected_evidence", []):
            path = anchor.get("path")
            count = line_counts.get(path, -1)
            if count < 0:
                problems.append(f"{path}: 不在 Snapshot 清单中或不可读")
                continue
            start = anchor.get("line_start")
            end = anchor.get("line_end")
            if start is not None and not (1 <= start <= count):
                problems.append(f"{path}:{start} 起始行超出 1-{count}")
            if start is not None and end is not None and not (start <= end <= count):
                problems.append(f"{path}:{end} 结束行超出 {start}-{count}")
        reports.append(
            AnchorReport(
                case_id=case["case_id"], ok=not problems, problems=tuple(problems)
            )
        )
    return tuple(reports)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


# 无行号的裸路径也是位置主张（模型常写 "见 src/tau_agent/loop.py"）；要求含
# 目录分隔符以降低把普通文件名单词当路径的误报。
_BARE_PATH_PATTERN = re.compile(r"[A-Za-z0-9_\-./\\]+\.[A-Za-z0-9]+")


def _claimed_paths(answer_text: str, tool_paths: tuple[str, ...]) -> tuple[str, ...]:
    paths: list[str] = list(tool_paths)
    for ref in extract_citations(answer_text):
        paths.append(ref.path)
    for match in _BARE_PATH_PATTERN.finditer(answer_text):
        raw = match.group(0).replace("\\", "/")
        if "/" in raw and any(char.isalpha() for char in raw):
            paths.append(raw)
    return tuple(dict.fromkeys(paths))


def _ground_truth(case: object) -> dict[str, Any]:
    if isinstance(case, dict):
        return case.get("ground_truth", {})
    return dict(getattr(case, "ground_truth", {}) or {})


def _case_id(case: object) -> str:
    if isinstance(case, dict):
        return str(case.get("case_id", ""))
    return str(getattr(case, "case_id", ""))


def _expected_paths(case: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            anchor["path"]
            for anchor in _ground_truth(case).get("expected_evidence", [])
            if anchor.get("path")
        )
    )


class Phase3Evaluator:
    """确定性评分：答案要点覆盖、证据精确率/召回率、证据不足题的诚实性。

    must_mention 命中是定性核对清单，不是经过校准的数值评分；分值只用于
    排序与对比，部分正确与错误的边界以逐题 details 供人工复核。
    """

    def evaluate(self, case: object, output: object) -> tuple[Any, ...]:
        from sparkle_navigator.domain.evaluation import EvaluationResult, MetricDirection

        ground_truth = _ground_truth(case)
        answer_text = getattr(output, "output", "") or ""
        tool_paths = tuple(getattr(output, "evidence_paths", ()) or ())
        claimed_paths = _claimed_paths(answer_text, tool_paths)
        expected = _expected_paths(case)
        normalized_answer = _normalize(answer_text)
        evaluator = ("sparkle.phase3-deterministic", "1")
        direction = MetricDirection.HIGHER_IS_BETTER
        results: list[Any] = []

        if ground_truth.get("rubric", {}).get("negation_required"):
            negation_hits = [
                phrase for phrase in _NEGATION_PHRASES if _normalize(phrase) in normalized_answer
            ]
            results.append(
                EvaluationResult(
                    metric_name="answer_honesty",
                    value=1.0 if negation_hits else 0.0,
                    details={
                        "negation_found": negation_hits,
                        "claimed_paths": sorted(set(claimed_paths)),
                        "expected_absent_topic": ground_truth.get("rubric", {}).get(
                            "expected_absent_topic"
                        ),
                        "note": "1=明确否认该能力存在；编造与混淆需人工复核 claimed_paths",
                    },
                    evaluator_id=evaluator[0],
                    evaluator_version=evaluator[1],
                    direction=direction,
                )
            )
            return tuple(results)

        must_mention = ground_truth.get("rubric", {}).get("must_mention", ())
        hits = {
            point: _normalize(point) in normalized_answer for point in must_mention
        }
        coverage = (
            sum(1 for hit in hits.values() if hit) / len(hits) if hits else None
        )
        results.append(
            EvaluationResult(
                metric_name="answer_points_coverage",
                value=coverage,
                details={
                    "must_mention_hits": hits,
                    "answer_points": ground_truth.get("answer_points", []),
                    "note": "子串命中为定性代理，未校准分项权重",
                },
                evaluator_id=evaluator[0],
                evaluator_version=evaluator[1],
                direction=direction,
            )
        )
        overlap = set(expected) & set(claimed_paths)
        precision = len(overlap) / len(claimed_paths) if claimed_paths else None
        recall = len(overlap) / len(expected) if expected else None
        results.append(
            EvaluationResult(
                metric_name="evidence_precision",
                value=precision,
                details={
                    "expected_paths": sorted(expected),
                    "tool_evidence_paths": sorted(set(tool_paths)),
                    "claimed_paths": sorted(set(claimed_paths)),
                },
                evaluator_id=evaluator[0],
                evaluator_version=evaluator[1],
                direction=direction,
            )
        )
        results.append(
            EvaluationResult(
                metric_name="evidence_recall",
                value=recall,
                details={
                    "expected_paths": sorted(expected),
                    "matched_paths": sorted(overlap),
                },
                evaluator_id=evaluator[0],
                evaluator_version=evaluator[1],
                direction=direction,
            )
        )
        return tuple(results)


def build_variant_session_config(
    variant: str,
    *,
    base: CodingSessionConfig,
    database_path: Path,
    snapshot_id: str,
    repository_id: str | None = None,
    context_database_path: Path | None = None,
) -> CodingSessionConfig:
    """按变体返回 Session 配置：A=基础只读运行时，B=A+Snapshot 工具，
    C=B+每轮自动上下文注入（ContextManager）。

    三个变体的工具 Profile 与信任设置完全一致；B/C 附加 Snapshot 工具，
    C 额外配置 turn_context_provider。差异全部记录在冻结配置里。

    context_database_path：ContextManager 读取的数据库（含 Repository 与
    项目记忆）。实验场景下与 Snapshot 工具的 database_path 不同——工具只
    依赖快照文件清单，可指向实验库；ContextManager 需查 Repository 记录，
    必须指向导入过该仓库的库（通常是用户产品库）。
    """

    if variant not in {VARIANT_A, VARIANT_B, VARIANT_C}:
        raise ValueError(f"未知实验变体：{variant}")
    if variant == VARIANT_A:
        system = _SYSTEM_PROMPT_A
    else:
        system = _SYSTEM_PROMPT_B if variant == VARIANT_B else _SYSTEM_PROMPT_C
    config = _with_readonly_policy(base, system_prompt=system)
    if variant in {VARIANT_B, VARIANT_C}:
        config = add_codebase_tools(
            config, database_path=database_path, snapshot_id=snapshot_id
        )
    if variant == VARIANT_C:
        if repository_id is None:
            raise ValueError("变体 C 需要 repository_id 以装配 ContextManager。")
        config = add_context_manager(
            config,
            ContextManager(context_database_path or database_path),
            repository_id=repository_id,
            snapshot_id=snapshot_id,
        )
    return config


def _with_readonly_policy(
    base: CodingSessionConfig, *, system_prompt: str
) -> CodingSessionConfig:
    from dataclasses import replace

    policy = ToolProfileRuntime(
        approver=None,
        repository_trusted=False,
        workspace_root=base.cwd,
    )
    return replace(
        base,
        tool_access_policy=policy,
        system=system_prompt,
        trust_default="never",
        trust_interactive=False,
        trust_prompt=None,
    )


def variant_frozen_summary(
    variant: str,
    *,
    model: str,
    provider_name: str,
    max_tokens: int,
    requests_per_run: int,
    timeout_seconds: float,
    max_retries: int,
    runs_cap: int,
) -> dict[str, Any]:
    """冻结配置快照：进入实验报告，保证可追溯。"""

    tool_notes = {
        VARIANT_A: "A=Tau 默认只读工具（无 Snapshot 工具）",
        VARIANT_B: "B=A+Sparkle Snapshot 五工具（tools/skill 关闭）",
        VARIANT_C: "C=B+每轮自动上下文注入（ContextManager，Turn Context 不入会话历史）",
    }
    systems = {
        VARIANT_A: _SYSTEM_PROMPT_A,
        VARIANT_B: _SYSTEM_PROMPT_B,
        VARIANT_C: _SYSTEM_PROMPT_C,
    }
    return {
        "variant": variant,
        "provider_name": provider_name,
        "model": model,
        "max_tokens": max_tokens,
        "provider_requests_per_run_cap": requests_per_run,
        "provider_timeout_seconds": timeout_seconds,
        "provider_max_retries": max_retries,
        "agent_runs_cap": runs_cap,
        "tool_profile": "explain (read-only, repository_trusted=False)",
        "visible_tools_note": tool_notes[variant],
        "system_prompt": systems[variant],
        "execution": "sequential, one CodingSession per AgentRun",
    }


__all__ = [
    "AnchorReport",
    "Phase3Evaluator",
    "VARIANT_A",
    "VARIANT_B",
    "VARIANT_C",
    "build_variant_session_config",
    "load_phase3_dataset",
    "variant_frozen_summary",
    "verify_evidence_anchors",
]
