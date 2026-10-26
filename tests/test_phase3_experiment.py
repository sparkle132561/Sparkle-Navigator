"""阶段三 A/B 实验骨架：变体配置、锚点核验、确定性评分与 Lab 集成。"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    AgentRunOutput,
    AgentRunStatus,
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.phase3_experiment import (
    Phase3Evaluator,
    build_variant_session_config,
    load_phase3_dataset,
    variant_frozen_summary,
    verify_evidence_anchors,
)
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from tau_agent.session import InMemorySessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSessionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "benchmarks" / "phase3-ab"


def _base_config(tmp_path: Path) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=FakeProvider([]),
        model="phase3-test",
        cwd=tmp_path,
        storage=InMemorySessionStorage(),
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="never",
    )


def test_dataset_loads_twenty_cases_with_consistent_manifest() -> None:
    manifest, cases = load_phase3_dataset(DATASET_DIR)
    assert manifest["case_count"] == len(cases) == 20
    assert manifest["repositories"][0]["source_commit"] == (
        "a8f18e7bc645d09d47d40b21bc602df97f2579fd"
    )
    insufficient = [
        c for c in cases if c["ground_truth"].get("rubric", {}).get("negation_required")
    ]
    assert len(insufficient) == 2


def test_variant_configs_share_policy_and_differ_in_snapshot_tools(
    tmp_path: Path,
) -> None:
    from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime

    base = _base_config(tmp_path)
    config_a = build_variant_session_config(
        "A", base=base, database_path=tmp_path / "db.sqlite3", snapshot_id="s" * 64
    )
    config_b = build_variant_session_config(
        "B", base=base, database_path=tmp_path / "db.sqlite3", snapshot_id="s" * 64
    )
    assert "Snapshot" not in config_a.system
    assert "Sparkle Snapshot" in config_b.system
    assert config_a.additional_tool_registrations == ()
    b_tool_names = {
        item.tool.name for item in config_b.additional_tool_registrations
    }
    assert {"overview_repository", "explain_code", "ask_codebase"} <= b_tool_names
    assert isinstance(config_a.tool_access_policy, ToolProfileRuntime)
    assert isinstance(config_b.tool_access_policy, ToolProfileRuntime)
    # 两变体信任与 Profile 语义一致：都拒绝未信任仓库的非只读工具。
    assert config_a.trust_default == config_b.trust_default == "never"

    try:
        build_variant_session_config(
            "Z", base=base, database_path=tmp_path / "db.sqlite3", snapshot_id="s" * 64
        )
    except ValueError as exc:
        assert "未知实验变体" in str(exc)
    else:
        raise AssertionError("未知变体应当被拒绝")


def test_evaluator_scores_coverage_and_evidence() -> None:
    case = {
        "case_id": "case-1",
        "ground_truth": {
            "answer_points": ["a", "b"],
            "expected_evidence": [
                {"path": "src/app.py", "line_start": 1, "line_end": 2},
                {"path": "src/other.py", "line_start": 3, "line_end": 4},
            ],
            "rubric": {"must_mention": ["_assistant_events", "TurnEndEvent"]},
        },
    }
    output = AgentRunOutput(
        output=(
            "流程依赖 _assistant_events，结束后有 TurnEndEvent。"
            "见 src/app.py:1-2 与 src/wrong.py:9。"
        ),
        evidence_paths=("src/app.py",),
    )
    results = {
        r.metric_name: r for r in Phase3Evaluator().evaluate(case, output)
    }
    assert results["answer_points_coverage"].value == 1.0
    # 工具证据 + 文本引用并集：app.py 命中，wrong.py 拉低精确率。
    assert results["evidence_precision"].value == 0.5
    assert results["evidence_recall"].value == 0.5
    details = results["evidence_precision"].details
    assert details["tool_evidence_paths"] == ["src/app.py"]
    assert "src/wrong.py" in details["claimed_paths"]


def test_evaluator_scores_insufficient_evidence_honesty() -> None:
    case = {
        "case_id": "case-2",
        "ground_truth": {
            "answer_points": [],
            "expected_evidence": [],
            "rubric": {
                "negation_required": True,
                "expected_absent_topic": "temperature",
            },
        },
    }
    honest = AgentRunOutput(output="没有找到 temperature 配置；当前版本不支持。")
    fabricated = AgentRunOutput(
        output="在 src/tau_ai/config.py:12 设置 temperature 字段即可。"
    )
    results = {
        r.metric_name: r
        for r in Phase3Evaluator().evaluate(case, honest)
    }
    assert results["answer_honesty"].value == 1.0
    fabricated_results = {
        r.metric_name: r
        for r in Phase3Evaluator().evaluate(case, fabricated)
    }
    assert fabricated_results["answer_honesty"].value == 0.0
    assert fabricated_results["answer_honesty"].details["claimed_paths"] == [
        "src/tau_ai/config.py"
    ]


def _register_repo(tmp_path: Path, files: dict[str, str]) -> SparklePresentationServices:
    services = SparklePresentationServices(
        database_path=tmp_path / "navigator.sqlite3",
        snapshots_root=tmp_path / "snapshots",
        working_copies_root=tmp_path / "copies",
        reports_root=tmp_path / "reports",
    )
    source = tmp_path / "source"
    source.mkdir()
    for relative_path, content in files.items():
        target = source.joinpath(*relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    asyncio.run(services.import_local_repository(source))
    return services


def test_verify_evidence_anchors_reports_ok_and_problems(tmp_path: Path) -> None:
    services = _register_repo(
        tmp_path, {"src/app.py": "one\ntwo\nthree\n"}
    )
    (repository,) = asyncio.run(services.list_repositories())
    snapshot = asyncio.run(
        services.get_repository_snapshot(
            repository.repository_id, repository.current_snapshot_id
        )
    )
    cases = [
        {
            "case_id": "ok-case",
            "ground_truth": {
                "expected_evidence": [
                    {"path": "src/app.py", "line_start": 1, "line_end": 3}
                ]
            },
        },
        {
            "case_id": "missing-file",
            "ground_truth": {
                "expected_evidence": [{"path": "src/gone.py", "line_start": 1}]
            },
        },
        {
            "case_id": "out-of-range",
            "ground_truth": {
                "expected_evidence": [
                    {"path": "src/app.py", "line_start": 1, "line_end": 99}
                ]
            },
        },
    ]
    reports = verify_evidence_anchors(
        cases,
        database_path=services.database_path,
        repository_id=repository.repository_id,
        snapshot_id=snapshot.snapshot_id,
    )
    by_id = {r.case_id: r for r in reports}
    assert by_id["ok-case"].ok
    assert not by_id["missing-file"].ok
    assert any("不在 Snapshot 清单" in p for p in by_id["missing-file"].problems)
    assert not by_id["out-of-range"].ok
    assert any("99" in p for p in by_id["out-of-range"].problems)


def test_evaluation_lab_runs_phase3_evaluator_end_to_end(tmp_path: Path) -> None:
    case = {
        "case_id": "lab-case",
        "user_input": "问题",
        "ground_truth": {
            "answer_points": ["x"],
            "expected_evidence": [{"path": "src/app.py", "line_start": 1}],
            "rubric": {"must_mention": ["identity"]},
        },
    }

    class _StubRunner:
        async def run(self, agent_input: object) -> AgentRunOutput:
            return AgentRunOutput(
                output="identity 定义在 src/app.py:1。",
                evidence_paths=("src/app.py",),
            )

    async def main() -> None:
        lab = EvaluationLab(
            database_path=tmp_path / "eval.sqlite3",
            runner=_StubRunner(),
            evaluator=Phase3Evaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="phase3-ab-test",
                snapshot_id=None,
                cases=(
                    EvaluationCase(
                        case_id=case["case_id"],
                        user_input=case["user_input"],
                        ground_truth=case["ground_truth"],
                    ),
                ),
            )
        )
        [event async for event in handle.events()]
        snapshot = await handle.snapshot()
        assert snapshot.experiment.status is ExperimentStatus.COMPLETED
        (run,) = snapshot.runs
        assert run.status is AgentRunStatus.COMPLETED
        metrics = {r.metric_name: r.value for r in run.results}
        assert metrics["answer_points_coverage"] == 1.0
        assert metrics["evidence_precision"] == 1.0
        assert metrics["evidence_recall"] == 1.0

    asyncio.run(main())


def test_frozen_summary_records_variant_differences() -> None:
    a = variant_frozen_summary(
        "A",
        model="agnes-2.5-flash",
        provider_name="agnes",
        max_tokens=2048,
        requests_per_run=12,
        timeout_seconds=45,
        max_retries=0,
        runs_cap=40,
    )
    b = variant_frozen_summary(
        "B",
        model="agnes-2.5-flash",
        provider_name="agnes",
        max_tokens=2048,
        requests_per_run=12,
        timeout_seconds=45,
        max_retries=0,
        runs_cap=40,
    )
    assert a["model"] == b["model"] == "agnes-2.5-flash"
    assert a["visible_tools_note"] != b["visible_tools_note"]
    assert a["system_prompt"] != b["system_prompt"]
    assert "agnes-2.5-flash" not in a["system_prompt"]


def test_variant_c_adds_context_injection_on_top_of_b(tmp_path: Path) -> None:
    config_c = build_variant_session_config(
        "C",
        base=_base_config(tmp_path),
        database_path=tmp_path / "db.sqlite3",
        snapshot_id="s" * 64,
        repository_id="r" * 64,
    )
    assert config_c.turn_context_provider is not None
    assert "参考上下文" in config_c.system
    assert "Sparkle Snapshot" in config_c.system
    c_tools = {item.tool.name for item in config_c.additional_tool_registrations}
    assert "explain_code" in c_tools

    # C 必须显式提供 repository_id。
    try:
        build_variant_session_config(
            "C",
            base=_base_config(tmp_path),
            database_path=tmp_path / "db.sqlite3",
            snapshot_id="s" * 64,
        )
    except ValueError as exc:
        assert "repository_id" in str(exc)
    else:
        raise AssertionError("缺少 repository_id 时应拒绝变体 C")


def test_frozen_summary_covers_all_variants() -> None:
    notes = set()
    for variant in ("A", "B", "C"):
        frozen = variant_frozen_summary(
            variant,
            model="m",
            provider_name="p",
            max_tokens=2048,
            requests_per_run=30,
            timeout_seconds=90,
            max_retries=0,
            runs_cap=40,
        )
        notes.add(frozen["visible_tools_note"])
        assert frozen["variant"] == variant
    assert len(notes) == 3
