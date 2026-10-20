"""使用固定的小型仓库验证初始关键词检索 Smoke Set。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.rank_fusion import reciprocal_rank_fusion
from sparkle_navigator.context.semantic_index import SemanticIndex
from sparkle_navigator.context.snapshot_explorer import SnapshotExplorer
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.evaluation import (
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.evaluation import (
    EvaluationLab,
    EvidencePathEvaluator,
    RunExplorer,
    SnapshotExplorerRunner,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore

SMOKE_SET_ROOT = Path(__file__).parent / "fixtures" / "smoke_set"
SMOKE_SET = json.loads((SMOKE_SET_ROOT / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    SMOKE_SET["cases"],
    ids=lambda case: case["case_id"],
)
def test_smoke_set_case_retrieves_expected_file(tmp_path: Path, case: dict) -> None:
    asyncio.run(_smoke_set_case_retrieves_expected_file(tmp_path, case))


def test_snapshot_explorer_smoke_set_has_repeatable_evidence_scores(
    tmp_path: Path,
) -> None:
    asyncio.run(_snapshot_explorer_smoke_set_has_repeatable_evidence_scores(tmp_path))


@pytest.mark.skipif(
    os.environ.get("SPARKLE_RUN_LOCAL_EMBEDDING_TESTS") != "1",
    reason="设置 SPARKLE_RUN_LOCAL_EMBEDDING_TESTS=1 才下载或运行本地语义模型。",
)
def test_local_semantic_smoke_set_ablation(tmp_path: Path) -> None:
    asyncio.run(_local_semantic_smoke_set_ablation(tmp_path))


async def _smoke_set_case_retrieves_expected_file(
    tmp_path: Path,
    case: dict,
) -> None:
    repository_root = SMOKE_SET_ROOT / "repository"
    snapshot_files = tuple(
        SnapshotFile(
            path=path.relative_to(repository_root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(repository_root.rglob("*"))
        if path.is_file()
    )
    snapshot = RepositorySnapshot.create(
        source_commit="5" * 40,
        files=snapshot_files,
        snapshot_root=repository_root,
    )
    database_path = tmp_path / "smoke-set.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"smoke-set-fixture").hexdigest(),
                name="codebase-retrieval-smoke",
                source_type="local",
                source_locator="test://codebase-retrieval-smoke",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    index = SnapshotTextIndex(database_path=database_path)
    await index.build(snapshot)

    hits = index.search_text(snapshot.snapshot_id, case["query"])
    actual_paths = {hit.path for hit in hits}

    assert set(case["expected_paths"]) <= actual_paths, (
        f"{case['case_id']}：{case['question']}；实际命中：{sorted(actual_paths)}"
    )


async def _snapshot_explorer_smoke_set_has_repeatable_evidence_scores(
    tmp_path: Path,
) -> None:
    repository_root = SMOKE_SET_ROOT / "repository"
    snapshot_files = tuple(
        SnapshotFile(
            path=path.relative_to(repository_root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(repository_root.rglob("*"))
        if path.is_file()
    )
    snapshot = RepositorySnapshot.create(
        source_commit="9" * 40,
        files=snapshot_files,
        snapshot_root=repository_root,
    )
    database_path = tmp_path / "evidence-smoke-set.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"evidence-smoke-set").hexdigest(),
                name="evidence-smoke-set",
                source_type="local",
                source_locator="test://evidence-smoke-set",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    await SnapshotTextIndex(database_path=database_path).build(snapshot)
    await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
    await SnapshotCodeGraph(database_path=database_path).build(snapshot)
    explorer = SnapshotExplorer(database_path=database_path)

    manifest_hashes = {item.path: item.sha256 for item in snapshot.files}
    for case in SMOKE_SET["cases"]:
        result = await explorer.ask(snapshot.snapshot_id, case["query"])
        retrieved_paths = {item.path for item in result.evidence.items}
        expected_paths = set(case["expected_paths"])
        assert expected_paths <= retrieved_paths, (
            f"{case['case_id']}：证据未覆盖预期文件 {sorted(expected_paths)}"
        )
        assert all(
            item.snapshot_id == snapshot.snapshot_id
            and item.sha256 == manifest_hashes[item.path]
            and item.line_start is not None
            and item.line_end is not None
            for item in result.evidence.items
        )

    lab = EvaluationLab(
        database_path=database_path,
        runner=SnapshotExplorerRunner(explorer),
        evaluator=EvidencePathEvaluator(),
    )
    request = EvaluationRequest(
        suite_id="retrieval-smoke-v1",
        snapshot_id=snapshot.snapshot_id,
        configuration={"retriever": "fts5+ast+code-graph"},
        cases=tuple(
            EvaluationCase(
                case_id=case["case_id"],
                user_input=case["query"],
                ground_truth={"expected_paths": case["expected_paths"]},
            )
            for case in SMOKE_SET["cases"]
        ),
    )
    handle = await lab.evaluate(request)
    events = [event async for event in handle.events()]
    experiment = await handle.snapshot()

    assert experiment.experiment.status is ExperimentStatus.COMPLETED
    assert len(experiment.runs) == len(SMOKE_SET["cases"])
    scores: dict[str, tuple[float | None, float | None]] = {}
    for run in experiment.runs:
        metric_values = {result.metric_name: result.value for result in run.results}
        scores[run.case_id] = (
            metric_values["evidence_precision"],
            metric_values["evidence_recall"],
        )
        detail = await RunExplorer(database_path=database_path).get_run(
            experiment.experiment.experiment_id,
            run.run_id,
        )
        expected_paths = next(
            case["expected_paths"]
            for case in SMOKE_SET["cases"]
            if case["case_id"] == run.case_id
        )
        assert set(expected_paths) <= {item.path for item in detail.evidence}
        assert all(
            item.snapshot_id == snapshot.snapshot_id
            and item.sha256 == manifest_hashes[item.path]
            and item.line_start is not None
            and item.line_end is not None
            for item in detail.evidence
        )
        assert detail.trace_events == ()
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))

    expected_scores = {
        "snapshot-manifest": (1.0, 1.0),
        "keyword-index": (1.0, 1.0),
        "changed-file": (1.0, 1.0),
    }
    assert scores == expected_scores


async def _local_semantic_smoke_set_ablation(tmp_path: Path) -> None:
    repository_root = SMOKE_SET_ROOT / "repository"
    snapshot_files = tuple(
        SnapshotFile(
            path=path.relative_to(repository_root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(repository_root.rglob("*"))
        if path.is_file()
    )
    snapshot = RepositorySnapshot.create(
        source_commit="6" * 40,
        files=snapshot_files,
        snapshot_root=repository_root,
    )
    database_path = tmp_path / "semantic-smoke.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"semantic-smoke-fixture").hexdigest(),
                name="semantic-smoke-set",
                source_type="local",
                source_locator="test://semantic-smoke-set",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    text_index = SnapshotTextIndex(database_path=database_path)
    semantic_index = SemanticIndex(
        database_path=database_path,
        local_files_only=True,
    )
    await text_index.build(snapshot)
    await semantic_index.build(snapshot)

    fts_hits = semantic_hits = hybrid_hits = 0
    for case in SMOKE_SET["cases"]:
        expected_paths = set(case["expected_paths"])
        keyword_paths = tuple(
            hit.path
            for hit in text_index.search_text(snapshot.snapshot_id, case["query"], limit=5)
        )
        semantic_paths = tuple(
            hit.path
            for hit in semantic_index.search(
                snapshot.snapshot_id,
                case["question"],
                limit=5,
            )
        )
        fused = reciprocal_rank_fusion((keyword_paths, semantic_paths))
        hybrid_paths = {hit.candidate_id for hit in fused[:5]}
        fts_found = expected_paths <= set(keyword_paths)
        semantic_found = expected_paths <= set(semantic_paths)
        hybrid_found = expected_paths <= hybrid_paths
        fts_hits += fts_found
        semantic_hits += semantic_found
        hybrid_hits += hybrid_found
        print(
            f"{case['case_id']}: FTS={fts_found} "
            f"Semantic={semantic_found} RRF={hybrid_found} "
            f"semantic_paths={list(semantic_paths)}"
        )
        assert hybrid_found, f"{case['case_id']}：融合后未召回 {sorted(expected_paths)}"

    print(
        f"Smoke Set recall@5: FTS={fts_hits}/{len(SMOKE_SET['cases'])}, "
        f"Semantic={semantic_hits}/{len(SMOKE_SET['cases'])}, "
        f"RRF={hybrid_hits}/{len(SMOKE_SET['cases'])}"
    )
