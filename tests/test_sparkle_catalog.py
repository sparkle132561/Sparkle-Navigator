"""Workspace 与 Evaluation 视图使用的只读目录接口。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    EvaluationCase,
    EvaluationRequest,
    ExperimentComparison,
    MetricDirection,
    RegressionResult,
)
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
)
from sparkle_navigator.evaluation.run_explorer import RunExplorer
from sparkle_navigator.persistence.evaluation_store import EvaluationStore
from sparkle_navigator.persistence.repository_store import RepositoryStore


def test_run_explorer_lists_experiments_and_reads_saved_comparison(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _run_explorer_lists_experiments_and_reads_saved_comparison(tmp_path)
    )


async def _run_explorer_lists_experiments_and_reads_saved_comparison(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "evaluation-catalog.sqlite3"
    store = EvaluationStore(database_path)
    request = EvaluationRequest(
        suite_id="retrieval-smoke",
        snapshot_id=None,
        cases=(
            EvaluationCase(
                case_id="find-symbol",
                user_input="Where is the symbol defined?",
                ground_truth={"expected_paths": ["src/example.py"]},
            ),
        ),
    )
    baseline_id = store.create_experiment(request)
    candidate_id = store.create_experiment(request)
    comparison = ExperimentComparison(
        comparison_id="comparison-1",
        baseline_experiment_id=baseline_id,
        candidate_experiment_id=candidate_id,
        thresholds={"evidence_recall": 0.05},
        results=(
            RegressionResult(
                metric_name="evidence_recall",
                evaluator_id="sparkle.evidence-path",
                evaluator_version="1",
                baseline_mean=0.9,
                candidate_mean=0.6,
                delta=-0.3,
                baseline_variance=0.0,
                candidate_variance=0.0,
                baseline_sample_count=1,
                candidate_sample_count=1,
                direction=MetricDirection.HIGHER_IS_BETTER,
                regression_threshold=0.05,
                is_regression=True,
            ),
        ),
        created_at=datetime.now(UTC),
    )
    store.save_comparison(comparison)

    explorer = RunExplorer(database_path=database_path)
    experiments = await explorer.list_experiments(limit=10)
    snapshot = await explorer.get_experiment(baseline_id)
    stored_comparison = await explorer.get_comparison("comparison-1")
    comparisons = await explorer.list_comparisons(limit=10)

    assert {item.experiment_id for item in experiments} == {baseline_id, candidate_id}
    assert snapshot.experiment.experiment_id == baseline_id
    assert snapshot.runs[0].case_id == "find-symbol"
    assert stored_comparison == comparison
    assert comparisons == (comparison,)


def test_repository_store_lists_imported_repositories(tmp_path: Path) -> None:
    database_path = tmp_path / "repository-catalog.sqlite3"
    snapshot = RepositorySnapshot.create(
        source_commit="a" * 40,
        files=(),
        snapshot_root=tmp_path / "frozen",
    )
    repository = Repository(
        repository_id="repository-1",
        name="Sparkle Example",
        source_type="local",
        source_locator=str(tmp_path / "source"),
        current_snapshot_id=snapshot.snapshot_id,
    )
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(repository=repository, snapshot=snapshot)
    )

    repositories = RepositoryStore(database_path).list_repositories()

    assert repositories == (repository,)
