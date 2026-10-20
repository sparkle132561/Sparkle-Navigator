"""版本化 Benchmark Dataset 清单与 EvaluationRequest 转换测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkle_navigator.evaluation.dataset import BenchmarkDataset


def test_dataset_groups_cases_by_repository_and_suite(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path)

    dataset = BenchmarkDataset.load(manifest_path)
    requests = dataset.to_evaluation_requests(
        snapshot_ids={"repo-a": "a" * 64, "repo-b": "b" * 64},
        repetitions=2,
        configuration={"provider_request_limit": 3},
    )

    assert len(requests) == 2
    assert [request.snapshot_id for request in requests] == ["a" * 64, "b" * 64]
    assert [len(request.cases) for request in requests] == [2, 1]
    assert requests[0].suite_id != requests[1].suite_id
    assert all(request.repetitions == 2 for request in requests)
    assert all(
        request.configuration["provider_request_limit"] == 3
        for request in requests
    )


def test_dataset_rejects_missing_or_unknown_snapshot_mapping(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.load(_write_dataset(tmp_path))

    with pytest.raises(ValueError, match="snapshot_ids"):
        dataset.to_evaluation_requests(
            snapshot_ids={"repo-a": "a" * 64},
            repetitions=1,
            configuration={},
        )
    with pytest.raises(ValueError, match="snapshot_ids"):
        dataset.to_evaluation_requests(
            snapshot_ids={
                "repo-a": "a" * 64,
                "repo-b": "b" * 64,
                "unknown": "c" * 64,
            },
            repetitions=1,
            configuration={},
        )


def test_dataset_rejects_cases_file_outside_manifest_directory(tmp_path: Path) -> None:
    manifest_path = _write_dataset(tmp_path, cases_file="../outside.json")

    with pytest.raises(ValueError, match="cases_file"):
        BenchmarkDataset.load(manifest_path)


def test_dataset_accepts_local_repository_without_claiming_a_license(
    tmp_path: Path,
) -> None:
    dataset = BenchmarkDataset.load(_write_dataset(tmp_path))

    local_repository = dataset.repositories[1]
    assert local_repository.url == "local://sparkle-navigator"
    assert local_repository.license == "NOASSERTION"


def test_tau_review_batch_loads_as_user_approved_round_gold_reference() -> None:
    repository_root = Path(__file__).parents[1]
    manifest_path = (
        repository_root
        / "data"
        / "benchmarks"
        / "post-alpha"
        / "tau-review-v0.1"
        / "manifest.json"
    )

    dataset = BenchmarkDataset.load(manifest_path)

    assert dataset.dataset_id == "sparkle-post-alpha-tau-review"
    assert dataset.split == "round-gold-reference"
    assert len(dataset.cases) == 16
    assert dataset.repositories[0].source_commit == (
        "a8f18e7bc645d09d47d40b21bc602df97f2579fd"
    )
    requests = dataset.to_evaluation_requests(
        snapshot_ids={
            "tau-v0.4.4": (
                "76f35b6ac5a23edab514ce454ae4f8d5644ace5acaa3ffa7579002d501b46754"
            )
        },
        repetitions=1,
        configuration={"tool_profile": "hybrid"},
    )
    assert len(requests) == 7
    assert sum(len(request.cases) for request in requests) == 16
    assert {request.snapshot_id for request in requests} == {
        "76f35b6ac5a23edab514ce454ae4f8d5644ace5acaa3ffa7579002d501b46754"
    }
    assert {
        case.provenance["verification_status"] for case in dataset.cases
    } == {"user_approved_ai_review"}
    assert all(case.ground_truth.get("expected_evidence") for case in dataset.cases)


def test_gold_candidate_batch_loads_as_pending_eighty_case_five_repo_set() -> None:
    repository_root = Path(__file__).parents[1]
    manifest_path = (
        repository_root
        / "data"
        / "benchmarks"
        / "post-alpha"
        / "gold-candidates-v0.1"
        / "manifest.json"
    )

    dataset = BenchmarkDataset.load(manifest_path)

    assert dataset.dataset_id == "sparkle-post-alpha-gold-candidates"
    assert dataset.split == "gold-candidate"
    assert len(dataset.cases) == 80
    assert len(dataset.repositories) == 5
    assert {
        repository.repository_id: sum(
            case.repository_id == repository.repository_id for case in dataset.cases
        )
        for repository in dataset.repositories
    } == {
        "tau-v0.4.4": 16,
        "fastapi-0.143.0": 16,
        "django-oscar-4.2.1": 16,
        "httpx-0.28.1": 16,
        "sparkle-navigator": 16,
    }
    assert {
        case.provenance["verification_status"] for case in dataset.cases
    } == {"needs_human_review"}
    assert all(case.ground_truth.get("expected_evidence") for case in dataset.cases)
    assert all(
        case.provenance["source_commit"]
        == next(
            repository.source_commit
            for repository in dataset.repositories
            if repository.repository_id == case.repository_id
        )
        for case in dataset.cases
    )


def test_silver_dataset_requires_machine_verified_case_status(tmp_path: Path) -> None:
    unverified_manifest = _write_dataset(
        tmp_path / "unverified",
        split="silver",
    )

    with pytest.raises(ValueError, match="auto_verified"):
        BenchmarkDataset.load(unverified_manifest)

    verified_cases = _example_cases()
    for case in verified_cases:
        case["provenance"]["verification_status"] = "auto_verified"
    verified_manifest = _write_dataset(
        tmp_path / "verified",
        cases=verified_cases,
        split="silver",
    )

    dataset = BenchmarkDataset.load(verified_manifest)

    assert dataset.split == "silver"
    assert len(dataset.cases) == len(verified_cases)
    assert {
        case.provenance["verification_status"] for case in dataset.cases
    } == {"auto_verified"}


def test_generated_silver_batch_loads_with_three_hundred_snapshot_bound_cases() -> None:
    repository_root = Path(__file__).parents[1]
    manifest_path = (
        repository_root
        / "data"
        / "benchmarks"
        / "post-alpha"
        / "silver-v0.1"
        / "manifest.json"
    )
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_snapshots = {
        repository["repository_id"]: repository["snapshot_id"]
        for repository in raw_manifest["repositories"]
    }

    dataset = BenchmarkDataset.load(manifest_path)

    assert dataset.dataset_id == "sparkle-post-alpha-silver"
    assert dataset.split == "silver"
    assert len(dataset.repositories) == 5
    assert len(dataset.cases) == 300
    assert {
        repository.repository_id: sum(
            case.repository_id == repository.repository_id for case in dataset.cases
        )
        for repository in dataset.repositories
    } == {
        "tau-v0.4.4": 60,
        "fastapi-0.143.0": 60,
        "django-oscar-4.2.1": 60,
        "httpx-0.28.1": 60,
        "sparkle-navigator": 60,
    }
    assert len({case.case_id for case in dataset.cases}) == 300
    assert {
        case.provenance["verification_status"] for case in dataset.cases
    } == {"auto_verified"}
    for case in dataset.cases:
        assert case.provenance["snapshot_id"] == expected_snapshots[
            case.repository_id
        ]
        assert case.provenance["source_commit"] == next(
            repository.source_commit
            for repository in dataset.repositories
            if repository.repository_id == case.repository_id
        )
        evidence = case.ground_truth["expected_evidence"]
        assert len(evidence) == 1
        assert evidence[0]["snapshot_id"] == expected_snapshots[case.repository_id]
        assert case.provenance["deterministic_input"]


def test_dataset_rejects_duplicate_case_ids_and_unknown_repository(
    tmp_path: Path,
) -> None:
    duplicate_cases = _example_cases()
    duplicate_cases[1]["case_id"] = duplicate_cases[0]["case_id"]
    manifest_path = _write_dataset(tmp_path / "duplicate", cases=duplicate_cases)

    with pytest.raises(ValueError, match="case_id"):
        BenchmarkDataset.load(manifest_path)

    unknown_repo_cases = _example_cases()
    unknown_repo_cases[0]["repository_id"] = "not-in-manifest"
    manifest_path = _write_dataset(tmp_path / "unknown", cases=unknown_repo_cases)

    with pytest.raises(ValueError, match="repository_id"):
        BenchmarkDataset.load(manifest_path)


def test_dataset_keeps_ground_truth_separate_from_public_input(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.load(_write_dataset(tmp_path))
    request = dataset.to_evaluation_requests(
        snapshot_ids={"repo-a": "a" * 64, "repo-b": "b" * 64},
        repetitions=1,
        configuration={},
    )[0]

    public_case = request.cases[0]
    assert public_case.user_input == "Explain the public call chain."
    assert "expected_evidence" not in public_case.user_input
    assert "src/internal.py" in repr(public_case.ground_truth)


def _write_dataset(
    directory: Path,
    *,
    cases: list[dict[str, object]] | None = None,
    cases_file: str = "cases.json",
    split: str = "gold-candidate",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    selected_cases = cases if cases is not None else _example_cases()
    (directory / "cases.json").write_text(
        json.dumps(selected_cases),
        encoding="utf-8",
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "post-alpha",
                "version": "1.0.0",
                "split": split,
                "repositories": [
                    {
                        "repository_id": "repo-a",
                        "url": "https://example.invalid/repo-a",
                        "source_commit": "a" * 40,
                        "license": "MIT",
                    },
                    {
                        "repository_id": "repo-b",
                        "url": "local://sparkle-navigator",
                        "source_commit": "b" * 40,
                        "license": "NOASSERTION",
                    },
                ],
                "cases_file": cases_file,
                "case_count": len(selected_cases),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _example_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": "case-a-1",
            "repository_id": "repo-a",
            "suite_type": "call-chain",
            "question_type": "explain_flow",
            "difficulty": "easy",
            "user_input": "Explain the public call chain.",
            "ground_truth": {
                "answer_points": ["The entry point delegates to a service."],
                "expected_evidence": [
                    {"path": "src/internal.py", "line_start": 1, "line_end": 4}
                ],
                "rubric": {"must_mention": ["service"]},
            },
            "provenance": {
                "verification_status": "needs_human_review",
                "generator": "manual-candidate-v1",
                "generator_version": "1",
            },
        },
        {
            "case_id": "case-a-2",
            "repository_id": "repo-a",
            "suite_type": "call-chain",
            "question_type": "find_symbol",
            "difficulty": "medium",
            "user_input": "Find the service implementation.",
            "ground_truth": {
                "answer_points": ["The implementation is in the service module."],
                "expected_evidence": [
                    {"path": "src/internal.py", "line_start": 5, "line_end": 8}
                ],
                "rubric": {"must_mention": ["implementation"]},
            },
            "provenance": {
                "verification_status": "needs_human_review",
                "generator": "manual-candidate-v1",
                "generator_version": "1",
            },
        },
        {
            "case_id": "case-b-1",
            "repository_id": "repo-b",
            "suite_type": "error-handling",
            "question_type": "explain_failure",
            "difficulty": "hard",
            "user_input": "Explain how this error is handled.",
            "ground_truth": {
                "answer_points": ["The error is translated at the boundary."],
                "expected_evidence": [
                    {"path": "src/errors.py", "line_start": 1, "line_end": 4}
                ],
                "rubric": {"must_mention": ["boundary"]},
            },
            "provenance": {
                "verification_status": "needs_human_review",
                "generator": "manual-candidate-v1",
                "generator_version": "1",
            },
        },
    ]
