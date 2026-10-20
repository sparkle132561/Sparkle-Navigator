"""从固定 Snapshot 生成 Silver 案例的公开行为测试。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.evaluation.silver_dataset import (
    SilverDatasetGenerationError,
    generate_silver_cases,
)


def test_silver_generator_returns_snapshot_bound_symbol_location_case(
    tmp_path: Path,
) -> None:
    source = "class Catalog:\n    def load(self):\n        return read_one()\n"
    snapshot = _create_snapshot(tmp_path, source)

    cases = generate_silver_cases("demo-repo", snapshot, limit=1)

    assert len(cases) == 1
    case = cases[0]
    assert case["repository_id"] == "demo-repo"
    assert case["question_type"] == "symbol_location"
    assert "Catalog" in case["user_input"]
    evidence = case["ground_truth"]["expected_evidence"][0]
    assert evidence["path"] == "src/catalog.py"
    assert evidence["line_start"] == 1
    assert evidence["line_end"] == 3
    assert evidence["sha256"] == sha256(source.encode("utf-8")).hexdigest()
    assert evidence["snapshot_id"] == snapshot.snapshot_id
    assert case["provenance"]["verification_status"] == "auto_verified"
    assert case["provenance"]["source_commit"] == snapshot.source_commit


def test_silver_generator_covers_import_and_call_site_facts(
    tmp_path: Path,
) -> None:
    source = (
        "import json\n"
        "class Catalog:\n"
        "    def load(self):\n"
        "        return json.loads('{}')\n"
    )
    snapshot = _create_snapshot(tmp_path, source)

    cases = generate_silver_cases("demo-repo", snapshot, limit=3)

    assert [case["question_type"] for case in cases] == [
        "symbol_location",
        "import_statement",
        "call_site",
    ]
    assert "import json" in cases[1]["ground_truth"]["answer_points"][0]
    assert "json.loads" in cases[2]["ground_truth"]["answer_points"][0]
    assert all(
        case["ground_truth"]["expected_evidence"][0]["snapshot_id"]
        == snapshot.snapshot_id
        for case in cases
    )


def test_silver_generator_is_deterministic_and_respects_limit(
    tmp_path: Path,
) -> None:
    source = "def first():\n    return second()\n\ndef second():\n    return 2\n"
    snapshot = _create_snapshot(tmp_path, source)

    first_run = generate_silver_cases("demo-repo", snapshot, limit=2)
    second_run = generate_silver_cases("demo-repo", snapshot, limit=2)

    assert first_run == second_run
    assert len(first_run) == 2
    assert len({case["case_id"] for case in first_run}) == 2


def test_silver_generator_rejects_changed_snapshot_file(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path, "class Catalog:\n    pass\n")
    (tmp_path / "snapshot" / "src" / "catalog.py").write_bytes(
        b"class Catalog:\n    changed = True\n"
    )

    with pytest.raises(SilverDatasetGenerationError, match="Snapshot 校验"):
        generate_silver_cases("demo-repo", snapshot)


def test_silver_generator_handles_strict_path_resolution_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_snapshot(tmp_path, "class Catalog:\n    pass\n")
    original_resolve = Path.resolve

    def reject_strict_resolution(path: Path, strict: bool = False) -> Path:
        if strict:
            raise PermissionError("strict path resolution is unavailable")
        return original_resolve(path, strict=False)

    monkeypatch.setattr(Path, "resolve", reject_strict_resolution)

    cases = generate_silver_cases("demo-repo", snapshot, limit=1)

    assert len(cases) == 1
    assert cases[0]["ground_truth"]["expected_evidence"][0]["path"] == (
        "src/catalog.py"
    )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_silver_generator_rejects_non_positive_or_boolean_limit(
    tmp_path: Path,
    limit: int,
) -> None:
    snapshot = _create_snapshot(tmp_path, "class Catalog:\n    pass\n")

    with pytest.raises(ValueError, match="limit"):
        generate_silver_cases("demo-repo", snapshot, limit=limit)


def _create_snapshot(tmp_path: Path, source: str) -> RepositorySnapshot:
    root = tmp_path / "snapshot"
    source_path = root / "src" / "catalog.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source.encode("utf-8"))
    file_hash = sha256(source_path.read_bytes()).hexdigest()
    return RepositorySnapshot.create(
        source_commit="a" * 40,
        files=(SnapshotFile(path="src/catalog.py", sha256=file_hash),),
        snapshot_root=root,
    )
