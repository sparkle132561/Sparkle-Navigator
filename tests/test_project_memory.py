"""Project Memory 的公开行为测试。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.memory.project_memory import (
    ProjectMemory,
    ProjectMemoryError,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


def test_project_fact_survives_reopening_its_sqlite_database(tmp_path: Path) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)

    fact = ProjectMemory(database_path).record_user_correction(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
        statement="greet 函数返回固定问候文本。",
        evidence=(evidence,),
    )

    reopened_facts = ProjectMemory(database_path).facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    )
    assert len(reopened_facts) == 1
    assert reopened_facts[0].fact_id == fact.fact_id
    assert reopened_facts[0].statement == "greet 函数返回固定问候文本。"
    assert reopened_facts[0].evidence == (evidence,)
    assert reopened_facts[0].status.value == "CONFIRMED"


def test_project_memory_rejects_evidence_from_another_snapshot_or_wrong_lines(
    tmp_path: Path,
) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)
    memory = ProjectMemory(database_path)

    with pytest.raises(ProjectMemoryError, match="Snapshot ID"):
        memory.record_user_correction(
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
            statement="错误快照的证据不能建立事实。",
            evidence=(replace(evidence, snapshot_id="f" * 64),),
        )

    with pytest.raises(ProjectMemoryError, match="片段"):
        memory.record_user_correction(
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
            statement="不匹配的代码片段不能建立事实。",
            evidence=(replace(evidence, snippet="return 'not the source'"),),
        )

    assert memory.facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    ) == ()


def test_project_memory_rejects_snapshot_file_hash_drift(tmp_path: Path) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)
    (snapshot.snapshot_root / "src" / "app.py").write_text(
        "def changed():\n    return 'different'\n",
        encoding="utf-8",
    )

    memory = ProjectMemory(database_path)
    with pytest.raises(ProjectMemoryError, match="无法验证"):
        memory.record_user_correction(
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
            statement="磁盘文件已经偏离快照。",
            evidence=(evidence,),
        )
    assert memory.facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    ) == ()


def test_importing_a_new_snapshot_marks_old_confirmed_facts_stale(
    tmp_path: Path,
) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)
    memory = ProjectMemory(database_path)
    fact = memory.record_user_correction(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
        statement="旧版本的 greet 函数返回固定问候文本。",
        evidence=(evidence,),
    )

    next_root = tmp_path / "next-snapshot"
    next_source_path = next_root / "src" / "app.py"
    next_source_path.parent.mkdir(parents=True)
    next_source_path.write_text("def greet():\n    return 'changed'\n", encoding="utf-8")
    next_snapshot = RepositorySnapshot.create(
        source_commit="b" * 40,
        files=(
            SnapshotFile(
                path="src/app.py",
                sha256=hashlib.sha256(next_source_path.read_bytes()).hexdigest(),
            ),
        ),
        snapshot_root=next_root,
    )
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=repository_id,
                name="project-memory-test",
                source_type="local",
                source_locator="test://project-memory",
                current_snapshot_id=next_snapshot.snapshot_id,
            ),
            snapshot=next_snapshot,
        )
    )

    with pytest.raises(ProjectMemoryError, match="当前的 Snapshot"):
        memory.record_user_correction(
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
            statement="旧快照不再接受新确认。",
            evidence=(evidence,),
        )

    old_facts = memory.facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    )
    assert len(old_facts) == 1
    assert old_facts[0].fact_id == fact.fact_id
    assert old_facts[0].status.value == "STALE"


def test_user_correction_supersedes_a_fact_and_fact_can_be_invalidated(
    tmp_path: Path,
) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)
    memory = ProjectMemory(database_path)
    original = memory.record_user_correction(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
        statement="旧的用户确认陈述。",
        evidence=(evidence,),
    )
    replacement = memory.record_user_correction(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
        statement="用户修正后的陈述。",
        evidence=(evidence,),
        supersedes_fact_id=original.fact_id,
    )

    facts = memory.facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    )
    statuses = {fact.fact_id: fact.status.value for fact in facts}
    assert statuses == {
        original.fact_id: "SUPERSEDED",
        replacement.fact_id: "CONFIRMED",
    }
    assert replacement.supersedes_fact_id == original.fact_id

    invalidated = memory.invalidate_fact(replacement.fact_id)
    assert invalidated.status.value == "INVALID"


def test_verified_project_facts_are_invalidated_when_snapshot_evidence_drifts(
    tmp_path: Path,
) -> None:
    snapshot, database_path, repository_id, evidence = _registered_snapshot(tmp_path)
    memory = ProjectMemory(database_path)
    fact = memory.record_user_correction(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
        statement="只有哈希仍匹配的代码事实可进入上下文。",
        evidence=(evidence,),
    )
    (snapshot.snapshot_root / "src" / "app.py").write_text(
        "def changed():\n    return 'tampered'\n",
        encoding="utf-8",
    )

    assert memory.verified_facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    ) == ()
    stored_facts = memory.facts_for_snapshot(
        repository_id=repository_id,
        snapshot_id=snapshot.snapshot_id,
    )
    assert len(stored_facts) == 1
    assert stored_facts[0].fact_id == fact.fact_id
    assert stored_facts[0].status.value == "INVALID"


def _registered_snapshot(
    tmp_path: Path,
) -> tuple[RepositorySnapshot, Path, str, CodeEvidence]:
    snapshot_root = tmp_path / "snapshot"
    source_path = snapshot_root / "src" / "app.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('def greet():\n    return "hello"\n', encoding="utf-8")
    file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    snapshot = RepositorySnapshot.create(
        source_commit="a" * 40,
        files=(SnapshotFile(path="src/app.py", sha256=file_hash),),
        snapshot_root=snapshot_root,
    )
    repository_id = hashlib.sha256(b"project-memory-repository").hexdigest()
    database_path = tmp_path / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=repository_id,
                name="project-memory-test",
                source_type="local",
                source_locator="test://project-memory",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    evidence = CodeEvidence(
        snapshot_id=snapshot.snapshot_id,
        path="src/app.py",
        line_start=1,
        line_end=1,
        snippet="def greet():",
        sha256=file_hash,
        source="symbol",
    )
    return snapshot, database_path, repository_id, evidence
