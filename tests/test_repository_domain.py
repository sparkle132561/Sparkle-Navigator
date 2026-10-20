"""Repository 与 RepositorySnapshot 的领域模型测试。"""

from __future__ import annotations

from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)


def test_repository_import_result_separates_trust_from_snapshot_version() -> None:
    repository = Repository(
        repository_id="repo-1",
        name="demo",
        source_type="local",
        source_locator="C:/projects/demo",
    )
    snapshot = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=(SnapshotFile(path="README.md", sha256="a" * 64),),
    )

    result = RepositoryImportResult(repository=repository, snapshot=snapshot)

    assert result.repository.trusted is False
    assert result.snapshot.snapshot_id == snapshot.snapshot_id
