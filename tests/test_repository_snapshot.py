"""RepositorySnapshot 公共领域行为测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile


def test_snapshot_id_is_independent_of_manifest_input_order() -> None:
    files = (
        SnapshotFile(path="src/main.py", sha256="a" * 64),
        SnapshotFile(path="README.md", sha256="b" * 64),
    )

    first = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=files,
        index_version=0,
    )
    second = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=tuple(reversed(files)),
        index_version=0,
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.files == second.files


def test_snapshot_file_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError):
        SnapshotFile(path="../outside.txt", sha256="c" * 64)


def test_snapshot_rejects_negative_index_version() -> None:
    with pytest.raises(ValueError):
        RepositorySnapshot.create(
            source_commit="1" * 40,
            files=(),
            index_version=-1,
        )


def test_snapshot_rejects_duplicate_manifest_paths() -> None:
    duplicate_files = (
        SnapshotFile(path="src/main.py", sha256="a" * 64),
        SnapshotFile(path="src/main.py", sha256="b" * 64),
    )

    with pytest.raises(ValueError):
        RepositorySnapshot.create(
            source_commit="1" * 40,
            files=duplicate_files,
        )


def test_snapshot_rejects_malformed_source_commit() -> None:
    with pytest.raises(ValueError):
        RepositorySnapshot.create(source_commit="not-a-git-hash", files=())


def test_snapshot_storage_path_does_not_change_snapshot_identity() -> None:
    first = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=(SnapshotFile(path="src/main.py", sha256="a" * 64),),
        snapshot_root=Path("managed/snapshot-one"),
    )
    second = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=(SnapshotFile(path="src/main.py", sha256="a" * 64),),
        snapshot_root=Path("managed/snapshot-two"),
    )

    assert first.snapshot_root == Path("managed/snapshot-one")
    assert first.snapshot_id == second.snapshot_id
