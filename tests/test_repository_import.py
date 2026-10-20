"""通过 RepositoryWorkspace 公开接口测试本地 Git 仓库导入。"""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sparkle_navigator.domain.repository import GitRepositorySource, LocalRepositorySource
from sparkle_navigator.workspace.repository_import import (
    RepositoryImportError,
    RepositoryWorkspace,
)


def test_local_import_freezes_worktree_and_persists_snapshot() -> None:
    asyncio.run(_import_local_worktree())


def test_git_url_import_checks_out_the_requested_commit() -> None:
    asyncio.run(_import_git_commit())


def test_repository_trust_decision_is_persisted_and_revocable() -> None:
    asyncio.run(_repository_trust_round_trip())


def test_working_copy_is_separate_and_survives_workspace_reopen() -> None:
    asyncio.run(_working_copy_is_separate_and_survives_workspace_reopen())


async def _import_local_worktree() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository_path = root / "source"
        repository_path.mkdir()
        _git(repository_path, "init")
        _git(repository_path, "config", "user.name", "Sparkle Test")
        _git(repository_path, "config", "user.email", "sparkle-test@example.invalid")

        tracked_file = repository_path / "tracked.txt"
        tracked_file.write_text("committed version\n", encoding="utf-8")
        (repository_path / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
        _git(repository_path, "add", ".gitignore", "tracked.txt")
        _git(repository_path, "commit", "-m", "initial repository")
        source_commit = _git(repository_path, "rev-parse", "HEAD")

        tracked_file.write_text("imported working-tree version\n", encoding="utf-8")
        untracked_file = repository_path / "notes.txt"
        untracked_file.write_text("untracked but not ignored\n", encoding="utf-8")
        (repository_path / "ignored.tmp").write_text("ignored\n", encoding="utf-8")
        source_status = _git(repository_path, "status", "--porcelain")

        database_path = root / "sparkle.sqlite3"
        snapshots_root = root / "managed-snapshots"
        working_copies_root = root / "working-copies"
        workspace = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        )
        result = await workspace.import_source(LocalRepositorySource(path=repository_path))
        snapshot = result.snapshot

        assert result.repository.trusted is False
        assert result.repository.current_snapshot_id == snapshot.snapshot_id
        assert snapshot.source_commit == source_commit
        assert snapshot.index_version == 0
        assert (snapshot.snapshot_root / "tracked.txt").read_text(encoding="utf-8") == (
            "imported working-tree version\n"
        )
        assert (snapshot.snapshot_root / "notes.txt").read_text(encoding="utf-8") == (
            "untracked but not ignored\n"
        )
        assert not (snapshot.snapshot_root / "ignored.tmp").exists()
        assert _git(repository_path, "status", "--porcelain") == source_status

        tracked_file.write_text("changed after import\n", encoding="utf-8")
        assert (snapshot.snapshot_root / "tracked.txt").read_text(encoding="utf-8") == (
            "imported working-tree version\n"
        )

        restored = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        ).get_snapshot(snapshot.snapshot_id)
        assert restored == snapshot


async def _import_git_commit() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_repository = root / "source"
        source_repository.mkdir()
        _git(source_repository, "init")
        _git(source_repository, "config", "user.name", "Sparkle Test")
        _git(source_repository, "config", "user.email", "sparkle-test@example.invalid")
        source_file = source_repository / "hello.txt"
        source_file.write_text("fixed remote version\n", encoding="utf-8")
        _git(source_repository, "add", "hello.txt")
        _git(source_repository, "commit", "-m", "remote version")
        source_commit = _git(source_repository, "rev-parse", "HEAD")

        bare_remote = root / "public.git"
        _git(root, "clone", "--bare", str(source_repository), str(bare_remote))
        workspace = RepositoryWorkspace(
            database_path=root / "sparkle.sqlite3",
            snapshots_root=root / "managed-snapshots",
            working_copies_root=root / "working-copies",
        )

        result = await workspace.import_source(
            GitRepositorySource(url=bare_remote.as_uri(), ref=source_commit)
        )

        assert result.repository.source_type == "git"
        assert result.repository.trusted is False
        assert result.snapshot.source_commit == source_commit
        assert (result.snapshot.snapshot_root / "hello.txt").read_text(
            encoding="utf-8"
        ) == "fixed remote version\n"


async def _repository_trust_round_trip() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository_path = root / "source"
        repository_path.mkdir()
        _git(repository_path, "init")
        _git(repository_path, "config", "user.name", "Sparkle Test")
        _git(repository_path, "config", "user.email", "sparkle-test@example.invalid")
        (repository_path / "README.md").write_text("repository\n", encoding="utf-8")
        _git(repository_path, "add", "README.md")
        _git(repository_path, "commit", "-m", "initial repository")

        database_path = root / "sparkle.sqlite3"
        snapshots_root = root / "managed-snapshots"
        working_copies_root = root / "working-copies"
        workspace = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        )
        imported = await workspace.import_source(
            LocalRepositorySource(path=repository_path)
        )
        assert imported.repository.trusted is False

        trusted = workspace.set_repository_trust(
            imported.repository.repository_id,
            trusted=True,
        )
        assert trusted.trusted is True
        assert trusted.current_snapshot_id == imported.snapshot.snapshot_id

        reopened = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        )
        restored = reopened.get_repository(imported.repository.repository_id)
        assert restored.trusted is True

        revoked = reopened.set_repository_trust(
            imported.repository.repository_id,
            trusted=False,
        )
        assert revoked.trusted is False


async def _working_copy_is_separate_and_survives_workspace_reopen() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository_path = root / "source"
        repository_path.mkdir()
        _git(repository_path, "init")
        _git(repository_path, "config", "user.name", "Sparkle Test")
        _git(repository_path, "config", "user.email", "sparkle-test@example.invalid")
        source_file = repository_path / "README.md"
        source_file.write_text("snapshot version\n", encoding="utf-8")
        _git(repository_path, "add", "README.md")
        _git(repository_path, "commit", "-m", "initial repository")

        database_path = root / "sparkle.sqlite3"
        snapshots_root = root / "managed-snapshots"
        working_copies_root = root / "working-copies"
        workspace = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        )
        imported = await workspace.import_source(
            LocalRepositorySource(path=repository_path)
        )
        snapshot_file = imported.snapshot.snapshot_root / "README.md"
        snapshot_bytes = snapshot_file.read_bytes()
        snapshot_hash = imported.snapshot.files[0].sha256

        working_copy = await workspace.open_working_copy(
            imported.repository.repository_id
        )
        working_file = working_copy.root / "README.md"
        assert working_copy.snapshot_id == imported.snapshot.snapshot_id
        assert working_copy.root != imported.snapshot.snapshot_root
        assert working_file.read_bytes() == snapshot_bytes

        working_file.write_text("agent edit\n", encoding="utf-8")
        assert snapshot_file.read_bytes() == snapshot_bytes
        assert hashlib.sha256(snapshot_file.read_bytes()).hexdigest() == snapshot_hash

        reopened_workspace = RepositoryWorkspace(
            database_path=database_path,
            snapshots_root=snapshots_root,
            working_copies_root=working_copies_root,
        )
        reopened = await reopened_workspace.open_working_copy(
            imported.repository.repository_id
        )
        assert reopened.root == working_copy.root
        assert (reopened.root / "README.md").read_text(encoding="utf-8") == (
            "agent edit\n"
        )

        snapshot_file.write_text("tampered snapshot\n", encoding="utf-8")
        with pytest.raises(RepositoryImportError, match="SHA-256"):
            await reopened_workspace.open_working_copy(
                imported.repository.repository_id
            )


def _git(repository_path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_path), *arguments],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()
