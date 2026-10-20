"""读取并校验不可变 Snapshot 中的文件内容。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile


class SnapshotFileScanError(RuntimeError):
    """Snapshot 身份、文件清单或磁盘内容校验失败。"""


def scan_verified_snapshot_files(
    snapshot: RepositorySnapshot,
) -> tuple[tuple[SnapshotFile, int, str | None], ...]:
    """验证 Snapshot 身份与文件，再返回元数据和可索引文本。"""

    _validate_snapshot_id(snapshot)
    snapshot_root = _verified_snapshot_root(snapshot)

    try:
        root = snapshot_root.resolve(strict=True)
        expected_paths = {item.path for item in snapshot.files}
        actual_paths: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SnapshotFileScanError("Snapshot 内容中不允许符号链接。")
            if path.is_file():
                actual_paths.add(path.relative_to(root).as_posix())
        if actual_paths != expected_paths:
            raise SnapshotFileScanError("Snapshot 磁盘文件路径与清单不匹配。")

        scanned: list[tuple[SnapshotFile, int, str | None]] = []
        for snapshot_file in snapshot.files:
            path = root.joinpath(*PurePosixPath(snapshot_file.path).parts)
            scanned.append(_read_verified_file(path, snapshot_file))
        return tuple(scanned)
    except SnapshotFileScanError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SnapshotFileScanError("无法安全扫描 Snapshot 文件。") from exc


def read_verified_snapshot_files(
    snapshot: RepositorySnapshot,
    paths: Iterable[str],
) -> tuple[tuple[SnapshotFile, int, str | None], ...]:
    """只读取并校验指定 Snapshot 清单中的文件。"""

    _validate_snapshot_id(snapshot)
    root = _verified_snapshot_root(snapshot)
    files_by_path = {item.path: item for item in snapshot.files}
    verified: list[tuple[SnapshotFile, int, str | None]] = []
    for relative_path in dict.fromkeys(paths):
        snapshot_file = files_by_path.get(relative_path)
        if snapshot_file is None:
            raise SnapshotFileScanError("待引用文件不在 Snapshot 清单中。")
        path = root
        try:
            for part in PurePosixPath(snapshot_file.path).parts:
                path = path / part
                if path.is_symlink():
                    raise SnapshotFileScanError("Snapshot 内容中不允许符号链接。")
            resolved_path = path.resolve(strict=False)
            if not resolved_path.is_relative_to(root) or not resolved_path.is_file():
                raise SnapshotFileScanError("Snapshot 文件路径无效。")
            verified.append(_read_verified_file(resolved_path, snapshot_file))
        except SnapshotFileScanError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise SnapshotFileScanError("无法安全读取 Snapshot 文件。") from exc
    return tuple(verified)


def _validate_snapshot_id(snapshot: RepositorySnapshot) -> None:
    expected_snapshot_id = RepositorySnapshot.create(
        source_commit=snapshot.source_commit,
        files=snapshot.files,
        index_version=snapshot.index_version,
    ).snapshot_id
    if snapshot.snapshot_id != expected_snapshot_id:
        raise SnapshotFileScanError("Snapshot ID 与文件清单不匹配。")


def _verified_snapshot_root(snapshot: RepositorySnapshot) -> Path:
    snapshot_root = snapshot.snapshot_root
    if snapshot_root is None or snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise SnapshotFileScanError("Snapshot 缺少安全的内容目录。")
    try:
        # Windows OneDrive/受限环境可能拒绝 strict realpath 查询；is_dir、文件包含关系和哈希校验仍会确认可读内容。
        return snapshot_root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SnapshotFileScanError("无法安全访问 Snapshot 内容目录。") from exc


def _read_verified_file(
    path: Path,
    snapshot_file: SnapshotFile,
) -> tuple[SnapshotFile, int, str | None]:
    content = path.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != snapshot_file.sha256:
        raise SnapshotFileScanError(
            f"Snapshot 文件 SHA-256 不匹配：{snapshot_file.path}"
        )
    try:
        text_content = None if b"\0" in content else content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = None
    return snapshot_file, len(content), text_content


__all__ = [
    "SnapshotFileScanError",
    "read_verified_snapshot_files",
    "scan_verified_snapshot_files",
]
