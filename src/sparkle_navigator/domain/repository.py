"""Repository Snapshot 的不可变领域值。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """快照中的一个相对路径及其内容 SHA-256。"""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        posix_path = PurePosixPath(self.path)
        invalid_path = (
            not self.path
            or "\\" in self.path
            or self.path != posix_path.as_posix()
            or posix_path.is_absolute()
            or re.match(r"^[A-Za-z]:", self.path) is not None
            or any(part in {".", ".."} for part in posix_path.parts)
        )
        if invalid_path:
            raise ValueError("Snapshot 文件路径必须是规范的相对路径。")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("Snapshot 文件 SHA-256 必须是 64 位小写十六进制字符串。")


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """由 Git commit、文件清单和索引版本确定的不可变仓库快照。"""

    snapshot_id: str
    source_commit: str
    files: tuple[SnapshotFile, ...]
    index_version: int
    snapshot_root: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        source_commit: str,
        files: Sequence[SnapshotFile],
        index_version: int = 0,
        snapshot_root: Path | None = None,
    ) -> RepositorySnapshot:
        """规范化文件清单顺序，并生成确定性的快照身份。"""

        if not isinstance(source_commit, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            source_commit,
        ) is None:
            raise ValueError("Snapshot source_commit 必须是规范的 Git commit 哈希。")
        if (
            not isinstance(index_version, int)
            or isinstance(index_version, bool)
            or index_version < 0
        ):
            raise ValueError("Snapshot index_version 必须是非负整数。")
        ordered_files = tuple(sorted(files, key=lambda item: item.path))
        paths = tuple(item.path for item in ordered_files)
        if len(paths) != len(set(paths)):
            raise ValueError("Snapshot 文件清单不能包含重复路径。")
        identity = {
            "source_commit": source_commit,
            "files": [
                {"path": item.path, "sha256": item.sha256}
                for item in ordered_files
            ],
            "index_version": index_version,
        }
        encoded_identity = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            snapshot_id=sha256(encoded_identity).hexdigest(),
            source_commit=source_commit,
            files=ordered_files,
            index_version=index_version,
            snapshot_root=snapshot_root,
        )


@dataclass(frozen=True, slots=True)
class RepositoryWorkingCopy:
    """从固定 Snapshot 派生、允许 Agent 修改的仓库内容副本。"""

    repository_id: str
    snapshot_id: str
    root: Path


@dataclass(frozen=True, slots=True)
class Repository:
    """已导入项目的稳定身份和当前信任状态。"""

    repository_id: str
    name: str
    source_type: Literal["local", "git"]
    source_locator: str
    trusted: bool = False
    current_snapshot_id: str | None = None


@dataclass(frozen=True, slots=True)
class RepositoryImportResult:
    """一次导入返回的仓库记录和固定版本。"""

    repository: Repository
    snapshot: RepositorySnapshot


@dataclass(frozen=True, slots=True)
class LocalRepositorySource:
    """指向本地 Git 工作树的导入来源。"""

    path: Path


@dataclass(frozen=True, slots=True)
class GitRepositorySource:
    """指向 Git URL 及可选 ref 的导入来源。"""

    url: str
    ref: str | None = None


RepositorySource = LocalRepositorySource | GitRepositorySource


__all__ = [
    "GitRepositorySource",
    "LocalRepositorySource",
    "Repository",
    "RepositoryImportResult",
    "RepositorySnapshot",
    "RepositoryWorkingCopy",
    "RepositorySource",
    "SnapshotFile",
]
