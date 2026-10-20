"""通过稳定 interface 导入 Git 仓库并冻结工作树内容。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from sparkle_navigator.domain.repository import (
    GitRepositorySource,
    LocalRepositorySource,
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    RepositoryWorkingCopy,
    RepositorySource,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


class RepositoryImportError(RuntimeError):
    """仓库来源无效或导入失败。"""


class RepositoryWorkspace:
    """导入仓库并恢复其 Repository/Snapshot 记录。"""

    def __init__(
        self,
        *,
        database_path: Path,
        snapshots_root: Path,
        working_copies_root: Path,
    ) -> None:
        self.database_path = database_path.expanduser()
        self.snapshots_root = snapshots_root.expanduser().resolve()
        self.working_copies_root = working_copies_root.expanduser().resolve()
        if _paths_overlap(self.snapshots_root, self.working_copies_root):
            raise ValueError("Snapshot 存储目录与可写工作副本目录必须分开。")
        if _is_same_or_child(self.database_path.resolve(), self.snapshots_root) or (
            _is_same_or_child(self.database_path.resolve(), self.working_copies_root)
        ):
            raise ValueError("SQLite 数据库不能放在 Snapshot 或工作副本目录中。")
        self._store: RepositoryStore | None = None

    async def import_source(self, source: RepositorySource) -> RepositoryImportResult:
        """从来源导入并冻结一个仓库版本。"""

        if isinstance(source, GitRepositorySource):
            return await self._import_git_source(source)
        if not isinstance(source, LocalRepositorySource):
            raise TypeError("source 必须是已知的 RepositorySource 类型。")

        selected_path = source.path.expanduser().resolve(strict=True)
        if not selected_path.is_dir():
            raise RepositoryImportError(f"本地导入来源不是目录：{selected_path}")
        repository_root = Path(
            await _git_output(selected_path, "rev-parse", "--show-toplevel")
        ).resolve(strict=True)
        self._validate_storage_paths(repository_root)
        source_commit = await _git_output(repository_root, "rev-parse", "HEAD")
        raw_paths = await _git_bytes(
            repository_root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        )
        try:
            relative_paths = tuple(
                sorted(
                    {
                        item.decode("utf-8")
                        for item in raw_paths.split(b"\0")
                        if item
                    }
                )
            )
        except UnicodeDecodeError as exc:
            raise RepositoryImportError("Git 文件路径不是有效的 UTF-8。") from exc

        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".repository-import-",
            dir=self.snapshots_root,
        ) as temporary_directory:
            staging_root = Path(temporary_directory) / "snapshot"
            staging_root.mkdir()
            files = await asyncio.to_thread(
                _copy_working_tree,
                repository_root,
                staging_root,
                relative_paths,
            )
            return self._persist_staged_snapshot(
                source_type="local",
                source_locator=str(repository_root),
                name=repository_root.name or repository_root.anchor,
                source_commit=source_commit,
                files=files,
                staging_root=staging_root,
            )

    async def _import_git_source(
        self,
        source: GitRepositorySource,
    ) -> RepositoryImportResult:
        url = _normalize_git_url(source.url)
        ref = source.ref
        if ref is not None and (
            not ref.strip() or ref.startswith("-") or any(ord(char) < 32 for char in ref)
        ):
            raise RepositoryImportError("Git ref 为空或包含不支持的控制字符。")

        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".git-source-") as temporary_directory:
            bare_repository = Path(temporary_directory) / "remote.git"
            await _git_output(
                None,
                "clone",
                "--bare",
                "--no-local",
                "--",
                url,
                str(bare_repository),
            )
            revision = f"{ref}^{{commit}}" if ref is not None else "HEAD^{commit}"
            source_commit = await _git_output(
                None,
                f"--git-dir={bare_repository}",
                "rev-parse",
                "--verify",
                "--end-of-options",
                revision,
            )
            tree = await _git_bytes(
                None,
                f"--git-dir={bare_repository}",
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                source_commit,
            )
            entries = _git_tree_entries(tree)

            with tempfile.TemporaryDirectory(
                prefix=".repository-import-",
                dir=self.snapshots_root,
            ) as staging_directory:
                staging_root = Path(staging_directory) / "snapshot"
                staging_root.mkdir()
                files = await _copy_git_blobs(bare_repository, entries, staging_root)
                return self._persist_staged_snapshot(
                    source_type="git",
                    source_locator=url,
                    name=_git_repository_name(url),
                    source_commit=source_commit,
                    files=files,
                    staging_root=staging_root,
                )

    def _validate_storage_paths(self, repository_root: Path) -> None:
        database_path = self.database_path.resolve()
        storage_roots = (self.snapshots_root, self.working_copies_root)
        if _is_same_or_child(database_path, repository_root) or any(
            _paths_overlap(storage_root, repository_root)
            for storage_root in storage_roots
        ):
            raise RepositoryImportError(
                "SQLite 数据库、Snapshot 和工作副本存储目录必须在导入仓库之外。"
            )

    def _persist_staged_snapshot(
        self,
        *,
        source_type: str,
        source_locator: str,
        name: str,
        source_commit: str,
        files: tuple[SnapshotFile, ...],
        staging_root: Path,
    ) -> RepositoryImportResult:
        snapshot_id = RepositorySnapshot.create(
            source_commit=source_commit,
            files=files,
            index_version=0,
        ).snapshot_id
        snapshot_path = self.snapshots_root.resolve() / snapshot_id
        snapshot = RepositorySnapshot.create(
            source_commit=source_commit,
            files=files,
            index_version=0,
            snapshot_root=snapshot_path,
        )

        created_snapshot_path = False
        if snapshot_path.exists():
            if not _snapshot_matches(snapshot_path, files):
                raise RepositoryImportError(
                    f"已存在的 Snapshot 副本与清单不匹配：{snapshot_path}"
                )
            shutil.rmtree(staging_root)
        else:
            staging_root.replace(snapshot_path)
            created_snapshot_path = True

        repository_id = hashlib.sha256(
            f"{source_type}\0{source_locator}".encode("utf-8")
        ).hexdigest()
        repository = Repository(
            repository_id=repository_id,
            name=name,
            source_type=source_type,
            source_locator=source_locator,
            trusted=False,
            current_snapshot_id=snapshot.snapshot_id,
        )
        result = RepositoryImportResult(repository=repository, snapshot=snapshot)
        try:
            return self._repository_store().save_import(result)
        except Exception:
            if created_snapshot_path:
                shutil.rmtree(snapshot_path)
            raise

    def get_snapshot(self, snapshot_id: str) -> RepositorySnapshot:
        """按 Snapshot ID 从 SQLite 恢复快照元数据。"""

        return self._repository_store().get_snapshot(snapshot_id)

    def get_repository(self, repository_id: str) -> Repository:
        """读取仓库来源及其持久化信任状态。"""

        return self._repository_store().get_repository(repository_id)

    async def open_working_copy(self, repository_id: str) -> RepositoryWorkingCopy:
        """为当前 Snapshot 创建或恢复独立的可写工作副本。"""

        repository = self.get_repository(repository_id)
        if repository.current_snapshot_id is None:
            raise RepositoryImportError("Repository 尚无当前 Snapshot。")
        snapshot = self.get_snapshot(repository.current_snapshot_id)
        return await asyncio.to_thread(
            self._materialize_working_copy,
            repository,
            snapshot,
        )

    def _materialize_working_copy(
        self,
        repository: Repository,
        snapshot: RepositorySnapshot,
    ) -> RepositoryWorkingCopy:
        """在线程中校验 Snapshot 并创建或恢复其可写副本。"""

        if repository.current_snapshot_id is None:
            raise RepositoryImportError("Repository 尚无当前 Snapshot。")
        if snapshot.snapshot_id != repository.current_snapshot_id:
            raise RepositoryImportError("Repository 当前 Snapshot 与读取结果不一致。")
        if snapshot.snapshot_root is None:
            raise RepositoryImportError("Snapshot 缺少冻结内容目录。")
        if not _snapshot_matches(snapshot.snapshot_root, snapshot.files):
            raise RepositoryImportError("Snapshot 磁盘内容与 SHA-256 清单不匹配。")

        self.working_copies_root.mkdir(parents=True, exist_ok=True)
        working_copy_path = (
            self.working_copies_root
            / repository.repository_id
            / snapshot.snapshot_id
        )
        if working_copy_path.exists() or working_copy_path.is_symlink():
            return self._working_copy_result(
                repository,
                snapshot,
                working_copy_path,
            )

        working_copy_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".working-copy-",
            dir=working_copy_path.parent,
        ) as temporary_directory:
            staging_root = Path(temporary_directory) / "contents"
            try:
                shutil.copytree(snapshot.snapshot_root, staging_root)
            except OSError as exc:
                raise RepositoryImportError(
                    f"无法从 Snapshot 创建工作副本：{exc}"
                ) from exc
            if not _snapshot_matches(staging_root, snapshot.files):
                raise RepositoryImportError("工作副本复制结果与 Snapshot 清单不匹配。")
            try:
                staging_root.rename(working_copy_path)
            except OSError as exc:
                if working_copy_path.exists():
                    return self._working_copy_result(
                        repository,
                        snapshot,
                        working_copy_path,
                    )
                raise RepositoryImportError(
                    f"无法保存仓库工作副本：{exc}"
                ) from exc

        return self._working_copy_result(repository, snapshot, working_copy_path)

    def _working_copy_result(
        self,
        repository: Repository,
        snapshot: RepositorySnapshot,
        path: Path,
    ) -> RepositoryWorkingCopy:
        if path.is_symlink() or not path.is_dir():
            raise RepositoryImportError("工作副本路径不是安全的目录。")
        try:
            root = path.resolve(strict=True)
            storage_root = self.working_copies_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryImportError("无法安全解析工作副本目录。") from exc
        if not _is_same_or_child(root, storage_root):
            raise RepositoryImportError("工作副本目录越出了受管理的存储根目录。")
        return RepositoryWorkingCopy(
            repository_id=repository.repository_id,
            snapshot_id=snapshot.snapshot_id,
            root=root,
        )

    def set_repository_trust(self, repository_id: str, *, trusted: bool) -> Repository:
        """持久化由宿主 UI 明确确认的信任决定。"""

        return self._repository_store().set_repository_trust(
            repository_id,
            trusted=trusted,
        )

    def _repository_store(self) -> RepositoryStore:
        if self._store is None:
            self._store = RepositoryStore(self.database_path)
        return self._store


def _is_same_or_child(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_same_or_child(first, second) or _is_same_or_child(second, first)


def _copy_working_tree(
    repository_root: Path,
    staging_root: Path,
    relative_paths: tuple[str, ...],
) -> tuple[SnapshotFile, ...]:
    files: list[SnapshotFile] = []
    for relative_path in relative_paths:
        posix_path = PurePosixPath(relative_path)
        if (
            posix_path.is_absolute()
            or "\\" in relative_path
            or any(part in {".", ".."} for part in posix_path.parts)
        ):
            raise RepositoryImportError(f"Git 返回了越界文件路径：{relative_path!r}")

        source_path = repository_root.joinpath(*posix_path.parts)
        if source_path.is_symlink():
            resolved_path = source_path.resolve(strict=True)
            if not _is_same_or_child(resolved_path, repository_root):
                raise RepositoryImportError(
                    f"符号链接指向仓库根目录之外：{relative_path}"
                )
            if not resolved_path.is_file():
                raise RepositoryImportError(
                    f"不支持指向目录的符号链接：{relative_path}"
                )
            content = resolved_path.read_bytes()
        elif not source_path.exists():
            # 已跟踪但在工作树中删除的文件不属于当前快照。
            continue
        elif source_path.is_file():
            content = source_path.read_bytes()
        else:
            raise RepositoryImportError(
                f"不支持目录或 Git submodule 条目：{relative_path}"
            )

        destination = staging_root.joinpath(*posix_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        files.append(
            SnapshotFile(
                path=posix_path.as_posix(),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def _snapshot_matches(snapshot_root: Path, files: tuple[SnapshotFile, ...]) -> bool:
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        return False
    expected_paths = {item.path for item in files}
    actual_paths: set[str] = set()
    try:
        for path in snapshot_root.rglob("*"):
            if path.is_symlink():
                return False
            if path.is_file():
                actual_paths.add(path.relative_to(snapshot_root).as_posix())
        if actual_paths != expected_paths:
            return False
        for snapshot_file in files:
            path = snapshot_root.joinpath(*PurePosixPath(snapshot_file.path).parts)
            if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot_file.sha256:
                return False
    except OSError:
        return False
    return True


def _normalize_git_url(url: str) -> str:
    if not url or any(ord(character) < 32 for character in url):
        raise RepositoryImportError("Git URL 为空或包含控制字符。")
    parsed = urlsplit(url)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise RepositoryImportError("file Git URL 的主机、查询和片段必须为空。")
        return url
    if parsed.scheme in {"https", "ssh"}:
        if (
            parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RepositoryImportError("Git URL 必须有主机且不能内嵌凭据。")
        return url
    if re.fullmatch(r"git@[^:/\s]+:[^\s]+", url):
        return url
    raise RepositoryImportError("仅支持 HTTPS、SSH 或 file Git URL。")


def _git_repository_name(url: str) -> str:
    parsed = urlsplit(url)
    path = url.split(":", 1)[1] if url.startswith("git@") else parsed.path
    name = PurePosixPath(path.rstrip("/")).name.removesuffix(".git")
    return name or "repository"


def _git_tree_entries(tree: bytes) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for record in tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryImportError("Git tree 含有无法解析的文件记录。") from exc
        if object_type == "commit":
            raise RepositoryImportError("当前 Snapshot 导入不支持 Git submodule。")
        if object_type != "blob":
            raise RepositoryImportError(f"Git tree 含有不支持的对象类型：{object_type}")
        if mode == "120000":
            raise RepositoryImportError(f"当前 Snapshot 导入不支持符号链接：{path}")
        if mode not in {"100644", "100755"}:
            raise RepositoryImportError(f"Git tree 含有不支持的文件模式：{mode}")
        entries.append((path, object_id))
    return tuple(sorted(entries))


async def _copy_git_blobs(
    bare_repository: Path,
    entries: tuple[tuple[str, str], ...],
    staging_root: Path,
) -> tuple[SnapshotFile, ...]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    process = await asyncio.create_subprocess_exec(
        "git",
        f"--git-dir={bare_repository}",
        "cat-file",
        "--batch",
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    files: list[SnapshotFile] = []
    try:
        for relative_path, object_id in entries:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            await process.stdin.drain()
            header = await asyncio.wait_for(process.stdout.readline(), timeout=30)
            fields = header.rstrip(b"\n").split()
            if len(fields) != 3 or fields[1] != b"blob":
                raise RepositoryImportError("Git 对象数据库没有返回文件 blob。")
            try:
                content_size = int(fields[2])
            except ValueError as exc:
                raise RepositoryImportError("Git blob 大小不是有效整数。") from exc
            framed_content = await asyncio.wait_for(
                process.stdout.readexactly(content_size + 1),
                timeout=30,
            )
            if framed_content[-1:] != b"\n":
                raise RepositoryImportError("Git blob 传输帧不完整。")
            content = framed_content[:-1]
            posix_path = PurePosixPath(relative_path)
            destination = staging_root.joinpath(*posix_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            files.append(
                SnapshotFile(
                    path=posix_path.as_posix(),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )

        process.stdin.close()
        await process.stdin.wait_closed()
        return_code = await asyncio.wait_for(process.wait(), timeout=30)
        stderr = await process.stderr.read()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RepositoryImportError(detail or "Git cat-file 执行失败。")
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return tuple(sorted(files, key=lambda item: item.path))


async def _git_output(cwd: Path | None, *arguments: str) -> str:
    return (await _git_bytes(cwd, *arguments)).decode("utf-8").strip()


async def _git_bytes(cwd: Path | None, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command = ["git"]
    if cwd is not None:
        command.extend(["-C", str(cwd)])
    command.extend(arguments)
    process = await asyncio.create_subprocess_exec(
        *command,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RepositoryImportError("Git 命令超过 30 秒，导入已取消。") from exc
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RepositoryImportError(detail or "Git 命令执行失败。")
    return stdout


__all__ = ["RepositoryImportError", "RepositoryWorkspace"]
