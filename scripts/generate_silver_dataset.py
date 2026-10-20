"""从 SQLite Snapshot 清单生成可重复的 Silver Dataset。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import subprocess
import sqlite3
from pathlib import Path
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import cast

from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.evaluation.silver_dataset import generate_silver_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--repositories-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-repository", type=int, default=60)
    arguments = parser.parse_args()

    if arguments.cases_per_repository < 1:
        parser.error("--cases-per-repository 必须是正整数。")
    if not arguments.output_dir.parent.is_dir():
        parser.error("--output-dir 的父目录必须已存在。")

    repository_manifest = json.loads(
        arguments.repositories_manifest.read_text(encoding="utf-8")
    )
    repositories = repository_manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        parser.error("来源 Manifest 必须包含非空 repositories 数组。")

    cases: list[dict[str, object]] = []
    for repository in repositories:
        if not isinstance(repository, dict):
            parser.error("来源 Manifest 中的 Repository 条目格式无效。")
        repository_id = repository.get("repository_id")
        snapshot_id = repository.get("snapshot_id")
        source_commit = repository.get("source_commit")
        if not all(isinstance(value, str) and value for value in (repository_id, snapshot_id, source_commit)):
            parser.error("Repository 条目缺少 repository_id、snapshot_id 或 source_commit。")

        snapshot = _load_snapshot(
            arguments.database_path,
            repository_id=repository_id,
            snapshot_id=snapshot_id,
        )
        if snapshot.source_commit != source_commit:
            parser.error(f"{repository_id} 的 Snapshot commit 与来源 Manifest 不一致。")
        source_root = arguments.source_cache_root / _source_cache_name(repository_id)
        source_head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if source_head != snapshot.source_commit:
            parser.error(f"{repository_id} 的 source-cache HEAD 与 Snapshot commit 不一致。")
        with TemporaryDirectory(
            prefix="sparkle-silver-snapshot-",
            dir=arguments.output_dir.parent,
        ) as temporary_root:
            staged_snapshot = _stage_python_files(
                snapshot,
                source_root,
                Path(temporary_root) / "snapshot",
            )
            generated = generate_silver_cases(
                repository_id,
                staged_snapshot,
                limit=arguments.cases_per_repository,
            )
        if len(generated) != arguments.cases_per_repository:
            parser.error(
                f"{repository_id} 只能生成 {len(generated)} 条，"
                f"目标为 {arguments.cases_per_repository} 条。"
            )
        cases.extend(cast(dict[str, object], case) for case in generated)

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        parser.error("生成结果包含重复 case_id。")

    first_provenance = cast(dict[str, object], cases[0]["provenance"])
    dataset_manifest = {
        "dataset_id": "sparkle-post-alpha-silver",
        "version": "0.1.0",
        "split": "silver",
        "target_case_count": len(cases),
        "generation_status": "complete",
        "generation": {
            "generator": first_provenance["generator"],
            "generator_version": first_provenance["generator_version"],
            "method": (
                "stdlib AST 提取类/函数声明、import 语句和调用表达式；"
                "按 Snapshot 路径排序并按题型轮转。"
            ),
            "deterministic_input": (
                "repository_id + source_commit + snapshot_id + Snapshot Manifest 中的 "
                "Python 路径/源码事实 + generator_version + cases_per_repository；不使用随机种子。"
            ),
            "evidence_validation": (
                "读取 HEAD 与 Snapshot commit 完全匹配的 source-cache；逐个 .py 文件"
                "按 Snapshot Manifest 校验 SHA-256，再由安全 Snapshot 读取器校验暂存内容；"
                "AST 语法解析成功后从相同文本提取行号与 snippet。"
            ),
            "cases_per_repository": arguments.cases_per_repository,
            "question_types": ["symbol_location", "import_statement", "call_site"],
        },
        "repositories": repositories,
        "cases_file": "cases.json",
        "case_count": len(cases),
    }
    _write_new_dataset(
        arguments.output_dir,
        dataset_manifest=dataset_manifest,
        cases=cases,
    )
    print(
        f"Generated {len(cases)} Silver cases for {len(repositories)} repositories "
        f"at {arguments.output_dir.resolve()}"
    )


def _load_snapshot(
    database_path: Path,
    *,
    repository_id: str,
    snapshot_id: str,
) -> RepositorySnapshot:
    catalog_name = _catalog_name_for_repository_id(repository_id)
    database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        row = connection.execute(
            """
            SELECT repository.name,
                   snapshot.source_commit,
                   snapshot.file_manifest_json,
                   snapshot.index_version,
                   snapshot.snapshot_root
            FROM repository_snapshots AS snapshot
            JOIN repository_snapshot_links AS link
              ON link.snapshot_id = snapshot.snapshot_id
            JOIN repositories AS repository
              ON repository.repository_id = link.repository_id
            WHERE snapshot.snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchall()
    if len(row) != 1 or row[0][0] != catalog_name:
        raise ValueError(
            f"数据库中的 Snapshot 关联与 Repository 不匹配：{repository_id}/{snapshot_id}"
        )

    _catalog_name, source_commit, raw_manifest, index_version, snapshot_root = row[0]
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, list):
        raise ValueError(f"Snapshot 文件清单格式无效：{snapshot_id}")
    files = tuple(
        SnapshotFile(path=item["path"], sha256=item["sha256"])
        for item in manifest
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    )
    if len(files) != len(manifest):
        raise ValueError(f"Snapshot 文件清单含无效记录：{snapshot_id}")

    snapshot = RepositorySnapshot.create(
        source_commit=source_commit,
        files=files,
        index_version=index_version,
        snapshot_root=Path(snapshot_root),
    )
    if snapshot.snapshot_id != snapshot_id:
        raise ValueError(f"Snapshot Manifest 无法重建其 ID：{snapshot_id}")
    return snapshot


def _catalog_name_for_repository_id(repository_id: str) -> str:
    catalog_names = {
        "tau-v0.4.4": "tau",
        "fastapi-0.143.0": "fastapi",
        "django-oscar-4.2.1": "django-oscar",
        "httpx-0.28.1": "httpx",
        "sparkle-navigator": "sparkle",
    }
    try:
        return catalog_names[repository_id]
    except KeyError as exc:
        raise ValueError(f"未知的 Benchmark Repository ID：{repository_id}") from exc


def _source_cache_name(repository_id: str) -> str:
    cache_names = {
        "tau-v0.4.4": "tau",
        "fastapi-0.143.0": "fastapi",
        "django-oscar-4.2.1": "django-oscar",
        "httpx-0.28.1": "httpx",
        "sparkle-navigator": "sparkle",
    }
    try:
        return cache_names[repository_id]
    except KeyError as exc:
        raise ValueError(f"未知的 Benchmark Repository ID：{repository_id}") from exc


def _stage_python_files(
    snapshot: RepositorySnapshot,
    source_root: Path,
    destination_root: Path,
) -> RepositorySnapshot:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source-cache 缺少安全的源码根目录。")
    resolved_source_root = source_root.resolve(strict=False)
    destination_root.mkdir(parents=True, exist_ok=False)

    for snapshot_file in snapshot.files:
        if not snapshot_file.path.lower().endswith(".py"):
            continue
        relative_path = PurePosixPath(snapshot_file.path)
        source_path = source_root
        for part in relative_path.parts:
            source_path = source_path / part
            if source_path.is_symlink():
                raise ValueError(f"Snapshot 中不允许符号链接：{snapshot_file.path}")
        resolved_source = source_path.resolve(strict=False)
        if (
            not resolved_source.is_relative_to(resolved_source_root)
            or not resolved_source.is_file()
        ):
            raise ValueError(f"Snapshot Python 文件路径无效：{snapshot_file.path}")
        contents = resolved_source.read_bytes()
        if sha256(contents).hexdigest() != snapshot_file.sha256:
            raise ValueError(f"Snapshot Python 文件哈希不匹配：{snapshot_file.path}")

        destination_path = destination_root.joinpath(*relative_path.parts)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(contents)

    staged_snapshot = RepositorySnapshot.create(
        source_commit=snapshot.source_commit,
        files=snapshot.files,
        index_version=snapshot.index_version,
        snapshot_root=destination_root,
    )
    if staged_snapshot.snapshot_id != snapshot.snapshot_id:
        raise ValueError("暂存后的 Snapshot 身份与来源不一致。")
    return staged_snapshot


def _write_new_dataset(
    output_dir: Path,
    *,
    dataset_manifest: dict[str, object],
    cases: list[dict[str, object]],
) -> None:
    manifest_path = output_dir / "manifest.json"
    cases_path = output_dir / "cases.json"
    if manifest_path.exists() or cases_path.exists():
        raise FileExistsError(
            f"输出文件已存在，不会覆盖：{output_dir}。请使用新的版本目录。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cases_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
