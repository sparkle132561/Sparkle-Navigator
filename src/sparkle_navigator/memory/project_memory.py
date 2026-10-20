"""保存经用户确认且有快照源码证据支持的项目事实。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    read_verified_snapshot_files,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.domain.memory import ProjectFact, ProjectFactStatus
from sparkle_navigator.domain.repository import RepositorySnapshot
from sparkle_navigator.persistence.database import open_sqlite_database
from sparkle_navigator.persistence.models import ProjectFactRow, RepositoryRow
from sparkle_navigator.persistence.repository_store import RepositoryStore


class ProjectMemoryError(ValueError):
    """Project Fact 缺少有效 Snapshot 或源码行证据。"""


class ProjectMemory:
    """以公开方法管理带证据的项目事实及其生命周期。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self._engine = open_sqlite_database(self.database_path)
        self._repository_store = RepositoryStore(self.database_path)

    def record_user_correction(
        self,
        *,
        repository_id: str,
        snapshot_id: str,
        statement: str,
        evidence: Sequence[CodeEvidence],
        supersedes_fact_id: str | None = None,
    ) -> ProjectFact:
        """校验用户确认的源码证据并持久化为 CONFIRMED Project Fact。"""

        normalized_statement = statement.strip()
        if not normalized_statement:
            raise ProjectMemoryError("项目事实陈述不能为空。")
        evidence_items = tuple(evidence)
        if not evidence_items or any(
            not isinstance(item, CodeEvidence) for item in evidence_items
        ):
            raise ProjectMemoryError("项目事实至少需要一条 Code Evidence。")

        repository = self._repository_store.get_repository(repository_id)
        if repository.current_snapshot_id != snapshot_id:
            raise ProjectMemoryError("Project Fact 只能绑定 Repository 当前的 Snapshot。")
        snapshot = self._repository_store.get_snapshot(snapshot_id)
        self._verify_evidence(snapshot, evidence_items)

        fact = ProjectFact(
            fact_id=uuid4().hex,
            repository_id=repository_id,
            snapshot_id=snapshot_id,
            statement=normalized_statement,
            evidence=evidence_items,
            status=ProjectFactStatus.CONFIRMED,
            supersedes_fact_id=supersedes_fact_id,
            created_at=datetime.now(UTC),
        )
        with Session(self._engine) as session, session.begin():
            current_repository = session.get(RepositoryRow, repository_id)
            if current_repository is None:
                raise KeyError(f"Repository 不存在：{repository_id}")
            if current_repository.current_snapshot_id != snapshot_id:
                raise ProjectMemoryError("记录事实期间 Repository 已切换到其他 Snapshot。")

            previous_fact = None
            if supersedes_fact_id is not None:
                previous_fact = session.get(ProjectFactRow, supersedes_fact_id)
                if previous_fact is None:
                    raise KeyError(f"待替代的 Project Fact 不存在：{supersedes_fact_id}")
                if previous_fact.repository_id != repository_id:
                    raise ProjectMemoryError("只能替代同一 Repository 的 Project Fact。")
                if previous_fact.status in {
                    ProjectFactStatus.INVALID.value,
                    ProjectFactStatus.SUPERSEDED.value,
                }:
                    raise ProjectMemoryError("INVALID 或 SUPERSEDED 的事实不能再次被替代。")
                previous_fact.status = ProjectFactStatus.SUPERSEDED.value

            row = ProjectFactRow(
                fact_id=fact.fact_id,
                repository_id=fact.repository_id,
                snapshot_id=fact.snapshot_id,
                statement=fact.statement,
                evidence_json=_evidence_json(fact.evidence),
                status=fact.status.value,
                supersedes_fact_id=fact.supersedes_fact_id,
                created_at=fact.created_at,
            )
            session.add(row)
            session.flush()
        return fact

    def facts_for_snapshot(
        self,
        *,
        repository_id: str,
        snapshot_id: str,
    ) -> tuple[ProjectFact, ...]:
        """读取某 Repository/Snapshot 的事实及其当前状态。"""

        statement = (
            select(ProjectFactRow)
            .where(
                ProjectFactRow.repository_id == repository_id,
                ProjectFactRow.snapshot_id == snapshot_id,
            )
            .order_by(ProjectFactRow.created_at, ProjectFactRow.fact_id)
        )
        with Session(self._engine) as session:
            rows = session.execute(statement).scalars().all()
        return tuple(_project_fact_from_row(row) for row in rows)

    def verified_facts_for_snapshot(
        self,
        *,
        repository_id: str,
        snapshot_id: str,
    ) -> tuple[ProjectFact, ...]:
        """重新验证当前快照证据，只返回仍有效的 CONFIRMED 事实。"""

        repository = self._repository_store.get_repository(repository_id)
        if repository.current_snapshot_id != snapshot_id:
            raise ProjectMemoryError("Project Fact 只能从 Repository 当前的 Snapshot 注入。")
        snapshot = self._repository_store.get_snapshot(snapshot_id)
        verified: list[ProjectFact] = []
        for fact in self.facts_for_snapshot(
            repository_id=repository_id,
            snapshot_id=snapshot_id,
        ):
            if fact.status is not ProjectFactStatus.CONFIRMED:
                continue
            try:
                self._verify_evidence(snapshot, fact.evidence)
            except ProjectMemoryError:
                self.invalidate_fact(fact.fact_id)
                continue
            verified.append(fact)
        return tuple(verified)

    def invalidate_fact(self, fact_id: str) -> ProjectFact:
        """将指定事实标记为 INVALID，并返回更新后的记录。"""

        with Session(self._engine) as session, session.begin():
            row = session.get(ProjectFactRow, fact_id)
            if row is None:
                raise KeyError(f"Project Fact 不存在：{fact_id}")
            row.status = ProjectFactStatus.INVALID.value
            session.flush()
            return _project_fact_from_row(row)

    @staticmethod
    def _verify_evidence(
        snapshot: RepositorySnapshot,
        evidence: tuple[CodeEvidence, ...],
    ) -> None:
        snapshot_files = {file.path: file for file in snapshot.files}
        for item in evidence:
            if item.snapshot_id != snapshot.snapshot_id:
                raise ProjectMemoryError("Code Evidence 与 Project Fact 的 Snapshot ID 不一致。")
            if (
                not isinstance(item.line_start, int)
                or isinstance(item.line_start, bool)
                or not isinstance(item.line_end, int)
                or isinstance(item.line_end, bool)
                or item.line_start < 1
                or item.line_end < item.line_start
            ):
                raise ProjectMemoryError("Code Evidence 必须包含有效的源码行范围。")
            snapshot_file = snapshot_files.get(item.path)
            if snapshot_file is None or item.sha256 != snapshot_file.sha256:
                raise ProjectMemoryError("Code Evidence 文件或 SHA-256 与 Snapshot 清单不一致。")

        try:
            verified_files = read_verified_snapshot_files(
                snapshot,
                (item.path for item in evidence),
            )
        except SnapshotFileScanError as exc:
            raise ProjectMemoryError("Code Evidence 对应的 Snapshot 文件无法验证。") from exc

        contents = {snapshot_file.path: content for snapshot_file, _size, content in verified_files}
        for item in evidence:
            content = contents[item.path]
            if content is None:
                raise ProjectMemoryError("二进制文件不能作为 Project Fact 的源码行证据。")
            source_lines = content.splitlines()
            if item.line_end > len(source_lines):
                raise ProjectMemoryError("Code Evidence 行范围超出 Snapshot 文件内容。")
            expected_snippet = "\n".join(source_lines[item.line_start - 1 : item.line_end])
            if not item.snippet.strip() or item.snippet != expected_snippet:
                raise ProjectMemoryError("Code Evidence 片段与 Snapshot 源码行不一致。")


def _evidence_json(evidence: tuple[CodeEvidence, ...]) -> str:
    return json.dumps(
        [asdict(item) for item in evidence],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _project_fact_from_row(row: ProjectFactRow) -> ProjectFact:
    raw_evidence = json.loads(row.evidence_json)
    if not isinstance(raw_evidence, list) or any(
        not isinstance(item, dict) for item in raw_evidence
    ):
        raise ProjectMemoryError("SQLite 中的 Project Fact 证据格式无效。")
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return ProjectFact(
        fact_id=row.fact_id,
        repository_id=row.repository_id,
        snapshot_id=row.snapshot_id,
        statement=row.statement,
        evidence=tuple(CodeEvidence(**item) for item in raw_evidence),
        status=ProjectFactStatus(row.status),
        supersedes_fact_id=row.supersedes_fact_id,
        created_at=created_at,
    )


__all__ = ["ProjectMemory", "ProjectMemoryError"]
