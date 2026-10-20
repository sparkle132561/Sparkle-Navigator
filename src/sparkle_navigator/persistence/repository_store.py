"""用 SQLite 保存 Repository、Snapshot 和导入关联。"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import and_, delete, or_, select, text, update
from sqlalchemy.orm import Session

from sparkle_navigator.domain.code_index import CodeRelationship, CodeSymbol, ImportRecord
from sparkle_navigator.domain.memory import ProjectFactStatus
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.domain.semantic import SemanticVectorRecord
from sparkle_navigator.persistence.database import open_sqlite_database
from sparkle_navigator.persistence.models import (
    ProjectFactRow,
    RepositoryRow,
    RepositorySnapshotFileRow,
    RepositorySnapshotImportRow,
    RepositorySnapshotLinkRow,
    RepositorySnapshotRelationshipRow,
    RepositorySnapshotRow,
    RepositorySnapshotSemanticVectorRow,
    RepositorySnapshotSymbolRow,
)


class RepositoryStoreError(RuntimeError):
    """Repository/Snapshot 持久化失败。"""


_RELATIONSHIP_SOURCE_BATCH_SIZE = 200


class RepositoryStore:
    """通过稳定的 Snapshot ID 保存并恢复 Repository 导入结果。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self._engine = open_sqlite_database(self.database_path)

    def save_import(self, result: RepositoryImportResult) -> RepositoryImportResult:
        """在一个事务中保存 Repository、Snapshot 及其关联。"""

        snapshot = result.snapshot
        if snapshot.snapshot_root is None:
            raise RepositoryStoreError("持久化 Snapshot 必须有内容目录。")
        manifest_json = _manifest_json(snapshot.files)

        with Session(self._engine) as session, session.begin():
            repository_row = session.get(RepositoryRow, result.repository.repository_id)
            if repository_row is None:
                repository_row = RepositoryRow(
                    repository_id=result.repository.repository_id,
                    name=result.repository.name,
                    source_type=result.repository.source_type,
                    source_locator=result.repository.source_locator,
                    trusted=False,
                    current_snapshot_id=None,
                )
                session.add(repository_row)
            else:
                if (
                    repository_row.source_type != result.repository.source_type
                    or repository_row.source_locator != result.repository.source_locator
                ):
                    raise RepositoryStoreError("Repository ID 与已保存来源不一致。")
                repository_row.name = result.repository.name

            snapshot_row = session.get(RepositorySnapshotRow, snapshot.snapshot_id)
            if snapshot_row is None:
                snapshot_row = RepositorySnapshotRow(
                    snapshot_id=snapshot.snapshot_id,
                    source_commit=snapshot.source_commit,
                    file_manifest_json=manifest_json,
                    index_version=snapshot.index_version,
                    snapshot_root=str(snapshot.snapshot_root),
                )
                session.add(snapshot_row)
            elif (
                snapshot_row.source_commit != snapshot.source_commit
                or snapshot_row.file_manifest_json != manifest_json
                or snapshot_row.index_version != snapshot.index_version
            ):
                raise RepositoryStoreError("Snapshot ID 已对应不同的快照内容。")

            link = session.get(
                RepositorySnapshotLinkRow,
                (result.repository.repository_id, snapshot.snapshot_id),
            )
            if link is None:
                session.add(
                    RepositorySnapshotLinkRow(
                        repository_id=result.repository.repository_id,
                        snapshot_id=snapshot.snapshot_id,
                    )
                )
            repository_row.current_snapshot_id = snapshot.snapshot_id
            session.flush()
            session.execute(
                update(ProjectFactRow)
                .where(
                    ProjectFactRow.repository_id == result.repository.repository_id,
                    ProjectFactRow.snapshot_id != snapshot.snapshot_id,
                    ProjectFactRow.status == ProjectFactStatus.CONFIRMED.value,
                )
                .values(status=ProjectFactStatus.STALE.value)
            )
            repository = _repository_from_row(repository_row)

        return RepositoryImportResult(repository=repository, snapshot=snapshot)

    def get_snapshot(self, snapshot_id: str) -> RepositorySnapshot:
        """从 SQLite 恢复 Snapshot 清单和冻结副本位置。"""

        with Session(self._engine) as session:
            row = session.get(RepositorySnapshotRow, snapshot_id)
            if row is None:
                raise KeyError(f"Snapshot 不存在：{snapshot_id}")
            raw_manifest = json.loads(row.file_manifest_json)
            if not isinstance(raw_manifest, list):
                raise RepositoryStoreError("Snapshot 文件清单格式无效。")
            files = tuple(
                SnapshotFile(path=item["path"], sha256=item["sha256"])
                for item in raw_manifest
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
            )
            if len(files) != len(raw_manifest):
                raise RepositoryStoreError("Snapshot 文件清单包含无效记录。")
            snapshot = RepositorySnapshot.create(
                source_commit=row.source_commit,
                files=files,
                index_version=row.index_version,
                snapshot_root=Path(row.snapshot_root),
            )
            if snapshot.snapshot_id != row.snapshot_id:
                raise RepositoryStoreError("Snapshot 清单与保存的 Snapshot ID 不匹配。")
            return snapshot

    def get_repository(self, repository_id: str) -> Repository:
        """从 SQLite 恢复 Repository 及其当前信任状态。"""

        with Session(self._engine) as session:
            row = session.get(RepositoryRow, repository_id)
            if row is None:
                raise KeyError(f"Repository 不存在：{repository_id}")
            return _repository_from_row(row)

    def list_repositories(self) -> tuple[Repository, ...]:
        """按名称读取已导入仓库目录及各自的当前 Snapshot ID。"""

        with Session(self._engine) as session:
            rows = session.execute(
                select(RepositoryRow).order_by(
                    RepositoryRow.name,
                    RepositoryRow.repository_id,
                )
            ).scalars().all()
        return tuple(_repository_from_row(row) for row in rows)

    def set_repository_trust(self, repository_id: str, *, trusted: bool) -> Repository:
        """持久化宿主明确作出的信任决定，不修改不可变 Snapshot。"""

        with Session(self._engine) as session, session.begin():
            row = session.get(RepositoryRow, repository_id)
            if row is None:
                raise KeyError(f"Repository 不存在：{repository_id}")
            row.trusted = trusted
            session.flush()
            return _repository_from_row(row)

    def replace_snapshot_file_index(
        self,
        snapshot_id: str,
        files: Sequence[tuple[SnapshotFile, int, str | None]],
    ) -> None:
        """原子替换一个 Snapshot 的文件元数据和 FTS5 文本。"""

        paths = tuple(snapshot_file.path for snapshot_file, _size, _content in files)
        if len(paths) != len(set(paths)):
            raise RepositoryStoreError("文件索引不能包含重复路径。")
        if any(size_bytes < 0 for _snapshot_file, size_bytes, _content in files):
            raise RepositoryStoreError("文件大小不能为负数。")

        with Session(self._engine) as session, session.begin():
            if session.get(RepositorySnapshotRow, snapshot_id) is None:
                raise KeyError(f"Snapshot 不存在：{snapshot_id}")
            session.execute(
                text("DELETE FROM repository_snapshot_files_fts WHERE snapshot_id = :snapshot_id"),
                {"snapshot_id": snapshot_id},
            )
            session.execute(
                delete(RepositorySnapshotFileRow).where(
                    RepositorySnapshotFileRow.snapshot_id == snapshot_id
                )
            )
            session.add_all(
                [
                    RepositorySnapshotFileRow(
                        snapshot_id=snapshot_id,
                        path=snapshot_file.path,
                        sha256=snapshot_file.sha256,
                        size_bytes=size_bytes,
                        indexed_text=content is not None,
                    )
                    for snapshot_file, size_bytes, content in files
                ]
            )
            text_rows = [
                {
                    "snapshot_id": snapshot_id,
                    "path": snapshot_file.path,
                    "content": content,
                }
                for snapshot_file, _size_bytes, content in files
                if content is not None
            ]
            if text_rows:
                session.execute(
                    text(
                        "INSERT INTO repository_snapshot_files_fts "
                        "(snapshot_id, path, content) "
                        "VALUES (:snapshot_id, :path, :content)"
                    ),
                    text_rows,
                )

    def search_snapshot_text(
        self,
        snapshot_id: str,
        match_query: str,
        *,
        limit: int,
    ) -> tuple[tuple[str, str, float], ...]:
        """查询指定 Snapshot 的 FTS5 文本并返回路径、片段和排序分。"""

        statement = text(
            "SELECT path, "
            "snippet(repository_snapshot_files_fts, 2, '[', ']', '…', 12), "
            "bm25(repository_snapshot_files_fts) "
            "FROM repository_snapshot_files_fts "
            "WHERE repository_snapshot_files_fts MATCH :match_query "
            "AND snapshot_id = :snapshot_id "
            "ORDER BY 3 LIMIT :limit"
        )
        with Session(self._engine) as session:
            rows = session.execute(
                statement,
                {
                    "snapshot_id": snapshot_id,
                    "match_query": match_query,
                    "limit": limit,
                },
            ).all()
        return tuple((path, snippet, float(score)) for path, snippet, score in rows)

    def replace_snapshot_symbol_index(
        self,
        snapshot_id: str,
        symbols: Sequence[tuple[str, CodeSymbol]],
        imports: Sequence[tuple[str, ImportRecord]],
    ) -> None:
        """在一个事务中替换指定 Snapshot 的符号和 import 记录。"""

        with Session(self._engine) as session, session.begin():
            if session.get(RepositorySnapshotRow, snapshot_id) is None:
                raise KeyError(f"Snapshot 不存在：{snapshot_id}")
            session.execute(
                delete(RepositorySnapshotSymbolRow).where(
                    RepositorySnapshotSymbolRow.snapshot_id == snapshot_id
                )
            )
            session.execute(
                delete(RepositorySnapshotImportRow).where(
                    RepositorySnapshotImportRow.snapshot_id == snapshot_id
                )
            )
            session.add_all(
                [
                    RepositorySnapshotSymbolRow(
                        snapshot_id=snapshot_id,
                        path=path,
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        kind=symbol.kind,
                        line_start=symbol.line_start,
                        line_end=symbol.line_end,
                    )
                    for path, symbol in symbols
                ]
            )
            session.add_all(
                [
                    RepositorySnapshotImportRow(
                        snapshot_id=snapshot_id,
                        path=path,
                        module=record.module,
                        imported_name=record.imported_name,
                        alias=record.alias,
                        relative_level=record.relative_level,
                        line=record.line,
                    )
                    for path, record in imports
                ]
            )

    def search_snapshot_symbols(
        self,
        snapshot_id: str,
        name: str,
    ) -> tuple[tuple[str, CodeSymbol], ...]:
        """按名称或限定名查询一个 Snapshot 的符号。"""

        statement = (
            select(RepositorySnapshotSymbolRow)
            .where(
                RepositorySnapshotSymbolRow.snapshot_id == snapshot_id,
                or_(
                    RepositorySnapshotSymbolRow.name == name,
                    RepositorySnapshotSymbolRow.qualified_name == name,
                ),
            )
            .order_by(
                RepositorySnapshotSymbolRow.path,
                RepositorySnapshotSymbolRow.line_start,
            )
        )
        with Session(self._engine) as session:
            rows = session.execute(statement).scalars().all()
        return tuple(
            (
                row.path,
                CodeSymbol(
                    name=row.name,
                    qualified_name=row.qualified_name,
                    kind=row.kind,
                    line_start=row.line_start,
                    line_end=row.line_end,
                ),
            )
            for row in rows
        )

    def get_snapshot_imports(
        self,
        snapshot_id: str,
        path: str,
    ) -> tuple[ImportRecord, ...]:
        """按 Snapshot 和文件路径读取 import 记录。"""

        statement = (
            select(RepositorySnapshotImportRow)
            .where(
                RepositorySnapshotImportRow.snapshot_id == snapshot_id,
                RepositorySnapshotImportRow.path == path,
            )
            .order_by(RepositorySnapshotImportRow.line, RepositorySnapshotImportRow.id)
        )
        with Session(self._engine) as session:
            rows = session.execute(statement).scalars().all()
        return tuple(
            ImportRecord(
                module=row.module,
                imported_name=row.imported_name,
                alias=row.alias,
                relative_level=row.relative_level,
                line=row.line,
            )
            for row in rows
        )

    def replace_snapshot_relationships(
        self,
        snapshot_id: str,
        relationships: Sequence[CodeRelationship],
    ) -> None:
        """在一个事务中替换指定 Snapshot 的全部代码关系。"""

        with Session(self._engine) as session, session.begin():
            if session.get(RepositorySnapshotRow, snapshot_id) is None:
                raise KeyError(f"Snapshot 不存在：{snapshot_id}")
            session.execute(
                delete(RepositorySnapshotRelationshipRow).where(
                    RepositorySnapshotRelationshipRow.snapshot_id == snapshot_id
                )
            )
            session.add_all(
                [
                    RepositorySnapshotRelationshipRow(
                        snapshot_id=snapshot_id,
                        relation_type=relationship.relation_type,
                        confidence=relationship.confidence,
                        source_path=relationship.source_path,
                        source_symbol=relationship.source_symbol,
                        target_path=relationship.target_path,
                        target_symbol=relationship.target_symbol,
                        evidence_line=relationship.evidence_line,
                    )
                    for relationship in relationships
                ]
            )

    def get_snapshot_relationships_from_many(
        self,
        snapshot_id: str,
        relation_type: str,
        sources: Sequence[tuple[str, str | None]],
    ) -> tuple[CodeRelationship, ...]:
        """批量读取指定 Snapshot 中一组节点的指定出边。"""

        unique_sources = tuple(dict.fromkeys(sources))
        if not unique_sources:
            return ()

        rows: list[RepositorySnapshotRelationshipRow] = []
        with Session(self._engine) as session:
            for start in range(0, len(unique_sources), _RELATIONSHIP_SOURCE_BATCH_SIZE):
                source_batch = unique_sources[
                    start : start + _RELATIONSHIP_SOURCE_BATCH_SIZE
                ]
                source_conditions = []
                for source_path, source_symbol in source_batch:
                    source_condition = RepositorySnapshotRelationshipRow.source_path == source_path
                    if source_symbol is None:
                        source_condition = and_(
                            source_condition,
                            RepositorySnapshotRelationshipRow.source_symbol.is_(None),
                        )
                    else:
                        source_condition = and_(
                            source_condition,
                            RepositorySnapshotRelationshipRow.source_symbol == source_symbol,
                        )
                    source_conditions.append(source_condition)
                statement = (
                    select(RepositorySnapshotRelationshipRow)
                    .where(
                        RepositorySnapshotRelationshipRow.snapshot_id == snapshot_id,
                        RepositorySnapshotRelationshipRow.relation_type == relation_type,
                        or_(*source_conditions),
                    )
                    .order_by(
                        RepositorySnapshotRelationshipRow.source_path,
                        RepositorySnapshotRelationshipRow.source_symbol,
                        RepositorySnapshotRelationshipRow.evidence_line,
                        RepositorySnapshotRelationshipRow.target_path,
                        RepositorySnapshotRelationshipRow.target_symbol,
                        RepositorySnapshotRelationshipRow.id,
                    )
                )
                rows.extend(session.execute(statement).scalars().all())
        return tuple(
            CodeRelationship(
                relation_type=row.relation_type,
                source_path=row.source_path,
                source_symbol=row.source_symbol,
                target_path=row.target_path,
                target_symbol=row.target_symbol,
                confidence=row.confidence,
                evidence_line=row.evidence_line,
            )
            for row in rows
        )

    def get_snapshot_relationships_to_many(
        self,
        snapshot_id: str,
        relation_type: str,
        targets: Sequence[tuple[str, str | None]],
    ) -> tuple[CodeRelationship, ...]:
        """批量读取指定 Snapshot 中一组节点的指定入边。"""

        unique_targets = tuple(dict.fromkeys(targets))
        if not unique_targets:
            return ()

        rows: list[RepositorySnapshotRelationshipRow] = []
        with Session(self._engine) as session:
            for start in range(0, len(unique_targets), _RELATIONSHIP_SOURCE_BATCH_SIZE):
                target_batch = unique_targets[
                    start : start + _RELATIONSHIP_SOURCE_BATCH_SIZE
                ]
                target_conditions = []
                for target_path, target_symbol in target_batch:
                    target_condition = RepositorySnapshotRelationshipRow.target_path == target_path
                    if target_symbol is None:
                        target_condition = and_(
                            target_condition,
                            RepositorySnapshotRelationshipRow.target_symbol.is_(None),
                        )
                    else:
                        target_condition = and_(
                            target_condition,
                            RepositorySnapshotRelationshipRow.target_symbol == target_symbol,
                        )
                    target_conditions.append(target_condition)
                statement = (
                    select(RepositorySnapshotRelationshipRow)
                    .where(
                        RepositorySnapshotRelationshipRow.snapshot_id == snapshot_id,
                        RepositorySnapshotRelationshipRow.relation_type == relation_type,
                        or_(*target_conditions),
                    )
                    .order_by(
                        RepositorySnapshotRelationshipRow.target_path,
                        RepositorySnapshotRelationshipRow.target_symbol,
                        RepositorySnapshotRelationshipRow.evidence_line,
                        RepositorySnapshotRelationshipRow.source_path,
                        RepositorySnapshotRelationshipRow.source_symbol,
                        RepositorySnapshotRelationshipRow.id,
                    )
                )
                rows.extend(session.execute(statement).scalars().all())
        return tuple(
            CodeRelationship(
                relation_type=row.relation_type,
                source_path=row.source_path,
                source_symbol=row.source_symbol,
                target_path=row.target_path,
                target_symbol=row.target_symbol,
                confidence=row.confidence,
                evidence_line=row.evidence_line,
            )
            for row in rows
        )

    def replace_snapshot_semantic_vectors(
        self,
        snapshot_id: str,
        model_id: str,
        documents: Sequence[SemanticVectorRecord],
    ) -> None:
        """在一个事务中替换 Snapshot 和模型对应的全部语义向量。"""

        if not model_id.strip():
            raise RepositoryStoreError("语义模型 ID 不能为空。")
        document_ids = tuple(document.document_id for document in documents)
        if len(document_ids) != len(set(document_ids)):
            raise RepositoryStoreError("语义文档 ID 不能重复。")
        dimensions = {len(document.embedding) for document in documents}
        if any(dimension < 1 or dimension > 16384 for dimension in dimensions):
            raise RepositoryStoreError("语义向量维度无效。")
        if len(dimensions) > 1:
            raise RepositoryStoreError("同一模型的语义向量维度必须一致。")

        with Session(self._engine) as session, session.begin():
            if session.get(RepositorySnapshotRow, snapshot_id) is None:
                raise KeyError(f"Snapshot 不存在：{snapshot_id}")
            session.execute(
                delete(RepositorySnapshotSemanticVectorRow).where(
                    RepositorySnapshotSemanticVectorRow.snapshot_id == snapshot_id,
                    RepositorySnapshotSemanticVectorRow.model_id == model_id,
                )
            )
            session.add_all(
                [
                    RepositorySnapshotSemanticVectorRow(
                        snapshot_id=snapshot_id,
                        model_id=model_id,
                        document_id=document.document_id,
                        path=document.path,
                        document_kind=document.document_kind,
                        symbol_name=document.symbol_name,
                        line_start=document.line_start,
                        line_end=document.line_end,
                        dimensions=len(document.embedding),
                        embedding=struct.pack(
                            f"<{len(document.embedding)}f",
                            *document.embedding,
                        ),
                    )
                    for document in documents
                ]
            )

    def get_snapshot_semantic_vectors(
        self,
        snapshot_id: str,
        model_id: str,
    ) -> tuple[SemanticVectorRecord, ...]:
        """读取指定 Snapshot 和模型的语义文档向量。"""

        statement = (
            select(RepositorySnapshotSemanticVectorRow)
            .where(
                RepositorySnapshotSemanticVectorRow.snapshot_id == snapshot_id,
                RepositorySnapshotSemanticVectorRow.model_id == model_id,
            )
            .order_by(
                RepositorySnapshotSemanticVectorRow.path,
                RepositorySnapshotSemanticVectorRow.document_kind,
                RepositorySnapshotSemanticVectorRow.symbol_name,
                RepositorySnapshotSemanticVectorRow.document_id,
            )
        )
        with Session(self._engine) as session:
            rows = session.execute(statement).scalars().all()

        documents: list[SemanticVectorRecord] = []
        for row in rows:
            if (
                row.dimensions < 1
                or row.dimensions > 16384
                or len(row.embedding) != row.dimensions * 4
            ):
                raise RepositoryStoreError("SQLite 中的语义向量数据损坏。")
            try:
                embedding = struct.unpack(f"<{row.dimensions}f", row.embedding)
            except struct.error as exc:
                raise RepositoryStoreError("SQLite 中的语义向量数据损坏。") from exc
            documents.append(
                SemanticVectorRecord(
                    document_id=row.document_id,
                    path=row.path,
                    document_kind=row.document_kind,
                    symbol_name=row.symbol_name,
                    line_start=row.line_start,
                    line_end=row.line_end,
                    embedding=embedding,
                )
            )
        return tuple(documents)

def _manifest_json(files: tuple[SnapshotFile, ...]) -> str:
    return json.dumps(
        [{"path": item.path, "sha256": item.sha256} for item in files],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _repository_from_row(row: RepositoryRow) -> Repository:
    return Repository(
        repository_id=row.repository_id,
        name=row.name,
        source_type=row.source_type,
        source_locator=row.source_locator,
        trusted=row.trusted,
        current_snapshot_id=row.current_snapshot_id,
    )


__all__ = ["RepositoryStore", "RepositoryStoreError"]
