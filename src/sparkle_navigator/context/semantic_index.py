"""在固定 Snapshot 上建立和查询文件级、符号级语义向量。"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    scan_verified_snapshot_files,
)
from sparkle_navigator.context.python_symbols import (
    ParseDiagnostic,
    PythonFileParseResult,
    parse_python_file,
)
from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.domain.semantic import (
    SemanticDocumentKind,
    SemanticSearchHit,
    SemanticVectorRecord,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MAX_FILE_DOCUMENT_CHARS = 3000
_MAX_SYMBOL_DOCUMENT_CHARS = 4000


class SemanticIndexError(RuntimeError):
    """Snapshot 验证、Embedding 或语义索引操作失败。"""


class EmbeddingBackend(Protocol):
    """将查询或文档文本转换为同一向量空间的后端接口。"""

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, documents: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, query: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class SemanticIndexBuildResult:
    """一次 Snapshot 语义索引构建的计数和解析诊断。"""

    snapshot_id: str
    model_id: str
    document_count: int
    file_document_count: int
    symbol_document_count: int
    python_file_count: int
    diagnostics: tuple[ParseDiagnostic, ...]


class FastEmbedLocalBackend:
    """使用本机 FastEmbed/ONNX Runtime 生成查询和文档向量。"""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise SemanticIndexError(
                "本地语义检索需要可选依赖 fastembed；请安装项目的 semantic extra。"
            ) from exc

        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise SemanticIndexError(
                f"无法加载本地 Embedding 模型 {model_name}；请确认模型已缓存。"
            ) from exc
        try:
            fastembed_version = version("fastembed")
        except PackageNotFoundError:
            fastembed_version = "unknown"
        self.model_id = f"fastembed:{fastembed_version}:{model_name}"

    def embed_documents(self, documents: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) for value in vector)
            for vector in self._model.passage_embed(documents)
        )

    def embed_query(self, query: str) -> tuple[float, ...]:
        vectors = tuple(self._model.query_embed(query))
        if len(vectors) != 1:
            raise SemanticIndexError("Embedding 后端必须为单个查询返回一个向量。")
        return tuple(float(value) for value in vectors[0])


@dataclass(frozen=True, slots=True)
class _SemanticDocumentDraft:
    document_id: str
    path: str
    document_kind: SemanticDocumentKind
    symbol_name: str | None
    line_start: int | None
    line_end: int | None
    content: str


class SemanticIndex:
    """验证 Snapshot、持久化本地语义向量并按快照检索候选。"""

    def __init__(
        self,
        *,
        database_path: Path,
        embedder: EmbeddingBackend | None = None,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        self._store = RepositoryStore(database_path)
        self._embedder = embedder or FastEmbedLocalBackend(
            model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    async def build(self, snapshot: RepositorySnapshot) -> SemanticIndexBuildResult:
        """校验并解析 Snapshot，为文件概览和 Python 符号建立向量。"""

        try:
            scanned_files = await asyncio.to_thread(scan_verified_snapshot_files, snapshot)
        except SnapshotFileScanError as exc:
            raise SemanticIndexError(str(exc)) from exc

        drafts, python_file_count, diagnostics = await asyncio.to_thread(
            _extract_semantic_documents,
            scanned_files,
        )
        if drafts:
            embeddings = await asyncio.to_thread(
                self._embedder.embed_documents,
                tuple(document.content for document in drafts),
            )
            records = _records_with_embeddings(drafts, embeddings)
        else:
            records = ()
        self._store.replace_snapshot_semantic_vectors(
            snapshot.snapshot_id,
            self._embedder.model_id,
            records,
        )
        file_document_count = sum(item.document_kind == "file" for item in records)
        symbol_document_count = sum(item.document_kind == "symbol" for item in records)
        return SemanticIndexBuildResult(
            snapshot_id=snapshot.snapshot_id,
            model_id=self._embedder.model_id,
            document_count=len(records),
            file_document_count=file_document_count,
            symbol_document_count=symbol_document_count,
            python_file_count=python_file_count,
            diagnostics=diagnostics,
        )

    def search(
        self,
        snapshot_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[SemanticSearchHit, ...]:
        """按归一化 Embedding 的余弦相似度返回指定 Snapshot 的候选。"""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit 必须是正整数。")
        if not query.strip():
            raise ValueError("语义查询不能为空。")

        documents = self._store.get_snapshot_semantic_vectors(
            snapshot_id,
            self._embedder.model_id,
        )
        if not documents:
            return ()
        query_vector = _normalize_embedding(self._embedder.embed_query(query))
        ranked: list[SemanticSearchHit] = []
        for document in documents:
            if len(document.embedding) != len(query_vector):
                raise SemanticIndexError("查询向量与已保存文档向量的维度不一致。")
            document_vector = _normalize_embedding(document.embedding)
            score = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector,
                    document_vector,
                    strict=True,
                )
            )
            ranked.append(
                SemanticSearchHit(
                    document_id=document.document_id,
                    path=document.path,
                    document_kind=document.document_kind,
                    symbol_name=document.symbol_name,
                    line_start=document.line_start,
                    line_end=document.line_end,
                    score=score,
                )
            )
        ranked.sort(
            key=lambda hit: (
                -hit.score,
                hit.path,
                hit.document_kind,
                hit.symbol_name or "",
                hit.document_id,
            )
        )
        return tuple(ranked[:limit])


def _extract_semantic_documents(
    scanned_files: tuple[tuple[SnapshotFile, int, str | None], ...],
) -> tuple[tuple[_SemanticDocumentDraft, ...], int, tuple[ParseDiagnostic, ...]]:
    drafts: list[_SemanticDocumentDraft] = []
    diagnostics: list[ParseDiagnostic] = []
    python_file_count = 0

    for snapshot_file, _size_bytes, content in scanned_files:
        if content is None:
            continue
        path = snapshot_file.path
        parsed: PythonFileParseResult | None = None
        if path.lower().endswith(".py"):
            python_file_count += 1
            parsed = parse_python_file(path, content)
            if parsed.diagnostic is not None:
                diagnostics.append(parsed.diagnostic)

        drafts.append(
            _SemanticDocumentDraft(
                document_id=_document_id(path, "file", None, None),
                path=path,
                document_kind="file",
                symbol_name=None,
                line_start=None,
                line_end=None,
                content=_file_outline(path, content, parsed),
            )
        )
        if parsed is None or parsed.diagnostic is not None:
            continue

        source_lines = content.splitlines()
        for symbol in parsed.symbols:
            code = "\n".join(source_lines[symbol.line_start - 1 : symbol.line_end])
            symbol_text = (
                f"File: {path}\n"
                f"Symbol: {symbol.qualified_name}\n"
                f"Kind: {symbol.kind}\n"
                f"Code:\n{code}"
            )[:_MAX_SYMBOL_DOCUMENT_CHARS]
            drafts.append(
                _SemanticDocumentDraft(
                    document_id=_document_id(
                        path,
                        "symbol",
                        symbol.qualified_name,
                        symbol.line_start,
                    ),
                    path=path,
                    document_kind="symbol",
                    symbol_name=symbol.qualified_name,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                    content=symbol_text,
                )
            )
    return tuple(drafts), python_file_count, tuple(diagnostics)


def _file_outline(
    path: str,
    content: str,
    parsed: PythonFileParseResult | None,
) -> str:
    lines = [f"File: {path}"]
    if parsed is not None and parsed.diagnostic is None:
        if parsed.symbols:
            lines.append("Declarations:")
            lines.extend(
                f"{symbol.kind} {symbol.qualified_name} lines {symbol.line_start}-{symbol.line_end}"
                for symbol in parsed.symbols
            )
        if parsed.imports:
            lines.append("Imports:")
            lines.extend(
                " ".join(
                    part
                    for part in (
                        record.module or "",
                        record.imported_name or "",
                        record.alias or "",
                    )
                    if part
                )
                for record in parsed.imports
            )
    lines.append("Text preview:")
    lines.append(content[:_MAX_FILE_DOCUMENT_CHARS])
    return "\n".join(lines)[:_MAX_FILE_DOCUMENT_CHARS]


def _document_id(
    path: str,
    document_kind: SemanticDocumentKind,
    symbol_name: str | None,
    line_start: int | None,
) -> str:
    identity = "\0".join(
        (path, document_kind, symbol_name or "", str(line_start or ""))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _records_with_embeddings(
    drafts: Sequence[_SemanticDocumentDraft],
    embeddings: Sequence[Sequence[float]],
) -> tuple[SemanticVectorRecord, ...]:
    if len(drafts) != len(embeddings):
        raise SemanticIndexError("Embedding 后端返回的向量数量与文档数量不一致。")
    records: list[SemanticVectorRecord] = []
    for draft, embedding in zip(drafts, embeddings, strict=True):
        records.append(
            SemanticVectorRecord(
                document_id=draft.document_id,
                path=draft.path,
                document_kind=draft.document_kind,
                symbol_name=draft.symbol_name,
                line_start=draft.line_start,
                line_end=draft.line_end,
                embedding=_normalize_embedding(embedding),
            )
        )
    dimensions = {len(record.embedding) for record in records}
    if len(dimensions) > 1:
        raise SemanticIndexError("Embedding 后端为同一批文档返回了不同维度的向量。")
    return tuple(records)


def _normalize_embedding(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise SemanticIndexError("Embedding 向量必须是非空有限数值。")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise SemanticIndexError("Embedding 向量不能全为零。")
    return tuple(value / norm for value in vector)


__all__ = [
    "DEFAULT_FASTEMBED_MODEL",
    "EmbeddingBackend",
    "FastEmbedLocalBackend",
    "SemanticIndex",
    "SemanticIndexBuildResult",
    "SemanticIndexError",
]
