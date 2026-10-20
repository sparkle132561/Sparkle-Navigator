"""语义索引使用的向量文档与候选记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticDocumentKind = Literal["file", "symbol"]


@dataclass(frozen=True, slots=True)
class SemanticVectorRecord:
    """固定 Snapshot 中一个文件或代码符号的归一化向量记录。"""

    document_id: str
    path: str
    document_kind: SemanticDocumentKind
    symbol_name: str | None
    line_start: int | None
    line_end: int | None
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SemanticSearchHit:
    """语义索引召回的文件或符号候选。"""

    document_id: str
    path: str
    document_kind: SemanticDocumentKind
    symbol_name: str | None
    line_start: int | None
    line_end: int | None
    score: float


__all__ = ["SemanticDocumentKind", "SemanticSearchHit", "SemanticVectorRecord"]
