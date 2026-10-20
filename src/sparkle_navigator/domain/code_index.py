"""代码符号索引使用的稳定领域记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SymbolKind = Literal["class", "function", "method"]
CallSiteKind = Literal["name", "attribute", "dynamic"]
RelationshipType = Literal[
    "DEFINES",
    "IMPORTS",
    "CALLS",
    "REFERENCES",
    "INHERITS",
    "TESTS",
    "DEPENDS_ON",
]
RelationshipConfidence = Literal["CONFIRMED", "LIKELY", "INFERRED"]


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    """一个 Python 类、函数或方法声明。"""

    name: str
    qualified_name: str
    kind: SymbolKind
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ImportRecord:
    """一条 import 语法记录，不表示目标模块已成功解析。"""

    module: str | None
    imported_name: str | None
    alias: str | None
    relative_level: int
    line: int


@dataclass(frozen=True, slots=True)
class CallSite:
    """源码中的函数调用表达式及其词法上下文。"""

    caller_qualified_name: str | None
    callee: str
    callee_kind: CallSiteKind
    line: int


@dataclass(frozen=True, slots=True)
class CodeRelationship:
    """固定 Snapshot 中有源码位置支持的一条有向代码关系。"""

    relation_type: RelationshipType
    source_path: str
    source_symbol: str | None
    target_path: str | None
    target_symbol: str | None
    confidence: RelationshipConfidence
    evidence_line: int


__all__ = [
    "CallSite",
    "CallSiteKind",
    "CodeRelationship",
    "CodeSymbol",
    "ImportRecord",
    "RelationshipConfidence",
    "RelationshipType",
    "SymbolKind",
]
