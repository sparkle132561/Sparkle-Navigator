"""固定 Repository Snapshot 的代码证据领域记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sparkle_navigator.domain.code_index import (
    CodeRelationship,
    RelationshipConfidence,
    RelationshipType,
)

EvidenceSource = Literal[
    "text",
    "symbol",
    "relationship",
    "semantic",
    "overview",
]


@dataclass(frozen=True, slots=True)
class CodeEvidence:
    """能够回到指定 Snapshot 文件位置复核的一条证据。"""

    snapshot_id: str
    path: str
    line_start: int | None
    line_end: int | None
    snippet: str
    sha256: str
    source: EvidenceSource
    confidence: RelationshipConfidence = "CONFIRMED"
    symbol_name: str | None = None
    relationship_type: RelationshipType | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """针对一个问题，从单一 Snapshot 选出的有序代码证据。"""

    snapshot_id: str
    query: str
    items: tuple[CodeEvidence, ...]


@dataclass(frozen=True, slots=True)
class SnapshotAnswer:
    """由可复核代码片段组成的确定性 Ask 返回值。"""

    snapshot_id: str
    question: str
    answer: str
    evidence: EvidenceSet


@dataclass(frozen=True, slots=True)
class SnapshotOverview:
    """从固定 Snapshot 文件清单和可读说明文件生成的结构概览。"""

    snapshot_id: str
    source_commit: str
    file_count: int
    python_file_count: int
    top_level_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    entry_point_candidates: tuple[str, ...]
    dependency_files: tuple[str, ...]
    evidence: EvidenceSet


@dataclass(frozen=True, slots=True)
class SnapshotExplanation:
    """针对一个文件或符号收集的源码和静态关系证据。"""

    snapshot_id: str
    target: str
    target_kind: Literal["file", "symbol", "not_found"]
    relationships: tuple[CodeRelationship, ...]
    diagnostics: tuple[str, ...]
    evidence: EvidenceSet


@dataclass(frozen=True, slots=True)
class SnapshotTrace:
    """从指定符号沿静态调用关系追踪得到的结果。"""

    snapshot_id: str
    start_path: str
    start_symbol: str
    max_depth: int
    relationships: tuple[CodeRelationship, ...]
    evidence: EvidenceSet


@dataclass(frozen=True, slots=True)
class SnapshotImpact:
    """按反向静态关系和测试文本候选计算的影响结果。"""

    snapshot_id: str
    target_path: str
    target_symbol: str | None
    max_depth: int
    callers: tuple[CodeRelationship, ...]
    dependents: tuple[CodeRelationship, ...]
    test_candidates: tuple[str, ...]
    evidence: EvidenceSet


__all__ = [
    "CodeEvidence",
    "EvidenceSet",
    "SnapshotAnswer",
    "SnapshotExplanation",
    "SnapshotImpact",
    "SnapshotOverview",
    "SnapshotTrace",
]
