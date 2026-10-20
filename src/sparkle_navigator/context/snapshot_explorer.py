"""从固定 Repository Snapshot 返回概览或带源码位置的检索结果。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    read_verified_snapshot_files,
)
from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.python_symbols import parse_python_file
from sparkle_navigator.context.rank_fusion import reciprocal_rank_fusion
from sparkle_navigator.context.semantic_index import SemanticIndex
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex, SymbolSearchHit
from sparkle_navigator.context.text_index import SearchHit, SnapshotTextIndex
from sparkle_navigator.domain.code_index import (
    CodeRelationship,
    RelationshipConfidence,
    RelationshipType,
)
from sparkle_navigator.domain.evidence import (
    CodeEvidence,
    EvidenceSource,
    EvidenceSet,
    SnapshotAnswer,
    SnapshotExplanation,
    SnapshotImpact,
    SnapshotOverview,
    SnapshotTrace,
)
from sparkle_navigator.domain.repository import SnapshotFile
from sparkle_navigator.domain.semantic import SemanticSearchHit
from sparkle_navigator.persistence.repository_store import RepositoryStore

_MAX_QUERY_SYMBOLS = 10
_MAX_EVIDENCE_ITEMS = 30
_MAX_EVIDENCE_LINES_PER_PATH = 3
_MAX_EXPLANATION_SYMBOLS = 12
_EVIDENCE_SOURCE_PRIORITY: dict[EvidenceSource, int] = {
    "relationship": 0,
    "symbol": 1,
    "semantic": 2,
    "text": 3,
    "overview": 4,
}


class SnapshotExplorerError(RuntimeError):
    """Snapshot 无法验证或浏览。"""


class SnapshotExplorer:
    """通过简洁的 Overview/Ask 接口浏览固定 Snapshot。"""

    def __init__(
        self,
        *,
        database_path: Path,
        semantic_index: SemanticIndex | None = None,
    ) -> None:
        self._store = RepositoryStore(database_path)
        self._text_index = SnapshotTextIndex(database_path=database_path)
        self._symbol_index = SnapshotSymbolIndex(database_path=database_path)
        self._code_graph = SnapshotCodeGraph(database_path=database_path)
        self._semantic_index = semantic_index

    async def overview(self, snapshot_id: str) -> SnapshotOverview:
        """从指定 Snapshot 生成确定性的目录概览和文档证据。"""

        snapshot = await asyncio.to_thread(
            self._store.get_snapshot,
            snapshot_id,
        )
        paths = tuple(item.path for item in snapshot.files)
        document_paths = _overview_document_paths(snapshot.files)
        try:
            scanned_documents = await asyncio.to_thread(
                read_verified_snapshot_files,
                snapshot,
                document_paths,
            )
        except SnapshotFileScanError as exc:
            raise SnapshotExplorerError(str(exc)) from exc
        top_level_paths = tuple(
            sorted({PurePosixPath(path).parts[0] for path in paths})
        )
        test_paths = tuple(
            path
            for path in paths
            if any(
                part.casefold() in {"test", "tests"}
                for part in PurePosixPath(path).parts[:-1]
            )
        )
        entry_point_names = {"__main__.py", "main.py", "app.py", "cli.py"}
        entry_point_candidates = tuple(
            path for path in paths if PurePosixPath(path).name.casefold() in entry_point_names
        )
        dependency_names = {
            "pyproject.toml",
            "requirements.txt",
            "setup.cfg",
            "setup.py",
            "pipfile",
            "poetry.lock",
            "uv.lock",
        }
        dependency_files = tuple(
            path
            for path in paths
            if PurePosixPath(path).name.casefold() in dependency_names
        )
        overview_evidence = _overview_evidence(snapshot_id, scanned_documents)
        evidence_set = EvidenceSet(
            snapshot_id=snapshot_id,
            query="repository overview",
            items=overview_evidence,
        )
        return SnapshotOverview(
            snapshot_id=snapshot_id,
            source_commit=snapshot.source_commit,
            file_count=len(paths),
            python_file_count=sum(path.lower().endswith(".py") for path in paths),
            top_level_paths=top_level_paths,
            test_paths=test_paths,
            entry_point_candidates=entry_point_candidates,
            dependency_files=dependency_files,
            evidence=evidence_set,
        )

    async def explain(self, snapshot_id: str, target: str) -> SnapshotExplanation:
        """为 Snapshot 文件或精确符号名收集源码与静态关系证据。"""

        if not isinstance(target, str) or not target.strip():
            raise ValueError("Explain 目标不能为空。")

        snapshot = await asyncio.to_thread(self._store.get_snapshot, snapshot_id)
        snapshot_paths = {item.path for item in snapshot.files}
        relationships: tuple[CodeRelationship, ...] = ()
        diagnostics: list[str] = []
        symbols: tuple[SymbolSearchHit, ...] = ()
        target_kind: Literal["file", "symbol", "not_found"]
        initial_evidence: CodeEvidence | None = None

        if target in snapshot_paths:
            target_kind = "file"
            file_path = target
            file_hits = ()
        else:
            target_kind = "symbol"
            file_path = ""
            file_hits = await asyncio.to_thread(
                self._symbol_index.search_symbols,
                snapshot_id,
                target,
            )
            if not file_hits:
                return SnapshotExplanation(
                    snapshot_id=snapshot_id,
                    target=target,
                    target_kind="not_found",
                    relationships=(),
                    diagnostics=(),
                    evidence=EvidenceSet(
                        snapshot_id=snapshot_id,
                        query=target,
                        items=(),
                    ),
                )
            file_path = ""

        if target_kind == "file":
            candidate_paths = (file_path,)
        else:
            candidate_paths = tuple(dict.fromkeys(hit.path for hit in file_hits))

        try:
            scanned_files = await asyncio.to_thread(
                read_verified_snapshot_files,
                snapshot,
                candidate_paths,
            )
        except SnapshotFileScanError as exc:
            raise SnapshotExplorerError(str(exc)) from exc
        contents = {
            snapshot_file.path: (snapshot_file.sha256, content)
            for snapshot_file, _size_bytes, content in scanned_files
            if content is not None
        }

        if target_kind == "file":
            file_content = contents.get(file_path)
            if file_content is None:
                diagnostics.append("目标文件不是可解码的 UTF-8 文本。")
            else:
                file_sha256, content = file_content
                symbols_parsed = ()
                if file_path.lower().endswith(".py"):
                    parsed = await asyncio.to_thread(parse_python_file, file_path, content)
                    if parsed.diagnostic is not None:
                        diagnostic = parsed.diagnostic
                        diagnostics.append(
                            f"{diagnostic.path}:{diagnostic.line}: {diagnostic.message}"
                        )
                    else:
                        symbols_parsed = parsed.symbols[:_MAX_EXPLANATION_SYMBOLS]
                if symbols_parsed:
                    symbols = tuple(
                        SymbolSearchHit(
                            path=file_path,
                            name=symbol.name,
                            qualified_name=symbol.qualified_name,
                            kind=symbol.kind,
                            line_start=symbol.line_start,
                            line_end=symbol.line_end,
                        )
                        for symbol in symbols_parsed
                    )
                else:
                    lines = content.splitlines()
                    if lines:
                        evidence_lines = lines[:20]
                        initial_evidence = CodeEvidence(
                            snapshot_id=snapshot_id,
                            path=file_path,
                            line_start=1,
                            line_end=len(evidence_lines),
                            snippet="\n".join(evidence_lines),
                            sha256=file_sha256,
                            source="overview",
                        )
                    else:
                        initial_evidence = None
                if symbols:
                    symbol_relations = await asyncio.to_thread(
                        _trace_symbol_calls,
                        self._code_graph,
                        snapshot_id,
                        symbols,
                    )
                    file_dependencies = await asyncio.to_thread(
                        self._code_graph.trace_dependencies,
                        snapshot_id,
                        file_path,
                        max_depth=1,
                    )
                    relationships = _unique_relationships(
                        (*symbol_relations, *file_dependencies)
                    )
                else:
                    relationships = await asyncio.to_thread(
                        self._code_graph.trace_dependencies,
                        snapshot_id,
                        file_path,
                        max_depth=1,
                    )
        else:
            symbols = tuple(file_hits[:_MAX_EXPLANATION_SYMBOLS])
            relationships = await asyncio.to_thread(
                _trace_symbol_calls,
                self._code_graph,
                snapshot_id,
                symbols,
            )

        evidence_by_location: dict[
            tuple[str, int | None, int | None],
            CodeEvidence,
        ] = {}
        for symbol in symbols:
            file_content = contents.get(symbol.path)
            if file_content is None:
                continue
            file_sha256, content = file_content
            item = _span_evidence(
                snapshot_id,
                symbol.path,
                content.splitlines(),
                file_sha256,
                symbol.line_start,
                symbol.line_end,
                source="symbol",
                symbol_name=symbol.qualified_name,
            )
            if item is not None:
                _add_evidence(evidence_by_location, item)
        for item in _relationship_evidence(snapshot_id, relationships, contents):
            _add_evidence(evidence_by_location, item)
        if target_kind == "file" and not symbols and initial_evidence is not None:
            _add_evidence(evidence_by_location, initial_evidence)

        evidence = tuple(
            sorted(
                evidence_by_location.values(),
                key=lambda item: (
                    item.path,
                    item.line_start or 0,
                    item.line_end or 0,
                ),
            )[:_MAX_EVIDENCE_ITEMS]
        )
        return SnapshotExplanation(
            snapshot_id=snapshot_id,
            target=target,
            target_kind=target_kind,
            relationships=relationships,
            diagnostics=tuple(diagnostics),
            evidence=EvidenceSet(
                snapshot_id=snapshot_id,
                query=target,
                items=evidence,
            ),
        )

    async def trace_call(
        self,
        snapshot_id: str,
        start_path: str,
        start_symbol: str,
        *,
        max_depth: int = 3,
    ) -> SnapshotTrace:
        """追踪静态调用边，并返回逐条经过哈希校验的调用点证据。"""

        snapshot = await asyncio.to_thread(self._store.get_snapshot, snapshot_id)
        relationships = await asyncio.to_thread(
            self._code_graph.trace_calls,
            snapshot_id,
            start_path,
            start_symbol,
            max_depth=max_depth,
        )
        candidate_paths = tuple(
            dict.fromkeys(
                [start_path] + [item.source_path for item in relationships]
            )
        )
        try:
            scanned_files = await asyncio.to_thread(
                read_verified_snapshot_files,
                snapshot,
                candidate_paths,
            )
        except SnapshotFileScanError as exc:
            raise SnapshotExplorerError(str(exc)) from exc
        contents = {
            item.path: (item.sha256, content)
            for item, _size, content in scanned_files
            if content is not None
        }
        evidence = _relationship_evidence(snapshot_id, relationships, contents)
        return SnapshotTrace(
            snapshot_id=snapshot_id,
            start_path=start_path,
            start_symbol=start_symbol,
            max_depth=max_depth,
            relationships=relationships,
            evidence=EvidenceSet(
                snapshot_id=snapshot_id,
                query=f"{start_path}:{start_symbol}",
                items=evidence,
            ),
        )

    async def impact(
        self,
        snapshot_id: str,
        target_path: str,
        *,
        target_symbol: str | None = None,
        max_depth: int = 3,
    ) -> SnapshotImpact:
        """收集目标的反向调用、文件依赖和文本命中的测试候选。"""

        if not target_path.strip():
            raise ValueError("Impact 目标文件路径不能为空。")
        if target_symbol is not None and not target_symbol.strip():
            raise ValueError("Impact 目标符号不能为空。")

        snapshot = await asyncio.to_thread(self._store.get_snapshot, snapshot_id)
        callers = (
            await asyncio.to_thread(
                self._code_graph.trace_callers,
                snapshot_id,
                target_path,
                target_symbol,
                max_depth=max_depth,
            )
            if target_symbol is not None
            else ()
        )
        dependents = await asyncio.to_thread(
            self._code_graph.trace_dependents,
            snapshot_id,
            target_path,
            max_depth=max_depth,
        )
        test_hits = (
            await asyncio.to_thread(
                self._text_index.search_text,
                snapshot_id,
                target_symbol,
                limit=20,
            )
            if target_symbol is not None
            else ()
        )
        test_hits = tuple(hit for hit in test_hits if _is_test_path(hit.path))
        relationships = _unique_relationships((*callers, *dependents))
        candidate_paths = tuple(
            dict.fromkeys(
                [target_path]
                + [item.source_path for item in relationships]
                + [hit.path for hit in test_hits]
            )
        )
        try:
            scanned_files = await asyncio.to_thread(
                read_verified_snapshot_files,
                snapshot,
                candidate_paths,
            )
        except SnapshotFileScanError as exc:
            raise SnapshotExplorerError(str(exc)) from exc
        contents = {
            item.path: (item.sha256, content)
            for item, _size, content in scanned_files
            if content is not None
        }

        evidence_by_location: dict[
            tuple[str, int | None, int | None],
            CodeEvidence,
        ] = {}
        for item in _relationship_evidence(snapshot_id, relationships, contents):
            _add_evidence(evidence_by_location, item)
        if target_symbol is not None:
            terms = _query_terms(target_symbol)
            for hit in test_hits:
                file_content = contents.get(hit.path)
                if file_content is None:
                    continue
                file_sha256, content = file_content
                for line_number, line in _matching_lines(content, terms):
                    _add_evidence(
                        evidence_by_location,
                        CodeEvidence(
                            snapshot_id=snapshot_id,
                            path=hit.path,
                            line_start=line_number,
                            line_end=line_number,
                            snippet=line,
                            sha256=file_sha256,
                            source="text",
                        ),
                    )
        evidence = tuple(
            sorted(
                evidence_by_location.values(),
                key=lambda item: (
                    item.path,
                    item.line_start or 0,
                    item.line_end or 0,
                ),
            )[:_MAX_EVIDENCE_ITEMS]
        )
        return SnapshotImpact(
            snapshot_id=snapshot_id,
            target_path=target_path,
            target_symbol=target_symbol,
            max_depth=max_depth,
            callers=callers,
            dependents=dependents,
            test_candidates=tuple(sorted({hit.path for hit in test_hits})),
            evidence=EvidenceSet(
                snapshot_id=snapshot_id,
                query=f"{target_path}:{target_symbol or ''}",
                items=evidence,
            ),
        )

    async def ask(self, snapshot_id: str, question: str) -> SnapshotAnswer:
        """返回多路检索候选对应的、哈希已验证的 Snapshot 源码证据。"""

        if not isinstance(question, str) or not _query_terms(question):
            raise ValueError("问题至少包含一个可检索词。")

        snapshot = await asyncio.to_thread(
            self._store.get_snapshot,
            snapshot_id,
        )
        text_hits, symbol_hits, relationships, semantic_hits = await asyncio.to_thread(
            self._search_candidates,
            snapshot_id,
            question,
        )
        candidate_paths = tuple(
            dict.fromkeys(
                [hit.path for hit in text_hits]
                + [hit.path for hit in symbol_hits]
                + [item.source_path for item in relationships]
                + [hit.path for hit in semantic_hits]
            )
        )
        try:
            scanned_candidates = await asyncio.to_thread(
                read_verified_snapshot_files,
                snapshot,
                candidate_paths,
            )
        except SnapshotFileScanError as exc:
            raise SnapshotExplorerError(str(exc)) from exc
        contents = {
            snapshot_file.path: (snapshot_file.sha256, content)
            for snapshot_file, _size_bytes, content in scanned_candidates
            if content is not None
        }
        rankings = tuple(
            ranking
            for ranking in (
                tuple(hit.path for hit in text_hits),
                tuple(hit.path for hit in symbol_hits),
                tuple(item.source_path for item in relationships),
                tuple(hit.path for hit in semantic_hits),
            )
            if ranking
        )
        fused = reciprocal_rank_fusion(rankings)
        path_ranks = {item.candidate_id: rank for rank, item in enumerate(fused)}
        evidence = _collect_evidence(
            snapshot_id=snapshot_id,
            question=question,
            contents=contents,
            text_hits=text_hits,
            symbol_hits=symbol_hits,
            relationships=relationships,
            semantic_hits=semantic_hits,
            path_ranks=path_ranks,
        )

        evidence_set = EvidenceSet(
            snapshot_id=snapshot_id,
            query=question,
            items=tuple(evidence),
        )
        answer = _render_answer(question, evidence_set.items)
        return SnapshotAnswer(
            snapshot_id=snapshot_id,
            question=question,
            answer=answer,
            evidence=evidence_set,
        )

    def _search_candidates(
        self,
        snapshot_id: str,
        question: str,
    ) -> tuple[
        tuple[SearchHit, ...],
        tuple[SymbolSearchHit, ...],
        tuple[CodeRelationship, ...],
        tuple[SemanticSearchHit, ...],
    ]:
        text_hits = self._text_index.search_text(snapshot_id, question, limit=10)
        symbol_hits = _search_symbols(self._symbol_index, snapshot_id, question)
        relationships = _trace_symbol_calls(
            self._code_graph,
            snapshot_id,
            symbol_hits,
        )
        semantic_hits = (
            self._semantic_index.search(snapshot_id, question, limit=10)
            if self._semantic_index is not None
            else ()
        )
        return text_hits, symbol_hits, relationships, semantic_hits


def _overview_evidence(
    snapshot_id: str,
    scanned_files: tuple[tuple[SnapshotFile, int, str | None], ...],
) -> tuple[CodeEvidence, ...]:
    file_by_name: dict[str, tuple[SnapshotFile, str]] = {}
    for snapshot_file, _size, content in scanned_files:
        if content is not None:
            file_by_name.setdefault(
                PurePosixPath(snapshot_file.path).name.casefold(),
                (snapshot_file, content),
            )
    selected_names = ("readme.md", "readme.rst", "readme", "pyproject.toml")
    evidence: list[CodeEvidence] = []
    for name in selected_names:
        selected = file_by_name.get(name)
        if selected is None:
            continue
        snapshot_file, content = selected
        lines = content.splitlines()
        if not lines:
            continue
        selected_lines = lines[:8]
        evidence.append(
            CodeEvidence(
                snapshot_id=snapshot_id,
                path=snapshot_file.path,
                line_start=1,
                line_end=len(selected_lines),
                snippet="\n".join(selected_lines),
                sha256=snapshot_file.sha256,
                source="overview",
            )
        )
    return tuple(evidence)


def _overview_document_paths(files: tuple[SnapshotFile, ...]) -> tuple[str, ...]:
    files_by_name: dict[str, SnapshotFile] = {}
    accepted_names = {"readme.md", "readme.rst", "readme", "pyproject.toml"}
    for snapshot_file in files:
        name = PurePosixPath(snapshot_file.path).name.casefold()
        if name in accepted_names:
            files_by_name.setdefault(name, snapshot_file)
    return tuple(
        files_by_name[name].path
        for name in ("readme.md", "readme.rst", "readme", "pyproject.toml")
        if name in files_by_name
    )


def _is_test_path(path: str) -> bool:
    pure_path = PurePosixPath(path)
    return pure_path.name.casefold().startswith("test_") or any(
        part.casefold() in {"test", "tests"}
        for part in pure_path.parts[:-1]
    )


def _search_symbols(
    index: SnapshotSymbolIndex,
    snapshot_id: str,
    question: str,
) -> tuple[SymbolSearchHit, ...]:
    found: dict[tuple[str, str, int], SymbolSearchHit] = {}
    for token in question.split():
        name = token.strip("`'\".,:;()[]{}")
        if not name.isidentifier():
            continue
        for hit in index.search_symbols(snapshot_id, name):
            found.setdefault((hit.path, hit.qualified_name, hit.line_start), hit)
        if len(found) >= _MAX_QUERY_SYMBOLS:
            break
    return tuple(found.values())


def _trace_symbol_calls(
    graph: SnapshotCodeGraph,
    snapshot_id: str,
    symbols: tuple[SymbolSearchHit, ...],
) -> tuple[CodeRelationship, ...]:
    found: dict[tuple[str, str | None, int, str | None], CodeRelationship] = {}
    for symbol in symbols:
        for relationship in graph.trace_calls(
            snapshot_id,
            symbol.path,
            symbol.qualified_name,
            max_depth=1,
        ):
            found.setdefault(
                (
                    relationship.source_path,
                    relationship.source_symbol,
                    relationship.evidence_line,
                    relationship.target_symbol,
                ),
                relationship,
            )
    return tuple(found.values())


def _unique_relationships(
    relationships: tuple[CodeRelationship, ...],
) -> tuple[CodeRelationship, ...]:
    return tuple(dict.fromkeys(relationships))[:_MAX_EVIDENCE_ITEMS]


def _relationship_evidence(
    snapshot_id: str,
    relationships: tuple[CodeRelationship, ...],
    contents: dict[str, tuple[str, str]],
) -> tuple[CodeEvidence, ...]:
    evidence: list[CodeEvidence] = []
    for relationship in relationships:
        file_content = contents.get(relationship.source_path)
        if file_content is None:
            continue
        file_sha256, content = file_content
        item = _span_evidence(
            snapshot_id,
            relationship.source_path,
            content.splitlines(),
            file_sha256,
            relationship.evidence_line,
            relationship.evidence_line,
            source="relationship",
            confidence=relationship.confidence,
            symbol_name=relationship.source_symbol,
            relationship_type=relationship.relation_type,
        )
        if item is not None:
            evidence.append(item)
    return tuple(evidence)


def _collect_evidence(
    *,
    snapshot_id: str,
    question: str,
    contents: dict[str, tuple[str, str]],
    text_hits: tuple[SearchHit, ...],
    symbol_hits: tuple[SymbolSearchHit, ...],
    relationships: tuple[CodeRelationship, ...],
    semantic_hits: tuple[SemanticSearchHit, ...],
    path_ranks: dict[str, int],
) -> tuple[CodeEvidence, ...]:
    terms = _query_terms(question)
    evidence_by_location: dict[
        tuple[str, int | None, int | None],
        CodeEvidence,
    ] = {}
    line_counts: dict[str, int] = {}

    for hit in text_hits:
        if line_counts.get(hit.path, 0) >= _MAX_EVIDENCE_LINES_PER_PATH:
            continue
        file_content = contents.get(hit.path)
        if file_content is None:
            continue
        file_sha256, content = file_content
        for line_number, line in _matching_lines(content, terms):
            _add_evidence(
                evidence_by_location,
                CodeEvidence(
                    snapshot_id=snapshot_id,
                    path=hit.path,
                    line_start=line_number,
                    line_end=line_number,
                    snippet=line,
                    sha256=file_sha256,
                    source="text",
                ),
            )
            line_counts[hit.path] = line_counts.get(hit.path, 0) + 1
            if line_counts[hit.path] >= _MAX_EVIDENCE_LINES_PER_PATH:
                break

    for hit in symbol_hits:
        file_content = contents.get(hit.path)
        if file_content is None:
            continue
        file_sha256, content = file_content
        source_lines = content.splitlines()
        item = _span_evidence(
            snapshot_id,
            hit.path,
            source_lines,
            file_sha256,
            hit.line_start,
            hit.line_end,
            source="symbol",
            symbol_name=hit.qualified_name,
        )
        if item is not None:
            _add_evidence(evidence_by_location, item)

    for relationship in relationships:
        file_content = contents.get(relationship.source_path)
        if file_content is None:
            continue
        file_sha256, content = file_content
        item = _span_evidence(
            snapshot_id,
            relationship.source_path,
            content.splitlines(),
            file_sha256,
            relationship.evidence_line,
            relationship.evidence_line,
            source="relationship",
            confidence=relationship.confidence,
            symbol_name=relationship.source_symbol,
            relationship_type=relationship.relation_type,
        )
        if item is not None:
            _add_evidence(evidence_by_location, item)

    for hit in semantic_hits:
        file_content = contents.get(hit.path)
        if file_content is None:
            continue
        file_sha256, content = file_content
        source_lines = content.splitlines()
        if hit.line_start is not None and hit.line_end is not None:
            item = _span_evidence(
                snapshot_id,
                hit.path,
                source_lines,
                file_sha256,
                hit.line_start,
                hit.line_end,
                source="semantic",
                symbol_name=hit.symbol_name,
            )
            if item is not None:
                _add_evidence(evidence_by_location, item)
        else:
            for line_number, line in _matching_lines(content, terms):
                _add_evidence(
                    evidence_by_location,
                    CodeEvidence(
                        snapshot_id=snapshot_id,
                        path=hit.path,
                        line_start=line_number,
                        line_end=line_number,
                        snippet=line,
                        sha256=file_sha256,
                        source="semantic",
                    ),
                )

    source_order = {"text": 0, "symbol": 1, "relationship": 2, "semantic": 3}
    return tuple(
        sorted(
            evidence_by_location.values(),
            key=lambda item: (
                path_ranks.get(item.path, len(path_ranks)),
                item.line_start if item.line_start is not None else 0,
                source_order.get(item.source, 4),
                item.path,
            ),
        )[:_MAX_EVIDENCE_ITEMS]
    )


def _matching_lines(
    content: str,
    terms: tuple[str, ...],
) -> tuple[tuple[int, str], ...]:
    if not terms:
        return ()
    return tuple(
        (line_number, line)
        for line_number, line in enumerate(content.splitlines(), start=1)
        if any(term in line.casefold() for term in terms)
    )


def _query_terms(question: str) -> tuple[str, ...]:
    return tuple(term.casefold() for term in re.findall(r"\w+", question))


def _span_evidence(
    snapshot_id: str,
    path: str,
    source_lines: list[str],
    file_sha256: str,
    line_start: int,
    line_end: int,
    *,
    source: EvidenceSource,
    confidence: RelationshipConfidence = "CONFIRMED",
    symbol_name: str | None = None,
    relationship_type: RelationshipType | None = None,
) -> CodeEvidence | None:
    if line_start < 1 or line_end < line_start or line_end > len(source_lines):
        return None
    return CodeEvidence(
        snapshot_id=snapshot_id,
        path=path,
        line_start=line_start,
        line_end=line_end,
        snippet="\n".join(source_lines[line_start - 1 : line_end]),
        sha256=file_sha256,
        source=source,
        confidence=confidence,
        symbol_name=symbol_name,
        relationship_type=relationship_type,
    )


def _add_evidence(
    evidence: dict[
        tuple[str, int | None, int | None],
        CodeEvidence,
    ],
    item: CodeEvidence,
) -> None:
    key = (item.path, item.line_start, item.line_end)
    existing = evidence.get(key)
    if (
        existing is None
        or _EVIDENCE_SOURCE_PRIORITY[item.source]
        < _EVIDENCE_SOURCE_PRIORITY[existing.source]
    ):
        evidence[key] = item


def _render_answer(question: str, evidence: tuple[CodeEvidence, ...]) -> str:
    if not evidence:
        return "当前 Snapshot 中没有找到可引用的代码证据。"
    citations = "\n".join(
        f"- {item.path}:{_line_label(item)}: {item.snippet}"
        for item in evidence
    )
    return f"问题：{question}\n以下是当前 Snapshot 中的直接源码证据：\n{citations}"


def _line_label(item: CodeEvidence) -> str:
    if item.line_end is not None and item.line_end != item.line_start:
        return f"{item.line_start}-{item.line_end}"
    return str(item.line_start)


__all__ = ["SnapshotExplorer", "SnapshotExplorerError"]
