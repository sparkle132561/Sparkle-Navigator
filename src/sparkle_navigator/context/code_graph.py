"""在固定 Repository Snapshot 上构建并遍历 Python 代码关系。"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    scan_verified_snapshot_files,
)
from sparkle_navigator.context.python_symbols import (
    ParseDiagnostic,
    PythonFileParseResult,
    parse_python_file,
)
from sparkle_navigator.domain.code_index import (
    CallSite,
    CodeRelationship,
    CodeSymbol,
    ImportRecord,
    RelationshipConfidence,
    RelationshipType,
)
from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.persistence.repository_store import RepositoryStore


class SnapshotCodeGraphError(RuntimeError):
    """Snapshot 校验失败或代码图构建失败。"""


@dataclass(frozen=True, slots=True)
class CodeGraphBuildResult:
    """一次 Snapshot 代码关系构建的计数和文件级诊断。"""

    snapshot_id: str
    python_file_count: int
    relationship_count: int
    definition_count: int
    import_count: int
    call_count: int
    dependency_count: int
    unresolved_call_count: int
    diagnostics: tuple[ParseDiagnostic, ...]


class SnapshotCodeGraph:
    """构建带置信度和源码行证据的关系，并追踪调用或本地依赖。"""

    def __init__(self, *, database_path: Path) -> None:
        self._store = RepositoryStore(database_path)

    async def build(self, snapshot: RepositorySnapshot) -> CodeGraphBuildResult:
        """验证 Snapshot，解析 Python 文件并原子替换其关系索引。"""

        try:
            scanned_files = await asyncio.to_thread(scan_verified_snapshot_files, snapshot)
        except SnapshotFileScanError as exc:
            raise SnapshotCodeGraphError(str(exc)) from exc

        build_data = await asyncio.to_thread(_build_relationships, scanned_files)
        relationships, python_file_count, unresolved_call_count, diagnostics = build_data
        self._store.replace_snapshot_relationships(snapshot.snapshot_id, relationships)
        counts = Counter(relationship.relation_type for relationship in relationships)
        return CodeGraphBuildResult(
            snapshot_id=snapshot.snapshot_id,
            python_file_count=python_file_count,
            relationship_count=len(relationships),
            definition_count=counts["DEFINES"],
            import_count=counts["IMPORTS"],
            call_count=counts["CALLS"],
            dependency_count=counts["DEPENDS_ON"],
            unresolved_call_count=unresolved_call_count,
            diagnostics=diagnostics,
        )

    def trace_calls(
        self,
        snapshot_id: str,
        start_path: str,
        start_qualified_name: str,
        *,
        max_depth: int = 3,
    ) -> tuple[CodeRelationship, ...]:
        """从指定函数或方法沿 CALLS 关系进行广度优先追踪。"""

        _validate_max_depth(max_depth)
        if not start_qualified_name.strip():
            raise ValueError("起始符号名称不能为空。")
        return self._trace(
            snapshot_id,
            relation_type="CALLS",
            start_path=start_path,
            start_symbol=start_qualified_name,
            max_depth=max_depth,
        )

    def trace_dependencies(
        self,
        snapshot_id: str,
        start_path: str,
        *,
        max_depth: int = 3,
    ) -> tuple[CodeRelationship, ...]:
        """从指定 Python 文件沿本地 DEPENDS_ON 关系追踪依赖链。"""

        _validate_max_depth(max_depth)
        return self._trace(
            snapshot_id,
            relation_type="DEPENDS_ON",
            start_path=start_path,
            start_symbol=None,
            max_depth=max_depth,
        )

    def trace_callers(
        self,
        snapshot_id: str,
        target_path: str,
        target_qualified_name: str,
        *,
        max_depth: int = 3,
    ) -> tuple[CodeRelationship, ...]:
        """从指定函数或方法沿 CALLS 入边反向追踪调用者。"""

        _validate_max_depth(max_depth)
        if not target_path.strip() or not target_qualified_name.strip():
            raise ValueError("目标文件路径和符号名称不能为空。")
        return self._trace_reverse(
            snapshot_id,
            relation_type="CALLS",
            target_path=target_path,
            target_symbol=target_qualified_name,
            max_depth=max_depth,
        )

    def trace_dependents(
        self,
        snapshot_id: str,
        target_path: str,
        *,
        max_depth: int = 3,
    ) -> tuple[CodeRelationship, ...]:
        """从指定文件沿 DEPENDS_ON 入边反向追踪依赖它的文件。"""

        _validate_max_depth(max_depth)
        if not target_path.strip():
            raise ValueError("目标文件路径不能为空。")
        return self._trace_reverse(
            snapshot_id,
            relation_type="DEPENDS_ON",
            target_path=target_path,
            target_symbol=None,
            max_depth=max_depth,
        )

    def _trace(
        self,
        snapshot_id: str,
        *,
        relation_type: RelationshipType,
        start_path: str,
        start_symbol: str | None,
        max_depth: int,
    ) -> tuple[CodeRelationship, ...]:
        frontier = [(start_path, start_symbol)]
        visited = {(start_path, start_symbol)}
        traced: list[CodeRelationship] = []

        for _depth in range(max_depth):
            if not frontier:
                break
            outgoing = self._store.get_snapshot_relationships_from_many(
                snapshot_id,
                relation_type,
                frontier,
            )
            next_frontier: list[tuple[str, str | None]] = []
            for relationship in outgoing:
                traced.append(relationship)
                if relationship.target_path is None:
                    continue
                target_symbol = (
                    relationship.target_symbol if relation_type == "CALLS" else None
                )
                target = (relationship.target_path, target_symbol)
                if target not in visited:
                    visited.add(target)
                    next_frontier.append(target)
            frontier = next_frontier
        return tuple(traced)

    def _trace_reverse(
        self,
        snapshot_id: str,
        *,
        relation_type: RelationshipType,
        target_path: str,
        target_symbol: str | None,
        max_depth: int,
    ) -> tuple[CodeRelationship, ...]:
        frontier = [(target_path, target_symbol)]
        visited = set(frontier)
        traced: list[CodeRelationship] = []

        for _depth in range(max_depth):
            if not frontier:
                break
            incoming = self._store.get_snapshot_relationships_to_many(
                snapshot_id,
                relation_type,
                frontier,
            )
            next_frontier: list[tuple[str, str | None]] = []
            for relationship in incoming:
                traced.append(relationship)
                caller_symbol = (
                    relationship.source_symbol if relation_type == "CALLS" else None
                )
                caller = (relationship.source_path, caller_symbol)
                if caller not in visited:
                    visited.add(caller)
                    next_frontier.append(caller)
            frontier = next_frontier
        return tuple(traced)


def _build_relationships(
    scanned_files: tuple[tuple[SnapshotFile, int, str | None], ...],
) -> tuple[
    tuple[CodeRelationship, ...],
    int,
    int,
    tuple[ParseDiagnostic, ...],
]:
    module_paths = _module_paths(scanned_files)
    parsed_files: dict[str, PythonFileParseResult] = {}
    diagnostics: list[ParseDiagnostic] = []
    python_file_count = 0

    for snapshot_file, _size_bytes, content in scanned_files:
        path = snapshot_file.path
        if not path.lower().endswith(".py") or content is None:
            continue
        python_file_count += 1
        result = parse_python_file(path, content)
        if result.diagnostic is not None:
            diagnostics.append(result.diagnostic)
            continue
        parsed_files[path] = result

    symbols_by_path = _symbols_by_path(parsed_files)
    relationships: list[CodeRelationship] = []
    for path, parsed_file in parsed_files.items():
        relationships.extend(
            CodeRelationship(
                relation_type="DEFINES",
                source_path=path,
                source_symbol=None,
                target_path=path,
                target_symbol=symbol.qualified_name,
                confidence="CONFIRMED",
                evidence_line=symbol.line_start,
            )
            for symbol in parsed_file.symbols
        )
        for import_record in parsed_file.imports:
            resolved_paths = _resolve_import_paths(import_record, path, module_paths)
            target_paths: tuple[str | None, ...] = tuple(resolved_paths) or (None,)
            target_name = _import_target_name(import_record, path)
            for target_path in target_paths:
                relationships.append(
                    CodeRelationship(
                        relation_type="IMPORTS",
                        source_path=path,
                        source_symbol=None,
                        target_path=target_path,
                        target_symbol=target_name,
                        confidence="CONFIRMED",
                        evidence_line=import_record.line,
                    )
                )
            for target_path in resolved_paths:
                if target_path != path:
                    relationships.append(
                        CodeRelationship(
                            relation_type="DEPENDS_ON",
                            source_path=path,
                            source_symbol=None,
                            target_path=target_path,
                            target_symbol=None,
                            confidence="CONFIRMED",
                            evidence_line=import_record.line,
                        )
                    )

    unresolved_call_count = 0
    for path, parsed_file in parsed_files.items():
        for call_site in parsed_file.call_sites:
            resolved = _resolve_call(
                path,
                call_site,
                parsed_file.imports,
                symbols_by_path,
                module_paths,
            )
            if resolved is None:
                unresolved_call_count += 1
                continue
            target_path, target_symbol, confidence = resolved
            relationships.append(
                CodeRelationship(
                    relation_type="CALLS",
                    source_path=path,
                    source_symbol=call_site.caller_qualified_name,
                    target_path=target_path,
                    target_symbol=target_symbol,
                    confidence=confidence,
                    evidence_line=call_site.line,
                )
            )

    return tuple(relationships), python_file_count, unresolved_call_count, tuple(diagnostics)


def _module_paths(
    scanned_files: tuple[tuple[SnapshotFile, int, str | None], ...],
) -> dict[str, tuple[str, ...]]:
    paths: dict[str, list[str]] = {}
    for snapshot_file, _size_bytes, content in scanned_files:
        if content is None or not snapshot_file.path.lower().endswith(".py"):
            continue
        for module_name in _module_names(snapshot_file.path):
            paths.setdefault(module_name, []).append(snapshot_file.path)
    return {name: tuple(sorted(set(items))) for name, items in paths.items()}


def _module_names(path: str) -> tuple[str, ...]:
    pure_path = PurePosixPath(path)
    parts = list(pure_path.parts)
    if pure_path.name == "__init__.py":
        parts.pop()
    else:
        parts[-1] = pure_path.stem
    names = [".".join(parts)] if parts else []
    if parts and parts[0] == "src":
        names.append(".".join(parts[1:]))
    return tuple(name for name in names if name)


def _resolve_import_paths(
    record: ImportRecord,
    source_path: str,
    module_paths: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    module_name = _resolved_import_module(record, source_path)
    candidates = [module_name] if module_name else []
    if record.imported_name and record.imported_name != "*" and module_name:
        candidates.append(f"{module_name}.{record.imported_name}")
    resolved: list[str] = []
    for candidate in candidates:
        matches = module_paths.get(candidate, ())
        if len(matches) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])
    return tuple(resolved)


def _resolved_import_module(record: ImportRecord, source_path: str) -> str:
    module_parts = record.module.split(".") if record.module else []
    if record.relative_level == 0:
        return ".".join(module_parts)
    package_parts = list(PurePosixPath(source_path).parts[:-1])
    levels_to_remove = record.relative_level - 1
    if levels_to_remove > len(package_parts):
        return ""
    package_parts = package_parts[: len(package_parts) - levels_to_remove]
    return ".".join((*package_parts, *module_parts))


def _import_target_name(record: ImportRecord, source_path: str) -> str | None:
    module_name = _resolved_import_module(record, source_path)
    if record.imported_name is None:
        return module_name or None
    return f"{module_name}.{record.imported_name}" if module_name else record.imported_name


def _symbols_by_path(
    parsed_files: dict[str, PythonFileParseResult],
) -> dict[str, dict[str, list[CodeSymbol]]]:
    symbols: dict[str, dict[str, list[CodeSymbol]]] = {}
    for path, parsed_file in parsed_files.items():
        by_name: dict[str, list[CodeSymbol]] = {}
        for symbol in parsed_file.symbols:
            by_name.setdefault(symbol.qualified_name, []).append(symbol)
        symbols[path] = by_name
    return symbols


def _resolve_call(
    source_path: str,
    call_site: CallSite,
    imports: Sequence[ImportRecord],
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
    module_paths: dict[str, tuple[str, ...]],
) -> tuple[str, str, RelationshipConfidence] | None:
    if call_site.callee_kind == "name":
        imported_target = _resolve_imported_name(
            source_path,
            call_site.callee,
            imports,
            symbols_by_path,
            module_paths,
        )
        if imported_target is not None:
            return imported_target[0], imported_target[1], "CONFIRMED"
        local_target = _resolve_local_name(
            source_path,
            call_site.caller_qualified_name,
            call_site.callee,
            symbols_by_path,
        )
        if local_target is not None:
            return source_path, local_target, "LIKELY"
        return None
    if call_site.callee_kind != "attribute":
        return None

    self_target = _resolve_self_attribute(
        source_path,
        call_site,
        symbols_by_path,
    )
    if self_target is not None:
        return source_path, self_target, "LIKELY"
    imported_module_target = _resolve_imported_module_attribute(
        source_path,
        call_site.callee,
        imports,
        symbols_by_path,
        module_paths,
    )
    if imported_module_target is not None:
        return imported_module_target[0], imported_module_target[1], "CONFIRMED"
    local_attribute = _unique_symbol(source_path, call_site.callee, symbols_by_path)
    if local_attribute is not None:
        return source_path, local_attribute.qualified_name, "LIKELY"
    return None


def _resolve_imported_name(
    source_path: str,
    name: str,
    imports: Sequence[ImportRecord],
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
    module_paths: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    for record in imports:
        if record.imported_name in {None, "*"}:
            continue
        if (record.alias or record.imported_name) != name:
            continue
        paths = _resolve_import_paths(record, source_path, module_paths)
        if len(paths) != 1:
            continue
        target = _unique_symbol(paths[0], record.imported_name, symbols_by_path)
        if target is not None:
            return paths[0], target.qualified_name
    return None


def _resolve_local_name(
    source_path: str,
    caller: str | None,
    name: str,
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
) -> str | None:
    candidates: list[str] = []
    if caller:
        parts = caller.split(".")
        for end in range(len(parts), 0, -1):
            scope = ".".join(parts[:end])
            scope_symbol = _unique_symbol(source_path, scope, symbols_by_path)
            if scope_symbol is not None and scope_symbol.kind in {"function", "method"}:
                candidates.append(f"{scope}.{name}")
    candidates.append(name)
    for candidate in dict.fromkeys(candidates):
        target = _unique_symbol(source_path, candidate, symbols_by_path)
        if target is not None:
            return target.qualified_name
    return None


def _resolve_self_attribute(
    source_path: str,
    call_site: CallSite,
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
) -> str | None:
    if call_site.caller_qualified_name is None:
        return None
    parts = call_site.callee.split(".")
    if len(parts) != 2 or parts[0] not in {"self", "cls"}:
        return None
    caller_parts = call_site.caller_qualified_name.split(".")
    for end in range(len(caller_parts), 0, -1):
        candidate = ".".join(caller_parts[:end])
        symbol = _unique_symbol(source_path, candidate, symbols_by_path)
        if symbol is not None and symbol.kind == "class":
            target_name = f"{candidate}.{parts[1]}"
            target = _unique_symbol(source_path, target_name, symbols_by_path)
            if target is not None:
                return target.qualified_name
    return None


def _resolve_imported_module_attribute(
    source_path: str,
    callee: str,
    imports: Sequence[ImportRecord],
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
    module_paths: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    for record in imports:
        if record.imported_name is not None:
            continue
        module_name = _resolved_import_module(record, source_path)
        if not module_name:
            continue
        if record.alias:
            prefix = record.alias
            if not callee.startswith(f"{prefix}."):
                continue
            target_name = callee[len(prefix) + 1 :]
        else:
            prefix = module_name
            if not callee.startswith(f"{prefix}."):
                continue
            target_name = callee[len(prefix) + 1 :]
        paths = module_paths.get(module_name, ())
        if len(paths) != 1:
            continue
        target = _unique_symbol(paths[0], target_name, symbols_by_path)
        if target is not None:
            return paths[0], target.qualified_name
    return None


def _unique_symbol(
    path: str,
    qualified_name: str,
    symbols_by_path: dict[str, dict[str, list[CodeSymbol]]],
) -> CodeSymbol | None:
    matches = symbols_by_path.get(path, {}).get(qualified_name, ())
    return matches[0] if len(matches) == 1 else None


def _validate_max_depth(max_depth: int) -> None:
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth 必须是正整数。")


__all__ = [
    "CodeGraphBuildResult",
    "SnapshotCodeGraph",
    "SnapshotCodeGraphError",
]
