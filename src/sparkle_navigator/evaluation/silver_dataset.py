"""从已固定的 Repository Snapshot 确定性生成 Silver 案例。"""

from __future__ import annotations

from hashlib import sha256
from typing import TypeAlias

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    read_verified_snapshot_files,
)
from sparkle_navigator.context.python_symbols import parse_python_file
from sparkle_navigator.domain.code_index import CallSite, CodeSymbol, ImportRecord
from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile

SilverCase: TypeAlias = dict[str, object]

_GENERATOR_ID = "sparkle-ast-silver"
_GENERATOR_VERSION = "0.1.0"
_QUESTION_TYPES = ("symbol_location", "import_statement", "call_site")


class SilverDatasetGenerationError(RuntimeError):
    """Snapshot 内容无法安全验证或生成 Silver 案例。"""


def generate_silver_cases(
    repository_id: str,
    snapshot: RepositorySnapshot,
    limit: int = 60,
) -> tuple[SilverCase, ...]:
    """从 Snapshot 中确定性提取符号位置题，并验证对应源码证据。"""

    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ValueError("repository_id 必须是非空字符串。")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit 必须是正整数。")

    python_files = tuple(
        sorted(
            (item for item in snapshot.files if item.path.lower().endswith(".py")),
            key=lambda item: item.path,
        )
    )
    try:
        verified_files = read_verified_snapshot_files(
            snapshot,
            (item.path for item in python_files),
        )
    except SnapshotFileScanError as exc:
        raise SilverDatasetGenerationError(
            "无法从未通过 Snapshot 校验的源码生成 Silver 案例。"
        ) from exc

    candidates: dict[str, list[SilverCase]] = {
        question_type: [] for question_type in _QUESTION_TYPES
    }
    for snapshot_file, _size_bytes, source in verified_files:
        if source is None:
            continue
        parsed = parse_python_file(snapshot_file.path, source)
        if parsed.diagnostic is not None:
            continue
        source_lines = source.splitlines()
        for ordinal, symbol in enumerate(parsed.symbols):
            if len(candidates["symbol_location"]) < limit:
                candidates["symbol_location"].append(
                    _symbol_case(
                        repository_id=repository_id,
                        snapshot=snapshot,
                        snapshot_file=snapshot_file,
                        source_lines=source_lines,
                        symbol=symbol,
                        ordinal=ordinal,
                    )
                )
        for ordinal, record in enumerate(parsed.imports):
            if len(candidates["import_statement"]) < limit:
                candidates["import_statement"].append(
                    _import_case(
                        repository_id=repository_id,
                        snapshot=snapshot,
                        snapshot_file=snapshot_file,
                        source_lines=source_lines,
                        record=record,
                        ordinal=ordinal,
                    )
                )
        for ordinal, record in enumerate(parsed.call_sites):
            if len(candidates["call_site"]) < limit:
                candidates["call_site"].append(
                    _call_case(
                        repository_id=repository_id,
                        snapshot=snapshot,
                        snapshot_file=snapshot_file,
                        source_lines=source_lines,
                        record=record,
                        ordinal=ordinal,
                    )
                )

    cases: list[SilverCase] = []
    position = 0
    while len(cases) < limit:
        added_candidate = False
        for question_type in _QUESTION_TYPES:
            if position < len(candidates[question_type]):
                cases.append(candidates[question_type][position])
                added_candidate = True
                if len(cases) == limit:
                    break
        if not added_candidate:
            break
        position += 1
    return tuple(cases)


def _symbol_case(
    *,
    repository_id: str,
    snapshot: RepositorySnapshot,
    snapshot_file: SnapshotFile,
    source_lines: list[str],
    symbol: CodeSymbol,
    ordinal: int,
) -> SilverCase:
    deterministic_input = (
        f"symbol:{snapshot_file.path}:{symbol.qualified_name}:"
        f"{symbol.line_start}:{symbol.line_end}:{ordinal}"
    )
    return _fact_case(
        repository_id=repository_id,
        snapshot=snapshot,
        snapshot_file=snapshot_file,
        source_lines=source_lines,
        question_type="symbol_location",
        deterministic_input=deterministic_input,
        line_start=symbol.line_start,
        line_end=symbol.line_end,
        symbols=(symbol.qualified_name,),
        user_input=(
            f"在固定代码快照中，符号 `{symbol.qualified_name}` "
            "的声明类型和源码行范围是什么？"
        ),
        answer_point=(
            f"`{symbol.qualified_name}` 是 {symbol.kind}，定义于 "
            f"`{snapshot_file.path}` 第 {symbol.line_start}-{symbol.line_end} 行。"
        ),
        rubric_points=(
            symbol.kind,
            snapshot_file.path,
            f"{symbol.line_start}-{symbol.line_end}",
        ),
    )


def _import_case(
    *,
    repository_id: str,
    snapshot: RepositorySnapshot,
    snapshot_file: SnapshotFile,
    source_lines: list[str],
    record: ImportRecord,
    ordinal: int,
) -> SilverCase:
    statement = _import_statement(record)
    imported_symbols = tuple(
        item
        for item in (record.module, record.imported_name, record.alias)
        if item
    ) or ("import",)
    deterministic_input = (
        f"import:{snapshot_file.path}:{record.line}:{statement}:{ordinal}"
    )
    return _fact_case(
        repository_id=repository_id,
        snapshot=snapshot,
        snapshot_file=snapshot_file,
        source_lines=source_lines,
        question_type="import_statement",
        deterministic_input=deterministic_input,
        line_start=record.line,
        line_end=record.line,
        symbols=imported_symbols,
        user_input=(
            f"固定代码快照中的 `{snapshot_file.path}` 第 {record.line} 行，"
            "Python import 语句导入了什么名称？"
        ),
        answer_point=f"该行的导入语句是 `{statement}`。",
        rubric_points=(statement,),
    )


def _call_case(
    *,
    repository_id: str,
    snapshot: RepositorySnapshot,
    snapshot_file: SnapshotFile,
    source_lines: list[str],
    record: CallSite,
    ordinal: int,
) -> SilverCase:
    caller = record.caller_qualified_name
    context = f"函数 `{caller}`" if caller is not None else "模块级代码"
    deterministic_input = (
        f"call:{snapshot_file.path}:{record.line}:{caller or '<module>'}:"
        f"{record.callee}:{record.callee_kind}:{ordinal}"
    )
    symbols = ((caller,) if caller is not None else ()) + (record.callee,)
    return _fact_case(
        repository_id=repository_id,
        snapshot=snapshot,
        snapshot_file=snapshot_file,
        source_lines=source_lines,
        question_type="call_site",
        deterministic_input=deterministic_input,
        line_start=record.line,
        line_end=record.line,
        symbols=symbols,
        user_input=(
            f"固定代码快照中的 `{snapshot_file.path}` 第 {record.line} 行，"
            f"{context} 写出了哪个调用表达式？"
        ),
        answer_point=(
            f"该行的词法调用表达式为 `{record.callee}`，"
            f"AST 类型是 {record.callee_kind}；这表示源码形式，不保证运行时目标已解析。"
        ),
        rubric_points=(record.callee, record.callee_kind),
    )


def _fact_case(
    *,
    repository_id: str,
    snapshot: RepositorySnapshot,
    snapshot_file: SnapshotFile,
    source_lines: list[str],
    question_type: str,
    deterministic_input: str,
    line_start: int,
    line_end: int,
    symbols: tuple[str, ...],
    user_input: str,
    answer_point: str,
    rubric_points: tuple[str, ...],
) -> SilverCase:
    if line_start < 1 or line_end < line_start or line_end > len(source_lines):
        raise SilverDatasetGenerationError(
            f"AST 事实行范围超出 Snapshot 文件：{snapshot_file.path}"
        )
    snippet = "\n".join(source_lines[line_start - 1 : line_end])
    if not snippet.strip():
        raise SilverDatasetGenerationError(
            f"AST 事实没有对应的源码证据：{snapshot_file.path}"
        )
    case_digest = sha256(
        f"{repository_id}:{snapshot.snapshot_id}:{deterministic_input}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return {
        "case_id": f"silver-{case_digest}",
        "repository_id": repository_id,
        "suite_type": "code-structure",
        "question_type": question_type,
        "difficulty": "easy",
        "user_input": user_input,
        "ground_truth": {
            "answer_points": [answer_point],
            "expected_evidence": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "path": snapshot_file.path,
                    "sha256": snapshot_file.sha256,
                    "line_start": line_start,
                    "line_end": line_end,
                    "symbols": list(symbols),
                    "snippet": snippet,
                }
            ],
            "rubric": {"must_mention": list(rubric_points)},
        },
        "provenance": {
            "verification_status": "auto_verified",
            "generator": _GENERATOR_ID,
            "generator_version": _GENERATOR_VERSION,
            "source_commit": snapshot.source_commit,
            "snapshot_id": snapshot.snapshot_id,
            "deterministic_input": deterministic_input,
            "evidence_validation": "snapshot_sha256_and_ast_line_range",
        },
    }


def _import_statement(record: ImportRecord) -> str:
    if record.imported_name is None and record.relative_level == 0:
        statement = f"import {record.module or ''}"
    else:
        module = "." * record.relative_level + (record.module or "")
        statement = f"from {module} import {record.imported_name or ''}"
    if record.alias is not None:
        statement += f" as {record.alias}"
    return statement.strip()


__all__ = [
    "SilverCase",
    "SilverDatasetGenerationError",
    "generate_silver_cases",
]
