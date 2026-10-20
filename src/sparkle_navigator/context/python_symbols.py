"""从 Python 源码提取声明符号与导入记录。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from sparkle_navigator.domain.code_index import (
    CallSite,
    CallSiteKind,
    CodeSymbol,
    ImportRecord,
    SymbolKind,
)


@dataclass(frozen=True, slots=True)
class ParseDiagnostic:
    """一个 Python 文件的语法解析诊断。"""

    path: str
    line: int | None
    column: int | None
    message: str


@dataclass(frozen=True, slots=True)
class PythonFileParseResult:
    """一次纯 Python 源码解析的记录集合或文件级诊断。"""

    path: str
    symbols: tuple[CodeSymbol, ...]
    imports: tuple[ImportRecord, ...]
    call_sites: tuple[CallSite, ...]
    diagnostic: ParseDiagnostic | None


class _PythonRecordVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[CodeSymbol] = []
        self.imports: list[ImportRecord] = []
        self.call_sites: list[CallSite] = []
        self._scope_names: list[str] = []
        self._scope_kinds: list[Literal["class", "function"]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._append_symbol(node, "class")
        self._visit_in_scope(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        kind: SymbolKind = (
            "method" if self._scope_kinds and self._scope_kinds[-1] == "class" else "function"
        )
        self._append_symbol(node, kind)
        self._visit_in_scope(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(
                    module=alias.name,
                    imported_name=None,
                    alias=alias.asname,
                    relative_level=0,
                    line=getattr(alias, "lineno", node.lineno),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(
                    module=node.module,
                    imported_name=alias.name,
                    alias=alias.asname,
                    relative_level=node.level,
                    line=getattr(alias, "lineno", node.lineno),
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        callee = _dotted_name(node.func)
        caller = (
            ".".join(self._scope_names)
            if self._scope_kinds and self._scope_kinds[-1] == "function"
            else None
        )
        if callee is None:
            callee = ast.unparse(node.func)
            callee_kind: CallSiteKind = "dynamic"
        else:
            callee_kind = "name" if isinstance(node.func, ast.Name) else "attribute"
        self.call_sites.append(
            CallSite(
                caller_qualified_name=caller,
                callee=callee,
                callee_kind=callee_kind,
                line=node.lineno,
            )
        )
        self.generic_visit(node)

    def _append_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
    ) -> None:
        qualified_parts = (*self._scope_names, node.name)
        self.symbols.append(
            CodeSymbol(
                name=node.name,
                qualified_name=".".join(qualified_parts),
                kind=kind,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            )
        )

    def _visit_in_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: Literal["class", "function"],
    ) -> None:
        self._scope_names.append(node.name)
        self._scope_kinds.append(kind)
        for statement in node.body:
            self.visit(statement)
        self._scope_kinds.pop()
        self._scope_names.pop()


def parse_python_file(path: str, source: str) -> PythonFileParseResult:
    """解析源码文本；不导入、不编译也不执行目标代码。"""

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return PythonFileParseResult(
            path=path,
            symbols=(),
            imports=(),
            call_sites=(),
            diagnostic=ParseDiagnostic(
                path=path,
                line=exc.lineno,
                column=exc.offset,
                message=exc.msg,
            ),
        )

    visitor = _PythonRecordVisitor()
    visitor.visit(tree)
    return PythonFileParseResult(
        path=path,
        symbols=tuple(visitor.symbols),
        imports=tuple(visitor.imports),
        call_sites=tuple(visitor.call_sites),
        diagnostic=None,
    )


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


__all__ = [
    "CodeSymbol",
    "CallSite",
    "CallSiteKind",
    "ImportRecord",
    "ParseDiagnostic",
    "PythonFileParseResult",
    "SymbolKind",
    "parse_python_file",
]
