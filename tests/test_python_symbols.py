"""通过 Python 文件解析接口验证代码符号和导入记录。"""

from __future__ import annotations

from sparkle_navigator.context.python_symbols import (
    CallSite,
    CodeSymbol,
    ImportRecord,
    parse_python_file,
)


def test_parse_python_file_extracts_declarations_and_import_aliases() -> None:
    source = """from pathlib import Path as PathAlias

class SnapshotReader:
    def read(self, path):
        return PathAlias(path)

async def refresh_reader(reader):
    return reader
"""

    result = parse_python_file("src/cache.py", source)

    assert result.path == "src/cache.py"
    assert result.symbols == (
        CodeSymbol(
            name="SnapshotReader",
            qualified_name="SnapshotReader",
            kind="class",
            line_start=3,
            line_end=5,
        ),
        CodeSymbol(
            name="read",
            qualified_name="SnapshotReader.read",
            kind="method",
            line_start=4,
            line_end=5,
        ),
        CodeSymbol(
            name="refresh_reader",
            qualified_name="refresh_reader",
            kind="function",
            line_start=7,
            line_end=8,
        ),
    )
    assert result.imports == (
        ImportRecord(
            module="pathlib",
            imported_name="Path",
            alias="PathAlias",
            relative_level=0,
            line=1,
        ),
    )
    assert result.diagnostic is None


def test_parse_python_file_returns_diagnostic_for_invalid_syntax() -> None:
    result = parse_python_file("src/broken.py", "def broken(:\n    pass\n")

    assert result.path == "src/broken.py"
    assert result.symbols == ()
    assert result.imports == ()
    assert result.diagnostic is not None
    assert result.diagnostic.line == 1
    assert result.diagnostic.message


def test_parse_python_file_records_absolute_and_relative_import_forms() -> None:
    source = """import os
from .models import Entry as ModelEntry
from .. import shared
"""

    result = parse_python_file("src/cache.py", source)

    assert result.imports == (
        ImportRecord(
            module="os",
            imported_name=None,
            alias=None,
            relative_level=0,
            line=1,
        ),
        ImportRecord(
            module="models",
            imported_name="Entry",
            alias="ModelEntry",
            relative_level=1,
            line=2,
        ),
        ImportRecord(
            module=None,
            imported_name="shared",
            alias=None,
            relative_level=2,
            line=3,
        ),
    )


def test_parse_python_file_does_not_mark_nested_function_as_method() -> None:
    source = """class Outer:
    def method(self):
        def helper():
            return None
        return helper()
"""

    result = parse_python_file("src/outer.py", source)

    assert result.symbols == (
        CodeSymbol(
            name="Outer",
            qualified_name="Outer",
            kind="class",
            line_start=1,
            line_end=5,
        ),
        CodeSymbol(
            name="method",
            qualified_name="Outer.method",
            kind="method",
            line_start=2,
            line_end=5,
        ),
        CodeSymbol(
            name="helper",
            qualified_name="Outer.method.helper",
            kind="function",
            line_start=3,
            line_end=4,
        ),
    )


def test_parse_python_file_records_call_sites_and_enclosing_symbols() -> None:
    source = """class Reader:
    def load(self):
        self.refresh()

def refresh_reader(reader):
    validate(reader)
    Reader.load(reader)
"""

    result = parse_python_file("src/reader.py", source)

    assert result.call_sites == (
        CallSite(
            caller_qualified_name="Reader.load",
            callee="self.refresh",
            callee_kind="attribute",
            line=3,
        ),
        CallSite(
            caller_qualified_name="refresh_reader",
            callee="validate",
            callee_kind="name",
            line=6,
        ),
        CallSite(
            caller_qualified_name="refresh_reader",
            callee="Reader.load",
            callee_kind="attribute",
            line=7,
        ),
    )


def test_parse_python_file_keeps_dynamic_call_sites_unresolved() -> None:
    result = parse_python_file(
        "src/factory.py",
        "def build(factory):\n    factory()()\n",
    )

    assert CallSite(
        caller_qualified_name="build",
        callee="factory()",
        callee_kind="dynamic",
        line=2,
    ) in result.call_sites
