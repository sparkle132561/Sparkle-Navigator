"""验证回答引用解析、Snapshot 核对与状态行呈现。"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from sparkle_navigator.domain.repository import RepositoryImportResult
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.session_binding import WorkspaceSessions
from sparkle_navigator.presentation.tui.turn_status import build_turn_status_formatter
from sparkle_navigator.runtime.citation_verification import (
    extract_citations,
    format_citation_checks,
    verify_citations,
)
from sparkle_navigator.runtime.run_status import format_run_status, run_turn_stats
from tau_agent.messages import AssistantMessage, TextContent
from tau_agent.session import InMemorySessionStorage, JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.paths import TauPaths
from tau_coding.session_manager import SessionManager


async def _register_snapshot(
    root: Path, source_files: dict[str, str]
) -> tuple[Path, RepositoryImportResult]:
    """导入仓库并建立 Snapshot 数据库；返回 (数据库路径, 导入结果)。"""

    source = root / "source"
    source.mkdir()
    for relative_path, content in source_files.items():
        target = source.joinpath(*relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    services = SparklePresentationServices(
        database_path=root / "navigator.sqlite3",
        snapshots_root=root / "snapshots",
        working_copies_root=root / "copies",
        reports_root=root / "reports",
    )
    imported = await services.import_local_repository(source)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
    return services.database_path, imported


def test_extract_citations_dedupes_and_normalizes_separators() -> None:
    text = (
        "定义在 src/app.py:1-2。另见 src\\app.py:1-2 与 docs/guide.md:10，"
        "以及重复 src/app.py:1-2。版本号 1.2:3 不算。"
    )
    refs = extract_citations(text)
    paths = [(ref.path, ref.line_start, ref.line_end) for ref in refs]
    assert ("src/app.py", 1, 2) in paths
    assert ("docs/guide.md", 10, None) in paths
    # 反斜杠路径归一化为 posix，去重后只剩一份。
    assert paths.count(("src/app.py", 1, 2)) == 1
    assert all(ref.path != "1.2" for ref in refs)


def test_extract_citations_ignores_urls_and_times() -> None:
    refs = extract_citations("见 https://example.com/docs:8080 与 12:30。")
    assert refs == ()


def test_verify_citations_marks_valid_missing_and_out_of_range(
    tmp_path: Path,
) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(
            tmp_path,
            {"src/app.py": "def identity():\n    return 'alpha'\n"},
        )
    )
    citations = extract_citations(
        "见 src/app.py:1-2、src/app.py:99、missing.py:3 与 src/app.py:2-1。"
    )
    checks = verify_citations(
        database_path,
        repository_id=imported.repository.repository_id,
        snapshot_id=imported.snapshot.snapshot_id,
        citations=citations,
    )
    by_status = {check.ref.path + str(check.ref.line_start): check for check in checks}
    assert by_status["src/app.py1"].status == "verified"
    assert by_status["src/app.py99"].status == "out_of_range"
    assert by_status["missing.py3"].status == "missing_path"
    assert by_status["src/app.py2"].status == "out_of_range"
    rendered = format_citation_checks(checks)
    assert rendered is not None
    assert "src/app.py:1-2 有效" in rendered
    assert "src/app.py:99 行号超出文件范围" in rendered
    assert "missing.py:3 未在当前 Snapshot 中找到" in rendered
    # 用户可见的固定边界：核对位置存在 ≠ 证据支持结论。
    assert "不代表证据支持回答结论" in rendered


def test_no_citations_formats_to_none() -> None:
    assert format_citation_checks(()) is None


def test_turn_status_formatter_verifies_citations_for_bound_session(
    tmp_path: Path,
) -> None:
    asyncio.run(_turn_status_formatter_verifies_citations_for_bound_session(tmp_path))


async def _turn_status_formatter_verifies_citations_for_bound_session(
    tmp_path: Path,
) -> None:
    database_path, imported = await _register_snapshot(
        tmp_path,
        {"src/app.py": "def identity():\n    return 'alpha'\n"},
    )
    services = SparklePresentationServices(
        database_path=database_path,
        snapshots_root=tmp_path / "snapshots",
        working_copies_root=tmp_path / "copies",
        reports_root=tmp_path / "reports",
    )
    binding = (
        imported.repository.repository_id,
        imported.snapshot.snapshot_id,
    )
    sessions = WorkspaceSessions(
        services=services,
        base_config=CodingSessionConfig(
            provider=FakeProvider([]),
            model="m",
            cwd=tmp_path,
            storage=InMemorySessionStorage(),
        ),
        approval_handler=None,
        initial_binding=binding,
    )
    manager = SessionManager(paths=TauPaths(home=tmp_path / "tau"))
    record = manager.create_session(cwd=tmp_path, model="m", title="citations")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="m",
            cwd=tmp_path,
            storage=JsonlSessionStorage(record.path),
            session_id=record.id,
            session_manager=manager,
            extensions_enabled=False,
            skills_enabled=False,
            trust_default="never",
        )
    )
    try:
        await session.append_custom_entry(
            "sparkle.repository-binding",
            {"repository_id": binding[0], "snapshot_id": binding[1]},
        )
        formatter = build_turn_status_formatter(
            services, sessions_lookup=lambda: sessions
        )
        tool = AssistantMessage(
            content=[],
            api="test",
            provider="fake",
            model="m",
            stop_reason="toolUse",
        )
        final = AssistantMessage(
            content=[
                TextContent(
                    text="identity 定义在 src/app.py:1-2，错误引用 missing.py:3。"
                )
            ],
            api="test",
            provider="fake",
            model="m",
            stop_reason="stop",
        )
        rendered = await formatter((tool, final), session)
        assert rendered is not None
        assert "已完成" in rendered
        assert "src/app.py:1-2 有效" in rendered
        assert "missing.py:3 未在当前 Snapshot 中找到" in rendered

        # 未绑定 Snapshot 的会话（无持久绑定记录）只显示运行统计。
        bare_record = manager.create_session(cwd=tmp_path, model="m", title="bare")
        bare = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="m",
                cwd=tmp_path,
                storage=JsonlSessionStorage(bare_record.path),
                session_id=bare_record.id,
                extensions_enabled=False,
                skills_enabled=False,
                trust_default="never",
            )
        )
        try:
            bare_rendered = await formatter((tool, final), bare)
            assert bare_rendered is not None
            assert "引用核对" not in bare_rendered
        finally:
            await bare.aclose()

        # 无引用时不追加引用行。
        no_citation = AssistantMessage(
            content=[TextContent(text="没有引用的回答。")],
            api="test",
            provider="fake",
            model="m",
            stop_reason="stop",
        )
        plain = await formatter((no_citation,), session)
        assert plain is not None
        assert "引用核对" not in plain
        stats_only = run_turn_stats((no_citation,))
        assert stats_only is not None and format_run_status(stats_only) in plain
    finally:
        await session.aclose()


def test_unverifiable_snapshot_content_is_reported_not_crashing(
    tmp_path: Path,
) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(
            tmp_path,
            {"src/app.py": "def identity():\n    return 'alpha'\n"},
        )
    )
    snapshot_root = tmp_path / "snapshots" / imported.snapshot.snapshot_id
    (snapshot_root / "src" / "app.py").write_text(
        "tampered content\n", encoding="utf-8"
    )
    citations = extract_citations("见 src/app.py:1。")
    checks = verify_citations(
        database_path,
        repository_id=imported.repository.repository_id,
        snapshot_id=imported.snapshot.snapshot_id,
        citations=citations,
    )
    assert checks[0].status == "unverifiable"
    rendered = format_citation_checks(checks)
    assert rendered is not None
    assert "无法核对" in rendered

def test_binary_file_citation_is_unverifiable_not_zero_lines(tmp_path: Path) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(tmp_path, {"src/logo.png": "\x00\x01\x02binary"})
    )
    citations = extract_citations("见 src/logo.png:1。")
    checks = verify_citations(
        database_path,
        repository_id=imported.repository.repository_id,
        snapshot_id=imported.snapshot.snapshot_id,
        citations=citations,
    )
    assert checks[0].status == "unverifiable"
    assert checks[0].detail is not None and "不是文本文件" in checks[0].detail


def test_missing_snapshot_in_database_is_unverifiable_without_crash(
    tmp_path: Path,
) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(tmp_path, {"src/app.py": "one\n"})
    )
    stale_snapshot_id = "ff" * 32
    citations = extract_citations("见 src/app.py:1。")
    checks = verify_citations(
        database_path,
        repository_id=imported.repository.repository_id,
        snapshot_id=stale_snapshot_id,
        citations=citations,
    )
    assert checks[0].status == "unverifiable"
    assert "不在本地数据库" in (checks[0].detail or "")
