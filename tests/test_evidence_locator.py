"""验证证据卡点击定位：key 解析、Snapshot 行视图与宿主动作接缝。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from tests.test_citation_verification import _register_snapshot
from textual.widget import Widget
from textual.widgets import Static

from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.presentation.tui.evidence_viewer import (
    format_evidence_key,
    parse_evidence_key,
    view_evidence,
)
from sparkle_navigator.runtime.evidence_rendering import (
    render_snapshot_evidence_result,
    snapshot_evidence_details,
)
from tau_agent.tools import AgentToolResult


def _evidence(path: str, start: int | None, end: int | None, snapshot_id: str) -> CodeEvidence:
    return CodeEvidence(
        snapshot_id=snapshot_id,
        path=path,
        line_start=start,
        line_end=end,
        snippet="content",
        sha256="a" * 64,
        source="symbol",
    )


def test_evidence_key_roundtrip_and_rejection() -> None:
    parsed = parse_evidence_key("ab" * 32 + "|src/app.py|1|2")
    assert parsed is not None
    assert format_evidence_key(parsed) == "ab" * 32 + "|src/app.py|1|2"
    assert parse_evidence_key("no-separator") is None
    assert parse_evidence_key("a|b|x|2") is None
    assert parse_evidence_key("a|b|2|1") is None
    assert parse_evidence_key("|src/app.py|1|1") is None


def test_evidence_card_embeds_clickable_host_action() -> None:
    snapshot_id = "ab" * 32
    no_evidence = AgentToolResult(content=[], details=None)
    assert render_snapshot_evidence_result(no_evidence, expanded=False) is None

    card_result = AgentToolResult(
        content=[],
        details=snapshot_evidence_details(
            (_evidence("src/app.py", 1, 2, snapshot_id),)
        ),
    )
    collapsed = render_snapshot_evidence_result(card_result, expanded=False)
    assert collapsed is not None
    assert f"app.view_product_evidence('{snapshot_id}|src/app.py|1|2')" in collapsed
    assert "[@click=" in collapsed
    expanded = render_snapshot_evidence_result(card_result, expanded=True)
    assert expanded is not None
    assert "app.view_product_evidence" in expanded


def test_view_evidence_shows_verified_lines(tmp_path: Path) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(
            tmp_path,
            {"src/app.py": "\n".join(f"line {number}" for number in range(1, 21))},
        )
    )
    parsed = parse_evidence_key(f"{imported.snapshot.snapshot_id}|src/app.py|5|7")
    assert parsed is not None
    title, body = view_evidence(database_path, format_evidence_key(parsed))
    assert "app.py" in title
    assert imported.snapshot.snapshot_id[:8] in title
    assert "src/app.py:5-7" in body
    assert "SHA-256 已核对" in body
    assert "    5 | line 5" in body
    assert "    7 | line 7" in body
    assert "line 8" not in body


def test_view_evidence_whole_file_and_missing_snapshot(tmp_path: Path) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(
            tmp_path, {"src/app.py": "def identity():\n    return 'alpha'\n"}
        )
    )
    whole = parse_evidence_key(f"{imported.snapshot.snapshot_id}|src/app.py|0|0")
    assert whole is not None
    _title, body = view_evidence(database_path, format_evidence_key(whole))
    assert "def identity():" in body
    assert "return 'alpha'" in body

    missing = parse_evidence_key("cd" * 32 + "|src/app.py|1|1")
    assert missing is not None
    _title, missing_body = view_evidence(database_path, format_evidence_key(missing))
    assert "已不存在" in missing_body


def test_view_evidence_out_of_range_and_tampered(tmp_path: Path) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(tmp_path, {"src/app.py": "one\ntwo\n"})
    )
    out_of_range = parse_evidence_key(f"{imported.snapshot.snapshot_id}|src/app.py|1|99")
    assert out_of_range is not None
    _title, body = view_evidence(database_path, format_evidence_key(out_of_range))
    assert "超出文件共 2 行" in body

    snapshot_root = tmp_path / "snapshots" / imported.snapshot.snapshot_id
    (snapshot_root / "src" / "app.py").write_text("tampered\n", encoding="utf-8")
    valid = parse_evidence_key(f"{imported.snapshot.snapshot_id}|src/app.py|1|1")
    assert valid is not None
    _title, tampered_body = view_evidence(database_path, format_evidence_key(valid))
    assert "无法核对" in tampered_body


def test_tau_host_action_opens_product_viewer() -> None:
    asyncio.run(_tau_host_action_opens_product_viewer())


async def _tau_host_action_opens_product_viewer() -> None:
    from tau_ai.fake import FakeProvider
    from tau_coding import CodingSession, CodingSessionConfig
    from tau_coding.tui import TauTuiApp, TuiProductPresentation

    viewed: list[str] = []

    async def loader(key: str) -> tuple[str, str] | None:
        viewed.append(key)
        return ("标题", "正文内容")

    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="viewer-test",
            cwd=Path("."),
            storage=InMemorySessionStorageLocal(),
            extensions_enabled=False,
            skills_enabled=False,
            trust_default="never",
        )
    )
    presentation = TuiProductPresentation(
        title="Sparkle Navigator",
        workspace_panel_factory=lambda context: _Panel(),
        evaluation_panel_factory=lambda context: _Panel(),
        evidence_viewer=loader,
    )
    app = TauTuiApp(session, product_presentation=presentation)
    app.animation_level = "none"
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            app.action_view_product_evidence("k1")
            await pilot.pause()
            assert viewed == ["k1"]
            body = app.screen.query_one("#command-output-body", Static)
            assert "正文内容" in str(body.render())
            await pilot.press("escape")
            await pilot.pause()

            # 未配置查看器时动作静默忽略，不报错也不打开。
            object.__setattr__(app, "product_presentation", None)
            app.action_view_product_evidence("k2")
            await pilot.pause()
            assert viewed == ["k1"]
    finally:
        await app.close_product_sessions()


class _Panel(Widget):
    pass


def InMemorySessionStorageLocal():
    from tau_agent.session import InMemorySessionStorage

    return InMemorySessionStorage()


def test_unsafe_path_characters_fall_back_to_plain_location() -> None:
    snapshot_id = "ab" * 32
    card_result = AgentToolResult(
        content=[],
        details=snapshot_evidence_details(
            (_evidence("it's.py", 1, 1, snapshot_id),),
        ),
    )
    markup = render_snapshot_evidence_result(card_result, expanded=False)
    assert markup is not None
    # 引号会破坏动作串：该位置降级为纯文本，不携带 @click。
    assert "app.view_product_evidence" not in markup
    assert "it's.py:1" in Text.from_markup(markup).plain
