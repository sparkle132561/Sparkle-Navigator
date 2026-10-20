"""Sparkle 产品页直接使用 Tau TUI 宿主和原 CodingSession/Event 链。"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from textual.widgets import Collapsible, Input, Label

from tau_agent.messages import AssistantMessage, TextContent
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent, TextDeltaEvent
from tau_agent.session import JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.tui import TauTuiApp, TuiProductPresentation, TuiSettings

from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.views import EvaluationView, WorkspaceView
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime


def test_sparkle_workspace_agent_and_evaluation_stay_together(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asyncio.run(
        asyncio.wait_for(
            _sparkle_workspace_agent_and_evaluation_stay_together(tmp_path, monkeypatch),
            timeout=30,
        )
    )


async def _sparkle_workspace_agent_and_evaluation_stay_together(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    readme_path = repository_root / "README.md"
    readme_path.write_text(
        "# Sparkle Navigator\n\nThe Context Manager assembles evidence for each turn.\n",
        encoding="utf-8",
    )
    source_path = repository_root / "src" / "context_manager.py"
    source_path.parent.mkdir()
    source_path.write_text("def build_context():\n    return []\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "add",
            "README.md",
            "src/context_manager.py",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Sparkle Tests",
            "-c",
            "user.email=sparkle-tests@example.invalid",
            "commit",
            "-qm",
            "Sparkle TUI fixture",
        ],
        check=True,
    )

    reply = "Tau Agent Event 已通过 Sparkle 界面显示。"
    assistant = AssistantMessage(
        content=[TextContent(text=reply)],
        api="test",
        provider="fake",
        model="sparkle-tui-test-v1",
        stop_reason="stop",
    )
    provider = FakeProvider(
        [[
            AssistantStartEvent(partial=assistant.model_copy(deep=True)),
            TextDeltaEvent(
                content_index=0,
                delta=reply,
                partial=assistant.model_copy(deep=True),
            ),
            AssistantDoneEvent(reason="stop", message=assistant),
        ]]
    )
    session_path = tmp_path / "sparkle-session.jsonl"
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="sparkle-tui-test-v1",
            storage=JsonlSessionStorage(session_path),
            cwd=repository_root,
            session_id="sparkle-tui-session",
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
            tool_access_policy=ToolProfileRuntime(
                repository_trusted=False,
                workspace_root=repository_root,
            ),
        )
    )
    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle-tui.sqlite3",
        snapshots_root=tmp_path / "sparkle-snapshots",
        working_copies_root=tmp_path / "sparkle-working-copies",
        reports_root=tmp_path / "sparkle-reports",
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
    presentation = TuiProductPresentation(
        title="Sparkle Navigator",
        workspace_panel_factory=lambda context: WorkspaceView(
            context=context,
            services=services,
        ),
        evaluation_panel_factory=lambda _context: EvaluationView(services=services),
    )
    app = TauTuiApp(
        session,
        tui_settings=TuiSettings(),
        product_presentation=presentation,
    )
    app.animation_level = "none"

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.title == "Sparkle Navigator"
            assert app.query_one("#product-agent-header").render().plain == "Agent · Sparkle Navigator"
            assert app.query_one("#prompt-prefix").render().plain == "\\|/\n-+-\n/|\\"
            assert app.query_one("#prompt-prefix").content_size.width == 3
            assert app.query_one("#prompt-prefix").content_size.height == 3
            assert not app.query_one("#sidebar").display
            for selector in (
                "#product-workspace-panel",
                "#transcript",
                "#prompt-row",
                "#product-evaluation-panel",
            ):
                panel = app.query_one(selector)
                assert panel.display
                assert panel.region.width > 0

            workspace_region = app.query_one("#product-workspace-panel").region
            agent_region = app.query_one("#main-pane").region
            evaluation_region = app.query_one("#product-evaluation-panel").region
            assert workspace_region.x < agent_region.x < evaluation_region.x
            assert workspace_region.x + workspace_region.width <= agent_region.x
            assert agent_region.x + agent_region.width <= evaluation_region.x
            for width, height in ((80, 24), (160, 48), (120, 40)):
                await pilot.resize_terminal(width, height)
                panels = [
                    app.query_one(selector).region
                    for selector in (
                        "#product-workspace-panel", "#main-pane", "#product-evaluation-panel"
                    )
                ]
                assert all(
                    region.width > 0 and 0 <= region.x < region.right <= width
                    for region in panels
                )
                assert panels[0].right <= panels[1].x < panels[1].right <= panels[2].x

            assert app.query_one("#sparkle-workspace-view")
            assert app.query_one("#sparkle-evaluation-view")
            assert app.query_one("#transcript").display
            assert imported.repository.name in app.query_one("#workspace-status").render().plain
            assert "src" in app.query_one("#workspace-file-tree").render().plain
            assert "README.md" in app.query_one("#workspace-evidence").render().plain
            workspace_controls = app.query_one(
                "#workspace-repository-controls",
                Collapsible,
            )
            assert workspace_controls.collapsed
            assert app.query_one("#workspace-status").display
            assert app.query_one("#workspace-status").region.y < workspace_controls.region.y
            assert app.query_one("#workspace-session-details", Collapsible).collapsed

            await pilot.click("#workspace-repository-controls CollapsibleTitle")
            assert not workspace_controls.collapsed
            await pilot.pause()

            source_input = app.query_one("#workspace-source", Input)
            source_input.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click("#workspace-source", offset=(2, 1))
            assert app.focused is source_input
            await pilot.press(*tuple("C:/sample-repository"))
            assert source_input.value == "C:/sample-repository"
            assert app.query_one("#prompt").text == ""
            source_input.value = str(repository_root)
            app.query_one("#workspace-import").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await asyncio.wait_for(pilot.click("#workspace-import", offset=(2, 1)), timeout=5)
            await pilot.pause()
            assert source_input.value == "", app.query_one("#workspace-status").render().plain
            assert "README.md" in app.query_one("#workspace-evidence").render().plain

            # 在同一真实 UI 中反复确认/取消，覆盖信任与撤销，捕捉缓存陈旧。
            for keys, expected in (
                (("escape",), False),
                (("down", "enter"), False),
                (("down", "up", "enter"), True),
                (("escape",), True),
                (("down", "enter"), True),
                (("enter",), False),
                (None, True),
            ):
                app.query_one("#workspace-trust").scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                await asyncio.wait_for(pilot.click("#workspace-trust"), timeout=3)
                await pilot.pause()
                assert len(app.screen_stack) == 2
                assert (
                    app.screen.query_one("#extension-confirm-title").render().plain
                    == "更改 Repository 信任状态"
                )
                if keys is None:
                    yes = app.screen.query_one("#extension-confirm-list").children[0].query_one(Label)
                    assert await asyncio.wait_for(pilot.click(yes, offset=(1, 0)), timeout=3)
                else:
                    await asyncio.wait_for(pilot.press(*keys), timeout=3)
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
                await pilot.pause()
                assert len(app.screen_stack) == 1
                # 新服务实例重新读取 SQLite，不能只验证当前 UI 的内存状态。
                reopened = SparklePresentationServices(
                    database_path=tmp_path / "sparkle-tui.sqlite3",
                    snapshots_root=tmp_path / "sparkle-snapshots",
                    working_copies_root=tmp_path / "sparkle-working-copies",
                    reports_root=tmp_path / "sparkle-reports",
                )
                assert (
                    await reopened.get_repository(imported.repository.repository_id)
                ).trusted is expected
                assert ("信任：已信任" if expected else "信任：未信任") in (
                    app.query_one("#workspace-status").render().plain
                )

            # 模拟确认弹框期间 Agent 正在运行；Yes 不得先写 SQLite 再让策略更新失败。
            monkeypatch.setattr(CodingSession, "is_running", property(lambda _session: True))
            await pilot.click("#workspace-trust")
            await pilot.pause()
            assert len(app.screen_stack) == 2
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert len(app.screen_stack) == 1
            assert "Agent 正在运行" in app.query_one("#workspace-status").render().plain
            still_trusted = await services.get_repository(imported.repository.repository_id)
            assert still_trusted.trusted is True
            assert app.query_one("#workspace-trust").display

            # 模拟确认之后运行时策略更新失败；SQLite 必须恢复为修改前的值。
            monkeypatch.setattr(CodingSession, "is_running", property(lambda _session: False))

            def reject_runtime_update(
                _runtime: ToolProfileRuntime,
                _session,
                *,
                trusted: bool,
            ) -> None:
                raise RuntimeError("simulated runtime update race")

            monkeypatch.setattr(
                ToolProfileRuntime,
                "set_repository_trusted",
                reject_runtime_update,
            )
            await pilot.click("#workspace-trust")
            await pilot.pause()
            assert len(app.screen_stack) == 2
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert "信任状态未更新" in app.query_one("#workspace-status").render().plain
            after_failed_runtime_update = await services.get_repository(
                imported.repository.repository_id
            )
            assert after_failed_runtime_update.trusted is True

            evaluation_controls = app.query_one(
                "#evaluation-run-controls",
                Collapsible,
            )
            assert evaluation_controls.collapsed
            assert app.query_one("#evaluation-run-progress").display
            assert app.query_one("#evaluation-run-summary").display
            assert app.query_one("#evaluation-run-list", Collapsible).collapsed
            assert app.query_one("#evaluation-run-summary").region.y < evaluation_controls.region.y
            await pilot.click("#evaluation-run-controls CollapsibleTitle")
            assert not evaluation_controls.collapsed
            assert app.query_one("#evaluation-experiment-controls", Collapsible).collapsed
            assert app.query_one("#evaluation-run-details", Collapsible).collapsed
            assert app.query_one("#evaluation-comparison-controls", Collapsible).collapsed
            evaluation_input = app.query_one("#evaluation-input", Input)
            evaluation_input.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click("#evaluation-input", offset=(2, 1))
            assert app.focused is evaluation_input
            await pilot.press(*tuple("Snapshot evidence"))
            assert evaluation_input.value == "Snapshot evidence"
            assert app.query_one("#prompt").text == ""

            for selector in (
                "#workspace-source",
                "#workspace-import",
                "#workspace-repository",
                "#workspace-index",
                "#workspace-copy",
                "#workspace-trust",
                "#evaluation-snapshot",
                "#evaluation-input",
                "#evaluation-run-case",
                "#evaluation-live-run-case",
            ):
                panel_region = (
                    workspace_region
                    if selector.startswith("#workspace-")
                    else evaluation_region
                )
                control_region = app.query_one(selector).region
                assert control_region.x >= panel_region.x
                assert control_region.x + control_region.width <= (
                    panel_region.x + panel_region.width
                )

            assert app.query_one("#workspace-profile-controls", Collapsible).collapsed
            assert app.query_one("#workspace-query-controls", Collapsible).collapsed

            await pilot.click("#prompt")
            await pilot.press(*tuple("请回答"), "enter")
            await pilot.pause(0.3)

            assert any(
                item.role == "assistant" and item.text == reply
                for item in app.state.items
            )
            assert any(
                message.role == "assistant" and message.text == reply
                for message in session.messages
            )
            assert await session.session_entries()
            assert session_path.exists()
            assert app.query_one("#prompt-prefix").render().plain == "\\|/\n-+-\n/|\\"
    finally:
        await session.aclose()


def test_sparkle_cli_starts_product_tui_without_sparkle_extension(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    import tau_coding.cli as tau_cli
    import sparkle_navigator.presentation.cli as sparkle_cli

    paths = (
        tmp_path / "sparkle.sqlite3",
        tmp_path / "snapshots",
        tmp_path / "working-copies",
        tmp_path / "reports",
    )
    monkeypatch.setattr(sparkle_cli, "default_sparkle_paths", lambda _home: paths)
    captured: dict[str, object] = {}

    async def capture_tui_launch(**kwargs) -> str:
        captured.update(kwargs)
        return "sparkle-test-session"

    monkeypatch.setattr(tau_cli, "run_tui_app", capture_tui_launch)
    monkeypatch.setattr(tau_cli, "startup_release_notes_notice", lambda _version: None)
    result = CliRunner().invoke(sparkle_cli.app, ["--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    presentation = captured["product_presentation"]
    assert isinstance(presentation, TuiProductPresentation)
    assert presentation.title == "Sparkle Navigator"
    assert callable(presentation.workspace_panel_factory)
    assert callable(presentation.evaluation_panel_factory)
    assert "session_config_transform" in captured
    assert captured.get("extension_paths", ()) == ()


def test_tau_tui_without_product_presentation_keeps_native_chat(
    tmp_path: Path,
) -> None:
    asyncio.run(_tau_tui_without_product_presentation_keeps_native_chat(tmp_path))


async def _tau_tui_without_product_presentation_keeps_native_chat(
    tmp_path: Path,
) -> None:
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="tau-native-tui-test-v1",
            storage=JsonlSessionStorage(tmp_path / "tau-native-session.jsonl"),
            cwd=tmp_path,
            session_id="tau-native-tui-session",
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )
    )
    app = TauTuiApp(session, tui_settings=TuiSettings())

    try:
        async with app.run_test(size=(120, 40)):
            assert app.query_one("#sidebar").region.width > 0
            assert app.query_one("#transcript").display
            assert app.query_one("#prompt-row").display
            assert app.query_one("#prompt-prefix").render().plain == "τ"
    finally:
        await session.aclose()
