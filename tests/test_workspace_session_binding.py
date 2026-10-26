"""通过产品界面验证 Repository、Snapshot 与 Tau Session 的绑定。"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from textual.widgets import Collapsible, Input, Select

from sparkle_navigator.domain.tool_profile import ToolProfile
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.session_binding import WorkspaceSessions
from sparkle_navigator.presentation.tui.session_config import SparkleSessionConfigTransform
from sparkle_navigator.presentation.tui.views import EvaluationView, WorkspaceView
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from tau_agent.messages import AssistantMessage, TextContent
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import ToolCall
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.paths import TauPaths
from tau_coding.session_manager import SessionManager
from tau_coding.tui import TauTuiApp, TuiProductPresentation


def _repository(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "app.py").write_text(f"def identity():\n    return '{name}'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "app.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
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
    return path


def test_import_creates_bound_session_without_restart(tmp_path: Path) -> None:
    asyncio.run(_import_creates_bound_session_without_restart(tmp_path))


async def _import_creates_bound_session_without_restart(tmp_path: Path) -> None:
    source = _repository(tmp_path, "alpha")
    app, services, initial = await _workbench(tmp_path, FakeProvider([]))
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, source)
            (repository,) = await services.list_repositories()
            assert app.session is not initial, app.query_one("#workspace-status").render().plain
            assert app.session.cwd == (
                services.working_copies_root
                / repository.repository_id
                / repository.current_snapshot_id
            )
            assert "explain_code" in app.session.visible_tool_names
            assert "Session 已绑定" in app.query_one("#workspace-status").render().plain
            assert repository.current_snapshot_id in (
                app.query_one("#workspace-agent-binding").render().plain
            )
    finally:
        await app.close_product_sessions()


def test_index_failure_retains_old_session_and_retry_activates_selection(tmp_path: Path) -> None:
    asyncio.run(_index_failure_retains_old_session_and_retry_activates_selection(tmp_path))


async def _index_failure_retains_old_session_and_retry_activates_selection(tmp_path: Path) -> None:
    alpha = _repository(tmp_path, "alpha")
    beta = _repository(tmp_path, "beta")
    app, services, _ = await _workbench(tmp_path, FakeProvider([]))
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            old = app.session
            (repository_a,) = await services.list_repositories()
            await services.set_repository_trust(repository_a.repository_id, trusted=True)
            policy = old.tool_access_policy
            assert isinstance(policy, ToolProfileRuntime)
            policy.set_repository_trusted(old, trusted=True)
            await policy.set_profile(old, ToolProfile.CODING)
            imported_b = await services.import_local_repository(beta)
            snapshot_file = imported_b.snapshot.snapshot_root / "app.py"
            original = snapshot_file.read_bytes()
            snapshot_file.write_text("corrupted snapshot", encoding="utf-8")
            await app.query_one(WorkspaceView).refresh_repositories()
            await _select(app, pilot, imported_b.repository.repository_id)
            assert app.session is old
            assert "Snapshot 已创建但索引失败" in app.query_one("#workspace-status").render().plain
            assert imported_b.snapshot.snapshot_id not in (
                app.query_one("#workspace-agent-binding").render().plain
            )
            app.query_one("#workspace-trust").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-trust", offset=(2, 1))
            await pilot.pause()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            assert "write" in old.visible_tool_names
            snapshot_file.write_bytes(original)
            app.query_one("#workspace-index").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-index", offset=(2, 1))
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.session is not old
            assert imported_b.snapshot.snapshot_id in (
                app.query_one("#workspace-agent-binding").render().plain
            )
    finally:
        await app.close_product_sessions()


class _BlockingProvider(FakeProvider):
    def __init__(self):
        super().__init__(_reply_streams("alpha"))
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def stream_response(self, **kwargs):
        async def stream():
            self.entered.set()
            await self.release.wait()
            async for event in super(_BlockingProvider, self).stream_response(**kwargs):
                yield event

        return stream()


def test_running_agent_blocks_repository_switch_until_cancelled(tmp_path: Path) -> None:
    asyncio.run(_running_agent_blocks_repository_switch_until_cancelled(tmp_path))


async def _running_agent_blocks_repository_switch_until_cancelled(tmp_path: Path) -> None:
    alpha = _repository(tmp_path, "alpha")
    beta = _repository(tmp_path, "beta")
    provider = _BlockingProvider()
    app, services, _ = await _workbench(tmp_path, provider)
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            old = app.session
            (repository_a,) = await services.list_repositories()
            imported_b = await services.import_local_repository(beta)
            await app.query_one(WorkspaceView).refresh_repositories()
            prompt = app.query_one("#prompt")
            prompt.text = "Inspect identity"
            app.set_focus(prompt)
            await pilot.press("enter")
            await asyncio.wait_for(provider.entered.wait(), timeout=5)
            app.query_one(
                "#workspace-repository", Select
            ).value = imported_b.repository.repository_id
            await pilot.pause()
            assert app.session is old
            assert "Agent 正在运行" in app.query_one("#workspace-status").render().plain
            assert (
                app.query_one("#workspace-repository", Select).value == repository_a.repository_id
            )
            app.set_focus(prompt)
            await pilot.press("escape")
            provider.release.set()
            await app.workers.wait_for_complete()
            await _select(app, pilot, imported_b.repository.repository_id)
            assert app.session is not old
    finally:
        provider.release.set()
        await app.close_product_sessions()


async def _workbench(tmp_path: Path, provider, *, base_config: CodingSessionConfig | None = None):
    services = SparklePresentationServices(
        database_path=tmp_path / "navigator.sqlite3",
        snapshots_root=tmp_path / "snapshots",
        working_copies_root=tmp_path / "copies",
        reports_root=tmp_path / "reports",
    )
    base = CodingSessionConfig(
        provider=provider,
        model="binding-test",
        cwd=tmp_path,
        storage=JsonlSessionStorage(tmp_path / "initial.jsonl"),
        session_id="initial",
        session_manager=SessionManager(paths=TauPaths(home=tmp_path / "tau")),
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="never",
    )
    if base_config is not None:
        base = replace(
            base_config,
            provider=provider,
            cwd=tmp_path,
            session_id="initial",
            storage=JsonlSessionStorage(tmp_path / "initial.jsonl"),
        )
    transform = SparkleSessionConfigTransform(
        database_path=services.database_path,
        snapshot_id=None,
        repository_trusted=False,
        approval_handler=None,
    )
    initial = await CodingSession.load(transform(base))
    sessions = WorkspaceSessions(services=services, base_config=base, approval_handler=None)
    presentation = TuiProductPresentation(
        title="Sparkle Navigator",
        workspace_panel_factory=lambda context: WorkspaceView(
            context=context,
            services=services,
            sessions=sessions,
        ),
        evaluation_panel_factory=lambda context: EvaluationView(services=services),
        before_session_activate=sessions.revalidate_trust,
    )
    app = TauTuiApp(initial, product_presentation=presentation)
    app.animation_level = "none"
    return app, services, initial


async def _import(app, pilot, source: Path) -> None:
    app.query_one("#workspace-repository-controls", Collapsible).collapsed = False
    app.query_one("#workspace-source", Input).value = str(source)
    await pilot.pause()
    app.query_one("#workspace-import").scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    assert await pilot.click("#workspace-import", offset=(2, 1))
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _select(app, pilot, repository_id: str) -> None:
    app.query_one("#workspace-repository", Select).value = repository_id
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _ask(app, pilot, question: str) -> None:
    prompt = app.query_one("#prompt")
    prompt.text = question
    app.set_focus(prompt)
    await pilot.press("enter")
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def _reply_streams(label: str):
    tool = AssistantMessage(
        content=[
            ToolCall(id=f"identity-{label}", name="explain_code", arguments={"target": "identity"})
        ],
        api="test",
        provider="fake",
        model="binding-test",
        stop_reason="toolUse",
    )
    final = AssistantMessage(
        content=[TextContent(text=f"Verified {label} in app.py:1-2")],
        api="test",
        provider="fake",
        model="binding-test",
        stop_reason="stop",
    )
    return [
        [
            AssistantStartEvent(partial=message),
            AssistantDoneEvent(reason=message.stop_reason, message=message),
        ]
        for message in (tool, final)
    ]


def test_switching_repositories_isolates_tools_history_and_trust(tmp_path: Path) -> None:
    asyncio.run(_switching_repositories_isolates_tools_history_and_trust(tmp_path))


async def _switching_repositories_isolates_tools_history_and_trust(tmp_path: Path) -> None:
    alpha = _repository(tmp_path, "alpha")
    beta = _repository(tmp_path, "beta")
    provider = FakeProvider(_reply_streams("alpha") + _reply_streams("beta"))
    app, services, _ = await _workbench(tmp_path, provider)
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            a = app.session
            (repository_a,) = await services.list_repositories()
            await services.set_repository_trust(repository_a.repository_id, trusted=True)
            policy = a.tool_access_policy
            assert isinstance(policy, ToolProfileRuntime)
            policy.set_repository_trusted(a, trusted=True)
            await policy.set_profile(a, ToolProfile.CODING)
            await _ask(app, pilot, "Inspect alpha identity")
            a_messages = a.messages
            assert "Verified alpha" in " ".join(item.text for item in app.state.items)

            await _import(app, pilot, beta)
            b = app.session
            assert b is not a
            assert b.messages == ()
            assert b.tool_access_policy is not a.tool_access_policy
            assert "write" not in b.visible_tool_names
            assert "Verified alpha" not in " ".join(item.text for item in app.state.items)
            await _ask(app, pilot, "Inspect beta identity")
            entries = await b.session_entries()
            serialized = "\n".join(entry.model_dump_json() for entry in entries)
            assert "return 'beta'" in serialized
            assert "return 'alpha'" not in serialized
            assert repository_a.current_snapshot_id not in serialized
            assert all("alpha" not in message.model_dump_json() for message in provider.calls[2][2])
            await _ask(app, pilot, f"/resume {a.session_id}")
            assert app.session is a
            assert repository_a.current_snapshot_id in (
                app.query_one("#workspace-agent-binding").render().plain
            )
            await _select(app, pilot, repository_a.repository_id)
            assert app.session is a
            assert a.messages == a_messages
            assert "write" in a.visible_tool_names
            assert "Verified alpha" in " ".join(item.text for item in app.state.items)
            assert "Verified beta" not in " ".join(item.text for item in app.state.items)
            assert repository_a.current_snapshot_id in (
                app.query_one("#workspace-agent-binding").render().plain
            )
            session_path = a.storage.path
            assert any(
                json.loads(line).get("type") == "message"
                for line in session_path.read_text(encoding="utf-8").splitlines()
            )
    finally:
        await app.close_product_sessions()


def test_cancel_import_retains_session_and_allows_retry(tmp_path: Path, monkeypatch) -> None:
    asyncio.run(_cancel_import_retains_session_and_allows_retry(tmp_path, monkeypatch))


async def _cancel_import_retains_session_and_allows_retry(tmp_path: Path, monkeypatch) -> None:
    alpha = _repository(tmp_path, "alpha")
    beta = _repository(tmp_path, "beta")
    app, services, _ = await _workbench(tmp_path, FakeProvider([]))
    entered = asyncio.Event()
    release = asyncio.Event()
    subprocess_exec = asyncio.create_subprocess_exec

    async def stalled_git(*args, **kwargs):
        if str(beta) in args:
            entered.set()
            await release.wait()
        return await subprocess_exec(*args, **kwargs)

    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            old = app.session
            monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_git)
            app.query_one("#workspace-source", Input).value = str(beta)
            app.query_one("#workspace-import").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-import", offset=(2, 1))
            await asyncio.wait_for(entered.wait(), timeout=5)
            app.query_one("#workspace-cancel").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-cancel", offset=(2, 1))
            await pilot.pause()
            assert app.session is old
            assert "已取消准备" in app.query_one("#workspace-status").render().plain
            release.set()
            await _import(app, pilot, beta)
            assert app.session is not old
            assert len(await services.list_repositories()) == 2
    finally:
        release.set()
        await app.close_product_sessions()


def test_import_and_session_preparation_failures_preserve_old_session(tmp_path: Path) -> None:
    asyncio.run(_import_and_session_preparation_failures_preserve_old_session(tmp_path))


async def _import_and_session_preparation_failures_preserve_old_session(tmp_path: Path) -> None:
    alpha = _repository(tmp_path, "alpha")
    beta = _repository(tmp_path, "beta")
    app, services, _ = await _workbench(tmp_path, FakeProvider([]))
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            old = app.session
            await _import(app, pilot, tmp_path / "missing")
            assert app.session is old
            assert "失败" in app.query_one("#workspace-status").render().plain
            imported_b = await services.import_local_repository(beta)
            blocking_path = services.working_copies_root / imported_b.repository.repository_id
            blocking_path.write_text("not a directory", encoding="utf-8")
            await app.query_one(WorkspaceView).refresh_repositories()
            await _select(app, pilot, imported_b.repository.repository_id)
            assert app.session is old
            assert "Session 准备失败" in app.query_one("#workspace-status").render().plain
            blocking_path.unlink()
            app.query_one("#workspace-copy").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-copy", offset=(2, 1))
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.session is not old
    finally:
        await app.close_product_sessions()


def test_resuming_same_snapshot_keeps_snapshot_tools(tmp_path: Path) -> None:
    asyncio.run(_resuming_same_snapshot_keeps_snapshot_tools(tmp_path))


async def _resuming_same_snapshot_keeps_snapshot_tools(tmp_path: Path) -> None:
    alpha = _repository(tmp_path, "alpha")
    app, _, _ = await _workbench(tmp_path, FakeProvider(_reply_streams("alpha")))
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, alpha)
            await _ask(app, pilot, "Inspect identity")
            a = app.session
            messages = a.messages
            manager = a.session_manager
            record = manager.get_session(a.session_id)
            await _ask(app, pilot, "/new")
            assert a.messages == ()
            assert "explain_code" in a.visible_tool_names
            # Tau 兼容旧索引中未记录 Provider 的 Session。
            manager.index_session(replace(record, provider_name=None))
            await _ask(app, pilot, f"/resume {record.id}")
            assert a.messages == messages
            assert "explain_code" in a.visible_tool_names
    finally:
        await app.close_product_sessions()


def test_resuming_old_snapshot_rechecks_revoked_repository_trust(tmp_path: Path) -> None:
    asyncio.run(_resuming_old_snapshot_rechecks_revoked_repository_trust(tmp_path))


async def _resuming_old_snapshot_rechecks_revoked_repository_trust(tmp_path: Path) -> None:
    source = _repository(tmp_path, "alpha")
    app, services, _ = await _workbench(tmp_path, FakeProvider([]))
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, source)
            old = app.session
            (repository,) = await services.list_repositories()
            await services.set_repository_trust(repository.repository_id, trusted=True)
            policy = old.tool_access_policy
            policy.set_repository_trusted(old, trusted=True)
            await policy.set_profile(old, ToolProfile.CODING)
            (source / "app.py").write_text("def identity():\n    return 'new'\n", encoding="utf-8")
            await _import(app, pilot, source)
            assert app.session is not old
            app.query_one("#workspace-trust").scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#workspace-trust", offset=(2, 1))
            await pilot.pause()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            assert not (await services.get_repository(repository.repository_id)).trusted
            await _ask(app, pilot, f"/resume {old.session_id}")
            assert app.session is old
            assert "write" not in old.visible_tool_names
    finally:
        await app.close_product_sessions()


def test_workspace_session_dispatches_extension_start_and_closes_provider_once(
    tmp_path: Path,
) -> None:
    asyncio.run(_workspace_session_dispatches_extension_start_and_closes_provider_once(tmp_path))


async def _workspace_session_dispatches_extension_start_and_closes_provider_once(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path, "alpha")
    extension = tmp_path / "lifecycle_probe.py"
    extension.write_text(
        "def setup(tau):\n"
        "    async def started(event, context):\n"
        "        await context.session_control.append_custom_entry(\n"
        "            'test.started', {'cwd': str(context.cwd)})\n"
        "    tau.on('session_start', started)\n",
        encoding="utf-8",
    )

    class ClosingProvider(FakeProvider):
        def __init__(self):
            super().__init__([])
            self.closes = 0

        async def aclose(self):
            self.closes += 1

    provider = ClosingProvider()
    config = CodingSessionConfig(
        provider=provider,
        model="binding-test",
        cwd=tmp_path,
        storage=JsonlSessionStorage(tmp_path / "initial.jsonl"),
        owns_initial_provider=True,
        extension_paths=(extension,),
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="never",
    )
    app, services, _ = await _workbench(tmp_path, provider, base_config=config)
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _import(app, pilot, source)
            starts = [
                entry for entry in app.session.custom_entries if entry.namespace == "test.started"
            ]
            assert len(starts) == 1
            assert starts[0].data["cwd"] == str(app.session.cwd)
            old = app.session
            (repository_a,) = await services.list_repositories()
            await _import(app, pilot, _repository(tmp_path, "beta"))
            await _select(app, pilot, repository_a.repository_id)
            assert app.session is old
            starts = [entry for entry in old.custom_entries if entry.namespace == "test.started"]
            assert len(starts) == 2
    finally:
        await app.close_product_sessions()
    assert provider.closes == 1


def test_cli_restores_original_snapshot_after_repository_reimport(
    tmp_path: Path, monkeypatch
) -> None:
    from typer.testing import CliRunner

    import sparkle_navigator.presentation.cli as sparkle_cli
    import tau_coding.cli as tau_cli
    from sparkle_navigator.presentation.tui.services import default_sparkle_paths

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    database, snapshots, copies, reports = default_sparkle_paths(TauPaths().home)
    services = SparklePresentationServices(
        database_path=database,
        snapshots_root=snapshots,
        working_copies_root=copies,
        reports_root=reports,
    )

    async def prepare():
        source = _repository(tmp_path, "alpha")
        imported = await services.import_local_repository(source)
        await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
        working_copy = await services.open_working_copy(imported.repository.repository_id)
        manager = SessionManager()
        record = manager.create_session(cwd=working_copy.root, model="binding-test")
        transform = SparkleSessionConfigTransform(
            database_path=database,
            snapshot_id=imported.snapshot.snapshot_id,
            repository_trusted=False,
            approval_handler=None,
        )
        session = await CodingSession.load(
            transform(
                CodingSessionConfig(
                    provider=FakeProvider(_reply_streams("alpha")),
                    model="binding-test",
                    cwd=working_copy.root,
                    storage=JsonlSessionStorage(record.path),
                    session_id=record.id,
                    session_manager=manager,
                    skills_enabled=False,
                    extensions_enabled=False,
                    trust_default="never",
                )
            )
        )
        try:
            await session.append_custom_entry(
                "sparkle.repository-binding",
                {
                    "repository_id": imported.repository.repository_id,
                    "snapshot_id": imported.snapshot.snapshot_id,
                },
            )
            async for _event in session.prompt("Original question"):
                pass
        finally:
            await session.aclose()
        (source / "app.py").write_text("def identity():\n    return 'new'\n", encoding="utf-8")
        await services.import_local_repository(source)
        return record, imported.snapshot.snapshot_id

    record, snapshot_id = asyncio.run(prepare())

    async def run_headless(**kwargs):
        base = CodingSessionConfig(
            provider=FakeProvider(_reply_streams("alpha")),
            model="binding-test",
            cwd=record.cwd,
            storage=JsonlSessionStorage(record.path),
            session_id=record.id,
            session_manager=SessionManager(),
            skills_enabled=False,
            extensions_enabled=False,
            trust_default="never",
        )
        session = await CodingSession.load(kwargs["session_config_transform"](base))
        try:
            assert "explain_code" in session.visible_tool_names
            assert any(
                message.role == "user" and message.text == "Original question"
                for message in session.messages
            )
            before = len(session.messages)
            async for _event in session.prompt("Inspect identity again"):
                pass
            evidence = "\n".join(message.model_dump_json() for message in session.messages[before:])
            assert snapshot_id in evidence
            assert "return 'alpha'" in evidence
            return record.id
        finally:
            await session.aclose()

    monkeypatch.setattr(tau_cli, "run_tui_app", run_headless)
    monkeypatch.setattr(tau_cli, "startup_release_notes_notice", lambda _version: None)
    result = CliRunner().invoke(sparkle_cli.app, ["--session", record.id])
    assert result.exit_code == 0, (result.output, result.exception)


def test_native_resume_rejects_other_snapshot_in_same_source_directory(tmp_path: Path) -> None:
    asyncio.run(_native_resume_rejects_other_snapshot_in_same_source_directory(tmp_path))


async def _native_resume_rejects_other_snapshot_in_same_source_directory(tmp_path: Path) -> None:
    source = _repository(tmp_path, "alpha")
    services = SparklePresentationServices(
        database_path=tmp_path / "navigator.sqlite3",
        snapshots_root=tmp_path / "snapshots",
        working_copies_root=tmp_path / "copies",
        reports_root=tmp_path / "reports",
    )
    imported = await services.import_local_repository(source)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
    manager = SessionManager(paths=TauPaths(home=tmp_path / "tau"))
    original_record = manager.create_session(cwd=source, model="binding-test", title="original")
    provider = FakeProvider(_reply_streams("alpha"))
    base = CodingSessionConfig(
        provider=provider,
        model="binding-test",
        cwd=source,
        storage=JsonlSessionStorage(original_record.path),
        session_id=original_record.id,
        session_manager=manager,
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="never",
    )

    def transform(snapshot_id):
        return SparkleSessionConfigTransform(
            database_path=services.database_path,
            snapshot_id=snapshot_id,
            repository_trusted=False,
            approval_handler=None,
        )

    original = await CodingSession.load(transform(imported.snapshot.snapshot_id)(base))
    await original.append_custom_entry(
        "sparkle.repository-binding",
        {
            "repository_id": imported.repository.repository_id,
            "snapshot_id": imported.snapshot.snapshot_id,
        },
    )
    (source / "app.py").write_text("def identity():\n    return 'new'\n", encoding="utf-8")
    newer = await services.import_local_repository(source)
    await services.build_snapshot_indexes(newer.snapshot.snapshot_id)
    target_record = manager.create_session(cwd=source, model="binding-test", title="target")
    target = await CodingSession.load(
        transform(newer.snapshot.snapshot_id)(
            replace(
                base,
                storage=JsonlSessionStorage(target_record.path),
                session_id=target_record.id,
            )
        )
    )
    try:
        await target.append_custom_entry(
            "sparkle.repository-binding",
            {
                "repository_id": newer.repository.repository_id,
                "snapshot_id": newer.snapshot.snapshot_id,
            },
        )
    finally:
        await target.aclose()
    manager.index_session(replace(manager.get_session(target_record.id), provider_name=None))
    sessions = WorkspaceSessions(
        services=services,
        base_config=base,
        approval_handler=None,
        initial_binding=(imported.repository.repository_id, imported.snapshot.snapshot_id),
    )
    app = TauTuiApp(
        original,
        product_presentation=TuiProductPresentation(
            title="Sparkle Navigator",
            workspace_panel_factory=lambda context: WorkspaceView(
                context=context,
                services=services,
                sessions=sessions,
            ),
            evaluation_panel_factory=lambda context: EvaluationView(services=services),
            before_session_activate=sessions.revalidate_trust,
        ),
    )
    app.animation_level = "none"
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await _ask(app, pilot, f"/resume {target_record.id}")
            assert original.session_id == original_record.id
            assert provider.calls == []
            await _ask(app, pilot, "Inspect identity")
            evidence = "\n".join(message.model_dump_json() for message in original.messages)
            assert imported.snapshot.snapshot_id in evidence
            assert "return 'alpha'" in evidence
    finally:
        await app.close_product_sessions()
