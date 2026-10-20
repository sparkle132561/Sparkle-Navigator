"""Sparkle 产品页面及其可选 Tau Extension 兼容入口。"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, Select, Static

from tau_agent.messages import AssistantMessage, TextContent, ToolCall
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSessionConfig
from tau_coding.extensions import ExtensionRuntime, NullUiBridge
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths
from tau_coding.tui.config import TAU_DARK_THEME

from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.views import EvaluationView, SparkleMainView


def test_sparkle_command_opens_tau_main_view_and_agent_action_restores_host() -> None:
    extension_directory = (
        Path(__file__).parents[1] / "src" / "sparkle_navigator" / "presentation" / "tui"
    )
    tau_home = Path(__file__).parent / "tau-session-control-unused"
    paths = TauPaths(home=tau_home, agents_home=tau_home / "agents")
    host_ui = _CapturingUiBridge()
    runtime = ExtensionRuntime(
        ui=host_ui,
        built_in_extensions=(),
        paths=paths,
    )
    runtime.load(
        TauResourcePaths(
            root=tau_home,
            agents_root=tau_home / "agents",
            paths=paths,
            project_resources_enabled=False,
        ),
        extra_paths=(extension_directory,),
        include_resource_dirs=False,
        include_project_dir=False,
        include_user_dir=False,
    )

    result = runtime.build_command_registry().execute(
        object(),
        "/sparkle",
    )

    assert result.handled
    assert isinstance(host_ui.main_view, SparkleMainView)
    assert host_ui.main_view.id == "sparkle-main-view"

    host_ui.main_view.action_show_agent_chat()
    assert host_ui.handle.closed


def test_evaluation_view_runs_local_case_and_displays_completed_run(
    tmp_path: Path,
) -> None:
    asyncio.run(_evaluation_view_runs_local_case_and_displays_completed_run(tmp_path))


async def _evaluation_view_runs_local_case_and_displays_completed_run(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "evaluation-ui-repository"
    source_path = repository_root / "src" / "answer.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def answer():\n    return 'sparkle answer'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "add", "src/answer.py"],
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
            "Evaluation UI fixture",
        ],
        check=True,
    )
    live_providers: list[FakeProvider] = []
    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle-evaluation-ui.sqlite3",
        snapshots_root=tmp_path / "evaluation-ui-snapshots",
        working_copies_root=tmp_path / "evaluation-ui-working-copies",
        reports_root=tmp_path / "evaluation-ui-reports",
        live_session_config_factory=_live_fake_session_config_factory(
            tmp_path,
            repository_root,
            live_providers,
        ),
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)

    app = _EvaluationViewApp(services)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.click("#evaluation-run-controls CollapsibleTitle")
        snapshot_selector = app.query_one("#evaluation-snapshot", Select)
        assert snapshot_selector.value == imported.snapshot.snapshot_id
        app.query_one("#evaluation-input", Input).value = (
            "Where is the answer function implemented?"
        )
        app.query_one("#evaluation-expected-paths", Input).value = "src/answer.py"
        assert await pilot.click("#evaluation-run-case")

        for _ in range(40):
            await pilot.pause(0.1)
            progress = app.query_one("#evaluation-run-progress", Static).render().plain
            if "COMPLETED" in progress:
                break

        assert "COMPLETED" in progress
        await pilot.click("#evaluation-run-list CollapsibleTitle")
        assert app.query_one("#evaluation-runs", DataTable).row_count == 1

        assert await pilot.click("#evaluation-live-run-case")
        for _ in range(40):
            await pilot.pause(0.1)
            progress = app.query_one("#evaluation-run-progress", Static).render().plain
            if "COMPLETED" in progress:
                break

        assert "COMPLETED" in progress
        assert app.query_one("#evaluation-runs", DataTable).row_count == 1
        experiment_id = app.query_one("#evaluation-experiment", Select).value
        assert isinstance(experiment_id, str)
        live_experiment = await services.get_experiment(experiment_id)
        assert live_experiment.experiment.configuration["runner"] == (
            "tau_coding_session"
        )
        assert live_experiment.experiment.configuration["provider"] == (
            "注入的测试 Provider"
        )
        assert live_experiment.experiment.status.value == "COMPLETED"
        assert len(live_providers) == 1
        assert len(live_providers[0].calls) == 2


class _EvaluationViewApp(App):
    def __init__(self, services: SparklePresentationServices) -> None:
        super().__init__()
        self._services = services

    def compose(self) -> ComposeResult:
        yield EvaluationView(services=self._services)


def _live_fake_session_config_factory(
    tmp_path: Path,
    cwd: Path,
    providers: list[FakeProvider],
):
    tool_message = AssistantMessage(
        content=[
            ToolCall(
                id="explain-ui-answer",
                name="explain_code",
                arguments={"target": "answer"},
            )
        ],
        api="test",
        provider="fake",
        model="tui-live-fake-v1",
        stop_reason="toolUse",
    )
    final_message = AssistantMessage(
        content=[TextContent(text="`src/answer.py:1` defines `answer`.")],
        api="test",
        provider="fake",
        model="tui-live-fake-v1",
        stop_reason="stop",
    )

    def provider_events(message: AssistantMessage):
        return [
            AssistantStartEvent(partial=message.model_copy(deep=True)),
            AssistantDoneEvent(reason=message.stop_reason, message=message),
        ]

    def create_config(agent_input) -> CodingSessionConfig:
        provider = FakeProvider(
            [provider_events(tool_message), provider_events(final_message)]
        )
        providers.append(provider)
        return CodingSessionConfig(
            provider=provider,
            model="tui-live-fake-v1",
            storage=JsonlSessionStorage(
                tmp_path / "tau-sessions" / f"{agent_input.run_id}.jsonl"
            ),
            cwd=cwd,
            system="Use only the verified Snapshot tools to answer.",
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    return create_config




class _CapturingUiBridge(NullUiBridge):
    @property
    def supports_components(self) -> bool:
        return True

    @property
    def theme(self):
        return TAU_DARK_THEME

    def open_main_view(self, factory):
        self.handle = _FakeMainViewHandle()
        self.main_view = factory(
            self.handle,
            self.theme,
        )
        return self.handle


class _FakeMainViewHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self, _result=None) -> None:
        self.closed = True

    async def wait(self):
        return None

    @property
    def is_open(self) -> bool:
        return not self.closed
