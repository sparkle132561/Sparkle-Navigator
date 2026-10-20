"""使用临时 Git 仓库与 FakeProvider 启动实际 Sparkle Navigator Textual 界面。"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from tau_agent.session import JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.tui import TauTuiApp, TuiProductPresentation, TuiSettings

from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.views import EvaluationView, WorkspaceView
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sparkle-tui-preview-") as temporary:
        root = Path(temporary)
        repository = root / "sparkle-preview"
        (repository / "src").mkdir(parents=True)
        (repository / "README.md").write_text(
            "# Sparkle Preview\n\nOffline Snapshot evidence.\n",
            encoding="utf-8",
        )
        (repository / "src" / "context_manager.py").write_text(
            "def build_context():\n    return []\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Preview",
                "-c",
                "user.email=preview@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

        def fake_config(session_id: str) -> CodingSessionConfig:
            return CodingSessionConfig(
                provider=FakeProvider([]),
                model="offline-preview",
                storage=JsonlSessionStorage(root / f"{session_id}.jsonl"),
                cwd=repository,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
                tool_access_policy=ToolProfileRuntime(
                    repository_trusted=False,
                    workspace_root=repository,
                ),
            )

        services = SparklePresentationServices(
            database_path=root / "data.sqlite3",
            snapshots_root=root / "snapshots",
            working_copies_root=root / "copies",
            reports_root=root / "reports",
            live_session_config_factory=lambda request: fake_config(request.run_id),
        )
        imported = await services.import_local_repository(repository)
        await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
        session = await CodingSession.load(fake_config("chat"))
        app = TauTuiApp(
            session,
            tui_settings=TuiSettings(),
            product_presentation=TuiProductPresentation(
                title="Sparkle Navigator",
                workspace_panel_factory=lambda context: WorkspaceView(
                    context=context,
                    services=services,
                ),
                evaluation_panel_factory=lambda _context: EvaluationView(
                    services=services
                ),
            ),
        )
        try:
            await app.run_async()
        finally:
            await session.aclose()


if __name__ == "__main__":
    asyncio.run(main())
