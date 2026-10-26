"""验证 ContextManager 经产品配置变换接入且默认关闭。"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from sparkle_navigator.domain.repository import RepositoryImportResult
from sparkle_navigator.presentation.tui.session_config import (
    SparkleSessionConfigTransform,
)
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from tau_agent.session import InMemorySessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig


async def _register_snapshot(
    root: Path, source_files: dict[str, str]
) -> tuple[Path, RepositoryImportResult]:
    from sparkle_navigator.presentation.tui.services import SparklePresentationServices

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


def _base_config(provider: FakeProvider, tmp_path: Path) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,
        model="context-wiring-test",
        cwd=tmp_path,
        storage=InMemorySessionStorage(),
        extensions_enabled=False,
        skills_enabled=False,
        trust_default="never",
    )


def test_transform_wires_context_only_when_enabled(tmp_path: Path) -> None:
    database_path, imported = asyncio.run(
        _register_snapshot(tmp_path, {"src/app.py": "def identity():\n    return 'x'\n"})
    )
    provider = FakeProvider([])
    enabled = SparkleSessionConfigTransform(
        database_path=database_path,
        snapshot_id=imported.snapshot.snapshot_id,
        repository_trusted=False,
        approval_handler=None,
        context_repository_id=imported.repository.repository_id,
    )
    wired = enabled(_base_config(provider, tmp_path))
    assert wired.turn_context_provider is not None

    disabled = SparkleSessionConfigTransform(
        database_path=database_path,
        snapshot_id=imported.snapshot.snapshot_id,
        repository_trusted=False,
        approval_handler=None,
    )
    unwired = disabled(_base_config(provider, tmp_path))
    assert unwired.turn_context_provider is None
    assert isinstance(unwired.tool_access_policy, ToolProfileRuntime)


def test_context_reaches_provider_without_becoming_session_message(
    tmp_path: Path,
) -> None:
    asyncio.run(_context_reaches_provider_without_becoming_session_message(tmp_path))


async def _context_reaches_provider_without_becoming_session_message(
    tmp_path: Path,
) -> None:
    database_path, imported = await _register_snapshot(
        tmp_path,
        {"src/app.py": "def identity():\n    return 'alpha'\n"},
    )
    provider = FakeProvider([])
    transform = SparkleSessionConfigTransform(
        database_path=database_path,
        snapshot_id=imported.snapshot.snapshot_id,
        repository_trusted=False,
        approval_handler=None,
        context_repository_id=imported.repository.repository_id,
    )
    session = await CodingSession.load(transform(_base_config(provider, tmp_path)))
    try:
        async for _event in session.prompt("identity 函数做什么"):
            pass
        assert provider.calls, "Provider 未被调用"
        provider_payload = json.dumps(
            [(call[0], call[1]) for call in provider.calls],
            ensure_ascii=False,
        )
        assert "sparkle_context" in provider_payload
        # ContextManager 对片段做 HTML 转义后注入。
        assert "return" in provider_payload and "alpha" in provider_payload
        # 注入的上下文不进入 Session 消息历史。
        persisted = "\n".join(m.model_dump_json() for m in session.messages)
        assert "sparkle_context" not in persisted
    finally:
        await session.aclose()
