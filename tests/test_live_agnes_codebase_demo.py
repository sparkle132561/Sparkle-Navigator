"""可选的真实 Agnes API 代码解释 Smoke Test。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest
from tau_agent.events import ToolExecutionEndEvent
from tau_agent.session import InMemorySessionStorage
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding import CodingSession, CodingSessionConfig

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.evaluation.live_provider import LIVE_PROVIDER_MAX_TOKENS
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime


@pytest.mark.skipif(
    os.environ.get("SPARKLE_RUN_AGNES_LIVE_TESTS") != "1",
    reason="设置 SPARKLE_RUN_AGNES_LIVE_TESTS=1 才运行真实 Agnes API Smoke Test。",
)
def test_live_agnes_agent_explains_a_verified_snapshot(tmp_path: Path) -> None:
    asyncio.run(_live_agnes_agent_explains_a_verified_snapshot(tmp_path))


async def _live_agnes_agent_explains_a_verified_snapshot(tmp_path: Path) -> None:
    required_names = ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL")
    settings = {name: os.environ.get(name) for name in required_names}
    missing = tuple(name for name, value in settings.items() if not value)
    if missing:
        pytest.skip(f"真实 Agnes 配置缺少进程环境变量：{', '.join(missing)}")

    model = settings["LLM_MODEL_ID"]
    api_key = settings["LLM_API_KEY"]
    base_url = _normalize_base_url(settings["LLM_BASE_URL"])
    assert model is not None and api_key is not None and base_url is not None

    snapshot_root = tmp_path / "snapshot"
    source = "def start():\n    return 'ready'\n"
    source_path = snapshot_root / "src" / "app.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    files = (
        SnapshotFile(
            path="src/app.py",
            sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        ),
    )
    snapshot = RepositorySnapshot.create(
        source_commit="e" * 40,
        files=files,
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "agnes-live.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"agnes-live-codebase").hexdigest(),
                name="agnes-live-codebase",
                source_type="local",
                source_locator="test://agnes-live-codebase",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    await SnapshotTextIndex(database_path=database_path).build(snapshot)
    await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
    await SnapshotCodeGraph(database_path=database_path).build(snapshot)

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=api_key,
            provider_name="Agnes API",
            base_url=base_url,
            api="openai-completions",
            timeout_seconds=45,
            max_retries=0,
            max_tokens=LIVE_PROVIDER_MAX_TOKENS,
        )
    )
    config = add_codebase_tools(
        CodingSessionConfig(
            provider=provider,
            model=model,
            storage=InMemorySessionStorage(),
            cwd=snapshot_root,
            system="使用只读代码工具核实证据；不得猜测未找到的实现。",
            tool_access_policy=ToolProfileRuntime(),
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        ),
        database_path=database_path,
        snapshot_id=snapshot.snapshot_id,
    )
    session = await CodingSession.load(config)
    try:
        try:
            events = [
                event
                async for event in session.prompt(
                    "请调用 explain_code，目标是符号 start。然后用中文解释并引用文件路径和行号。"
                )
            ]
        except Exception as exc:
            raise AssertionError(
                f"Agnes live request failed: {type(exc).__name__}; secret values are redacted."
            ) from None

        tool_results = [
            event.result
            for event in events
            if isinstance(event, ToolExecutionEndEvent) and not event.is_error
        ]
        assert any(
            result.text and "src/app.py" in result.text and "start" in result.text
            for result in tool_results
        ), "Agnes run completed without returning verified explain_code evidence."
    finally:
        await session.aclose()
        await provider.aclose()


def _normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url[: -len("/chat/completions")]
    return base_url
