"""验证 CodingSession 的临时上下文注入不改变 Tau Session 历史。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sparkle_navigator.context.context_manager import ContextManager
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.domain.memory import LearnerProfile
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.memory import LearnerProfileStore, ProjectMemory
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.runtime.context_injection import add_context_manager
from tau_agent.messages import AssistantMessage, TextContent, message_text
from tau_agent.session import CompactionEntry, InMemorySessionStorage, MessageEntry
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_ai.fake import FakeProvider
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
    TauResourcePaths,
    TurnContextRequest,
)


def test_turn_context_reaches_provider_without_becoming_a_session_message() -> None:
    asyncio.run(_turn_context_reaches_provider_without_becoming_a_session_message())


def test_context_manager_runs_through_coding_session_and_provider() -> None:
    asyncio.run(_context_manager_runs_through_coding_session_and_provider())


def test_tau_skill_and_compaction_summary_remain_in_provider_context() -> None:
    asyncio.run(_tau_skill_and_compaction_summary_remain_in_provider_context())


def test_context_provider_failure_does_not_start_an_uncontextualized_turn() -> None:
    asyncio.run(_context_provider_failure_does_not_start_an_uncontextualized_turn())


def test_cancelling_context_build_does_not_start_or_persist_a_turn() -> None:
    asyncio.run(_cancelling_context_build_does_not_start_or_persist_a_turn())


async def _turn_context_reaches_provider_without_becoming_a_session_message() -> None:
    model = "context-injection-test-model"
    first_reply = _assistant_reply(model, "first reply")
    second_reply = _assistant_reply(model, "second reply")
    provider = FakeProvider(
        [
            _assistant_stream(first_reply),
            _assistant_stream(second_reply),
        ]
    )
    storage = InMemorySessionStorage()
    context_requests: list[TurnContextRequest] = []

    async def provide_turn_context(request: TurnContextRequest) -> str:
        context_requests.append(request)
        return f"Supplemental context for: {request.prompt}"

    with TemporaryDirectory() as temp_dir:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=Path(temp_dir),
                system="Base system prompt.",
                tools=[],
                turn_context_provider=provide_turn_context,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        try:
            [event async for event in session.prompt("first request")]
            [event async for event in session.prompt("second request")]

            assert provider.calls[0][1] == (
                "Base system prompt.\n\nSupplemental context for: first request"
            )
            assert provider.calls[1][1] == (
                "Base system prompt.\n\nSupplemental context for: second request"
            )
            assert [request.prompt for request in context_requests] == [
                "first request",
                "second request",
            ]
            assert context_requests[0].messages == ()
            assert len(context_requests[1].messages) == 2
            assert all(request.token_budget > 0 for request in context_requests)
            assert session.system_prompt == "Base system prompt."
            assert len(session.messages) == 4
            assert all(
                "Supplemental context" not in message_text(message)
                for message in session.messages
            )
            stored_messages = [
                entry.message
                for entry in await storage.read_all()
                if isinstance(entry, MessageEntry)
            ]
            assert len(stored_messages) == 4
            assert all(
                "Supplemental context" not in message_text(message)
                for message in stored_messages
            )
        finally:
            await session.aclose()


async def _context_manager_runs_through_coding_session_and_provider() -> None:
    model = "context-manager-runtime-model"
    reply = _assistant_reply(model, "Evidence received.")
    continuation_reply = _assistant_reply(model, "Continuation received.")
    provider = FakeProvider(
        [_assistant_stream(reply), _assistant_stream(continuation_reply)]
    )

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot_root = root / "snapshot"
        source_path = snapshot_root / "src" / "app.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "def calculate_total(values):\n    return sum(values)  # context_marker\n",
            encoding="utf-8",
        )
        file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        snapshot = RepositorySnapshot.create(
            source_commit="e" * 40,
            files=(SnapshotFile(path="src/app.py", sha256=file_hash),),
            snapshot_root=snapshot_root,
        )
        repository_id = hashlib.sha256(b"context-injection-runtime").hexdigest()
        database_path = root / "sparkle.sqlite3"
        RepositoryStore(database_path).save_import(
            RepositoryImportResult(
                repository=Repository(
                    repository_id=repository_id,
                    name="context-injection-runtime",
                    source_type="local",
                    source_locator="test://context-injection-runtime",
                    current_snapshot_id=snapshot.snapshot_id,
                ),
                snapshot=snapshot,
            )
        )
        await SnapshotTextIndex(database_path=database_path).build(snapshot)
        evidence = CodeEvidence(
            snapshot_id=snapshot.snapshot_id,
            path="src/app.py",
            line_start=2,
            line_end=2,
            snippet="    return sum(values)  # context_marker",
            sha256=file_hash,
            source="text",
        )
        ProjectMemory(database_path).record_user_correction(
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
            statement="calculate_total 返回输入数值的总和。",
            evidence=(evidence,),
        )
        LearnerProfileStore(database_path).save(
            LearnerProfile(
                explanation_preferences=("中文解释，先讲输入和输出",),
            )
        )

        storage = InMemorySessionStorage()
        session_config = add_context_manager(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=snapshot_root,
                system="Base coding system.",
                tools=[],
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            ),
            ContextManager(database_path),
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
        )
        session = await CodingSession.load(session_config)
        try:
            [event async for event in session.prompt("Explain context_marker")]
            [event async for event in session.continue_()]

            provider_system = provider.calls[0][1]
            assert "context_marker" in provider_system
            assert "calculate_total 返回输入数值的总和。" in provider_system
            assert "中文解释，先讲输入和输出" in provider_system
            assert session.system_prompt == "Base coding system."
            assert all(
                "sparkle_context" not in message_text(message)
                for message in session.messages
            )
            assert any(
                message_text(message) == "Explain context_marker"
                for message in provider.calls[0][2]
            )
            assert "context_marker" in provider.calls[1][1]
            assert any(
                message_text(message) == "Explain context_marker"
                for message in provider.calls[1][2]
            )
        finally:
            await session.aclose()


async def _tau_skill_and_compaction_summary_remain_in_provider_context() -> None:
    model = "context-skill-summary-model"
    provider = FakeProvider([_assistant_stream(_assistant_reply(model, "Done."))])

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot_root = root / "snapshot"
        source_path = snapshot_root / "src" / "app.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(
            "def calculate_total(values):\n    return sum(values)  # context_marker\n",
            encoding="utf-8",
        )
        file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        snapshot = RepositorySnapshot.create(
            source_commit="9" * 40,
            files=(SnapshotFile(path="src/app.py", sha256=file_hash),),
            snapshot_root=snapshot_root,
        )
        repository_id = hashlib.sha256(b"context-skill-summary").hexdigest()
        database_path = root / "sparkle.sqlite3"
        RepositoryStore(database_path).save_import(
            RepositoryImportResult(
                repository=Repository(
                    repository_id=repository_id,
                    name="context-skill-summary",
                    source_type="local",
                    source_locator="test://context-skill-summary",
                    current_snapshot_id=snapshot.snapshot_id,
                ),
                snapshot=snapshot,
            )
        )
        await SnapshotTextIndex(database_path=database_path).build(snapshot)

        resource_root = root / "tau-resources"
        skill_path = resource_root / "skills" / "explain-code" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\nname: explain-code\n"
            "description: Explain code with inputs and outputs.\n---\n"
            "Explain code with a short call-chain walkthrough.\n",
            encoding="utf-8",
        )
        storage = InMemorySessionStorage(
            (
                CompactionEntry(
                    summary="The repository contains calculate_total.",
                    first_kept_entry_id=None,
                ),
            )
        )
        session_config = add_context_manager(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=snapshot_root,
                system="Base coding system.",
                tools=[],
                resource_paths=TauResourcePaths(
                    root=resource_root,
                    agents_root=None,
                    project_resources_enabled=False,
                ),
                extensions_enabled=False,
                project_extensions_enabled=False,
            ),
            ContextManager(database_path),
            repository_id=repository_id,
            snapshot_id=snapshot.snapshot_id,
        )
        session = await CodingSession.load(session_config)
        try:
            [event async for event in session.prompt("/skill:explain-code context_marker")]

            provider_messages = provider.calls[0][2]
            message_texts = [message_text(message) for message in provider_messages]
            assert any("Previous conversation summary:" in text for text in message_texts)
            assert any(
                "Explain code with a short call-chain walkthrough." in text
                for text in message_texts
            )
            assert any("context_marker" in text for text in message_texts)
            assert "context_marker" in provider.calls[0][1]
            assert all(
                "<sparkle_context" not in text and "<code_evidence" not in text
                for text in message_texts
            )
        finally:
            await session.aclose()


async def _context_provider_failure_does_not_start_an_uncontextualized_turn() -> None:
    model = "context-provider-error-model"
    provider = FakeProvider([_assistant_stream(_assistant_reply(model, "Should not run."))])
    storage = InMemorySessionStorage()

    async def fail_to_build_context(request: TurnContextRequest) -> str:
        del request
        raise RuntimeError("Context retrieval failed")

    with TemporaryDirectory() as temp_dir:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=Path(temp_dir),
                system="Base coding system.",
                tools=[],
                turn_context_provider=fail_to_build_context,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        try:
            with pytest.raises(RuntimeError, match="Context retrieval failed"):
                [event async for event in session.prompt("Find evidence")]

            assert provider.calls == []
            assert session.messages == ()
            assert not any(
                isinstance(entry, MessageEntry)
                for entry in await storage.read_all()
            )
        finally:
            await session.aclose()


async def _cancelling_context_build_does_not_start_or_persist_a_turn() -> None:
    model = "context-provider-cancel-model"
    provider = FakeProvider([_assistant_stream(_assistant_reply(model, "Should not run."))])
    storage = InMemorySessionStorage()
    context_started = asyncio.Event()

    async def wait_for_context(request: TurnContextRequest) -> str:
        del request
        context_started.set()
        await asyncio.Future()
        return "unreachable"

    with TemporaryDirectory() as temp_dir:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=Path(temp_dir),
                system="Base coding system.",
                tools=[],
                turn_context_provider=wait_for_context,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )

        async def run_prompt() -> None:
            async for _event in session.prompt("Cancel during retrieval"):
                pass

        task = asyncio.create_task(run_prompt())
        try:
            await context_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert provider.calls == []
            assert session.messages == ()
            assert not any(
                isinstance(entry, MessageEntry)
                for entry in await storage.read_all()
            )
        finally:
            if not task.done():
                task.cancel()
            await session.aclose()


def _assistant_reply(model: str, text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="test",
        provider="fake",
        model=model,
        stop_reason="stop",
    )


def _assistant_stream(message: AssistantMessage) -> list[object]:
    return [
        AssistantStartEvent(partial=message.model_copy(deep=True)),
        AssistantDoneEvent(reason="stop", message=message),
    ]
