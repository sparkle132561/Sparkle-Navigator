"""Compaction/Memory 指标通过 Tau CodingSession 的真实压缩入口验证。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sparkle_navigator.context.context_manager import ContextManager
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.evaluation.compaction_memory import (
    CompactionMemoryCase,
    CompactionMemoryEvaluator,
)
from sparkle_navigator.memory import ProjectMemory
from sparkle_navigator.persistence.repository_store import RepositoryStore
from tau_agent.messages import AssistantMessage, TextContent, UserMessage, message_text
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import CompactionEntry, InMemorySessionStorage, MessageEntry
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig


def test_evaluator_scores_the_context_replayed_after_tau_compaction() -> None:
    asyncio.run(_evaluator_scores_the_context_replayed_after_tau_compaction())


def test_failed_compaction_does_not_append_a_summary_entry() -> None:
    asyncio.run(_failed_compaction_does_not_append_a_summary_entry())


async def _evaluator_scores_the_context_replayed_after_tau_compaction() -> None:
    model = "compaction-memory-evaluation-model"
    summary = _assistant_message(
        model,
        "Summary: CRITICAL_FACT_KEEP remained important. "
        "OLD_SNAPSHOT_FACT_LEAK was repeated by the summarizer.",
    )
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=summary.model_copy(deep=True)),
                AssistantDoneEvent(reason="stop", message=summary),
            ]
        ]
    )

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        database_path = root / "sparkle.sqlite3"
        repository_id = hashlib.sha256(b"compaction-memory-evaluation").hexdigest()
        old_snapshot = _snapshot(
            root / "old-snapshot",
            "a" * 40,
            "def old_api():\n    return 'old'\n",
        )
        repository_store = RepositoryStore(database_path)
        repository_store.save_import(
            RepositoryImportResult(
                repository=_repository(repository_id, old_snapshot.snapshot_id),
                snapshot=old_snapshot,
            )
        )
        old_evidence = CodeEvidence(
            snapshot_id=old_snapshot.snapshot_id,
            path="src/app.py",
            line_start=1,
            line_end=1,
            snippet="def old_api():",
            sha256=old_snapshot.files[0].sha256,
            source="symbol",
        )
        memory = ProjectMemory(database_path)
        memory.record_user_correction(
            repository_id=repository_id,
            snapshot_id=old_snapshot.snapshot_id,
            statement="OLD_SNAPSHOT_FACT_LEAK is not valid in the new snapshot.",
            evidence=(old_evidence,),
        )

        current_snapshot = _snapshot(
            root / "current-snapshot",
            "b" * 40,
            "def current_api():\n    return 'current'\n",
        )
        repository_store.save_import(
            RepositoryImportResult(
                repository=_repository(repository_id, current_snapshot.snapshot_id),
                snapshot=current_snapshot,
            )
        )
        old_facts = memory.facts_for_snapshot(
            repository_id=repository_id,
            snapshot_id=old_snapshot.snapshot_id,
        )
        assert old_facts[0].status.value == "STALE"

        current_sparkle_context = await ContextManager(database_path).build(
            repository_id=repository_id,
            snapshot_id=current_snapshot.snapshot_id,
            query=None,
            token_budget=512,
        )
        assert current_sparkle_context is None

        old_history = MessageEntry(
            message=UserMessage(
                content=(
                    "Prior discussion: CRITICAL_FACT_KEEP must survive. "
                    "OLD_SNAPSHOT_FACT_LEAK was true only in the old snapshot."
                )
            )
        )
        recent_history = MessageEntry(
            parent_id=old_history.id,
            message=UserMessage(content="Recent suffix " + ("x" * 90_000)),
        )
        storage = InMemorySessionStorage((old_history, recent_history))
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=current_snapshot.snapshot_root,
                system="Summarize without inventing facts.",
                tools=[],
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        case = CompactionMemoryCase(
            case_id="hidden-evaluation-case-13",
            critical_fact_anchors=("CRITICAL_FACT_KEEP",),
            stale_fact_anchors=("OLD_SNAPSHOT_FACT_LEAK",),
        )
        try:
            before_context = "\n".join(message_text(message) for message in session.messages)
            compaction = await session.compact_detailed()
            after_context = "\n".join(message_text(message) for message in session.messages)
            if current_sparkle_context:
                after_context = f"{after_context}\n{current_sparkle_context}"

            result = CompactionMemoryEvaluator().evaluate(
                case,
                before_context=before_context,
                after_context=after_context,
            )

            assert compaction.replaced_entry_count == 1
            assert result.critical_fact_retention == 1.0
            assert result.stale_fact_leakage == 1.0
            assert "hidden-evaluation-case-13" not in message_text(provider.calls[0][2][0])
            assert any(isinstance(entry, CompactionEntry) for entry in await storage.read_all())
        finally:
            await session.aclose()


async def _failed_compaction_does_not_append_a_summary_entry() -> None:
    model = "compaction-memory-failure-model"
    provider = FakeProvider([[]])
    with TemporaryDirectory() as temp_dir:
        first_message = MessageEntry(
            message=UserMessage(content="Critical fact: CRITICAL_FACT_KEEP."),
        )
        recent_message = MessageEntry(
            parent_id=first_message.id,
            message=UserMessage(content="recent suffix " + ("x" * 90_000)),
        )
        storage = InMemorySessionStorage((first_message, recent_message))
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=model,
                storage=storage,
                cwd=Path(temp_dir),
                system="Compaction failure test.",
                tools=[],
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        before_messages = session.messages
        before_entry_ids = tuple(entry.id for entry in await storage.read_all())
        try:
            with pytest.raises(RuntimeError, match="empty summary"):
                await session.compact_detailed()

            after_entries = await storage.read_all()
            assert tuple(entry.id for entry in after_entries) == before_entry_ids
            assert not any(isinstance(entry, CompactionEntry) for entry in after_entries)
            assert session.messages == before_messages
        finally:
            await session.aclose()


def _assistant_message(model: str, text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="test",
        provider="fake",
        model=model,
        stop_reason="stop",
    )


def _snapshot(root: Path, commit: str, source: str) -> RepositorySnapshot:
    source_path = root / "src" / "app.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    return RepositorySnapshot.create(
        source_commit=commit,
        files=(
            SnapshotFile(
                path="src/app.py",
                sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            ),
        ),
        snapshot_root=root,
    )


def _repository(repository_id: str, snapshot_id: str) -> Repository:
    return Repository(
        repository_id=repository_id,
        name="compaction-memory-evaluation",
        source_type="local",
        source_locator="test://compaction-memory-evaluation",
        current_snapshot_id=snapshot_id,
    )
