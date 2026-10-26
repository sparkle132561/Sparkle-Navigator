"""把回合运行统计与回答引用核对组装为聊天状态行。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.runtime.citation_verification import (
    extract_citations,
    format_citation_checks,
    verify_citations_async,
)
from sparkle_navigator.runtime.run_status import (
    final_answer_message,
    format_run_status,
    run_turn_stats,
)
from tau_agent.messages import AssistantMessage, TextContent

if TYPE_CHECKING:
    from sparkle_navigator.presentation.tui.session_binding import WorkspaceSessions
    from tau_coding import CodingSession


def build_turn_status_formatter(
    services: SparklePresentationServices,
    *,
    sessions_lookup: Callable[[], WorkspaceSessions | None],
) -> Callable[[Sequence[AssistantMessage], CodingSession], Awaitable[str | None]]:
    """构造状态行格式化器；按当前 Session 绑定核对回答引用。"""

    async def format_turn_status(
        messages: Sequence[AssistantMessage], session: CodingSession,
    ) -> str | None:
        lines: list[str] = []
        message_tuple = tuple(messages)
        stats = run_turn_stats(message_tuple)
        if stats is not None:
            lines.append(format_run_status(stats))
        final = final_answer_message(message_tuple)
        sessions = sessions_lookup()
        if sessions is not None and final is not None:
            binding = sessions.binding_for(session)
            answer_text = "".join(
                block.text
                for block in final.content
                if isinstance(block, TextContent)
            )
            citations = extract_citations(answer_text)
            if binding is not None and citations:
                checks = await verify_citations_async(
                    services.database_path,
                    repository_id=binding[0],
                    snapshot_id=binding[1],
                    citations=citations,
                )
                citation_line = format_citation_checks(checks)
                if citation_line is not None:
                    lines.append(citation_line)
        return "\n".join(lines) if lines else None

    return format_turn_status


__all__ = ["build_turn_status_formatter"]
