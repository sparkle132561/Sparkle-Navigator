"""把 Sparkle Context Manager 接到 Tau CodingSession 的每轮上下文接缝。"""

from __future__ import annotations

from dataclasses import replace

from sparkle_navigator.context.context_manager import ContextManager
from tau_agent.messages import CustomMessage, UserMessage, message_text
from tau_coding import CodingSessionConfig, TurnContextRequest
from tau_coding.context_window import COMPACTION_SUMMARY_PREFIX
from tau_coding.skills import parse_skill_invocation

_BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:"
)


def add_context_manager(
    config: CodingSessionConfig,
    context_manager: ContextManager,
    *,
    repository_id: str,
    snapshot_id: str,
) -> CodingSessionConfig:
    """返回绑定 Repository/Snapshot 的 CodingSession 配置副本。"""

    if config.turn_context_provider is not None:
        raise ValueError("CodingSession 已配置 turn_context_provider。")

    async def provide_context(request: TurnContextRequest) -> str | None:
        return await context_manager.build(
            repository_id=repository_id,
            snapshot_id=snapshot_id,
            query=_query_for_turn(request),
            token_budget=request.token_budget,
        )

    return replace(config, turn_context_provider=provide_context)


def _query_for_turn(request: TurnContextRequest) -> str | None:
    if request.prompt is not None:
        return _query_from_expanded_prompt(request.prompt)

    for message in reversed(request.messages):
        if not isinstance(message, (UserMessage, CustomMessage)):
            continue
        text = message_text(message).strip()
        if not text or text.startswith(
            (COMPACTION_SUMMARY_PREFIX, _BRANCH_SUMMARY_PREFIX)
        ):
            continue
        return _query_from_expanded_prompt(text)
    return None


def _query_from_expanded_prompt(prompt: str) -> str | None:
    invocation = parse_skill_invocation(prompt)
    if invocation is not None:
        prompt = invocation.additional_instructions or invocation.name
    normalized = prompt.strip()
    return normalized or None


__all__ = ["add_context_manager"]
