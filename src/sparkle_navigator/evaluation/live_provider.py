"""为单个 AgentRun 限制 Provider 调用次数和单次等待时间。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from dataclasses import replace
from os import environ
from pathlib import Path

from tau_agent.messages import AgentMessage
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.provider_events import AssistantMessageEvent
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentTool
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding import CodingSessionConfig
from tau_coding.credentials import FileCredentialStore
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    load_provider_settings,
    openai_compatible_config_from_provider,
)

from sparkle_navigator.domain.evaluation import AgentRunInput
from sparkle_navigator.evaluation.native_backend import CodingSessionConfigFactory


LIVE_MAX_AGENT_RUNS = 30
LIVE_MAX_PROVIDER_REQUESTS_PER_RUN = 20
LIVE_PROVIDER_TIMEOUT_SECONDS = 45
LIVE_PROVIDER_MAX_RETRIES = 0
LIVE_PROVIDER_MAX_TOKENS = 2048


class ProviderRequestLimitExceeded(RuntimeError):
    """当前 AgentRun 已达到 Provider 请求上限。"""


class BoundedModelProvider:
    """在 Tau Provider seam 上执行单 Run 请求数与响应超时限制。"""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_requests: int,
        timeout_seconds: float,
    ) -> None:
        if (
            not isinstance(max_requests, int)
            or isinstance(max_requests, bool)
            or max_requests < 1
        ):
            raise ValueError("max_requests 必须是正整数。")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须是有限正数。")
        self._provider = provider
        self._max_requests = max_requests
        self._timeout_seconds = timeout_seconds
        self._request_count = 0

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        async def stream() -> AsyncIterator[AssistantMessageEvent]:
            if self._request_count >= self._max_requests:
                raise ProviderRequestLimitExceeded(
                    "AgentRun 已达到 Provider 请求数上限。"
                )
            self._request_count += 1
            async with asyncio.timeout(self._timeout_seconds):
                async for event in self._provider.stream_response(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    signal=signal,
                    session_id=session_id,
                ):
                    yield event

        return stream()

    async def aclose(self) -> None:
        close_provider = getattr(self._provider, "aclose", None)
        if close_provider is not None:
            await close_provider()


def create_tau_live_session_config_factory(
    database_path: Path,
) -> tuple[CodingSessionConfigFactory, str, str, str]:
    """从 Tau 本地 Provider 设置与凭据库创建单 Run 配置工厂。"""

    settings = load_provider_settings()
    provider_config = settings.get_provider()
    if not isinstance(provider_config, OpenAICompatibleProviderConfig):
        raise RuntimeError("Tau 当前默认 Provider 不是 OpenAI-compatible 配置。")
    model = provider_config.default_model
    credential_store = FileCredentialStore()
    api_key = (
        credential_store.get(provider_config.credential_name)
        if provider_config.credential_name
        else None
    ) or environ.get(provider_config.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Tau Provider「{provider_config.name}」没有本地凭据。"
        )
    runtime_config = openai_compatible_config_from_provider(
        provider_config,
        credential_reader=credential_store,
        model=model,
    )

    def create_config(agent_input: AgentRunInput) -> CodingSessionConfig:
        provider = OpenAICompatibleProvider(
            replace(
                runtime_config,
                timeout_seconds=LIVE_PROVIDER_TIMEOUT_SECONDS,
                max_retries=LIVE_PROVIDER_MAX_RETRIES,
                max_tokens=LIVE_PROVIDER_MAX_TOKENS,
            )
        )
        return CodingSessionConfig(
            provider=provider,
            model=model,
            storage=JsonlSessionStorage(
                database_path.parent
                / "tau-sessions"
                / f"{agent_input.run_id}.jsonl"
            ),
            cwd=database_path.parent,
            system=(
                "使用 Sparkle Snapshot 只读工具查找并核实代码证据。"
                "不得修改仓库文件；不得假设未检索到的实现。"
            ),
            provider_name=provider_config.name,
            owns_initial_provider=True,
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
            trust_default="deny",
            trust_interactive=False,
        )

    return create_config, provider_config.name, model, api_key


__all__ = [
    "BoundedModelProvider",
    "LIVE_MAX_AGENT_RUNS",
    "LIVE_MAX_PROVIDER_REQUESTS_PER_RUN",
    "LIVE_PROVIDER_MAX_RETRIES",
    "LIVE_PROVIDER_MAX_TOKENS",
    "LIVE_PROVIDER_TIMEOUT_SECONDS",
    "ProviderRequestLimitExceeded",
    "create_tau_live_session_config_factory",
]
