"""第 1 天：加载 Tau 持久化 coding session 的运行时边界。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tau_agent.provider import ModelProvider
    from tau_agent.session import SessionStorage
    from tau_agent.tools import AgentTool
    from tau_coding.session import CodingSession


TAU_VERSION = "0.4.4"


@dataclass(frozen=True, slots=True)
class TauSessionRequest:
    """构造一个 Tau ``CodingSession`` 所需的输入。"""

    cwd: Path
    model: str
    provider: ModelProvider
    storage: SessionStorage
    session_id: str | None = None
    tools: tuple[AgentTool, ...] = ()


class TauRuntimeAdapter:
    """Sparkle Navigator 与 Tau 之间第 1 天唯一的构造边界。

    Provider、storage 和 tools 由外部注入，因此此适配器不拥有凭据、产品持久化
    或后续功能模块。事件标准化有意留到下一阶段；调用方通过 ``prompt`` 接收
    Tau 原生的 ``CodingSession`` 事件流。
    """

    async def open_session(self, request: TauSessionRequest) -> CodingSession:
        """使用固定版本的 Tau API 加载一个持久化 session。"""

        from tau_coding.session import CodingSession, CodingSessionConfig

        config = CodingSessionConfig(
            provider=request.provider,
            model=request.model,
            storage=request.storage,
            cwd=request.cwd,
            session_id=request.session_id,
            tools=list(request.tools),
        )
        return await CodingSession.load(config)
