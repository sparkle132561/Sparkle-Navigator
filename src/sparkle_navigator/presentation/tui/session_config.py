"""把 Sparkle Tool Profile 接到 Tau Textual 启动配置。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sparkle_navigator.context.context_manager import ContextManager
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.context_injection import add_context_manager
from sparkle_navigator.runtime.tool_profile import ApprovalHandler, ToolProfileRuntime
from tau_coding.session import CodingSessionConfig


class SparkleSessionConfigTransform:
    """在 Tau 基础配置上注入一个 Profile 与可选 Snapshot 工具集。"""

    def __init__(
        self,
        *,
        database_path: Path,
        snapshot_id: str | None,
        repository_trusted: bool,
        approval_handler: ApprovalHandler | None,
        context_repository_id: str | None = None,
    ) -> None:
        self._database_path = database_path.expanduser().resolve()
        self._snapshot_id = snapshot_id
        self._repository_trusted = repository_trusted
        self._approval_handler = approval_handler
        self._context_repository_id = context_repository_id
        self.tool_profile_runtime: ToolProfileRuntime | None = None
        self.base_config: CodingSessionConfig | None = None

    def __call__(self, config: CodingSessionConfig) -> CodingSessionConfig:
        """保留 Tau 默认工具，只附加显式的 Sparkle Profile 与读工具。"""

        if config.tool_access_policy is not None:
            raise ValueError(
                "SparkleSessionConfigTransform 不会覆盖已有 ToolAccessPolicy。"
            )
        self.base_config = config
        profile_runtime = ToolProfileRuntime(
            approver=self._approval_handler,
            repository_trusted=self._repository_trusted,
            workspace_root=config.cwd,
        )
        transformed = replace(config, tool_access_policy=profile_runtime)
        if self._snapshot_id is not None:
            transformed = add_codebase_tools(
                transformed,
                database_path=self._database_path,
                snapshot_id=self._snapshot_id,
            )
        if self._context_repository_id is not None and self._snapshot_id is not None:
            # 实验开关（默认关闭）：为每轮注入 Snapshot 证据与项目事实，
            # 质量收益待阶段三验证；不改变 Session 消息历史。
            transformed = add_context_manager(
                transformed,
                ContextManager(self._database_path),
                repository_id=self._context_repository_id,
                snapshot_id=self._snapshot_id,
            )
        self.tool_profile_runtime = profile_runtime
        return transformed


__all__ = ["SparkleSessionConfigTransform"]
