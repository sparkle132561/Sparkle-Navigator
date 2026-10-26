"""保留各 Repository Snapshot 的独立聊天 Session。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from sparkle_navigator.domain.repository import Repository
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.presentation.tui.session_config import SparkleSessionConfigTransform
from sparkle_navigator.runtime.tool_profile import ApprovalHandler, ToolProfileRuntime
from tau_agent.session import CustomEntry, JsonlSessionStorage, SessionState
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.session_manager import CodingSessionRecord
from tau_coding.session_preparation import prepare_coding_session
from tau_coding.tui.product import TuiProductContext


class WorkspaceSessions:
    """候选 Session 就绪后才切换宿主，缓存按 Repository 与 Snapshot 隔离。"""

    def __init__(
        self,
        *,
        services: SparklePresentationServices,
        base_config: CodingSessionConfig,
        approval_handler: ApprovalHandler | None,
        initial_binding: tuple[str, str] | None = None,
        context_wiring: bool = False,
    ) -> None:
        self._services = services
        self._base_config = base_config
        self._approval_handler = approval_handler
        self._initial_binding = initial_binding
        self._context_wiring = context_wiring
        self._sessions: dict[tuple[str, str], CodingSession] = {}

    async def bind_initial(self, context: TuiProductContext) -> None:
        if self._initial_binding is not None:
            recorded = _recorded_binding(context.session.custom_entries)
            if recorded is not None and recorded != self._initial_binding:
                raise ValueError("启动绑定与 Session 持久记录不同。")
            self._sessions.setdefault(self._initial_binding, context.session)
            await self.revalidate_trust(context.session)
            if recorded is None:
                await context.session.append_custom_entry(
                    "sparkle.repository-binding",
                    {
                        "repository_id": self._initial_binding[0],
                        "snapshot_id": self._initial_binding[1],
                    },
                )

    def binding_for(self, session: CodingSession) -> tuple[str, str] | None:
        return _recorded_binding(session.custom_entries) or next(
            (key for key, value in self._sessions.items() if value is session),
            None,
        )

    @staticmethod
    async def restore_binding(
        services: SparklePresentationServices,
        path: Path,
        cwd: Path,
    ) -> tuple[Repository, str] | None:
        """从 Tau 活跃分支恢复固定 Snapshot，兼容旧受管目录中的会话。"""
        state = SessionState.from_entries(await JsonlSessionStorage(path).read_all())
        binding = _recorded_binding(state.custom_entries)
        if binding is None:
            try:
                parts = cwd.resolve().relative_to(services.working_copies_root).parts
            except ValueError:
                return None
            if len(parts) != 2:
                return None
            binding = parts[0], parts[1]
        repository = await _validate_binding(services, binding, cwd)
        return repository, binding[1]

    async def revalidate_trust(
        self,
        session: CodingSession,
        destination: CodingSessionRecord | None = None,
    ) -> None:
        """所有激活入口都重新读取仓库信任，包括宿主原生的 Session 恢复。"""
        binding = self.binding_for(session)
        if destination is not None:
            target = await self.restore_binding(
                self._services,
                destination.path,
                destination.cwd,
            )
            target_binding = (target[0].repository_id, target[1]) if target is not None else None
            if target_binding != binding:
                raise ValueError(
                    "目标 Session 的 Snapshot 不同，请用 "
                    f"sparkle --session {destination.id} 恢复完整绑定。"
                )
        if binding is None:
            return
        policy = session.tool_access_policy
        if isinstance(policy, ToolProfileRuntime):
            policy.set_repository_trusted(session, trusted=False)
        repository = await _validate_binding(self._services, binding, session.cwd)
        expected = (self._services.working_copies_root / binding[0] / binding[1]).resolve()
        if isinstance(policy, ToolProfileRuntime):
            policy.set_repository_trusted(
                session,
                trusted=repository.trusted and session.cwd.resolve() == expected,
            )

    async def activate(self, repository_id: str, context: TuiProductContext) -> None:
        """恢复已有绑定，或为指定 Snapshot 创建全新的工具、历史与权限。"""
        if context.is_busy:
            raise ValueError("Agent 正在运行，请先取消或等待完成后切换 Repository。")
        repository = await self._services.get_repository(repository_id)
        if repository.current_snapshot_id is None:
            raise ValueError("所选 Repository 尚无 Snapshot。")
        key = (repository_id, repository.current_snapshot_id)
        cached = self._sessions.get(key)
        if cached is not None:
            await context.replace_session(cached)
            return

        working_copy = await self._services.open_working_copy(repository_id)
        if working_copy.snapshot_id != key[1]:
            raise ValueError("准备期间 Snapshot 已变化，请重新选择 Repository。")
        current = context.session
        base = self._base_config
        session_id = uuid4().hex
        manager = base.session_manager
        record = (
            manager.prepare_session(
                cwd=working_copy.root,
                model=current.model,
                provider_name=current.provider_name,
            )
            if manager is not None
            else None
        )
        # 配置型 Provider 由 Tau 重新解析；显式注入的 Provider 保持由原 Session 拥有。
        config = replace(
            base,
            provider=None if base.provider_settings is not None else current.provider,
            owns_initial_provider=False,
            cwd=working_copy.root,
            model=current.model,
            provider_name=current.provider_name,
            requested_provider=current.provider_name
            if base.provider_settings is not None
            else None,
            requested_model=current.model if base.provider_settings is not None else None,
            runtime_provider_config=current.provider_config(current.provider_name),
            storage=JsonlSessionStorage(
                record.path
                if record is not None
                else self._services.database_path.parent / "tau-sessions" / f"{session_id}.jsonl"
            ),
            session_id=record.id if record is not None else session_id,
            session_provider_name=current.provider_name,
            index_on_first_persist=manager is not None,
            tool_access_policy=None,
            additional_tool_registrations=(),
            turn_context_provider=None,
            context_files=(),
            resource_paths=None,
            extension_runtime=None,
            project_trust_coordinator=None,
            trust_override="decline",
            trust_default="never",
            trust_interactive=False,
            trust_prompt=None,
        )
        transform = SparkleSessionConfigTransform(
            database_path=self._services.database_path,
            snapshot_id=key[1],
            repository_trusted=repository.trusted,
            approval_handler=self._approval_handler,
            context_repository_id=key[0] if self._context_wiring else None,
        )
        prepared = await prepare_coding_session(transform(config))
        adopted = False
        try:
            await prepared.session.append_custom_entry(
                "sparkle.repository-binding",
                {"repository_id": key[0], "snapshot_id": key[1]},
            )
            if context.is_busy:
                raise ValueError("Agent 正在运行，请先取消或等待完成后切换 Repository。")
            candidate = await prepared.adopt()
            adopted = True
            if manager is not None:
                await candidate.set_session_name(f"{repository.name} · {key[1][:8]}")
            await context.replace_session(candidate)
            self._sessions[key] = candidate
        except BaseException:
            if adopted:
                await prepared.session.aclose()
            else:
                await prepared.abort()
            raise


__all__ = ["WorkspaceSessions"]


def _recorded_binding(entries: Sequence[CustomEntry]) -> tuple[str, str] | None:
    for entry in reversed(entries):
        if entry.namespace == "sparkle.repository-binding":
            repository_id = entry.data.get("repository_id")
            snapshot_id = entry.data.get("snapshot_id")
            if not isinstance(repository_id, str) or not isinstance(snapshot_id, str):
                raise ValueError("Session Repository/Snapshot 绑定记录无效。")
            return repository_id, snapshot_id
    return None


async def _validate_binding(
    services: SparklePresentationServices,
    binding: tuple[str, str],
    cwd: Path,
) -> Repository:
    repository = await services.get_repository(binding[0])
    await services.get_repository_snapshot(*binding)
    expected = (services.working_copies_root / binding[0] / binding[1]).resolve()
    source = (
        Path(repository.source_locator).resolve() if repository.source_type == "local" else None
    )
    if cwd.resolve() != expected and cwd.resolve() != source:
        raise ValueError("Session 工作目录与持久化 Repository/Snapshot 绑定不同。")
    return repository
