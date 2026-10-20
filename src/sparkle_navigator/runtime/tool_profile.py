"""把 Sparkle Tool Profile 接入 Tau CodingSession 的通用工具 seam。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from tau_agent.messages import ToolCall
from tau_agent.session.entries import CustomEntry
from tau_agent.tools import AgentTool
from tau_agent.types import JSONValue
from tau_coding.tools import (
    CodingToolRegistration,
    apply_edits_to_normalized_content,
    generate_unified_patch,
    normalize_to_lf,
)

from sparkle_navigator.domain.tool_profile import (
    ToolAuthorizationRequest,
    ToolDecision,
    ToolProfile,
    ToolProfilePolicy,
    ToolRegistration,
    ToolRisk,
)

TOOL_PROFILE_NAMESPACE = "sparkle.navigator.tool-profile"
_TOOL_PROFILE_VERSION = 1


class ApprovalOutcome(StrEnum):
    """用户对这一次审批请求的明确结果。"""

    APPROVED = "approved"
    DENIED = "denied"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    """交给 TUI 或测试审批器的单次、完整审批材料。"""

    call_id: str
    registration_id: str
    tool_name: str
    source: str
    arguments: Mapping[str, JSONValue]
    risk: ToolRisk
    working_directory: Path
    target_path: Path | None = None
    diff_preview: str | None = None
    original_sha256: str | None = None
    command: str | None = None
    risk_level: str | None = None


@dataclass(frozen=True, slots=True)
class _AuthorizationReceipt:
    """一次性执行凭证，仅供 Tau 扩展参数 hook 后的完整性校验。"""

    tool_name: str
    registration_id: str
    profile: ToolProfile
    arguments_fingerprint: str
    approval_request: ToolApprovalRequest | None


ApprovalHandler = Callable[[ToolApprovalRequest], Awaitable[ApprovalOutcome]]


class ProfileSessionControl(Protocol):
    """Tool Profile 变更需要的 Tau Session 最小接口。"""

    @property
    def is_running(self) -> bool: ...

    @property
    def custom_entries(self) -> Sequence[CustomEntry]: ...

    @property
    def visible_tool_names(self) -> tuple[str, ...]: ...

    @property
    def tool_access_policy(self) -> object | None: ...

    def set_visible_tools(self, tool_names: Sequence[str]) -> None: ...

    async def append_custom_entry(
        self,
        namespace: str,
        data: dict[str, JSONValue],
    ) -> None: ...


class ToolProfileRuntime:
    """筛选 Provider 工具并在 Tau Agent Loop 执行前进行授权。"""

    def __init__(
        self,
        *,
        approver: ApprovalHandler | None = None,
        policy: ToolProfilePolicy | None = None,
        repository_trusted: bool | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._approver = approver
        self._policy = policy or ToolProfilePolicy()
        self._authorized_calls: dict[str, _AuthorizationReceipt] = {}
        self._registrations: tuple[CodingToolRegistration, ...] = ()
        self._repository_trusted = repository_trusted
        self._workspace_root = workspace_root

    def filter_tools(
        self,
        registrations: Sequence[CodingToolRegistration],
        session_entries: Sequence[CustomEntry],
    ) -> tuple[AgentTool, ...]:
        """按已恢复的 Profile 返回 Provider 可见的 Tau AgentTool。"""

        self._registrations = tuple(registrations)
        profile = _profile_from_entries(session_entries)
        domain_registrations = _domain_registrations(registrations)
        visible_registrations = self._policy.visible_tools(profile, domain_registrations)
        if self._repository_trusted is False:
            visible_registrations = tuple(
                registration
                for registration in visible_registrations
                if registration.risk is ToolRisk.READ_ONLY
            )
        visible_ids = {registration.identity for registration in visible_registrations}
        return tuple(
            registration.tool
            for registration in registrations
            if registration.identity in visible_ids
        )

    async def before_tool_call(
        self,
        call: ToolCall,
        registrations: Sequence[CodingToolRegistration],
        session_entries: Sequence[CustomEntry],
        *,
        cwd: Path,
        shell_command_prefix: str | None,
    ) -> tuple[bool, str | None]:
        """在真实工具 executor 运行前返回 Tau 所需的阻止结果。"""

        self._authorized_calls.pop(call.id, None)
        matches = [item for item in registrations if item.tool.name == call.name]
        if len(matches) != 1:
            return True, "工具没有唯一、有效的当前注册记录。"

        registration = matches[0]
        profile = _profile_from_entries(session_entries)
        try:
            arguments_fingerprint = _arguments_fingerprint(call.arguments)
        except (TypeError, ValueError):
            return True, "工具参数无法稳定校验，调用已拒绝。"
        domain_registration = _domain_registration(registration)
        if (
            self._repository_trusted is False
            and domain_registration.risk is ToolRisk.SIDE_EFFECT
        ):
            return True, "未信任的 Repository 不允许执行副作用工具。"
        if _requires_workspace_path(registration):
            path_error = _workspace_path_error(
                call.arguments,
                workspace_root=self._workspace_root,
                repository_trusted=self._repository_trusted,
            )
            if path_error is not None:
                return True, path_error
        decision = self._policy.decide(
            ToolAuthorizationRequest(
                profile=profile,
                registration=domain_registration,
                call_id=call.id,
                arguments=call.arguments,
            )
        )
        if decision.decision is ToolDecision.ALLOW:
            self._remember_authorization(
                call.id,
                _AuthorizationReceipt(
                    tool_name=call.name,
                    registration_id=registration.identity,
                    profile=profile,
                    arguments_fingerprint=arguments_fingerprint,
                    approval_request=None,
                ),
            )
            return False, None
        if decision.decision is ToolDecision.DENY:
            return True, decision.reason
        if self._approver is None:
            return True, "没有可用的审批器，副作用调用已拒绝。"

        try:
            request = _approval_request(
                registration,
                call,
                cwd=cwd,
                shell_command_prefix=shell_command_prefix,
            )
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            return True, f"无法安全生成审批材料，工具调用已拒绝：{exc}"
        if request is None:
            return True, "审批所需的差异预览或风险信息不可用，工具调用已拒绝。"

        try:
            arguments_fingerprint = _arguments_fingerprint(request.arguments)
        except (TypeError, ValueError):
            return True, "审批参数无法稳定校验，工具调用已拒绝。"

        try:
            outcome = await self._approver(request)
        except Exception as exc:  # 审批失败时保持关闭状态。
            return True, f"审批失败，工具调用已拒绝：{type(exc).__name__}"
        if outcome is not ApprovalOutcome.APPROVED:
            return True, "用户未批准这次工具调用。"
        try:
            approved_arguments_fingerprint = _arguments_fingerprint(request.arguments)
        except (TypeError, ValueError):
            return True, "审批期间调用参数无法继续校验，工具调用已拒绝。"
        if approved_arguments_fingerprint != arguments_fingerprint:
            return True, "审批期间调用参数发生变化，工具调用已拒绝。"
        if not _approval_snapshot_is_current(request):
            return True, "审批后目标文件已变化，工具调用已拒绝。"
        self._remember_authorization(
            call.id,
            _AuthorizationReceipt(
                tool_name=call.name,
                registration_id=registration.identity,
                profile=profile,
                arguments_fingerprint=arguments_fingerprint,
                approval_request=request,
            ),
        )
        return False, None

    async def validate_tool_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        original_arguments: Mapping[str, JSONValue],
        effective_arguments: Mapping[str, JSONValue],
        registrations: Sequence[CodingToolRegistration],
        session_entries: Sequence[CustomEntry],
        *,
        cwd: Path,
        shell_command_prefix: str | None,
    ) -> tuple[bool, str | None]:
        """消费一次性凭证，确保扩展 hook 没改动获批调用。"""

        receipt = self._authorized_calls.pop(tool_call_id, None)
        if receipt is None:
            return True, "缺少当前 Tool Profile 的一次性执行凭证。"
        if receipt.tool_name != tool_name:
            return True, "执行工具与审批工具不一致。"
        if _profile_from_entries(session_entries) is not receipt.profile:
            return True, "执行期间 Tool Profile 已变化，调用已拒绝。"

        matches = [item for item in registrations if item.tool.name == tool_name]
        if len(matches) != 1 or matches[0].identity != receipt.registration_id:
            return True, "执行工具注册来源已变化，调用已拒绝。"
        if _requires_workspace_path(matches[0]):
            for arguments in (original_arguments, effective_arguments):
                path_error = _workspace_path_error(
                    arguments,
                    workspace_root=self._workspace_root,
                    repository_trusted=self._repository_trusted,
                )
                if path_error is not None:
                    return True, path_error
        try:
            original_fingerprint = _arguments_fingerprint(original_arguments)
            effective_fingerprint = _arguments_fingerprint(effective_arguments)
        except (TypeError, ValueError):
            return True, "扩展 hook 后的工具参数无法校验，调用已拒绝。"
        if (
            original_fingerprint != receipt.arguments_fingerprint
            or effective_fingerprint != receipt.arguments_fingerprint
        ):
            return True, "扩展 hook 改写了已授权参数，调用已拒绝。"
        request = receipt.approval_request
        if request is not None:
            try:
                current_directory = cwd.resolve(strict=True)
            except (OSError, RuntimeError):
                return True, "执行前工作目录不可验证，调用已拒绝。"
            if request.working_directory != current_directory:
                return True, "执行前工作目录与审批时不一致，调用已拒绝。"
            if request.command is not None:
                command = original_arguments.get("command")
                if not isinstance(command, str):
                    return True, "执行前 Shell 命令不可验证，调用已拒绝。"
                expected_command = (
                    f"{shell_command_prefix.strip()}\n{command}"
                    if shell_command_prefix and shell_command_prefix.strip()
                    else command
                )
                if request.command != expected_command:
                    return True, "执行前 Shell 命令与审批内容不一致，调用已拒绝。"
            if not _approval_snapshot_is_current(request):
                return True, "执行前目标文件或批准范围已变化，调用已拒绝。"
        return False, None

    def _remember_authorization(self, call_id: str, receipt: _AuthorizationReceipt) -> None:
        self._authorized_calls[call_id] = receipt
        while len(self._authorized_calls) > 256:
            oldest_call_id = next(iter(self._authorized_calls))
            self._authorized_calls.pop(oldest_call_id, None)

    async def set_profile(
        self,
        session: ProfileSessionControl,
        profile: ToolProfile,
    ) -> None:
        """更新当前 Session 的工具表并把 Profile 追加到 Tau Session。"""

        if session.tool_access_policy is not self:
            raise ValueError("CodingSession 未配置此 ToolProfileRuntime")
        if profile is ToolProfile.CODING and self._repository_trusted is False:
            raise PermissionError("未信任的 Repository 不能开启 Coding Profile。")
        if session.is_running:
            raise RuntimeError("Tool Profile 只能在 Agent 空闲时切换")

        current = _profile_from_entries(session.custom_entries)
        if current is profile:
            return

        previous_tools = session.visible_tool_names
        selected_tools = self.filter_tools(
            self._registrations,
            _entries_with_profile(session.custom_entries, profile),
        )
        selected_tool_names = tuple(tool.name for tool in selected_tools)
        session.set_visible_tools(selected_tool_names)
        try:
            await session.append_custom_entry(
                TOOL_PROFILE_NAMESPACE,
                {"version": _TOOL_PROFILE_VERSION, "profile": profile.value},
            )
        except BaseException:
            session.set_visible_tools(previous_tools)
            raise

    def current_profile(self, session: ProfileSessionControl) -> ToolProfile:
        """读取 Tau Session 中已经持久化的当前 Profile。"""

        if session.tool_access_policy is not self:
            raise ValueError("CodingSession 未配置此 ToolProfileRuntime")
        return _profile_from_entries(session.custom_entries)

    def set_repository_trusted(
        self,
        session: ProfileSessionControl,
        *,
        trusted: bool,
    ) -> None:
        """更新当前 Session 的信任门，并按原 Profile 重新筛选工具。"""

        if session.tool_access_policy is not self:
            raise ValueError("CodingSession 未配置此 ToolProfileRuntime")
        if not isinstance(trusted, bool):
            raise TypeError("trusted 必须是 bool。")
        if session.is_running:
            raise RuntimeError("Repository 信任状态只能在 Agent 空闲时切换")

        previous_trust = self._repository_trusted
        previous_tools = session.visible_tool_names
        self._repository_trusted = trusted
        try:
            selected_tools = self.filter_tools(
                self._registrations,
                session.custom_entries,
            )
            session.set_visible_tools(tuple(tool.name for tool in selected_tools))
        except BaseException:
            self._repository_trusted = previous_trust
            session.set_visible_tools(previous_tools)
            raise


def _profile_from_entries(entries: Sequence[CustomEntry]) -> ToolProfile:
    for entry in reversed(entries):
        if entry.namespace != TOOL_PROFILE_NAMESPACE:
            continue
        if entry.data.get("version") != _TOOL_PROFILE_VERSION:
            return ToolProfile.EXPLAIN
        value = entry.data.get("profile")
        try:
            return ToolProfile(value)
        except (TypeError, ValueError):
            return ToolProfile.EXPLAIN
    return ToolProfile.EXPLAIN


def _entries_with_profile(
    entries: Sequence[CustomEntry],
    profile: ToolProfile,
) -> tuple[CustomEntry, ...]:
    staged = CustomEntry(
        namespace=TOOL_PROFILE_NAMESPACE,
        data={"version": _TOOL_PROFILE_VERSION, "profile": profile.value},
    )
    return (*entries, staged)


def _domain_registrations(
    registrations: Sequence[CodingToolRegistration],
) -> tuple[ToolRegistration, ...]:
    return tuple(_domain_registration(item) for item in registrations)


def _domain_registration(registration: CodingToolRegistration) -> ToolRegistration:
    if registration.capabilities & {
        "side_effect",
        "file_write",
        "file_edit",
        "shell",
    }:
        risk = ToolRisk.SIDE_EFFECT
    elif registration.capabilities & {"read_only", "file_read"}:
        risk = ToolRisk.READ_ONLY
    else:
        risk = ToolRisk.UNKNOWN
    return ToolRegistration(
        identity=registration.identity,
        tool_name=registration.tool.name,
        source=registration.source,
        risk=risk,
    )


def _requires_workspace_path(registration: CodingToolRegistration) -> bool:
    return bool(
        registration.capabilities & {"file_read", "file_write", "file_edit"}
    )


def _workspace_path_error(
    arguments: Mapping[str, JSONValue],
    *,
    workspace_root: Path | None,
    repository_trusted: bool | None,
) -> str | None:
    repository_bound = repository_trusted is not None or workspace_root is not None
    if not repository_bound:
        return None
    if workspace_root is None:
        return "缺少 Repository Working Copy 根目录，文件工具已拒绝。"
    try:
        root = workspace_root.resolve(strict=True)
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return "文件工具路径无效，调用已拒绝。"
        target = Path(raw_path).expanduser()
        if not target.is_absolute():
            target = root / target
        resolved_target = target.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return "无法安全解析工作副本中的文件路径，调用已拒绝。"
    if not resolved_target.is_relative_to(root):
        return "文件路径超出 Repository Working Copy 根目录，调用已拒绝。"
    return None


def _approval_request(
    registration: CodingToolRegistration,
    call: ToolCall,
    *,
    cwd: Path,
    shell_command_prefix: str | None,
) -> ToolApprovalRequest | None:
    working_directory = cwd.resolve(strict=True)
    arguments: Mapping[str, JSONValue] = deepcopy(dict(call.arguments))
    domain_registration = _domain_registration(registration)

    has_file_write = "file_write" in registration.capabilities
    has_file_edit = "file_edit" in registration.capabilities
    if has_file_write and has_file_edit:
        return None
    if has_file_write or has_file_edit:
        target, original_text, original_hash = _file_snapshot(arguments, working_directory)
        diff_base = original_text
        if has_file_write:
            content = arguments.get("content")
            if not isinstance(content, str):
                return None
            updated_text = content
        else:
            raw_edits = arguments.get("edits")
            if not isinstance(raw_edits, list) or not raw_edits:
                return None
            edits: list[dict[str, str]] = []
            for item in raw_edits:
                if not isinstance(item, dict):
                    return None
                old_text = item.get("oldText")
                new_text = item.get("newText")
                if not isinstance(old_text, str) or not isinstance(new_text, str):
                    return None
                edits.append({"oldText": old_text, "newText": new_text})
            diff_base = normalize_to_lf(original_text)
            _base_text, updated_text = apply_edits_to_normalized_content(
                diff_base,
                edits,
                str(target),
            )
        diff_preview = generate_unified_patch(str(target), diff_base, updated_text)
        if not diff_preview:
            return None
        return ToolApprovalRequest(
            call_id=call.id,
            registration_id=registration.identity,
            tool_name=call.name,
            source=registration.source,
            arguments=arguments,
            risk=domain_registration.risk,
            working_directory=working_directory,
            target_path=target,
            diff_preview=diff_preview,
            original_sha256=original_hash,
        )

    if "shell" in registration.capabilities:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        effective_command = (
            f"{shell_command_prefix.strip()}\n{command}"
            if shell_command_prefix and shell_command_prefix.strip()
            else command
        )
        return ToolApprovalRequest(
            call_id=call.id,
            registration_id=registration.identity,
            tool_name=call.name,
            source=registration.source,
            arguments=arguments,
            risk=domain_registration.risk,
            working_directory=working_directory,
            command=effective_command,
            risk_level="high",
        )

    if call.name in {"write", "edit", "bash"}:
        return None

    return ToolApprovalRequest(
        call_id=call.id,
        registration_id=registration.identity,
        tool_name=call.name,
        source=registration.source,
        arguments=arguments,
        risk=domain_registration.risk,
        working_directory=working_directory,
        risk_level="registered-side-effect",
    )


def _file_snapshot(
    arguments: Mapping[str, JSONValue],
    cwd: Path,
) -> tuple[Path, str, str | None]:
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("工具路径无效")
    candidate = Path(raw_path).expanduser()
    target = (candidate if candidate.is_absolute() else cwd / candidate).resolve()
    if not target.is_relative_to(cwd):
        raise ValueError("目标文件必须位于当前工作目录内")
    if not target.exists():
        return target, "", None
    if not target.is_file():
        raise ValueError("目标路径不是普通文件")
    contents = target.read_bytes()
    return target, contents.decode("utf-8"), hashlib.sha256(contents).hexdigest()


def _approval_snapshot_is_current(request: ToolApprovalRequest) -> bool:
    if request.target_path is None:
        return True
    try:
        raw_path = request.arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return False
        candidate = Path(raw_path).expanduser()
        current_target = (
            candidate
            if candidate.is_absolute()
            else request.working_directory / candidate
        ).resolve()
        if (
            current_target != request.target_path
            or not current_target.is_relative_to(request.working_directory)
        ):
            return False
        if request.original_sha256 is None:
            return not current_target.exists()
        if not current_target.is_file():
            return False
        digest = hashlib.sha256(current_target.read_bytes()).hexdigest()
    except OSError:
        return False
    return digest == request.original_sha256


def _arguments_fingerprint(arguments: Mapping[str, JSONValue]) -> str:
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ApprovalHandler",
    "ApprovalOutcome",
    "TOOL_PROFILE_NAMESPACE",
    "ToolApprovalRequest",
    "ToolProfileRuntime",
]
