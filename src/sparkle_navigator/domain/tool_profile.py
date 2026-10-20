"""Sparkle Navigator 工具配置档的纯策略模型。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class ToolProfile(StrEnum):
    """会话可选择的工具配置档。"""

    EXPLAIN = "explain"
    CODING = "coding"


class ToolRisk(StrEnum):
    """工具注册时声明的能力风险。"""

    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"
    UNKNOWN = "unknown"


class ToolDecision(StrEnum):
    """策略对单次调用给出的结果。"""

    ALLOW = "allow"
    APPROVAL_REQUIRED = "approval_required"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """一个已注册工具的运行时身份、来源和能力。"""

    identity: str
    tool_name: str
    source: str
    risk: ToolRisk


@dataclass(frozen=True, slots=True)
class ToolAuthorizationRequest:
    """执行前策略判断需要的 Profile、注册身份和调用数据。"""

    profile: ToolProfile
    registration: ToolRegistration
    call_id: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolPolicyDecision:
    """策略模块返回的无副作用决定。"""

    registration_id: str
    tool_name: str
    source: str
    risk: ToolRisk
    decision: ToolDecision
    reason: str


class ToolProfilePolicy:
    """按工具注册能力筛选模型可见工具并判断执行授权。"""

    def visible_tools(
        self,
        profile: ToolProfile,
        registrations: Sequence[ToolRegistration],
    ) -> tuple[ToolRegistration, ...]:
        """返回当前 Profile 可提供给模型的已知工具。"""

        if profile is ToolProfile.EXPLAIN:
            return tuple(item for item in registrations if item.risk is ToolRisk.READ_ONLY)
        return tuple(
            item
            for item in registrations
            if item.risk in {ToolRisk.READ_ONLY, ToolRisk.SIDE_EFFECT}
        )

    def decide(self, request: ToolAuthorizationRequest) -> ToolPolicyDecision:
        """返回允许、要求审批或拒绝的策略决定。"""

        registration = request.registration
        if registration.risk is ToolRisk.UNKNOWN:
            decision = ToolDecision.DENY
            reason = "来源或能力未明确登记的工具默认拒绝。"
        elif request.profile is ToolProfile.EXPLAIN:
            if registration.risk is ToolRisk.READ_ONLY:
                decision = ToolDecision.ALLOW
                reason = "Explain Profile 只允许已登记的只读工具。"
            else:
                decision = ToolDecision.DENY
                reason = "Explain Profile 不允许副作用工具。"
        elif registration.risk is ToolRisk.READ_ONLY:
            decision = ToolDecision.ALLOW
            reason = "Coding Profile 可以直接使用已登记的只读工具。"
        else:
            decision = ToolDecision.APPROVAL_REQUIRED
            reason = "Coding Profile 的副作用工具需要本次调用的明确审批。"

        return ToolPolicyDecision(
            registration_id=registration.identity,
            tool_name=registration.tool_name,
            source=registration.source,
            risk=registration.risk,
            decision=decision,
            reason=reason,
        )


__all__ = [
    "ToolAuthorizationRequest",
    "ToolDecision",
    "ToolPolicyDecision",
    "ToolProfile",
    "ToolProfilePolicy",
    "ToolRegistration",
    "ToolRisk",
]
