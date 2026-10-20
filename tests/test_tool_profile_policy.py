"""Tool Profile 策略按注册来源和能力筛选工具。"""

from __future__ import annotations

from sparkle_navigator.domain.tool_profile import (
    ToolAuthorizationRequest,
    ToolDecision,
    ToolProfile,
    ToolProfilePolicy,
    ToolRegistration,
    ToolRisk,
)


def test_tau_legacy_imports_forward_to_sparkle_policy_types() -> None:
    from tau_coding import ToolProfile as PackageToolProfile
    from tau_coding.tool_profile import ToolProfile as ModuleToolProfile

    assert PackageToolProfile is ToolProfile
    assert ModuleToolProfile is ToolProfile


def test_explain_profile_does_not_trust_same_name_extension_tool() -> None:
    policy = ToolProfilePolicy()
    builtin_read = ToolRegistration(
        identity="tau-coding:read:1",
        tool_name="read",
        source="tau-coding-builtin",
        risk=ToolRisk.READ_ONLY,
    )
    extension_read = ToolRegistration(
        identity="extension:custom:read:2",
        tool_name="read",
        source="extension:custom",
        risk=ToolRisk.UNKNOWN,
    )
    builtin_write = ToolRegistration(
        identity="tau-coding:write:3",
        tool_name="write",
        source="tau-coding-builtin",
        risk=ToolRisk.SIDE_EFFECT,
    )

    visible = policy.visible_tools(
        ToolProfile.EXPLAIN,
        (builtin_read, extension_read, builtin_write),
    )

    assert visible == (builtin_read,)
    assert (
        policy.decide(
            ToolAuthorizationRequest(
                profile=ToolProfile.EXPLAIN,
                registration=extension_read,
                call_id="call-1",
                arguments={},
            )
        ).decision
        is ToolDecision.DENY
    )


def test_coding_profile_exposes_registered_side_effect_tools_for_approval() -> None:
    policy = ToolProfilePolicy()
    read = ToolRegistration(
        identity="tau-coding:read:1",
        tool_name="read",
        source="tau-coding-builtin",
        risk=ToolRisk.READ_ONLY,
    )
    write = ToolRegistration(
        identity="tau-coding:write:2",
        tool_name="write",
        source="tau-coding-builtin",
        risk=ToolRisk.SIDE_EFFECT,
    )
    unknown = ToolRegistration(
        identity="extension:custom:tool:3",
        tool_name="custom_tool",
        source="extension:custom",
        risk=ToolRisk.UNKNOWN,
    )

    visible = policy.visible_tools(ToolProfile.CODING, (read, write, unknown))

    assert visible == (read, write)
    write_decision = policy.decide(
        ToolAuthorizationRequest(
            profile=ToolProfile.CODING,
            registration=write,
            call_id="call-1",
            arguments={"path": "answer.txt", "content": "hello"},
        )
    )
    unknown_decision = policy.decide(
        ToolAuthorizationRequest(
            profile=ToolProfile.CODING,
            registration=unknown,
            call_id="call-2",
            arguments={},
        )
    )

    assert write_decision.decision is ToolDecision.APPROVAL_REQUIRED
    assert unknown_decision.decision is ToolDecision.DENY
