"""Day 3 Tool Profile 领域策略的测试。"""

from __future__ import annotations

from sparkle_navigator.domain.tool_profile import (
    ToolAuthorizationRequest,
    ToolDecision,
    ToolProfile,
    ToolProfilePolicy,
    ToolRegistration,
    ToolRisk,
)


def _request(profile: ToolProfile, tool_name: str, risk: ToolRisk) -> ToolAuthorizationRequest:
    source = "tau-coding-builtin" if risk is not ToolRisk.UNKNOWN else "unregistered"
    return ToolAuthorizationRequest(
        profile=profile,
        registration=ToolRegistration(
            identity=f"{source}:{tool_name}",
            tool_name=tool_name,
            source=source,
            risk=risk,
        ),
        call_id="call-1",
        arguments={},
    )


def test_explain_profile_allows_only_explicit_read_tools() -> None:
    policy = ToolProfilePolicy()

    read_decision = policy.decide(
        _request(ToolProfile.EXPLAIN, "read", ToolRisk.READ_ONLY)
    )
    write_decision = policy.decide(
        _request(ToolProfile.EXPLAIN, "write", ToolRisk.SIDE_EFFECT)
    )
    unknown_decision = policy.decide(
        _request(ToolProfile.EXPLAIN, "mcp_custom", ToolRisk.UNKNOWN)
    )

    assert (read_decision.risk, read_decision.decision) == (
        ToolRisk.READ_ONLY,
        ToolDecision.ALLOW,
    )
    assert write_decision.decision is ToolDecision.DENY
    assert unknown_decision.decision is ToolDecision.DENY


def test_coding_profile_requires_approval_for_side_effect_tools() -> None:
    policy = ToolProfilePolicy()

    read_decision = policy.decide(
        _request(ToolProfile.CODING, "read", ToolRisk.READ_ONLY)
    )
    write_decision = policy.decide(
        _request(ToolProfile.CODING, "write", ToolRisk.SIDE_EFFECT)
    )
    bash_decision = policy.decide(
        _request(ToolProfile.CODING, "bash", ToolRisk.SIDE_EFFECT)
    )
    unknown_decision = policy.decide(
        _request(ToolProfile.CODING, "mcp_custom", ToolRisk.UNKNOWN)
    )

    assert read_decision.decision is ToolDecision.ALLOW
    assert write_decision.decision is ToolDecision.APPROVAL_REQUIRED
    assert bash_decision.decision is ToolDecision.APPROVAL_REQUIRED
    assert unknown_decision.decision is ToolDecision.DENY
