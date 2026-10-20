"""Day 3 旧导入路径的兼容转发模块。

策略归 Sparkle Navigator 所有；Tau CodingSession 只消费通用工具策略接口。
新代码应从 ``sparkle_navigator.domain`` 或 ``sparkle_navigator.runtime`` 导入。
"""

from sparkle_navigator.domain.tool_profile import (
    ToolAuthorizationRequest,
    ToolDecision,
    ToolPolicyDecision,
    ToolProfile,
    ToolProfilePolicy,
    ToolRegistration,
    ToolRisk,
)
from sparkle_navigator.runtime.tool_profile import (
    ApprovalOutcome,
    ToolApprovalRequest,
    ToolProfileRuntime,
)

__all__ = [
    "ApprovalOutcome",
    "ToolApprovalRequest",
    "ToolAuthorizationRequest",
    "ToolDecision",
    "ToolPolicyDecision",
    "ToolProfile",
    "ToolProfilePolicy",
    "ToolProfileRuntime",
    "ToolRegistration",
    "ToolRisk",
]
