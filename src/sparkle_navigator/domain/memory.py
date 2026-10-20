"""项目记忆与学习者画像的领域类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from sparkle_navigator.domain.evidence import CodeEvidence


class ProjectFactStatus(StrEnum):
    """项目事实相对于其来源 Snapshot 的确认状态。"""

    CONFIRMED = "CONFIRMED"
    STALE = "STALE"
    INVALID = "INVALID"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class ProjectFact:
    """经用户确认、绑定仓库快照及源码证据的项目事实。"""

    fact_id: str
    repository_id: str
    snapshot_id: str
    statement: str
    evidence: tuple[CodeEvidence, ...]
    status: ProjectFactStatus
    supersedes_fact_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class LearnerProfile:
    """由用户控制的学习进度、疑难点与讲解偏好。"""

    profile_id: str = "local"
    learning_progress: tuple[str, ...] = ()
    confusion_points: tuple[str, ...] = ()
    explanation_preferences: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("Learner Profile ID 不能为空。")


__all__ = ["LearnerProfile", "ProjectFact", "ProjectFactStatus"]
