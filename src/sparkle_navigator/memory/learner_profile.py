"""持久化用户可管理的 Learner Profile。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from sparkle_navigator.domain.memory import LearnerProfile
from sparkle_navigator.persistence.database import open_sqlite_database
from sparkle_navigator.persistence.models import LearnerProfileRow


class LearnerProfileStore:
    """通过 SQLite 提供 Learner Profile 的读取、保存和删除。"""

    def __init__(self, database_path: Path) -> None:
        self._engine = open_sqlite_database(database_path)

    def get(self, profile_id: str = "local") -> LearnerProfile | None:
        """读取指定画像；尚未保存时返回 None。"""

        with Session(self._engine) as session:
            row = session.get(LearnerProfileRow, profile_id)
        return None if row is None else _learner_profile_from_row(row)

    def save(self, profile: LearnerProfile) -> LearnerProfile:
        """创建或完整替换一个 Learner Profile。"""

        saved_profile = replace(profile, updated_at=datetime.now(UTC))
        with Session(self._engine) as session, session.begin():
            row = session.get(LearnerProfileRow, profile.profile_id)
            if row is None:
                row = LearnerProfileRow(profile_id=profile.profile_id)
                session.add(row)
            row.learning_progress_json = _json_array(profile.learning_progress)
            row.confusion_points_json = _json_array(profile.confusion_points)
            row.explanation_preferences_json = _json_array(
                profile.explanation_preferences
            )
            row.updated_at = saved_profile.updated_at
            session.flush()
        return saved_profile

    def delete(self, profile_id: str = "local") -> bool:
        """删除指定画像；不存在时返回 False。"""

        with Session(self._engine) as session, session.begin():
            row = session.get(LearnerProfileRow, profile_id)
            if row is None:
                return False
            session.delete(row)
            return True


def _json_array(items: tuple[str, ...]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _learner_profile_from_row(row: LearnerProfileRow) -> LearnerProfile:
    updated_at = row.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return LearnerProfile(
        profile_id=row.profile_id,
        learning_progress=tuple(json.loads(row.learning_progress_json)),
        confusion_points=tuple(json.loads(row.confusion_points_json)),
        explanation_preferences=tuple(
            json.loads(row.explanation_preferences_json)
        ),
        updated_at=updated_at,
    )


__all__ = ["LearnerProfileStore"]
