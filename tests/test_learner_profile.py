"""Learner Profile 的公开行为测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sparkle_navigator.domain.memory import LearnerProfile
from sparkle_navigator.memory.learner_profile import LearnerProfileStore


def test_learner_profile_can_be_created_read_updated_and_deleted(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "learner-profile.sqlite3"
    store = LearnerProfileStore(database_path)
    assert store.get() is None

    saved = store.save(
        LearnerProfile(
            profile_id="local",
            learning_progress=("Day 1：理解 Tau 三层职责",),
            confusion_points=("sequence 的用途",),
            explanation_preferences=("先讲调用链，再看测试",),
        )
    )
    assert LearnerProfileStore(database_path).get() == saved

    updated = store.save(
        replace(
            saved,
            learning_progress=("Day 1：通过", "Day 2：学习 Trace"),
        )
    )
    assert store.get() == updated
    assert store.delete() is True
    assert store.get() is None
    assert store.delete() is False
