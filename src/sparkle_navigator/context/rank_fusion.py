"""用 Reciprocal Rank Fusion 合并多个检索排序列表。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """一个按倒数排名融合后的候选及其可解释分数。"""

    candidate_id: str
    score: float


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> tuple[FusedCandidate, ...]:
    """按每个列表中的名次累加 1 / (k + rank)，rank 从 1 开始。"""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k 必须是正整数。")

    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for candidate_id in ranking:
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("候选 ID 必须是非空字符串。")
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            rank += 1
            scores[candidate_id] += 1.0 / (k + rank)

    return tuple(
        FusedCandidate(candidate_id=candidate_id, score=score)
        for candidate_id, score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


__all__ = ["FusedCandidate", "reciprocal_rank_fusion"]
