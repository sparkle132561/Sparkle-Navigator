"""通过 RRF 公共接口验证多路候选排序的确定性融合。"""

from __future__ import annotations

import pytest

from sparkle_navigator.context.rank_fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_combines_ranks_and_ignores_duplicates() -> None:
    fused = reciprocal_rank_fusion(
        (
            ("file-a", "file-a", "file-b"),
            ("file-b", "file-c"),
        ),
    )

    assert [hit.candidate_id for hit in fused] == ["file-b", "file-a", "file-c"]
    assert [hit.score for hit in fused] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61, 1 / 62]
    )


def test_reciprocal_rank_fusion_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="正整数"):
        reciprocal_rank_fusion((("file-a",),), k=0)
