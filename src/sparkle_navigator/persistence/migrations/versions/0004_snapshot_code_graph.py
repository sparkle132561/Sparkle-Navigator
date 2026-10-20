"""为 Snapshot 增加带置信度和源码证据的代码关系表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_snapshot_code_graph"
down_revision = "0003_snapshot_symbol_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repository_snapshot_relationships",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_symbol", sa.Text(), nullable=True),
        sa.Column("target_path", sa.Text(), nullable=True),
        sa.Column("target_symbol", sa.Text(), nullable=True),
        sa.Column("evidence_line", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_repository_snapshot_relationships_source",
        "repository_snapshot_relationships",
        ["snapshot_id", "relation_type", "source_path", "source_symbol"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_repository_snapshot_relationships_source",
        table_name="repository_snapshot_relationships",
    )
    op.drop_table("repository_snapshot_relationships")
