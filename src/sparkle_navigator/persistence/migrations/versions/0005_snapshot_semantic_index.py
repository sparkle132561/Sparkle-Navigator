"""为 Snapshot 增加文件与符号级语义向量表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_snapshot_semantic_index"
down_revision = "0004_snapshot_code_graph"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repository_snapshot_semantic_vectors",
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=512), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("document_kind", sa.String(length=16), nullable=False),
        sa.Column("symbol_name", sa.Text(), nullable=True),
        sa.Column("line_start", sa.Integer(), nullable=True),
        sa.Column("line_end", sa.Integer(), nullable=True),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", "model_id", "document_id"),
    )


def downgrade() -> None:
    op.drop_table("repository_snapshot_semantic_vectors")
