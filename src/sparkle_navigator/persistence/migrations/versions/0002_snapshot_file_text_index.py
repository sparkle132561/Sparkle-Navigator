"""为 Snapshot 文件建立元数据表和 FTS5 全文索引。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_snapshot_file_text_index"
down_revision = "0001_repository_snapshot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repository_snapshot_files",
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("indexed_text", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", "path"),
    )
    op.execute(
        "CREATE VIRTUAL TABLE repository_snapshot_files_fts "
        "USING fts5(snapshot_id UNINDEXED, path UNINDEXED, content)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE repository_snapshot_files_fts")
    op.drop_table("repository_snapshot_files")
