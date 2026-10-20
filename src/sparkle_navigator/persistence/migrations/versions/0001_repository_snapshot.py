"""创建 Repository、Snapshot 和关联表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_repository_snapshot"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repository_snapshots",
        sa.Column("snapshot_id", sa.String(length=64), primary_key=True),
        sa.Column("source_commit", sa.String(length=64), nullable=False),
        sa.Column("file_manifest_json", sa.Text(), nullable=False),
        sa.Column("index_version", sa.Integer(), nullable=False),
        sa.Column("snapshot_root", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "repositories",
        sa.Column("repository_id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column(
            "trusted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("current_snapshot_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["current_snapshot_id"],
            ["repository_snapshots.snapshot_id"],
        ),
    )
    op.create_table(
        "repository_snapshot_links",
        sa.Column("repository_id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["repository_id"],
            ["repositories.repository_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("repository_id", "snapshot_id"),
    )


def downgrade() -> None:
    op.drop_table("repository_snapshot_links")
    op.drop_table("repositories")
    op.drop_table("repository_snapshots")
