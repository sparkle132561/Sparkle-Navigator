"""为 Snapshot 增加 Python 符号和 import 索引表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_snapshot_symbol_index"
down_revision = "0002_snapshot_file_text_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repository_snapshot_symbols",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("qualified_name", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=False),
        sa.Column("line_end", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_repository_snapshot_symbols_lookup",
        "repository_snapshot_symbols",
        ["snapshot_id", "name"],
    )
    op.create_table(
        "repository_snapshot_imports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("module", sa.Text(), nullable=True),
        sa.Column("imported_name", sa.Text(), nullable=True),
        sa.Column("alias", sa.Text(), nullable=True),
        sa.Column("relative_level", sa.Integer(), nullable=False),
        sa.Column("line", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_repository_snapshot_imports_file",
        "repository_snapshot_imports",
        ["snapshot_id", "path", "line"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_repository_snapshot_imports_file",
        table_name="repository_snapshot_imports",
    )
    op.drop_table("repository_snapshot_imports")
    op.drop_index(
        "ix_repository_snapshot_symbols_lookup",
        table_name="repository_snapshot_symbols",
    )
    op.drop_table("repository_snapshot_symbols")
