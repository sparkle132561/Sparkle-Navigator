"""增加项目事实与学习者画像表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_project_memory_learner_profile"
down_revision = "0005_snapshot_semantic_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_facts",
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column("repository_id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("supersedes_fact_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["repository_id", "snapshot_id"],
            [
                "repository_snapshot_links.repository_id",
                "repository_snapshot_links.snapshot_id",
            ],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["supersedes_fact_id"], ["project_facts.fact_id"]),
        sa.PrimaryKeyConstraint("fact_id"),
    )
    op.create_index(
        "ix_project_facts_repository_snapshot_status",
        "project_facts",
        ["repository_id", "snapshot_id", "status"],
    )
    op.create_table(
        "learner_profiles",
        sa.Column("profile_id", sa.String(length=128), nullable=False),
        sa.Column("learning_progress_json", sa.Text(), nullable=False),
        sa.Column("confusion_points_json", sa.Text(), nullable=False),
        sa.Column("explanation_preferences_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("profile_id"),
    )


def downgrade() -> None:
    op.drop_table("learner_profiles")
    op.drop_index(
        "ix_project_facts_repository_snapshot_status",
        table_name="project_facts",
    )
    op.drop_table("project_facts")
