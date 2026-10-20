"""创建 Evaluation Core 的 Experiment、AgentRun、Event 和 Result 表。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_evaluation_core"
down_revision = "0006_project_memory_learner_profile"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_experiments",
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("suite_id", sa.String(length=255), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=True),
        sa.Column("configuration_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["repository_snapshots.snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("experiment_id"),
    )
    op.create_table(
        "evaluation_cases",
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=255), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("ground_truth_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["evaluation_experiments.experiment_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("experiment_id", "case_id"),
    )
    op.create_table(
        "evaluation_agent_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=255), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("tau_session_id", sa.String(length=255), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id", "case_id"],
            ["evaluation_cases.experiment_id", "evaluation_cases.case_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id"),
        sa.UniqueConstraint(
            "experiment_id",
            "case_id",
            "attempt",
            name="uq_evaluation_agent_run_attempt",
        ),
    )
    op.create_index(
        "ix_evaluation_agent_runs_experiment_status",
        "evaluation_agent_runs",
        ["experiment_id", "status"],
    )
    op.create_table(
        "evaluation_results",
        sa.Column("result_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("evaluator_id", sa.String(length=128), nullable=False),
        sa.Column("evaluator_version", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("result_id"),
        sa.UniqueConstraint(
            "run_id",
            "metric_name",
            "evaluator_id",
            "evaluator_version",
            name="uq_evaluation_result_metric_version",
        ),
    )
    op.create_table(
        "evaluation_events",
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["evaluation_experiments.experiment_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("experiment_id", "sequence"),
    )
    op.create_index(
        "ix_evaluation_events_run",
        "evaluation_events",
        ["run_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_events_run", table_name="evaluation_events")
    op.drop_table("evaluation_events")
    op.drop_table("evaluation_results")
    op.drop_index(
        "ix_evaluation_agent_runs_experiment_status",
        table_name="evaluation_agent_runs",
    )
    op.drop_table("evaluation_agent_runs")
    op.drop_table("evaluation_cases")
    op.drop_table("evaluation_experiments")
