"""加入独立 repetition 与持久化比较/回归结果。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0009_evaluation_repetitions_and_comparison"
down_revision = "0008_run_explorer_details"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_experiments") as batch:
        batch.add_column(
            sa.Column(
                "repetitions",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
    with op.batch_alter_table("evaluation_experiments") as batch:
        batch.alter_column("repetitions", server_default=None)

    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.add_column(
            sa.Column(
                "repetition",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
        batch.drop_constraint(
            "uq_evaluation_agent_run_attempt",
            type_="unique",
        )
        batch.create_unique_constraint(
            "uq_evaluation_agent_run_repetition_attempt",
            ["experiment_id", "case_id", "repetition", "attempt"],
        )
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.alter_column("repetition", server_default=None)

    with op.batch_alter_table("evaluation_results") as batch:
        batch.add_column(
            sa.Column(
                "direction",
                sa.String(length=32),
                nullable=False,
                server_default="higher_is_better",
            )
        )
    with op.batch_alter_table("evaluation_results") as batch:
        batch.alter_column("direction", server_default=None)

    op.create_table(
        "evaluation_comparisons",
        sa.Column("comparison_id", sa.String(length=36), nullable=False),
        sa.Column("baseline_experiment_id", sa.String(length=36), nullable=False),
        sa.Column("candidate_experiment_id", sa.String(length=36), nullable=False),
        sa.Column("thresholds_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["baseline_experiment_id"],
            ["evaluation_experiments.experiment_id"],
        ),
        sa.ForeignKeyConstraint(
            ["candidate_experiment_id"],
            ["evaluation_experiments.experiment_id"],
        ),
        sa.PrimaryKeyConstraint("comparison_id"),
    )
    op.create_table(
        "evaluation_regression_results",
        sa.Column("comparison_id", sa.String(length=36), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("evaluator_id", sa.String(length=128), nullable=False),
        sa.Column("evaluator_version", sa.String(length=64), nullable=False),
        sa.Column("baseline_mean", sa.Float(), nullable=False),
        sa.Column("candidate_mean", sa.Float(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=False),
        sa.Column("baseline_variance", sa.Float(), nullable=False),
        sa.Column("candidate_variance", sa.Float(), nullable=False),
        sa.Column("baseline_sample_count", sa.Integer(), nullable=False),
        sa.Column("candidate_sample_count", sa.Integer(), nullable=False),
        sa.Column("direction", sa.String(length=32), nullable=False),
        sa.Column("regression_threshold", sa.Float(), nullable=False),
        sa.Column("is_regression", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["comparison_id"],
            ["evaluation_comparisons.comparison_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "comparison_id",
            "metric_name",
            "evaluator_id",
            "evaluator_version",
        ),
    )


def downgrade() -> None:
    op.drop_table("evaluation_regression_results")
    op.drop_table("evaluation_comparisons")
    with op.batch_alter_table("evaluation_results") as batch:
        batch.drop_column("direction")
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.drop_constraint(
            "uq_evaluation_agent_run_repetition_attempt",
            type_="unique",
        )
        batch.drop_column("repetition")
        batch.create_unique_constraint(
            "uq_evaluation_agent_run_attempt",
            ["experiment_id", "case_id", "attempt"],
        )
    with op.batch_alter_table("evaluation_experiments") as batch:
        batch.drop_column("repetitions")
