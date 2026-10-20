"""增加 Run Explorer 所需的 Trace、Evidence、Usage、Timing 与 Failure。"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0008_run_explorer_details"
down_revision = "0007_evaluation_core"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.add_column(sa.Column("failure_message", sa.Text(), nullable=True))
        batch.add_column(sa.Column("usage_json", sa.Text(), nullable=True))
        batch.add_column(sa.Column("timing_json", sa.Text(), nullable=True))

    op.create_table(
        "evaluation_run_trace_events",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "sequence"),
    )
    op.create_index(
        "ix_evaluation_run_trace_event_type",
        "evaluation_run_trace_events",
        ["run_id", "event_type"],
    )
    op.create_table(
        "evaluation_run_evidence",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=True),
        sa.Column("line_end", sa.Integer(), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("symbol_name", sa.Text(), nullable=True),
        sa.Column("relationship_type", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "ordinal"),
    )
    op.create_index(
        "ix_evaluation_run_evidence_path",
        "evaluation_run_evidence",
        ["run_id", "path"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluation_run_evidence_path",
        table_name="evaluation_run_evidence",
    )
    op.drop_table("evaluation_run_evidence")
    op.drop_index(
        "ix_evaluation_run_trace_event_type",
        table_name="evaluation_run_trace_events",
    )
    op.drop_table("evaluation_run_trace_events")
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.drop_column("timing_json")
        batch.drop_column("usage_json")
        batch.drop_column("failure_message")
