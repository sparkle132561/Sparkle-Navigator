"""记录 AgentRun 最终 stop_reason 以区分输出截断与正常完成。"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_agent_run_final_stop_reason"
down_revision = "0009_evaluation_repetitions_and_comparison"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.add_column(
            sa.Column("final_stop_reason", sa.String(length=32), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluation_agent_runs") as batch:
        batch.drop_column("final_stop_reason")
