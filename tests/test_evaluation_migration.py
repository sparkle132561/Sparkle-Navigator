"""验证 v7 Evaluation Core 数据能升级到 Run Explorer schema。"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import URL

from sparkle_navigator.evaluation import RunExplorer
from sparkle_navigator.domain.evaluation import MetricDirection


def test_populated_v7_evaluation_records_survive_run_detail_migration(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "evaluation-v7.sqlite3"
    migration_path = (
        Path(__file__).parents[1]
        / "src"
        / "sparkle_navigator"
        / "persistence"
        / "migrations"
    )
    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(migration_path))
    database_url = URL.create("sqlite", database=str(database_path.resolve()))
    alembic_config.set_main_option(
        "sqlalchemy.url",
        database_url.render_as_string(hide_password=False).replace("%", "%%"),
    )
    command.upgrade(alembic_config, "0007_evaluation_core")

    experiment_id = str(uuid4())
    run_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO evaluation_experiments "
            "(experiment_id, suite_id, snapshot_id, configuration_json, status, "
            "failure_reason, created_at, updated_at) "
            "VALUES (?, ?, NULL, ?, ?, NULL, ?, ?)",
            (experiment_id, "legacy-suite", "{}", "COMPLETED", now, now),
        )
        connection.execute(
            "INSERT INTO evaluation_cases "
            "(experiment_id, case_id, ordinal, user_input, ground_truth_json) "
            "VALUES (?, ?, 0, ?, ?)",
            (experiment_id, "legacy-case", "legacy question", "{}"),
        )
        connection.execute(
            "INSERT INTO evaluation_agent_runs "
            "(run_id, experiment_id, case_id, attempt, status, output, tau_session_id, "
            "failure_reason, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?, ?, NULL, ?, ?)",
            (
                run_id,
                experiment_id,
                "legacy-case",
                "COMPLETED",
                "legacy persisted answer",
                run_id,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO evaluation_results "
            "(result_id, run_id, metric_name, evaluator_id, evaluator_version, value, details_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                run_id,
                "faithfulness",
                "sparkle.test",
                "1",
                0.5,
                "{}",
            ),
        )

    detail = asyncio.run(RunExplorer(database_path=database_path).get_run(experiment_id, run_id))

    assert detail.run.output == "legacy persisted answer"
    assert detail.run.tau_session_id == run_id
    assert detail.run.repetition == 1
    assert detail.run.results[0].value == 0.5
    assert detail.run.results[0].direction is MetricDirection.HIGHER_IS_BETTER
    assert detail.run.usage is None
    assert detail.run.timing is None
    assert detail.trace_events == ()
    assert detail.evidence == ()
