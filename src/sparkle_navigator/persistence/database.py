"""Sparkle SQLite 数据库的初始化与迁移入口。"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool


def open_sqlite_database(database_path: Path) -> Engine:
    """创建 SQLite Engine，并将数据库升级到当前迁移版本。"""

    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create("sqlite", database=str(resolved_path)),
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )

    config = Config()
    migration_path = Path(__file__).with_name("migrations")
    config.set_main_option("script_location", str(migration_path))
    database_url = URL.create("sqlite", database=str(resolved_path))
    config.set_main_option(
        "sqlalchemy.url",
        database_url.render_as_string(hide_password=False).replace("%", "%%"),
    )
    command.upgrade(config, "head")
    return engine


__all__ = ["open_sqlite_database"]
