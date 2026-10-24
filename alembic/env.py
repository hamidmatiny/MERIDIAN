"""Alembic uses DATABASE_URL when it is set."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from meridian.audit.schema import metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata
url = os.environ.get("DATABASE_URL", "").strip()
if url.startswith("postgresql://"):
    url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
elif url.startswith("postgres://"):
    url = "postgresql+psycopg://" + url.removeprefix("postgres://")
if url:
    config.set_main_option("sqlalchemy.url", url)


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
