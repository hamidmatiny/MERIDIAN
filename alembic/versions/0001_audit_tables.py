"""Create the append-only audit tables.

Revision ID: 0001
Revises:
Create Date: 2026-10-24
"""

from alembic import op

from meridian.audit.schema import create_all

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    create_all(op.get_bind())


def downgrade() -> None:
    for name in ("runs", "tool_calls", "llm_calls", "audit_events"):
        op.drop_table(name)
