"""booking slots: confirmation flag and a lookup index for the availability check

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "leads",
        sa.Column("needs_confirmation", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.create_index("ix_leads_scheduled_at", "leads", ["scheduled_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_leads_scheduled_at", table_name="leads")
    op.drop_column("leads", "needs_confirmation")
