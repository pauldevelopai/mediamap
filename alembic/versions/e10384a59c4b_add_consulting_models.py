"""Add consulting models

Revision ID: e10384a59c4b
Revises: 215fd4424302
Create Date: 2025-08-30 12:08:47.188071

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e10384a59c4b'
down_revision: Union[str, None] = '215fd4424302'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
