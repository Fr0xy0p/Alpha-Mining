"""
@deprecated: Migrated to agent.multi.expression_fingerprint
Please use: from agent.multi.expression_fingerprint import expression_fingerprint
"""

from agent.multi.expression_fingerprint import (  # noqa: F401
    expression_fingerprint,
    structure_distance,
    field_fingerprint,
)
