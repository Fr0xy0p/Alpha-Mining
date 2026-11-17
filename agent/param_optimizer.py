"""
@deprecated: Migrated to agent.multi.param_optimizer
Please use: from agent.multi.param_optimizer import ParamOptimizer
"""

from agent.multi.param_optimizer import (  # noqa: F401
    ParamOptimizer,
    LLMOptimizer,
    OptimizerSuggestion,
)
