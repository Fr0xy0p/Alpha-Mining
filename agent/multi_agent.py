"""
@deprecated: Migrated to agent.multi.cli
Please use: python -m agent.multi.cli
"""

from agent.multi.cli import build_parser, main  # noqa: F401

if __name__ == "__main__":
    main()
