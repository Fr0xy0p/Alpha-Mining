"""
@deprecated: Migrated to agent.direct.cli
Please use: python -m agent.direct.cli
"""

from agent.direct.cli import build_parser, main  # noqa: F401

if __name__ == "__main__":
    main()
