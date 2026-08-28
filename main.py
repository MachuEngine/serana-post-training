"""CLI entry point. One command, config toggles everything (DESIGN.md §1,
CLAUDE.md single-pipeline rule) -- e.g.:

    uv run main.py --config config/experiments/b.yaml
    uv run main.py --config config/experiments/dpo.yaml --turn "Who are you?"
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.serve.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. config/experiments/b.yaml")
    parser.add_argument("--turn", default="Who are you?", help="user turn to send")
    args = parser.parse_args()

    config = load_config(args.config)
    print(run(config, args.turn))


if __name__ == "__main__":
    main()
