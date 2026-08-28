"""Training CLI. `--set dotted.path=value` overrides one config field,
letting one base diagnostics config express all four §3.6b breaks
without four separate files (see config/diagnostics/tiny_local.yaml's
comment block for the exact overrides each break uses):

    uv run scripts/train.py --config config/diagnostics/tiny_local.yaml \\
        --set train.learning_rate=2.0e-2   # divergence break
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.finetune.train import run


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        path, _, raw_value = item.partition("=")
        keys = path.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(raw_value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides", help="dotted.path=value, repeatable"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)

    report = run(config)
    print(
        f"done: method={report['method']} device={report['device']} "
        f"wall_clock={report['wall_clock_s']}s final_train_loss={report['final_train_loss']:.4f}"
    )
    if report["downgrade_notes"]:
        print("downgrades:", report["downgrade_notes"])


if __name__ == "__main__":
    main()
