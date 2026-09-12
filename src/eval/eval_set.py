"""Resolve the eval-set directory and per-version artifact names.

The eval set is versioned (`data/eval/eval_set_v1`, `..._v2`, ...). v1 is
frozen and shipped; v2 (artifacts/runs/p8_plan.md) is added alongside it,
not in place of it. Every eval-runner script takes `--eval-version` and
routes through here so v1's files and the shipped results tables stay
byte-identical when nothing asks for v2.

- `eval_set_dir(v)`  -> data/eval/eval_set_<v>
- `artifact_name("eval", "sft", "v1")` -> "eval_sft"      (unchanged)
- `artifact_name("eval", "sft", "v2")` -> "eval_v2_sft"    (parallel)
- `style_reference_lines(v)` -> v's style_reference.jsonl, or v1's if the
  version doesn't ship one (it is a fixed target -- p8_plan.md step 2).
- `judge_config(v)` -> the `judge` (v1) or `judge_v2` block of eval.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

_EVAL_YAML = Path("config/eval.yaml")
DEFAULT_VERSION = "v1"


def eval_set_dir(version: str = DEFAULT_VERSION) -> Path:
    d = Path(f"data/eval/eval_set_{version}")
    if not d.is_dir():
        raise SystemExit(f"eval set not found: {d} (build it first)")
    return d


def artifact_name(prefix: str, config_name: str, version: str = DEFAULT_VERSION) -> str:
    """Stem for artifacts/runs/<...>.json. v1 keeps the original naming so
    the shipped files are never shadowed; other versions get an infix."""
    if version == "v1":
        return f"{prefix}_{config_name}"
    return f"{prefix}_{version}_{config_name}"


def style_reference_lines(version: str = DEFAULT_VERSION) -> list[str]:
    path = eval_set_dir(version) / "style_reference.jsonl"
    if not path.exists():
        path = eval_set_dir("v1") / "style_reference.jsonl"
    return [json.loads(line)["line"] for line in path.open()]


def judge_config(version: str = DEFAULT_VERSION) -> dict:
    cfg = yaml.safe_load(_EVAL_YAML.read_text())
    key = "judge" if version == "v1" else f"judge_{version}"
    if key not in cfg:
        raise SystemExit(f"{_EVAL_YAML} has no '{key}' block for eval version {version}")
    return cfg[key]
