"""Config loading with `inherits` resolution (DESIGN.md §5).

Every config file may set `inherits: <relative path>`. The parent is
loaded first and the child is deep-merged on top of it -- child values
win, nested dicts merge key-by-key rather than replacing wholesale. This
is the only place that walks the config/ layout; everything else takes a
plain resolved dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override의 값으로 base를 덮어쓰되, 같은 키가 양쪽 다 dict면 그 안까지 재귀적으로 병합한다."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """config 파일(.yaml)을 로드하고, 파이썬 dict으로 변환한다.
    만약 inherits라는게 존재하면 지우고, 부모 경로를 통해 부모 config 파일도 재귀적으로 처리한다.
    마지막으로 _deep_merge를 통해 하나로 합친다."""
    path = Path(path).resolve()
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    inherits = data.pop("inherits", None)
    if inherits is None:
        return data

    parent_path = (path.parent / inherits).resolve()
    parent = load_config(parent_path)
    return _deep_merge(parent, data)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.config <config path>")
    print(json.dumps(load_config(sys.argv[1]), indent=2, default=str))
