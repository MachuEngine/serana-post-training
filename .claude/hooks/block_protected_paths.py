#!/usr/bin/env python3
"""PreToolUse(Write|Edit): block writes to protected paths.

- data/eval/  — the frozen eval set, human labels, attack probes, style
  reference (DESIGN.md §4.6 leakage guard). If the agent can freely edit
  this, "the eval set never appears in training data" can't be trusted.
- data/raw/   — ingested wiki dialogue. Should only be produced by
  src/data/ ingestion scripts run via Bash, not hand-edited.
- the secrets file — same protection as Read(./.env) in settings.json,
  extended to Write/Edit.

Read is still allowed (needed to diagnose eval/leakage issues). Only
writes are blocked. Generate these paths via scripts/ (Bash), not by
writing the files directly.
"""
import json
import os
import sys

PROTECTED_DIRS = ("data/eval/", "data/raw/")
_SECRETS_FILE = "." + "env"


def is_protected_secrets_file(basename: str) -> bool:
    if basename == _SECRETS_FILE + ".example":
        return False
    return basename == _SECRETS_FILE or basename.startswith(_SECRETS_FILE + ".")


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return 0

    normalized = file_path.replace(os.sep, "/")
    basename = os.path.basename(file_path)

    for protected in PROTECTED_DIRS:
        if protected in normalized:
            print(
                f"BLOCKED: '{protected}' is a protected path (DESIGN.md "
                "leakage guard / repo layout). Generate it via a scripts/ "
                "run (Bash), not by writing the file directly. If the eval "
                "set itself looks wrong, report it instead of editing it.",
                file=sys.stderr,
            )
            return 2

    if is_protected_secrets_file(basename):
        print(
            "BLOCKED: secrets file is protected. Keep values there only "
            "(never committed); add new variable names to .env.example as "
            "placeholders.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
