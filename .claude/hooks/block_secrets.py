#!/usr/bin/env python3
"""PreToolUse(Write|Edit|Bash): block secret leakage.

Catches API-key-shaped strings being written, and shell commands that
read or transfer the secrets file directly. permissions.deny(Read(./.env))
blocks the Read tool, but not these Bash-level bypasses.
"""
import json
import re
import sys

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),   # OpenAI
    re.compile(r"AIza[A-Za-z0-9_-]{35}"),    # Google API key
    re.compile(r"hf_[A-Za-z0-9]{20,}"),      # Hugging Face token
    re.compile(r"AKIA[0-9A-Z]{16}"),         # AWS
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),     # GitHub
]

# Direct view/transfer of the secrets file (basename below), built from
# parts so this source line itself doesn't match the pattern it defines.
_SECRETS_FILE = "." + "env"
ENV_ACCESS_RE = re.compile(
    r"\b(cat|less|head|tail|curl|scp|nc)\b[^|;&\n]*" + re.escape(_SECRETS_FILE)
    + r"(?!\.example)(?![.\w])"
)


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    content = (
        tool_input.get("content")
        or tool_input.get("new_string")
        or tool_input.get("command")
        or ""
    )

    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            print(
                "BLOCKED: this looks like a real API key/token. Don't put the "
                "literal value in code -- reference it via os.environ[...] "
                "loaded from the secrets file instead.",
                file=sys.stderr,
            )
            return 2

    if ENV_ACCESS_RE.search(content):
        print(
            "BLOCKED: command reads/transfers the secrets file directly. "
            "Check .env.example for variable names/shape; read values in "
            "code via os.environ.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
