#!/usr/bin/env python3
"""PreToolUse(Bash): block GCP commands targeting a region/zone outside
Seoul.

CLAUDE.md: "Hard constraint: compute and data stay in kr-west. No code
path ships data outside the region except text-only calls to the
judge/translation API." (kr-west == GCP asia-northeast3.)

Limitation: only catches commands that pass --region/--zone/-l
explicitly. A command relying on a pre-set `gcloud config` default region
won't be caught here -- this hook narrows the risk, it doesn't guarantee
it. Verify `gcloud config get-value compute/region` separately.
"""
import json
import re
import sys

ALLOWED_REGION = "asia-northeast3"

GCP_CMD_RE = re.compile(r"\b(gcloud|gsutil)\b")
REGION_FLAG_RE = re.compile(r"--(?:region|zone)[= ]([^\s]+)")
GSUTIL_LOCATION_RE = re.compile(r"-l\s+([^\s]+)")


def main() -> int:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")

    if not GCP_CMD_RE.search(command):
        return 0

    # GSUTIL_LOCATION_RE only applies when gsutil is actually being invoked --
    # otherwise it false-positives on unrelated "-l" flags (e.g. `wc -l`)
    # that happen to appear inside a `gcloud compute ssh --command="..."`
    # remote payload, which also matches GCP_CMD_RE via the outer `gcloud`.
    values = REGION_FLAG_RE.findall(command)
    if re.search(r"\bgsutil\b", command):
        values += GSUTIL_LOCATION_RE.findall(command)
    for value in values:
        if ALLOWED_REGION not in value:
            print(
                f"BLOCKED: region/zone '{value}' is outside {ALLOWED_REGION} "
                "(Seoul). CLAUDE.md hard constraint: compute and data stay "
                "in kr-west. If this is intentional, ask the user to run it "
                "directly.",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
