"""Renders the Grafana dashboard's panels to PNG files.

A dashboard that only exists while a cluster is up is not a deliverable --
the cluster is deleted between sessions, and asking a reader to recreate
it to see what the monitoring looked like is asking too much. Grafana's
render API (backed by the `grafana-renderer` deployment) turns each panel
into a file that outlives the cluster and lives in the repo.

Panels are read from the checked-in dashboard JSON rather than listed
here, so adding a panel to `deploy/grafana/serana-dashboard.json` is the
only edit needed to have it captured too.

Requires the port-forward Grafana is reachable on:

    kubectl port-forward svc/grafana 3000:3000
    uv run scripts/render_dashboard.py --out artifacts/runs/p9_dashboard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

DASHBOARD_JSON = Path("deploy/grafana/serana-dashboard.json")


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:3000")
    parser.add_argument("--out", default="artifacts/runs/p9_dashboard")
    parser.add_argument(
        "--range",
        default="now-30m",
        help="dashboard time range start; the load test is minutes long, so the "
        "default window has to be short enough that the traffic is not a spike "
        "at the right edge",
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=500)
    args = parser.parse_args()

    dash = json.loads(DASHBOARD_JSON.read_text())
    uid = dash["uid"]
    # Grafana assigns panel ids when a dashboard is saved; the checked-in
    # JSON has none, so ask the running instance for the version it loaded.
    resp = httpx.get(f"{args.base}/api/dashboards/uid/{uid}", timeout=30)
    resp.raise_for_status()
    live = resp.json()["dashboard"]
    panels = [p for p in live["panels"] if p.get("type") != "row"]
    if not panels:
        raise SystemExit("no non-row panels found on the live dashboard")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for panel in panels:
        params = {
            "orgId": 1,
            "panelId": panel["id"],
            "from": args.range,
            "to": "now",
            "width": args.width,
            "height": args.height,
            "tz": "Asia/Seoul",
        }
        url = f"{args.base}/render/d-solo/{uid}/serana?{urllib.parse.urlencode(params)}"
        # Chromium cold-starts on the first render; later ones are quick.
        r = httpx.get(url, timeout=120)
        if r.status_code != 200 or not r.content.startswith(b"\x89PNG"):
            print(f"  FAILED {panel['title']}: HTTP {r.status_code} {r.text[:120]}")
            continue
        path = out_dir / f"{panel['id']:02d}-{slugify(panel['title'])}.png"
        path.write_bytes(r.content)
        written.append(path)
        print(f"  {path}  ({len(r.content) // 1024} KB)")
        time.sleep(1)

    print(f"\n{len(written)}/{len(panels)} panels rendered into {out_dir}")
    if len(written) != len(panels):
        sys.exit(1)


if __name__ == "__main__":
    main()
