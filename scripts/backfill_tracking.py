"""Replays past runs into W&B from the `run_report.json` files they left
behind.

Tracking arrived in P9, but P2/P4/P7 already happened -- CPT, SFT, two DPO
attempts, and both DDP legs. Their loss histories exist, in
`trainer.state.log_history` as saved inside each run report. Without this,
the tracking project would start empty and the runs that matter most (the
DPO null result, the 2x L4 scaling pair) would be the only ones missing.

Each backfilled run is tagged `backfill` and carries its source path, so
nobody mistakes a replayed history for one that was streamed live. Step
numbers and metric names are whatever the original Trainer recorded; no
values are recomputed or cleaned up here.

    uv run scripts/backfill_tracking.py --dry-run     # list what would go
    uv run scripts/backfill_tracking.py               # send them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEARCH_GLOBS = [
    "artifacts/lora/*/run_report.json",
    "artifacts/runs/ddp_reports/report_*.json",
]


def discover() -> list[Path]:
    found: list[Path] = []
    for pattern in SEARCH_GLOBS:
        found.extend(sorted(Path().glob(pattern)))
    return found


def describe(path: Path) -> dict:
    report = json.loads(path.read_text())
    history = report.get("log_history") or []
    return {
        "path": str(path),
        # The adapter directory name is the most legible identifier these
        # runs have -- serana-sft, serana-dpo-v3, report_2gpu.
        "name": path.parent.name if path.name == "run_report.json" else path.stem,
        "method": report.get("method"),
        "device": report.get("device"),
        "steps_logged": len(history),
        "final_train_loss": report.get("final_train_loss"),
        "peak_mem_gb": report.get("peak_mem_gb"),
        "wall_clock_s": report.get("wall_clock_s"),
        "report": report,
        "history": history,
    }


def send(item: dict, project: str, entity: str | None) -> str:
    import wandb

    report = item["report"]
    run = wandb.init(
        project=project,
        entity=entity,
        name=f"{item['name']} (backfill)",
        # Distinct id per source file, so re-running this script updates
        # the same run rather than creating duplicates.
        id=f"backfill-{item['name']}",
        resume="allow",
        tags=["backfill", item["method"] or "unknown"],
        config={
            "source_report": item["path"],
            "method": item["method"],
            "device": item["device"],
            "backfilled": True,
        },
    )
    for entry in item["history"]:
        step = entry.get("step")
        metrics = {k: v for k, v in entry.items() if isinstance(v, int | float) and k != "step"}
        if metrics:
            run.log(metrics, step=step)
    for key in ("peak_mem_gb", "wall_clock_s", "final_train_loss", "resumed_from"):
        if report.get(key) is not None:
            run.summary[key] = report[key]
    url = run.url
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="serana-post-training")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = discover()
    if not paths:
        raise SystemExit("no run_report.json found under artifacts/")

    items = [describe(p) for p in paths]
    print(f"{len(items)} past run(s) found:\n")
    for it in items:
        print(
            f"  {it['name']:<22} method={str(it['method']):<6} "
            f"steps={it['steps_logged']:<4} final_loss={it['final_train_loss']} "
            f"peak={it['peak_mem_gb']}GB"
        )
    if args.dry_run:
        print("\n--dry-run: nothing sent")
        return

    print()
    for it in items:
        if not it["history"]:
            print(f"  skip {it['name']}: no log_history in the report")
            continue
        print(f"  sending {it['name']} ({it['steps_logged']} points)... ", end="", flush=True)
        print(send(it, args.project, args.entity))


if __name__ == "__main__":
    main()
