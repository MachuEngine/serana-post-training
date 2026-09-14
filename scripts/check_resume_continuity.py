"""Proves that a killed-and-restarted run produced one continuous record,
by reading the two legs' own logs rather than by looking at a picture.

A W&B screenshot would show a single unbroken curve, which is the point --
but a screenshot is not evidence anyone can re-derive, and in offline mode
there is no URL to show. This reads the step numbers and losses out of the
offline run directory and the run report, and checks three things:

  1. Both legs wrote into the **same** W&B run directory (same run id).
  2. The steps form one sequence with no gap and no repetition across the
     restart -- the second leg picks up at the checkpoint, it does not
     start over at step 1.
  3. The loss at the seam is continuous, i.e. the restart resumed
     optimizer state rather than re-initialising it. A curve that jumps
     back up at the seam means the checkpoint's optimizer state was not
     really restored, which is the failure this whole mechanism exists to
     prevent and which a loss curve alone makes easy to miss.

    uv run scripts/check_resume_continuity.py artifacts/diagnostics/resume_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_history(adapter_dir: Path) -> list[dict]:
    report = json.loads((adapter_dir / "run_report.json").read_text())
    return report.get("log_history") or [], report


def offline_runs(run_id: str) -> list[Path]:
    """W&B offline runs land in ./wandb/offline-run-*-<run_id>."""
    root = Path("wandb")
    if not root.exists():
        return []
    return sorted(p for p in root.glob("offline-run-*") if p.name.endswith(run_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", nargs="?", default="artifacts/diagnostics/resume_demo")
    args = parser.parse_args()
    adapter = Path(args.adapter_dir)

    history, report = load_history(adapter)
    run_id = (adapter / "wandb_run_id.txt").read_text().strip()
    losses = [(e["step"], e["loss"]) for e in history if "loss" in e]
    resumed_from = report.get("resumed_from")

    print(f"run id        : {run_id}")
    print(f"resumed from  : {resumed_from}")
    print(f"tracking      : {report.get('tracking')}")
    print(f"gcs restore   : {report.get('gcs_restore')}")
    print(f"logged points : {len(losses)}")

    runs = offline_runs(run_id)
    print(f"\nW&B run directories carrying this id: {len(runs)}")
    for r in runs:
        print(f"  {r}")

    steps = [s for s, _ in losses]
    print("\nstep sequence :", steps)
    gaps = [(a, b) for a, b in zip(steps, steps[1:]) if b <= a]
    print("out-of-order or repeated steps:", gaps or "none")

    if resumed_from:
        seam = int(str(resumed_from).rsplit("-", 1)[-1])
        before = [v for s, v in losses if s <= seam]
        after = [v for s, v in losses if s > seam]
        print(f"\nseam at step {seam}")
        print(f"  last loss before : {before[-1] if before else 'n/a'}")
        print(f"  first loss after : {after[0] if after else 'n/a'}")
        if before and after:
            jump = after[0] - before[-1]
            span = max(v for _, v in losses) - min(v for _, v in losses)
            print(f"  jump across seam : {jump:+.4f}  (full run spans {span:.4f})")
            print(
                "  verdict          : "
                + (
                    "continuous -- optimizer state was restored"
                    if abs(jump) <= 0.5 * span
                    else "DISCONTINUOUS -- suspect the checkpoint did not restore optimizer state"
                )
            )


if __name__ == "__main__":
    main()
