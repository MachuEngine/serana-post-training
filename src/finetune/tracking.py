"""Experiment tracking, built around one requirement: a Spot preemption
must not split a training run into two records.

Before this, a run's history lived in `run_report.json` next to the
adapter, rewritten from scratch every time the script started. Resuming
worked -- `get_last_checkpoint` + `resume_from_checkpoint` -- but the
record of the first leg was overwritten by the second. On a run that was
preempted twice, the middle was simply gone.

W&B rather than MLflow, for reasons specific to this stack:

- Resume semantics are two environment variables (`WANDB_RUN_ID` and
  `WANDB_RESUME`), and TRL's Trainer honours them without any glue.
- MLflow would need a tracking server, or a file backend on GCS. A local
  file backend on a Spot VM is exactly the thing that disappears when the
  VM does, which is the failure this module exists to prevent.

The run id is generated once, written next to the adapter, and synced to
GCS with the checkpoints (`src/finetune/gcs_sync.py`). A resumed run --
even on a freshly created VM -- reads that file back and reattaches to
the same W&B run, so the loss curve is continuous across the gap.

`tracking.mode` in config chooses `online`, `offline` (records locally,
`wandb sync` uploads later -- what CI and a keyless machine use), or
`disabled` (no tracking at all; the run report is still written).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_FILE = "wandb_run_id.txt"


def config_hash(config: dict[str, Any]) -> str:
    """Stable 8-char digest of the resolved config. CLAUDE.md asks every
    run to log 'the resolved config + hash'; this is that hash, and it is
    what makes two runs comparable-or-not at a glance."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _new_run_id(config: dict[str, Any]) -> str:
    method = config.get("train", {}).get("method", "run")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{method}-{config_hash(config)}-{stamp}"


def load_or_create_run_id(output_dir: str | Path, config: dict[str, Any]) -> tuple[str, bool]:
    """Returns (run_id, resumed). The id is created on the first launch
    and read back on every later one, so what makes a resume attach to
    the original run is a file that travels with the checkpoints -- not a
    value the caller has to remember and pass in."""
    path = Path(output_dir) / RUN_ID_FILE
    if path.exists():
        return path.read_text().strip(), True
    run_id = _new_run_id(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id + "\n")
    return run_id, False


def init(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Sets the environment TRL's Trainer reads, and opens the run.

    Returns a dict describing what happened, which goes into the run
    report -- so a report always shows whether tracking was on, and under
    which run id, rather than leaving it to be inferred.
    """
    cfg = config.get("tracking", {})
    mode = cfg.get("mode", "disabled")
    if mode == "disabled":
        return {"enabled": False, "mode": mode, "run_id": None, "resumed": False}

    run_id, resumed = load_or_create_run_id(output_dir, config)

    import wandb

    os.environ["WANDB_RUN_ID"] = run_id
    # "allow" rather than "must": a first launch has nothing to resume,
    # and "must" would make that an error instead of the normal case.
    os.environ["WANDB_RESUME"] = "allow"
    if mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=cfg.get("project", "serana-post-training"),
        entity=cfg.get("entity") or None,
        id=run_id,
        resume="allow",
        # The whole resolved config, so a run page answers "what was this
        # trained with" without opening the repo at the right commit.
        config={**config, "config_hash": config_hash(config)},
        tags=[config.get("train", {}).get("method", "run"), f"cfg:{config_hash(config)}"],
    )
    return {"enabled": True, "mode": mode, "run_id": run_id, "resumed": resumed}


def finish(report: dict[str, Any], state: dict[str, Any]) -> None:
    """Writes the run's final numbers to the W&B summary. Peak VRAM and
    the device-downgrade notes go here deliberately: they are the two
    things that explain a run's numbers and are invisible in a loss
    curve."""
    if not state.get("enabled"):
        return
    import wandb

    for key in ("peak_mem_gb", "wall_clock_s", "final_train_loss", "device", "resumed_from"):
        if report.get(key) is not None:
            wandb.summary[key] = report[key]
    if report.get("downgrade_notes"):
        wandb.summary["downgrade_notes"] = "; ".join(report["downgrade_notes"])
    wandb.finish()
