"""Checkpoints to GCS, so a preemption can be survived by a *different*
machine.

`config/base.yaml` has declared `train.checkpoint_uri` since P0 and no
code has ever read it. `src/finetune/train.py` says so in its own
comment: local checkpoints were enough "because the VM's
--instance-termination-action=STOP keeps the boot disk alive across
preemption (a fresh VM would need a GCS sync step too, which this does
not do)". This is that step.

The distinction matters for what Spot actually guarantees. STOP keeps the
disk, so a restarted *same* VM resumes from local files. But a preemption
you respond to by creating a new VM -- which is what you do when the zone
has no capacity -- loses everything not in GCS. Checkpoints, and the
`wandb_run_id.txt` that reattaches the W&B run, both need to travel.

Uses `gcloud storage rsync` rather than a Python client: it is already a
hard dependency of the whole workflow (RUNBOOK.md), it is far faster for
many small files, and it needs no extra credentials handling on a VM that
is already authenticated.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _gcloud_available() -> bool:
    return shutil.which("gcloud") is not None


def remote_uri(config: dict[str, Any]) -> str | None:
    """`<checkpoint_uri>/<adapter name>`, or None when either the config
    does not set one or gcloud is unavailable (the M5 sandbox may have
    neither, and tracking must never be what breaks a local run)."""
    train_cfg = config.get("train", {})
    base = train_cfg.get("checkpoint_uri")
    output_dir = train_cfg.get("output_adapter")
    if not base or not output_dir or not _gcloud_available():
        return None
    return f"{base.rstrip('/')}/{Path(output_dir).name}"


def _rsync(src: str, dst: str) -> tuple[bool, str]:
    """One-way mirror. Returns (ok, message) rather than raising: losing a
    checkpoint upload should slow a run down, never kill it mid-training."""
    cmd = ["gcloud", "storage", "rsync", "--recursive", src, dst]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or "").strip()[:400]
    return True, f"{src} -> {dst}"


def push(config: dict[str, Any]) -> tuple[bool, str]:
    """Local adapter dir (checkpoints + run id) up to GCS."""
    uri = remote_uri(config)
    if uri is None:
        return False, "no checkpoint_uri configured, or gcloud not installed"
    return _rsync(config["train"]["output_adapter"], uri)


def pull(config: dict[str, Any]) -> tuple[bool, str]:
    """GCS down to the local adapter dir, before training starts. This is
    what makes a *fresh* VM able to resume: `get_last_checkpoint` then
    finds checkpoints that this machine never wrote."""
    uri = remote_uri(config)
    if uri is None:
        return False, "no checkpoint_uri configured, or gcloud not installed"
    local = config["train"]["output_adapter"]
    Path(local).mkdir(parents=True, exist_ok=True)
    return _rsync(uri, local)


class GCSCheckpointCallback(TrainerCallback):
    """Mirrors the adapter directory to GCS every time the Trainer saves.

    Attached only on the main process: under DDP every rank runs the same
    callbacks, and two ranks rsyncing the same directory to the same URI
    would race for no benefit.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.results: list[str] = []

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        ok, msg = push(self.config)
        line = f"step {state.global_step}: {'ok' if ok else 'FAILED'} {msg}"
        self.results.append(line)
        print(f"[gcs] {line}")
        return control
