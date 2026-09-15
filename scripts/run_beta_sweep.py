"""DPO β sweep (DESIGN.md §8) -- VM-side orchestration.

Runs the two extra β values (0.05 and 0.3; 0.1 is the already-shipped
serana-dpo) end to end on the training VM: train each adapter continuing
the SFT adapter, then generate eval replies for it through the real vLLM
serving path. Judge scoring is deliberately NOT here -- it runs locally,
API-only, after the raw files are pulled off the VM (CLAUDE.md cost
discipline; same generate-on-GPU / score-on-CPU split as
generate_eval_replies.py + score_eval_replies.py).

Preemption-safe. A β whose adapter dir already holds
adapter_model.safetensors + run_report.json is treated as done and
skipped; a partially-trained one resumes from its last checkpoint via
run()'s existing get_last_checkpoint logic. Re-run with no arguments
after a Spot preemption and it picks up where it left off.

No GCS calls: the training VM's service account is devstorage.read_only
(see artifacts/runs/p4_progress.md), so every artifact is written to the
VM's local disk and pulled with `gcloud compute scp` afterwards -- the
exact procedure and cost prediction are in
artifacts/runs/beta_sweep_plan.md.

Usage (on the VM, repo synced, deps installed, from the repo root):
    python3 scripts/run_beta_sweep.py                     # train + generate, leave VM running
    python3 scripts/run_beta_sweep.py --stop-when-done    # ... then `sudo shutdown -h +1`
    python3 scripts/run_beta_sweep.py --dry-run           # plumbing check, no training / no serving

Outputs (VM local disk):
    artifacts/lora/serana-dpo-beta{005,030}/           trained adapters + run_report.json
    artifacts/runs/raw_eval_dpo_beta{005,030}.json     raw replies, for local scoring
    artifacts/runs/beta_sweep_report.json              training metrics for all three β
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

RUNS_DIR = Path("artifacts/runs")
DEFAULT_BETAS = ["005", "030"]
SERVE_HEALTH_URL = "http://localhost:8000/v1/models"
SERVE_TIMEOUT_S = 900  # FlashInfer JIT-compiles a kernel on first use (p5_progress note 4)


def adapter_is_done(out_dir: str) -> bool:
    p = Path(out_dir)
    return (p / "adapter_model.safetensors").exists() and (p / "run_report.json").exists()


def last_eval_row(log_history: list[dict]) -> dict:
    evals = [row for row in log_history if "eval_loss" in row]
    return evals[-1] if evals else {}


def summarize_report(beta: float, report: dict) -> dict:
    """One row of beta_sweep_report.json -- the numbers the write-up compares."""
    ev = last_eval_row(report.get("log_history", []))
    return {
        "beta": beta,
        "output_adapter": report.get("output_adapter"),
        "wall_clock_s": report.get("wall_clock_s"),
        "peak_mem_gb": report.get("peak_mem_gb"),
        "resumed_from": report.get("resumed_from"),
        "final_train_loss": report.get("final_train_loss"),
        "eval_loss": ev.get("eval_loss"),
        "eval_rewards_accuracies": ev.get("eval_rewards/accuracies"),
        "eval_rewards_margins": ev.get("eval_rewards/margins"),
    }


# ---- train phase -----------------------------------------------------------


def train_one(beta_tag: str, dry_run: bool) -> dict:
    cfg_path = f"config/train_runs/dpo_beta{beta_tag}.yaml"
    cfg = load_config(cfg_path)
    beta = cfg["train"]["dpo"]["beta"]
    out_dir = cfg["train"]["output_adapter"]
    assert cfg["train"]["method"] == "dpo", cfg_path
    assert cfg["train"]["init_adapter"] == "artifacts/lora/serana-sft", cfg_path

    if adapter_is_done(out_dir):
        print(f"[train β={beta}] already done at {out_dir} -- skipping")
        report = json.loads((Path(out_dir) / "run_report.json").read_text())
        return summarize_report(beta, report)

    print(f"[train β={beta}] {cfg_path} -> {out_dir}")
    if dry_run:
        print(f"[train β={beta}] dry-run: would call src.finetune.train.run()")
        return {"beta": beta, "output_adapter": out_dir, "dry_run": True}

    from src.finetune.train import run

    report = run(cfg)
    print(
        f"[train β={beta}] done: wall={report['wall_clock_s']}s "
        f"peak_vram={report['peak_mem_gb']}GB train_loss={report['final_train_loss']:.4f}"
    )
    return summarize_report(beta, report)


# ---- generate phase ------------------------------------------------------------


def wait_for_server(adapter_ids: list[str], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    want = set(adapter_ids)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(SERVE_HEALTH_URL, timeout=5) as resp:
                served = {m["id"] for m in json.loads(resp.read())["data"]}
            if want.issubset(served):
                print(f"[serve] ready, adapters registered: {sorted(want)}")
                return
            print(f"[serve] up, waiting for adapters {sorted(want - served)}...")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            print("[serve] not listening yet...")
        time.sleep(10)
    raise TimeoutError(f"vLLM server not ready with {adapter_ids} after {timeout_s}s")


def generate_all(beta_tags: list[str], dry_run: bool) -> None:
    exp_cfgs = {tag: f"config/experiments/dpo_beta{tag}.yaml" for tag in beta_tags}
    adapter_ids = [load_config(p)["lora_adapter_id"] for p in exp_cfgs.values()]

    if dry_run:
        print(f"[generate] dry-run: would serve + generate for {adapter_ids}")
        for tag, p in exp_cfgs.items():
            cfg = load_config(p)
            assert cfg["model_weights"] == "lora" and cfg["lora_adapter_id"], p
        return

    from scripts.generate_eval_replies import run_attack_probes, run_quality_prompts

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "scripts/serve_up.py"],
        stdout=open(RUNS_DIR / "beta_sweep_serve.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_server(adapter_ids, SERVE_TIMEOUT_S)
        for tag, cfg_path in exp_cfgs.items():
            config = load_config(cfg_path)
            name = Path(cfg_path).stem
            print(f"[generate] {name} ({config['lora_adapter_id']})")
            quality = run_quality_prompts(config)
            attack = run_attack_probes(config)
            out = {
                "config": name,
                "base_model": config["model"]["base_id"],
                "quantization": config.get("serving", {}).get("quantization"),
                "quality_raw": quality,
                "attack_raw": attack,
            }
            out_path = RUNS_DIR / f"raw_eval_{name}.json"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"[generate] wrote {out_path}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


# ---- shipped-β baseline ------------------------------------------------------


def shipped_dpo_summary() -> dict:
    """β=0.1 row: the already-shipped serana-dpo. Reads its run_report.json
    if present locally, else falls back to the numbers recorded in
    artifacts/runs/p4_progress.md so the table is always complete."""
    rp = Path("artifacts/lora/serana-dpo/run_report.json")
    if rp.exists():
        return summarize_report(0.1, json.loads(rp.read_text()))
    return {
        "beta": 0.1,
        "output_adapter": "artifacts/lora/serana-dpo",
        "source": "p4_progress.md (run_report.json not on this disk)",
        "wall_clock_s": 2760,  # ~46 min, two segments across one preemption
        "peak_mem_gb": "~15 (nvidia-smi)",
        "final_train_loss": 0.3467,
        "eval_loss": 0.6912,
        "eval_rewards_accuracies": 0.5952,
        "eval_rewards_margins": 0.004224,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", default=DEFAULT_BETAS, help="tags, e.g. 005 030")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="resolve configs only, no GPU work")
    parser.add_argument(
        "--stop-when-done", action="store_true", help="`sudo shutdown -h +1` after everything"
    )
    args = parser.parse_args()

    print(f"β sweep: tags={args.betas} dry_run={args.dry_run}")
    summaries = [shipped_dpo_summary()]

    if not args.skip_train:
        for tag in args.betas:
            summaries.append(train_one(tag, args.dry_run))
    if not args.skip_generate:
        generate_all(args.betas, args.dry_run)

    summaries.sort(key=lambda s: s["beta"])
    report_path = RUNS_DIR / "beta_sweep_report.json"
    if not args.dry_run:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"betas": summaries}, indent=2, default=str))
        print(f"wrote {report_path}")
    print(json.dumps(summaries, indent=2, default=str))

    if args.stop_when_done and not args.dry_run:
        print("[shutdown] scheduling `sudo shutdown -h +1`")
        subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)


if __name__ == "__main__":
    main()
