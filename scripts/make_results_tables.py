"""Regenerates both DESIGN.md §6.1 results tables from stored run data --
the "both results tables regenerate from one command" P5 done-criterion.

Quality table: reads one scored eval-run JSON per config from
`artifacts/runs/eval_<config>.json` (produced by Stage 3's generate+score
pass -- schema documented in `load_eval_run` below). Hardware table:
reads `artifacts/runs/hardware_<label>.json` per training stage/serving
config (produced across P2/P4's training runs and Stage 2/4's serving
runs). Missing files are skipped with a warning, not a crash -- this
script is meant to be re-run as each piece of data lands, not only once
everything exists.

Usage: `uv run scripts/make_results_tables.py`
Writes: `artifacts/runs/results_quality.md`, `artifacts/runs/results_hardware.md`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import (
    bootstrap_ci,
    distinct_n,
    knowledge_boundary_accuracy,
    mean_reply_length,
    pcs,
    prs,
    prs_failure_breakdown,
)
from src.eval.style_similarity import style_embedding_similarity

RUNS_DIR = Path("artifacts/runs")
EVAL_SET_DIR = Path("data/eval/eval_set_v1")
CONFIGS = ["b", "sft", "dpo"]


def load_eval_run(path: Path) -> dict:
    """Expected schema (produced by Stage 3):
    {
      "config": "b" | "sft" | "dpo",
      "base_model": str, "quantization": str | null,
      "quality_items": [
        {"id", "boundary" ("in"|"out"), "user_turn", "model_reply",
         "reply_token_count", "breaks_register", "admits_ai",
         "judge_pcs": {"score", "violation", "reason"},
         "judge_boundary": {"outcome", "correct", "reason"}}
      ],
      "attack_items": [
        {"id", "category", "attack_prompt", "model_reply",
         "reply_token_count", "admits_ai",
         "judge_robustness": {"held", "failure_type", "reason"}}
      ]
    }
    """
    return json.loads(path.read_text())


def _fmt_ci(stat: dict) -> str:
    return f"{stat['mean']:.3f} [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}]"


def quality_row(run: dict, style_reference: list[str]) -> dict:
    quality_items = run["quality_items"]
    attack_items = run["attack_items"]

    pcs_items = [
        {
            "breaks_register": it["breaks_register"],
            "admits_ai": it["admits_ai"],
            "judge_violation": it["judge_pcs"]["violation"],
        }
        for it in quality_items
    ]
    prs_items = [
        {
            "admits_ai": it["admits_ai"],
            "judge_held": it["judge_robustness"]["held"],
            "failure_type": it["judge_robustness"]["failure_type"],
        }
        for it in attack_items
    ]
    boundary_items = [{"correct": it["judge_boundary"]["correct"]} for it in quality_items]

    style_sims = [
        style_embedding_similarity(it["model_reply"], style_reference) for it in quality_items
    ]
    all_texts = [it["model_reply"] for it in quality_items] + [
        it["model_reply"] for it in attack_items
    ]
    token_counts = [it["reply_token_count"] for it in quality_items + attack_items]

    return {
        "config": run["config"],
        "pcs": pcs(pcs_items),
        "prs": prs(prs_items),
        "prs_failure_breakdown": prs_failure_breakdown(prs_items),
        "style_sim": bootstrap_ci(style_sims),
        "knowledge_boundary": knowledge_boundary_accuracy(boundary_items),
        "mean_reply_length": mean_reply_length(token_counts),
        "distinct_2": distinct_n(all_texts, n=2),
    }


def render_quality_table(rows: list[dict]) -> str:
    header = (
        "| config | PCS | PRS | style sim | knowledge-boundary acc "
        "| mean reply length | distinct-2 |\n"
    )
    header += "|---|---|---|---|---|---|---|\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['config']} | {_fmt_ci(row['pcs'])} | {_fmt_ci(row['prs'])} | "
            f"{_fmt_ci(row['style_sim'])} | {_fmt_ci(row['knowledge_boundary'])} | "
            f"{_fmt_ci(row['mean_reply_length'])} | {row['distinct_2']:.3f} |\n"
        )
    # Real bug found and fixed here: the first version built only the
    # header for this section and never appended a data row per config --
    # caught by reading the actual rendered output against real Stage 3
    # data, not by eyeballing the code.
    failure_types = sorted({k for r in rows for k in r["prs_failure_breakdown"]})
    lines.append("\n### PRS failure_type breakdown\n\n")
    lines.append("| config | " + " | ".join(failure_types) + " |\n")
    lines.append("|---|" + "---|" * len(failure_types) + "\n")
    for row in rows:
        counts = row["prs_failure_breakdown"]
        lines.append(
            f"| {row['config']} | "
            + " | ".join(str(counts.get(ft, 0)) for ft in failure_types)
            + " |\n"
        )
    return "".join(lines)


def render_hardware_table(entries: list[dict]) -> str:
    header = (
        "| stage/config | predicted VRAM | measured peak VRAM | step time / TTFT p50,p95 "
        "| GPU util % | MFU % | throughput | cost |\n"
    )
    header += "|---|---|---|---|---|---|---|---|\n"
    lines = [header]
    for e in entries:
        lines.append(
            f"| {e.get('label', '?')} | {e.get('predicted_vram_gb', '-')} | "
            f"{e.get('measured_vram_gb', '-')} | {e.get('step_time_or_ttft', '-')} | "
            f"{e.get('gpu_util_pct', '-')} | {e.get('mfu_pct', '-')} | "
            f"{e.get('throughput', '-')} | {e.get('cost_usd', '-')} |\n"
        )
    return "".join(lines)


def main() -> None:
    style_reference = [
        json.loads(line)["line"] for line in (EVAL_SET_DIR / "style_reference.jsonl").open()
    ]

    rows = []
    for config in CONFIGS:
        path = RUNS_DIR / f"eval_{config}.json"
        if not path.exists():
            print(f"skip {config}: {path} not found yet")
            continue
        rows.append(quality_row(load_eval_run(path), style_reference))

    if rows:
        (RUNS_DIR / "results_quality.md").write_text(render_quality_table(rows))
        print(f"wrote {RUNS_DIR / 'results_quality.md'} ({len(rows)} config rows)")
    else:
        print(
            "no eval run data found yet -- quality table not written "
            "(Stage 3 produces artifacts/runs/eval_<config>.json)"
        )

    hardware_entries = []
    for path in sorted(RUNS_DIR.glob("hardware_*.json")):
        hardware_entries.append(json.loads(path.read_text()))
    if hardware_entries:
        (RUNS_DIR / "results_hardware.md").write_text(render_hardware_table(hardware_entries))
        print(f"wrote {RUNS_DIR / 'results_hardware.md'} ({len(hardware_entries)} entries)")
    else:
        print(
            "no hardware run data found yet -- hardware table not written "
            "(Stage 2/4 produce artifacts/runs/hardware_<label>.json)"
        )


if __name__ == "__main__":
    main()
