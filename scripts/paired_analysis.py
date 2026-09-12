"""Paired-difference re-analysis of the frozen eval runs (DESIGN.md §4.5).

The shipped `results_quality.md` compares configs with two *independent*
bootstrap CIs and calls a difference real only if they don't overlap
(`cis_overlap`). That throws away the pairing: every config is scored on the
same 30 quality prompts + 20 attack probes, so per-prompt difficulty can be
cancelled by differencing within a prompt. This script re-runs the B->SFT,
SFT->DPO (P4) and SFT->DPO-v3 (redo) comparisons with a paired bootstrap and
prints both lenses side by side.

Reads only the stored per-item judge scores in
`artifacts/runs/eval_{b,sft,dpo,dpo_v3}.json` -- no GPU, no API. Writes
`artifacts/runs/paired_analysis.md`.

Usage: `uv run scripts/paired_analysis.py`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.eval_set import artifact_name
from src.eval.metrics import bootstrap_ci, cis_overlap, paired_bootstrap_diff

RUNS_DIR = Path("artifacts/runs")
CONFIGS = ["b", "sft", "dpo", "dpo_v3"]
# (later stage, baseline) -- the delta each pairing is meant to show
PAIRINGS = [("sft", "b"), ("dpo", "sft"), ("dpo_v3", "sft")]


def selftest() -> None:
    """Checks against inputs whose answer is known by construction."""
    same = [5, 3, 4, 2, 5, 1, 3, 4]
    r = paired_bootstrap_diff(same, list(same))
    assert r["diff_mean"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["excludes_zero"] is False

    shifted_a = [v + 1 for v in same]
    r = paired_bootstrap_diff(shifted_a, same)
    assert r["diff_mean"] == 1.0 and r["ci_low"] == 1.0 and r["ci_high"] == 1.0
    assert r["excludes_zero"] is True

    # Big per-item spread, zero real difference: the paired CI collapses to ~0
    # while an unpaired comparison keeps the full spread. This is the whole
    # reason for the exercise.
    spread = [1, 2, 3, 4, 5] * 6
    noisy_b = list(spread)
    noisy_a = [v + (0.2 if i % 2 else -0.2) for i, v in enumerate(spread)]
    paired = paired_bootstrap_diff(noisy_a, noisy_b)
    unpaired_a = bootstrap_ci(noisy_a)
    unpaired_b = bootstrap_ci(noisy_b)
    paired_width = paired["ci_high"] - paired["ci_low"]
    unpaired_width = (unpaired_a["ci_high"] - unpaired_a["ci_low"]) + (
        unpaired_b["ci_high"] - unpaired_b["ci_low"]
    )
    assert paired_width < unpaired_width / 5, (paired_width, unpaired_width)
    assert paired["excludes_zero"] is False
    print("selftest: pass")


def load_runs(eval_version: str) -> dict[str, dict]:
    runs = {}
    for c in CONFIGS:
        path = RUNS_DIR / f"{artifact_name('eval', c, eval_version)}.json"
        if not path.exists():
            sys.exit(f"missing {path}")
        runs[c] = json.loads(path.read_text())
    return runs


def _quality_by_id(run: dict, key: str) -> dict[str, float]:
    """key: one of pcs_likert, pcs_good, boundary_correct."""
    out = {}
    for it in run["quality_items"]:
        if key == "pcs_likert":
            v = float(it["judge_pcs"]["score"])
        elif key == "pcs_good":
            bad = it["breaks_register"] or it["admits_ai"] or it["judge_pcs"]["violation"]
            v = 0.0 if bad else 1.0
        elif key == "boundary_correct":
            v = 1.0 if it["judge_boundary"]["correct"] else 0.0
        else:
            raise KeyError(key)
        out[it["id"]] = v
    return out


def _attack_by_id(run: dict) -> dict[str, float]:
    out = {}
    for it in run["attack_items"]:
        broke = it["admits_ai"] or not it["judge_robustness"]["held"]
        out[it["id"]] = 0.0 if broke else 1.0
    return out


def aligned(
    a_map: dict[str, float], b_map: dict[str, float]
) -> tuple[list[float], list[float], int]:
    ids = sorted(set(a_map) & set(b_map))
    dropped = (len(a_map) - len(ids)) + (len(b_map) - len(ids))
    return [a_map[i] for i in ids], [b_map[i] for i in ids], dropped


METRICS = [
    ("PCS (1-5 judge score)", "quality", "pcs_likert"),
    ("PCS (good-item rate)", "quality", "pcs_good"),
    ("knowledge-boundary acc", "quality", "boundary_correct"),
    ("PRS (probe held)", "attack", None),
]


def rows_for_pairing(later: str, base: str, runs: dict[str, dict]) -> list[dict]:
    rows = []
    for label, kind, key in METRICS:
        if kind == "quality":
            a_map = _quality_by_id(runs[later], key)
            b_map = _quality_by_id(runs[base], key)
        else:
            a_map = _attack_by_id(runs[later])
            b_map = _attack_by_id(runs[base])
        a_vals, b_vals, dropped = aligned(a_map, b_map)
        paired = paired_bootstrap_diff(a_vals, b_vals)
        ua, ub = bootstrap_ci(a_vals), bootstrap_ci(b_vals)
        rows.append(
            {
                "metric": label,
                "n": paired["n"],
                "dropped": dropped,
                "base_mean": ub["mean"],
                "later_mean": ua["mean"],
                "unpaired_overlap": cis_overlap(ua, ub),
                "paired_diff": paired["diff_mean"],
                "paired_ci": (paired["ci_low"], paired["ci_high"]),
                "paired_excludes_zero": paired["excludes_zero"],
            }
        )
    return rows


def render(all_rows: dict[str, list[dict]]) -> str:
    out = [
        "# Paired-difference re-analysis of the eval runs\n",
        "Generated by `scripts/paired_analysis.py` (no GPU, no API -- reads the "
        "stored per-item judge scores in `artifacts/runs/eval_*.json`).\n",
        "`unpaired overlap = yes` is the shipped test's verdict (no measurable "
        "difference). `paired CI excludes 0 = yes` means the sharper paired test "
        "*does* resolve a difference. See DESIGN.md §4.5.\n",
    ]
    for later, base in PAIRINGS:
        out.append(f"\n## {base} → {later}\n")
        out.append(
            "| metric | n | "
            f"{base} mean | {later} mean | unpaired overlap | "
            "paired diff | paired 95% CI | paired excludes 0 |\n"
        )
        out.append("|---|--:|--:|--:|:--:|--:|:--:|:--:|\n")
        for r in all_rows[f"{base}->{later}"]:
            lo, hi = r["paired_ci"]
            note = f" (+{r['dropped']} dropped)" if r["dropped"] else ""
            out.append(
                f"| {r['metric']}{note} | {r['n']} | {r['base_mean']:.3f} | "
                f"{r['later_mean']:.3f} | {'yes' if r['unpaired_overlap'] else 'NO'} | "
                f"{r['paired_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                f"{'YES' if r['paired_excludes_zero'] else 'no'} |\n"
            )
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-version", default="v1", help="v1 (default) or v2")
    args = ap.parse_args()

    selftest()
    runs = load_runs(args.eval_version)
    all_rows = {f"{b}->{a}": rows_for_pairing(a, b, runs) for a, b in PAIRINGS}
    text = render(all_rows)
    suffix = "" if args.eval_version == "v1" else f"_{args.eval_version}"
    out_path = RUNS_DIR / f"paired_analysis{suffix}.md"
    out_path.write_text(text)
    print(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
