"""Multi-turn length check (DESIGN.md §8) -- scoring half. DRAFT.

Reads artifacts/runs/raw_multiturn_<config>.json (from
generate_multiturn_replies.py) and scores the reply at each checkpoint
depth (default 1 / 3 / 10) with the SAME instruments the §6.1 quality
table uses: judge_pcs, the admits_ai / breaks_register rule checks, and
style-embedding similarity to the held-out style reference. API + CPU
only, no GPU (CLAUDE.md cost discipline).

The question it answers: does persona quality decay as the conversation
gets longer? A flat PCS / style-sim line across depths = the persona
holds; a downward slope = drift.

Usage: uv run python3 scripts/score_multiturn.py --config sft dpo
Writes: artifacts/runs/multiturn_<config>.json  (per-config detail)
        artifacts/runs/results_multiturn.md      (the drift table)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import RateLimitError

from src.eval.judge_pcs import judge_pcs
from src.eval.metrics import bootstrap_ci
from src.eval.rule_checks import admits_ai, breaks_register
from src.eval.style_similarity import style_embedding_similarity

RUNS_DIR = Path("artifacts/runs")
STYLE_REF = Path("data/eval/eval_set_v1/style_reference.jsonl")
DEFAULT_DEPTHS = [1, 3, 10]
MAX_WORKERS = 2  # same 30k-TPM ceiling as score_eval_replies.py
MAX_RETRIES = 8


def _with_retry(fn, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except RateLimitError:
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"{fn.__name__} exhausted {MAX_RETRIES} retries")


def score_turn(turn: dict, style_reference: list[str]) -> dict:
    reply = turn["model_reply"]
    pcs = _with_retry(judge_pcs, turn["user_turn"], reply)
    return {
        "depth": turn["depth"],
        "user_turn": turn["user_turn"],
        "model_reply": reply,
        "reply_token_count": turn["reply_token_count"],
        "judge_pcs_score": pcs["score"],
        "judge_pcs_violation": pcs["violation"],
        "admits_ai": admits_ai(reply),
        "breaks_register": breaks_register(reply),
        "style_sim": style_embedding_similarity(reply, style_reference),
    }


def score_config(config_name: str, depths: list[int], style_reference: list[str]) -> dict:
    raw = json.loads((RUNS_DIR / f"raw_multiturn_{config_name}.json").read_text())
    targets = [
        turn for conv in raw["conversations"] for turn in conv["turns"] if turn["depth"] in depths
    ]
    print(f"[{config_name}] scoring {len(targets)} turns at depths {depths}...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        scored = list(ex.map(lambda t: score_turn(t, style_reference), targets))

    by_depth = {}
    for d in depths:
        rows = [s for s in scored if s["depth"] == d]
        by_depth[d] = {
            "n": len(rows),
            "pcs_score": bootstrap_ci([r["judge_pcs_score"] for r in rows]),
            "violation_rate": bootstrap_ci([int(r["judge_pcs_violation"]) for r in rows]),
            "style_sim": bootstrap_ci([r["style_sim"] for r in rows]),
            "admits_ai_count": sum(r["admits_ai"] for r in rows),
            "breaks_register_count": sum(r["breaks_register"] for r in rows),
            "mean_reply_tokens": sum(r["reply_token_count"] for r in rows) / len(rows),
        }
    out = {"config": config_name, "depths": depths, "by_depth": by_depth, "turns": scored}
    (RUNS_DIR / f"multiturn_{config_name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str)
    )
    return out


def render_table(results: list[dict]) -> str:
    lines = [
        "# Multi-turn length check (DESIGN.md §8, DRAFT)\n",
        "Persona quality at conversation depth 1 / 3 / 10. Flat = holds, "
        "downward = drift. PCS is the 1-5 judge score; style sim is cosine "
        "to the held-out style reference.\n",
        "| config | depth | n | PCS score | violation rate | style sim | "
        "admits-AI | register breaks | mean reply tokens |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for res in results:
        for d in res["depths"]:
            s = res["by_depth"][d]
            p = s["pcs_score"]
            pcs_cell = f"{p['mean']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}]"
            lines.append(
                f"| {res['config']} | {d} | {s['n']} | {pcs_cell} | "
                f"{s['violation_rate']['mean']:.2f} | "
                f"{s['style_sim']['mean']:.3f} | "
                f"{s['admits_ai_count']} | {s['breaks_register_count']} | "
                f"{s['mean_reply_tokens']:.0f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="config names, e.g. sft dpo")
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    args = parser.parse_args()

    style_reference = [json.loads(line)["line"] for line in STYLE_REF.open()]
    results = [score_config(c, args.depths, style_reference) for c in args.config]

    out_path = RUNS_DIR / "results_multiturn.md"
    out_path.write_text(render_table(results))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
