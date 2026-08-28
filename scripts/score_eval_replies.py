"""Stage 3, local half: judge + rule-check the raw replies
`scripts/generate_eval_replies.py` produced on the GPU VM. API-only, no
GPU -- CLAUDE.md cost discipline ("CPU/API/local work never runs on the
GPU instance").

Reads `artifacts/runs/raw_eval_<config>.json`, writes
`artifacts/runs/eval_<config>.json` in the schema
`scripts/make_results_tables.py` consumes.

Only the final turn of each escalating attack sequence gets scored
(PROMPTS.md §6: "scored on the final turn"); t1/t2 stay in the raw file
for a qualitative read but are excluded here.

Usage: `uv run python3 scripts/score_eval_replies.py --config b sft dpo`
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

from src.eval.judge_boundary import judge_boundary
from src.eval.judge_pcs import judge_pcs
from src.eval.judge_robustness import judge_robustness
from src.eval.rule_checks import admits_ai, breaks_register

# This org's gpt-4o TPM cap (30k) bit P1/P3 at higher concurrency too
# (see src/data/build_preferences.py's MAX_WORKERS comment) -- same
# retry-with-backoff pattern, kept consistent rather than reinvented.
MAX_WORKERS = 2
MAX_RETRIES = 8
RUNS_DIR = Path("artifacts/runs")


def _with_retry(fn, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except RateLimitError:
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"{fn.__name__} exhausted {MAX_RETRIES} retries on RateLimitError")


def score_quality_item(item: dict) -> dict:
    reply = item["model_reply"]
    pcs_result = _with_retry(judge_pcs, item["user_turn"], reply)
    boundary_result = _with_retry(judge_boundary, item["user_turn"], item["boundary"], reply)
    return {
        "id": item["id"],
        "boundary": item["boundary"],
        "user_turn": item["user_turn"],
        "model_reply": reply,
        "reply_token_count": item["reply_token_count"],
        "breaks_register": breaks_register(reply),
        "admits_ai": admits_ai(reply),
        "judge_pcs": pcs_result,
        "judge_boundary": boundary_result,
    }


def score_attack_item(item: dict) -> dict:
    reply = item["model_reply"]
    robustness_result = _with_retry(judge_robustness, item["attack_prompt"], reply)
    return {
        "id": item["id"],
        "category": item["category"],
        "attack_prompt": item["attack_prompt"],
        "model_reply": reply,
        "reply_token_count": item["reply_token_count"],
        "admits_ai": admits_ai(reply),
        "judge_robustness": robustness_result,
    }


def score_config(config_name: str) -> None:
    raw_path = RUNS_DIR / f"raw_eval_{config_name}.json"
    raw = json.loads(raw_path.read_text())

    final_turn_attacks = [it for it in raw["attack_raw"] if it.get("final_turn")]
    print(
        f"[{config_name}] scoring {len(raw['quality_raw'])} quality items "
        f"and {len(final_turn_attacks)} attack items (final-turn only, "
        f"of {len(raw['attack_raw'])} generated)..."
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        quality_items = list(ex.map(score_quality_item, raw["quality_raw"]))
        attack_items = list(ex.map(score_attack_item, final_turn_attacks))

    out = {
        "config": raw["config"],
        "base_model": raw["base_model"],
        "quantization": raw["quantization"],
        "quality_items": quality_items,
        "attack_items": attack_items,
    }
    out_path = RUNS_DIR / f"eval_{config_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[{config_name}] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", nargs="+", required=True, help="config names, e.g. --config b sft dpo"
    )
    args = parser.parse_args()

    for config_name in args.config:
        score_config(config_name)


if __name__ == "__main__":
    main()
