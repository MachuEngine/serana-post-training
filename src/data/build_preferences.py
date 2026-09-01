"""P4 redo (artifacts/runs/p4_dpo_redo_plan.md): judge the reply groups
from scripts/generate_reply_groups.py -- each group is N replies the SFT
model produced for one prompt -- and emit the final (prompt, chosen,
rejected) preference set.

Each group's pair is (judge's best reply, judge's worst reply). A group
is dropped when:
  - the judge errored or returned a wrong-shape response (JudgeSchemaError),
  - best and worst are the same reply,
  - the best reply is one the judge flagged as breaking 반말 (v1's failure
    mode -- a register violation should never be the preferred output),
  - best and worst are near-identical after normalization,
  - the pick's confidence is not in KEEP_CONFIDENCE (the near-coin-flip
    picks are what gave P4's DPO run no learnable signal).

Default paths are the redo's own, so this does NOT overwrite the shipped
P3 artifacts (data/ko/prefs_1k.jsonl, artifacts/runs/p3_preference_report.json).
`--audit-sample N` also writes N judged groups to a readable file for a
by-eye sanity check of the judge's picks.

Runs off the GPU VM -- API-only (CLAUDE.md cost discipline).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from src.eval.preference_judge import JudgeSchemaError, preference_judge

MAX_WORKERS = 2  # gpt-4o TPM cap bit us in P1 at higher concurrency.
MAX_RETRIES = 8
NEAR_DUP_THRESHOLD = 0.92
KEEP_CONFIDENCE = ("high", "medium")  # drop "low"/missing -- the v3 signal filter

# Retried: rate limits, timeouts, connection drops, 5xx. NOT retried (fail
# fast): auth errors, a wrong judge model_id, bad-request -- those never
# recover, and the P3 lesson was that a silent "exit 0 with near-empty
# output" is worse than a crash. Note: a $0 balance surfaces as
# RateLimitError (insufficient_quota), so a persistent RateLimitError spike
# still means "check the OpenAI billing dashboard", as in P3.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def judge_with_retry(prompt: str, replies: list[str], seed: int) -> dict | None:
    """Judge result, or None if the group is unjudgeable (schema error).
    Transient API errors are retried; any other exception propagates so
    the run fails fast instead of silently dropping every group."""
    rng = random.Random(seed)
    for attempt in range(MAX_RETRIES):
        try:
            return preference_judge(prompt, replies, rng)
        except JudgeSchemaError:
            return None
        except RETRYABLE:
            time.sleep(min(2**attempt, 60))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="data/ko/raw/reply_groups_v3.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/ko/prefs_v3.jsonl")
    parser.add_argument(
        "--report", dest="report_path", default="artifacts/runs/p4redo_preference_report.json"
    )
    parser.add_argument(
        "--audit-sample",
        type=int,
        default=30,
        help="write this many judged groups to <out>.audit.json for a by-eye check",
    )
    args = parser.parse_args()

    groups = [json.loads(line) for line in open(args.in_path)]
    print(f"judging {len(groups)} reply groups from {args.in_path}...")

    judged: list[dict | None] = [None] * len(groups)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(judge_with_retry, g["prompt"], g["replies"], i): i
            for i, g in enumerate(groups)
        }
        done = 0
        for fut in as_completed(futs):
            judged[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(groups)}")

    n_error = n_same = n_register = n_near_dup = n_low_conf = 0
    conf_buckets = {"high": 0, "medium": 0, "low": 0, "missing": 0}
    preferences = []
    audit_rows = []
    for g, j in zip(groups, judged):
        if j is None:
            n_error += 1
            continue
        best_i, worst_i = j["best"], j["worst"]
        if best_i == worst_i:
            n_same += 1
            continue
        chosen, rejected = g["replies"][best_i], g["replies"][worst_i]

        if len(audit_rows) < args.audit_sample:
            audit_rows.append(
                {
                    "prompt": g["prompt"],
                    "replies": g["replies"],
                    "best_idx": best_i,
                    "worst_idx": worst_i,
                    "register_breaks_idx": sorted(j["register_breaks"]),
                    "confidence": j["confidence"],
                    "reason": j["reason"],
                }
            )

        conf = j["confidence"]
        conf_buckets[conf if conf in conf_buckets else "missing"] += 1

        if best_i in j["register_breaks"]:
            n_register += 1  # judge preferred a 반말-violating reply -- distrust the pick
            continue
        if (
            SequenceMatcher(None, normalize(chosen), normalize(rejected)).ratio()
            > NEAR_DUP_THRESHOLD
        ):
            n_near_dup += 1
            continue
        if conf not in KEEP_CONFIDENCE:
            n_low_conf += 1
            continue

        preferences.append(
            {
                "prompt": g["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "confidence": conf,
                "reason": j["reason"],
            }
        )

    with open(args.out_path, "w") as f:
        for p in preferences:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.audit_sample:
        audit_path = args.out_path.rsplit(".", 1)[0] + ".audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_rows, f, indent=2, ensure_ascii=False)

    # Length guard (DESIGN.md §4.3): DPO learns "longer = preferred" if the
    # judge has any length bias. If chosen is systematically longer than
    # rejected, revise the prompt before training.
    longer = sum(1 for p in preferences if len(p["chosen"]) > len(p["rejected"]))
    ratio = (
        round(
            sum(len(p["chosen"]) / max(len(p["rejected"]), 1) for p in preferences)
            / len(preferences),
            3,
        )
        if preferences
        else None
    )

    report = {
        "input": args.in_path,
        "n_groups": len(groups),
        "replies_per_group": len(groups[0]["replies"]) if groups else None,
        "n_judge_error": n_error,
        "n_best_equals_worst": n_same,
        "n_register_violation_discarded": n_register,
        "n_near_duplicate_discarded": n_near_dup,
        "n_low_confidence_discarded": n_low_conf,
        "kept_confidence": list(KEEP_CONFIDENCE),
        "confidence_buckets": conf_buckets,
        "n_final_preference_pairs": len(preferences),
        "length_guard": {
            "chosen_longer_frac": round(longer / len(preferences), 3) if preferences else None,
            "mean_chosen_over_rejected_len": ratio,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
