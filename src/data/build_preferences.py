"""P4 redo (artifacts/runs/p4_dpo_redo_plan.md): judge the reply groups
from scripts/generate_reply_groups.py -- each group is N replies the SFT
model produced for one prompt -- and emit the final (prompt, chosen,
rejected) preference set.

Each group's pair is (judge's best reply, judge's worst reply). Every
group lands in exactly one funnel bucket (they sum to n_groups):
  error            -- judge errored or returned a wrong-shape response
  best_equals_worst
  register         -- the best reply is one the judge flagged as breaking
                      반말 (v1's failure mode -- a register violation should
                      never be the preferred output)
  near_duplicate   -- best and worst are near-identical after normalization
  low_confidence   -- pick confidence not in KEEP_CONFIDENCE (near-coin-flip
                      picks are what gave P4's DPO run no learnable signal)
  kept

Exits non-zero if no pairs survive, rather than writing an empty training
file (the P1/P3 lesson: a silent "exit 0 with near-empty output" is worse
than a crash).

Default paths are the redo's own, so this does NOT overwrite the shipped
P3 artifacts (data/ko/prefs_1k.jsonl, artifacts/runs/p3_preference_report.json).
`--audit-sample N` writes N judged groups to <out>.audit.json for a
by-eye check of the judge's picks.

Runs off the GPU VM -- API-only (CLAUDE.md cost discipline).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from src.eval.preference_judge import JudgeSchemaError, preference_judge
from src.eval.rule_checks import breaks_register

MAX_WORKERS = 2  # gpt-4o TPM cap bit us in P1 at higher concurrency.
MAX_RETRIES = 8
NEAR_DUP_THRESHOLD = 0.92
KEEP_CONFIDENCE = ("high", "medium")  # drop "low"/missing -- the v3 signal filter

# Retried: genuine rate limits, timeouts, connection drops, 5xx. NOT retried
# (fail fast): auth errors, a wrong judge model_id, bad-request, and a $0
# balance -- openai raises insufficient_quota as RateLimitError, and that is
# the one that actually bit this project twice (P1, P3); retrying it just
# burns hours of backoff before writing nothing.
RETRYABLE = (APITimeoutError, APIConnectionError, InternalServerError)


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def judge_with_retry(prompt: str, replies: list[str], seed: int) -> dict | str:
    """A judge result dict, or a marker string:
      "schema"    -- unjudgeable group (bad response shape); fine to drop.
      "api_error" -- transient API error that outlasted MAX_RETRIES; an
                     infrastructure problem, counted separately so a run
                     that lost many groups to it is not mistaken for a
                     clean one.
    A non-retryable error (auth, wrong model, $0 balance) propagates."""
    rng = random.Random(seed)
    for attempt in range(MAX_RETRIES):
        try:
            return preference_judge(prompt, replies, rng)
        except JudgeSchemaError:
            return "schema"
        except RateLimitError as e:
            if getattr(e, "code", None) == "insufficient_quota":
                raise  # $0 balance -- no amount of retrying fixes it
            time.sleep(min(2**attempt, 60))
        except RETRYABLE:
            time.sleep(min(2**attempt, 60))
    return "api_error"


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

    judged: list[dict | str | None] = [None] * len(groups)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(judge_with_retry, g["prompt"], g["replies"], i): i
            for i, g in enumerate(groups)
        }
        try:
            done = 0
            for fut in as_completed(futs):
                judged[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(groups)}")
        except BaseException:
            for f in futs:  # a non-retryable error -> stop the queued API calls
                f.cancel()
            raise

    funnel = dict.fromkeys(
        (
            "schema_error",
            "api_error",
            "best_equals_worst",
            "register",
            "near_duplicate",
            "low_confidence",
            "kept",
        ),
        0,
    )
    # confidence distribution over every group with a valid best/worst pick,
    # independent of the later filters -- a separate view, not part of the funnel
    conf_dist = {"high": 0, "medium": 0, "low": 0, "missing": 0}
    reg_agree = reg_judge_only = reg_regex_only = reg_checked = 0
    preferences = []
    audit_rows = []
    for g, j in zip(groups, judged):
        if j == "schema":
            funnel["schema_error"] += 1
            continue
        if j == "api_error":
            funnel["api_error"] += 1
            continue
        best_i, worst_i = j["best"], j["worst"]
        if best_i == worst_i:
            funnel["best_equals_worst"] += 1
            continue
        chosen, rejected = g["replies"][best_i], g["replies"][worst_i]

        conf = j["confidence"]
        conf_dist[conf if conf in conf_dist else "missing"] += 1

        # Cross-check the judge's register call against the regex (rule_checks).
        # A high regex_only count means the judge is not doing its register job.
        for idx, reply in enumerate(g["replies"]):
            reg_checked += 1
            judge_flag = idx in j["register_breaks"]
            regex_flag = breaks_register(reply)
            if judge_flag == regex_flag:
                reg_agree += 1
            elif judge_flag:
                reg_judge_only += 1
            else:
                reg_regex_only += 1

        if len(audit_rows) < args.audit_sample:
            audit_rows.append(
                {
                    "prompt": g["prompt"],
                    "replies": g["replies"],
                    "best_idx": best_i,
                    "worst_idx": worst_i,
                    "chosen": chosen,
                    "rejected": rejected,
                    "register_breaks_idx": sorted(j["register_breaks"]),
                    "regex_flags_chosen": breaks_register(chosen),
                    "confidence": conf,
                    "reason": j["reason"],
                }
            )

        # Drop the pair if the best reply breaks 반말 by EITHER signal: the
        # judge's flag or the non-circular regex. The judge is ~73% reliable
        # and can miss it (v1's failure); the regex is the backstop.
        if best_i in j["register_breaks"] or breaks_register(chosen):
            funnel["register"] += 1
            continue
        if (
            SequenceMatcher(None, normalize(chosen), normalize(rejected)).ratio()
            > NEAR_DUP_THRESHOLD
        ):
            funnel["near_duplicate"] += 1
            continue
        if conf not in KEEP_CONFIDENCE:
            funnel["low_confidence"] += 1
            continue

        funnel["kept"] += 1
        preferences.append(
            {
                "prompt": g["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "confidence": conf,
                "reason": j["reason"],
            }
        )

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
        "funnel": funnel,  # sums to n_groups
        "confidence_of_valid_picks": conf_dist,
        "register_check_vs_regex": {
            "replies_checked": reg_checked,
            "agree": reg_agree,
            "judge_flagged_only": reg_judge_only,
            "regex_flagged_only": reg_regex_only,
        },
        "n_final_preference_pairs": len(preferences),
        "length_guard": {
            "chosen_longer_frac": round(longer / len(preferences), 3) if preferences else None,
            "mean_chosen_over_rejected_len": ratio,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Written before the empty-set check: when everything is filtered out,
    # the audit file is exactly the diagnostic you need.
    if args.audit_sample:
        audit_path = os.path.splitext(args.out_path)[0] + ".audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_rows, f, indent=2, ensure_ascii=False)

    api_error_limit = max(10, len(groups) // 20)  # 5%, floor 10
    if funnel["api_error"] > api_error_limit:
        sys.exit(
            f"{funnel['api_error']} groups lost to API errors (limit {api_error_limit}); "
            f"see {args.report_path}. not writing a partial training file -- rerun once "
            "the API is healthy."
        )

    if not preferences:
        sys.exit(
            f"no preference pairs survived ({args.in_path} -> 0); see {args.report_path} "
            f"and the audit file. not writing an empty training file."
        )

    with open(args.out_path, "w") as f:
        for p in preferences:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
