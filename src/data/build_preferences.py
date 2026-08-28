"""P3 step 2-4 (DESIGN.md §3.4): judge the (prompt, reply_a, reply_b)
pairs from scripts/generate_replies.py, discard ties and near-identical
pairs, emit the final (prompt, chosen, rejected) preference set.

Runs off the GPU VM -- API-only (CLAUDE.md cost discipline).
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from openai import RateLimitError

from src.eval.preference_judge import preference_judge

MAX_WORKERS = 2  # this account's gpt-4o TPM cap bit us in P1 at higher concurrency.
# NOTE on the 476/893 (53%) error spike seen when preference_judge went to v2:
# initially attributed to TPM throttling from the longer prompt (see
# judge_with_retry -- only RateLimitError is caught, and exit code 0 with no
# crash means every "error" really was retries exhausted, not a schema bug).
# That diagnosis is now suspect: the account's OpenAI credit hit zero around
# the same time, and openai-python raises insufficient_quota as RateLimitError
# too, so retrying it is never going to succeed no matter how patient. Lowering
# MAX_WORKERS/raising MAX_RETRIES here does NOT fix a $0-balance failure --
# check the OpenAI billing dashboard first if this error rate reappears.
MAX_RETRIES = 8
NEAR_DUP_THRESHOLD = 0.92


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def judge_with_retry(prompt: str, reply_a: str, reply_b: str, seed: int) -> dict:
    rng = random.Random(seed)
    for attempt in range(MAX_RETRIES):
        try:
            return preference_judge(prompt, reply_a, reply_b, rng)
        except RateLimitError:
            time.sleep(min(2**attempt, 60))
    return {"choice": None, "reason": "exhausted retries"}


def main() -> None:
    items = [json.loads(line) for line in open("data/ko/raw/reply_pairs.jsonl")]
    print(f"judging {len(items)} pairs...")

    judged = [None] * len(items)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(judge_with_retry, it["prompt"], it["reply_a"], it["reply_b"], i): i
            for i, it in enumerate(items)
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            judged[i] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(items)}")

    n_near_dup = 0
    n_tie = 0
    n_error = 0
    preferences = []
    for it, j in zip(items, judged):
        a_norm, b_norm = normalize(it["reply_a"]), normalize(it["reply_b"])
        if SequenceMatcher(None, a_norm, b_norm).ratio() > NEAR_DUP_THRESHOLD:
            n_near_dup += 1
            continue
        choice = j["choice"]
        if choice not in ("A", "B"):
            if choice == "tie":
                n_tie += 1
            else:
                n_error += 1
            continue
        chosen, rejected = (
            (it["reply_a"], it["reply_b"]) if choice == "A" else (it["reply_b"], it["reply_a"])
        )
        preferences.append(
            {"prompt": it["prompt"], "chosen": chosen, "rejected": rejected, "reason": j["reason"]}
        )

    with open("data/ko/prefs_1k.jsonl", "w") as f:
        for p in preferences:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "n_pairs_generated": len(items),
        "n_near_duplicate_discarded": n_near_dup,
        "n_tie_discarded": n_tie,
        "n_judge_error": n_error,
        "n_final_preference_pairs": len(preferences),
        "tie_rate": round(n_tie / len(items), 4),
        "near_dup_rate": round(n_near_dup / len(items), 4),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("artifacts/runs/p3_preference_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
