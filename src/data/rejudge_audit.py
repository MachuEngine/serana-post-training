"""One-off: re-run the fixed preference_judge (v2 -- voice_notes/
speech_level added, see PROMPTS.md §4 change note) against the same 30
audit_sample.jsonl items, to check whether the fix raises hand-audit
agreement before spending API budget re-judging all 861 pairs.

Does NOT touch human_choice (already filled in) or the original
p3_audit_key.json (v1 judge_choice) -- writes a separate v2 key so both
are comparable side by side.
"""
from __future__ import annotations

import json
import random

from src.eval.preference_judge import preference_judge


def main() -> None:
    items = [json.loads(line) for line in open("data/ko/audit_sample.jsonl")]

    n_agree = 0
    n_tie = 0
    results = {}
    for i, it in enumerate(items):
        rng = random.Random(1000 + i)  # distinct from build_audit_sample's own seed use
        j = preference_judge(it["prompt"], it["reply_a"], it["reply_b"], rng)
        results[it["id"]] = {"judge_choice_v2": j["choice"], "judge_reason_v2": j["reason"]}
        if j["choice"] == "tie":
            n_tie += 1
        elif j["choice"] == it["human_choice"]:
            n_agree += 1

    n_comparable = len(items) - n_tie
    agreement = n_agree / n_comparable if n_comparable else float("nan")

    print(json.dumps({"n": len(items), "n_judge_tie": n_tie, "n_comparable": n_comparable,
                       "n_agree": n_agree, "agreement_rate": round(agreement, 4)}, indent=2))

    with open("artifacts/runs/p3_audit_key_v2.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
