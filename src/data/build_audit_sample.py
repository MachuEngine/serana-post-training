"""P3 step 5 (DESIGN.md §3.4): build a blind ~30-pair sample from the
final preference set for a hand-audit -- "if your judgement often
disagrees with the judge, fix the preference prompt before spending GPU
time." This is the one P3 step that can't be automated, same reason
P1's human_labels.jsonl needed a person: using an LLM to check the LLM
judge would make the circularity guard circular.

Output is split in two so the audit stays blind:
  - data/ko/audit_sample.jsonl   -- prompt + reply_a/reply_b (chosen/
    rejected randomly reassigned to A/B per item), human_choice: null.
    This is the file to fill in by hand.
  - artifacts/runs/p3_audit_key.json -- the judge's real choice (in
    terms of this sample's A/B labels) and reason, per id. Not meant to
    be read before filling in audit_sample.jsonl -- it's the answer key
    for the agreement check afterward.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
N_AUDIT = 30


def main() -> None:
    rng = random.Random(SEED)
    prefs = [json.loads(line) for line in open("data/ko/prefs_1k.jsonl")]
    sample = rng.sample(prefs, min(N_AUDIT, len(prefs)))

    audit_items = []
    answer_key = {}
    for i, p in enumerate(sample):
        item_id = f"audit_{i:02d}"
        swapped = rng.random() < 0.5
        reply_a, reply_b = (p["rejected"], p["chosen"]) if swapped else (p["chosen"], p["rejected"])
        audit_items.append(
            {
                "id": item_id,
                "prompt": p["prompt"],
                "reply_a": reply_a,
                "reply_b": reply_b,
                "human_choice": None,  # fill in "A" / "B" / "tie"
            }
        )
        answer_key[item_id] = {
            "judge_choice": "B" if swapped else "A",
            "judge_reason": p["reason"],
        }

    Path("data/ko/audit_sample.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in audit_items) + "\n"
    )
    Path("artifacts/runs/p3_audit_key.json").write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False)
    )

    print(f"wrote {len(audit_items)} blind audit items to data/ko/audit_sample.jsonl")
    print("fill in human_choice (\"A\" / \"B\" / \"tie\") for each item by hand, then")
    print("compare against artifacts/runs/p3_audit_key.json (don't peek first).")


if __name__ == "__main__":
    main()
