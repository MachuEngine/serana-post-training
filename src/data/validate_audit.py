"""P3 step 5, second half (DESIGN.md §3.4): compare the hand-filled
data/ko/audit_sample.jsonl against artifacts/runs/p3_audit_key.json
(the preference judge's real choices) and report agreement. Run this
after human_choice is filled in for every item -- not before, or the
audit isn't blind.
"""

from __future__ import annotations

import json
from pathlib import Path

AGREEMENT_FLOOR = 0.7  # below this, DESIGN.md §3.4 says fix the preference prompt first


def main() -> None:
    items = [json.loads(line) for line in open("data/ko/audit_sample.jsonl")]
    key = json.loads(Path("artifacts/runs/p3_audit_key.json").read_text())

    unfilled = [it for it in items if it["human_choice"] is None]
    if unfilled:
        raise SystemExit(f"{len(unfilled)} items still unfilled -- finish the hand-audit first")

    n_agree = 0
    n_human_tie = 0
    disagreements = []
    for it in items:
        judge_choice = key[it["id"]]["judge_choice"]
        human_choice = it["human_choice"]
        if human_choice == "tie":
            n_human_tie += 1
            continue
        if human_choice == judge_choice:
            n_agree += 1
        else:
            disagreements.append(
                {
                    "id": it["id"],
                    "human_choice": human_choice,
                    "judge_choice": judge_choice,
                    "judge_reason": key[it["id"]]["judge_reason"],
                }
            )

    n_comparable = len(items) - n_human_tie
    agreement = n_agree / n_comparable if n_comparable else float("nan")

    report = {
        "n": len(items),
        "n_human_tie": n_human_tie,
        "n_comparable": n_comparable,
        "n_agree": n_agree,
        "agreement_rate": round(agreement, 4),
        "agreement_floor": AGREEMENT_FLOOR,
        "passes_floor": agreement >= AGREEMENT_FLOOR,
        "disagreements": disagreements,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("artifacts/runs/p3_audit_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not report["passes_floor"]:
        print(
            f"\nFAILED floor ({AGREEMENT_FLOOR}) -- revise the preference_judge prompt "
            "(PROMPTS.md §4) before spending GPU time on DPO (DESIGN.md §3.4)."
        )


if __name__ == "__main__":
    main()
