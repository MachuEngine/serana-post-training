"""DESIGN.md §4.4 judge validation: score the 50 human-labeled candidates
with `judge_pcs`, report Spearman correlation (human vs. judge) and
exact / ±1 agreement. Must clear `config/eval.yaml`'s `correlation_floor`
before any judge-backed number is trusted (CLAUDE.md, top-level
Evaluation section) -- this is the one gate, not a formality.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.eval.judge_pcs import judge_pcs

CORRELATION_FLOOR = yaml.safe_load(open("config/eval.yaml"))["judge"]["correlation_floor"]


def _rank(values: list[float]) -> list[float]:
    """Average-rank method for ties -- the standard Spearman tie handling."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    mean_ra, mean_rb = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_ra) * (y - mean_rb) for x, y in zip(ra, rb))
    var_a = sum((x - mean_ra) ** 2 for x in ra)
    var_b = sum((y - mean_rb) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / (var_a**0.5 * var_b**0.5)


def main() -> None:
    path = Path("data/eval/eval_set_v1/human_labels.jsonl")
    items = [json.loads(line) for line in path.open()]
    unscored = [it for it in items if it["human_score"] is None]
    if unscored:
        raise SystemExit(f"{len(unscored)} items still unscored -- finish human labeling first")

    judge_model_id = yaml.safe_load(open("config/eval.yaml"))["judge"]["model_id"]
    print(f"scoring {len(items)} items with judge_pcs ({judge_model_id})...")
    for i, it in enumerate(items):
        result = judge_pcs(it["user_turn"], it["reply"])
        it["judge_score"] = result.get("score")
        it["judge_violation"] = result.get("violation")
        it["judge_reason"] = result.get("reason")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)}")

    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    human_scores = [it["human_score"] for it in items]
    judge_scores = [it["judge_score"] for it in items]

    rho = spearman(human_scores, judge_scores)
    exact = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / len(items)
    within_1 = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / len(items)

    human_viol = [it["human_violation"] for it in items]
    judge_viol = [it["judge_violation"] for it in items]
    violation_agreement = sum(1 for h, j in zip(human_viol, judge_viol) if h == j) / len(items)

    report = {
        "n": len(items),
        "spearman": round(rho, 4),
        "exact_agreement": round(exact, 4),
        "within_1_agreement": round(within_1, 4),
        "violation_flag_agreement": round(violation_agreement, 4),
        "correlation_floor": CORRELATION_FLOOR,
        "passes_floor": rho >= CORRELATION_FLOOR,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("data/eval/eval_set_v1/judge_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not report["passes_floor"]:
        print(
            f"\nFAILED floor ({CORRELATION_FLOOR}) -- revise judge_pcs prompt before "
            "trusting any judge-backed number (DESIGN.md §4.4)."
        )


if __name__ == "__main__":
    main()
