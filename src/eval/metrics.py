"""Ties the judges + rule-checks into the four persona quality metrics
and two DPO guards (DESIGN.md §4.2/§4.3), plus the shared bootstrap-CI
helper every metric reports through (§4.5: mean +/- 95% CI, >=1000
resamples, "real" only if CIs for two configs don't overlap).

Pure aggregation -- every function here takes pre-computed per-item
results (already scored by the judges/rule-checks in this package), not
raw model output. That keeps this module testable without a GPU or an
API key, and keeps scoring (which needs a live model / OpenAI calls)
separate from aggregation (which doesn't).
"""

from __future__ import annotations

import random
from collections import Counter


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Bootstrap over `values` (one number per eval item -- a 0/1 for
    binary metrics like PCS/PRS/accuracy, a token count for reply
    length). >=1000 resamples per DESIGN.md §4.5."""
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return {
        "mean": sum(values) / n,
        "ci_low": means[max(lo_idx, 0)],
        "ci_high": means[min(hi_idx, n_resamples - 1)],
        "n": n,
    }


def cis_overlap(a: dict, b: dict) -> bool:
    """DESIGN.md §4.5: a difference is "real" only if the 95% CIs don't
    overlap. Otherwise report "no measurable difference" -- itself a finding."""
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


def paired_bootstrap_diff(
    a_values: list[float],
    b_values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of the per-item difference a_i - b_i (DESIGN.md §4.5).

    `a_values[i]` and `b_values[i]` must be the same eval item scored under
    two configs -- the caller aligns them by item id. Each resample draws the
    per-item *differences* with replacement, which is the same as drawing one
    item-index set and applying it to both configs: per-item difficulty
    cancels, so this CI is tighter than comparing two independent
    `bootstrap_ci` runs with `cis_overlap`. Use it on the raw 1-5 judge score
    (not the 0/1 violation collapse) where the extra resolution shows.

    Returns the mean paired difference, its 95% CI, and whether that CI
    excludes zero -- a difference the eval set can actually resolve.
    """
    if len(a_values) != len(b_values):
        raise ValueError("a_values and b_values must be the same length (paired)")
    if not a_values:
        raise ValueError("values must be non-empty")
    n = len(a_values)
    diffs = [a_values[i] - b_values[i] for i in range(n)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    ci_low = means[max(lo_idx, 0)]
    ci_high = means[min(hi_idx, n_resamples - 1)]
    return {
        "diff_mean": sum(diffs) / n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "excludes_zero": ci_low > 0 or ci_high < 0,
    }


def pcs(items: list[dict]) -> dict:
    """items: [{"breaks_register": bool, "admits_ai": bool, "judge_violation": bool}, ...]
    DESIGN.md §4.2: PCS = 1 - (violations/total). violation = union of the
    hard-constraint rule-checks and judge_pcs's soft-trait violation flag."""
    per_item_violation = [
        1.0 if (it["breaks_register"] or it["admits_ai"] or it["judge_violation"]) else 0.0
        for it in items
    ]
    result = bootstrap_ci(per_item_violation)
    for key in ("mean", "ci_low", "ci_high"):
        result[key] = 1.0 - result[key]
    result["ci_low"], result["ci_high"] = result["ci_high"], result["ci_low"]  # flip after 1-x
    return result


def prs(items: list[dict]) -> dict:
    """items: [{"admits_ai": bool, "judge_held": bool, "failure_type": str}, ...]
    DESIGN.md §4.2: PRS = fraction of attack probes held. broke = union of
    the rule-check firing or the judge saying it didn't hold; held = not broke."""
    per_item_held = [0.0 if (it["admits_ai"] or not it["judge_held"]) else 1.0 for it in items]
    return bootstrap_ci(per_item_held)


def prs_failure_breakdown(items: list[dict]) -> dict[str, int]:
    """Per-category failure_type counts (DESIGN.md §4.2.1: "report the
    failure_type breakdown, not just PRS"). Only counts items that broke."""
    counts: Counter[str] = Counter()
    for it in items:
        broke = it["admits_ai"] or not it["judge_held"]
        if broke:
            # rule-check catching an AI-admission the judge missed has no
            # judge-provided failure_type -- label it explicitly rather
            # than silently dropping it from the breakdown.
            counts[it.get("failure_type") or "admits_ai_rule_only"] += 1
    return dict(counts)


def knowledge_boundary_accuracy(items: list[dict]) -> dict:
    """items: [{"correct": bool}, ...] from judge_boundary."""
    return bootstrap_ci([1.0 if it["correct"] else 0.0 for it in items])


def mean_reply_length(token_counts: list[int]) -> dict:
    """DESIGN.md §4.3 DPO guard: mean output tokens per config."""
    return bootstrap_ci([float(c) for c in token_counts])


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Corpus-level distinct-n (Li et al. 2016): unique n-grams / total
    n-grams across all replies for one config. DESIGN.md §4.3's
    degeneration check ("distinct-n repetition rate") -- a low value
    flags repetitive/mode-collapsed output. Character n-grams, not word
    n-grams: Korean isn't whitespace-tokenized the way English is, so a
    word-level n-gram would undercount repetition within a word/phrase."""
    total = 0
    seen: set[str] = set()
    for text in texts:
        chars = text.replace(" ", "")
        for i in range(len(chars) - n + 1):
            seen.add(chars[i : i + n])
            total += 1
    if total == 0:
        return 0.0
    return len(seen) / total
