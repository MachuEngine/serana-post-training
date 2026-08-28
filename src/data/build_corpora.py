"""P1: assemble the final training files from translated + synthetic
data (DESIGN.md §3.1/§3.2/§3.3), and the style-reference eval component.

- data/ko/cpt_corpus.txt   -- raw text CLM (CPT pool only; style reference
                               is excluded per DESIGN.md §4.1: "excluded
                               from the CPT corpus too")
- data/ko/sft_3k.jsonl     -- real translated pairs + synthetic pairs,
                               deduplicated against each other
- data/ko/dpo_prompt_pool.jsonl -- cleaned P3 input: deduplicated (the raw
  pool had ~24 near-identical "what's your favorite magic" variants --
  independent per-batch generation has no memory of earlier batches, so
  topic collisions across batches are expected, not a one-off) and
  filtered against eval prompts (DESIGN.md §4.6 rule 4: "the DPO prompt
  pool excludes every eval prompt" -- caught for real by check_leakage.py
  finding an exact match against eval prompt ib14, not a hypothetical).
- data/eval/eval_set_v1/style_reference.jsonl -- held-out slice, eval-only
  (emitted via this script since data/eval/ is a protected path)

Paths match config/base.yaml train.data.{cpt_file,sft_file} and
config/base.yaml paths.eval_set.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def dedup_sft(pairs: list[dict]) -> tuple[list[dict], int]:
    """Exact dedup by (user, reply), then near-dup by reply text alone
    (PROMPTS.md §3: "a synthetic exchange that paraphrases a real one
    inflates effective epochs"). O(n^2) at ~3k is acceptable for a
    one-off P1 build."""
    seen = set()
    exact = []
    for p in pairs:
        key = (normalize(p["user"]), normalize(p["reply"]))
        if key not in seen:
            seen.add(key)
            exact.append(p)

    kept: list[dict] = []
    dropped_near = 0
    kept_norms: list[str] = []
    for p in exact:
        norm = normalize(p["reply"])
        if any(SequenceMatcher(None, norm, k).ratio() > 0.92 for k in kept_norms):
            dropped_near += 1
            continue
        kept.append(p)
        kept_norms.append(norm)
    return kept, (len(pairs) - len(exact)) + dropped_near


def dedup_flat(texts: list[str], threshold: float = 0.92) -> tuple[list[int], int]:
    """Indices to keep, exact-then-near-dup, same policy as dedup_sft but
    over a flat list of strings (used for the DPO prompt pool)."""
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        key = normalize(t)
        if key not in seen:
            seen[key] = i
    exact_idx = list(seen.values())

    kept_idx: list[int] = []
    kept_norms: list[str] = []
    dropped_near = 0
    for i in exact_idx:
        norm = normalize(texts[i])
        if any(SequenceMatcher(None, norm, k).ratio() > threshold for k in kept_norms):
            dropped_near += 1
            continue
        kept_idx.append(i)
        kept_norms.append(norm)
    return kept_idx, (len(texts) - len(exact_idx)) + dropped_near


def main() -> None:
    Path("data/ko").mkdir(parents=True, exist_ok=True)
    Path("data/eval/eval_set_v1").mkdir(parents=True, exist_ok=True)

    # --- CPT corpus (raw text, CPT pool only) ---
    cpt_lines = [json.loads(line) for line in open("data/ko/raw/cpt_pool.jsonl")]
    cpt_ok = [c["ko"] for c in cpt_lines if c.get("ko")]
    with open("data/ko/cpt_corpus.txt", "w") as f:
        f.write("\n\n".join(cpt_ok) + "\n")
    print(f"cpt_corpus.txt: {len(cpt_ok)}/{len(cpt_lines)} lines")

    # --- style reference (eval-only, excluded from all training) ---
    style_lines = [json.loads(line) for line in open("data/ko/raw/style_reference.jsonl")]
    style_ok = [{"line": s["ko"], "source": s["source"]} for s in style_lines if s.get("ko")]
    with open("data/eval/eval_set_v1/style_reference.jsonl", "w") as f:
        for s in style_ok:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"style_reference.jsonl: {len(style_ok)}/{len(style_lines)} lines")

    # --- SFT set: real translated pairs + synthetic ---
    real_pairs = [json.loads(line) for line in open("data/ko/raw/pairs.jsonl")]
    real_ok = [
        {"user": r["ko_user"], "reply": r["ko_reply"], "category": "real"}
        for r in real_pairs
        if r.get("ko_user") and r.get("ko_reply")
    ]
    synth_path = Path("data/ko/raw/synthetic_pairs.jsonl")
    synth = [json.loads(line) for line in synth_path.open()] if synth_path.exists() else []

    combined = real_ok + synth
    deduped, n_dropped = dedup_sft(combined)
    with open("data/ko/sft_3k.jsonl", "w") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_real_kept = sum(1 for p in deduped if p["category"] == "real")
    report = {
        "real_pairs_translated": len(real_ok),
        "synthetic_pairs_generated": len(synth),
        "combined_before_dedup": len(combined),
        "dropped_exact_or_near_dup": n_dropped,
        "sft_set_final_size": len(deduped),
        "real_pairs_in_final_set": n_real_kept,
        "real_pair_ratio_of_sft_set": round(n_real_kept / len(deduped), 4) if deduped else 0.0,
    }

    # --- DPO prompt pool: dedup + filter out anything overlapping an eval prompt ---
    dpo_raw_path = Path("data/ko/raw/dpo_prompt_pool.jsonl")
    dpo_raw = (
        [json.loads(line)["prompt"] for line in dpo_raw_path.open()]
        if dpo_raw_path.exists()
        else []
    )
    keep_idx, dpo_dropped_dup = dedup_flat(dpo_raw)
    dpo_deduped = [dpo_raw[i] for i in keep_idx]

    eval_path = Path("data/eval/eval_set_v1/eval_prompts.jsonl")
    eval_norms = (
        [normalize(json.loads(line)["prompt_ko"]) for line in eval_path.open()]
        if eval_path.exists()
        else []
    )
    dpo_final = [
        p
        for p in dpo_deduped
        if not any(SequenceMatcher(None, normalize(p), e).ratio() > 0.92 for e in eval_norms)
    ]
    dpo_dropped_eval_overlap = len(dpo_deduped) - len(dpo_final)
    with open("data/ko/dpo_prompt_pool.jsonl", "w") as f:
        for p in dpo_final:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    report["dpo_prompt_pool"] = {
        "raw": len(dpo_raw),
        "dropped_exact_or_near_dup": dpo_dropped_dup,
        "dropped_eval_prompt_overlap": dpo_dropped_eval_overlap,
        "final_size": len(dpo_final),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("data/ko/build_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
