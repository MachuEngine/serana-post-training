"""P1: leakage guard (DESIGN.md §4.6). Run before trusting the built
corpora, and again before each training run per the same section.

Checks, each exact + near-duplicate (SequenceMatcher ratio > 0.92):
1. Eval prompts / attack probes / style reference never appear in the
   CPT corpus or SFT set.
2. Attack probes never appear in the SFT set (hard cases teach, probes
   measure -- §4.6.3).
3. The DPO prompt pool excludes every eval prompt (§4.6.4).

Exits nonzero on any violation so it can gate a training run.
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

THRESHOLD = 0.92


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def near_dup(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() > THRESHOLD


def find_overlaps(needles: list[tuple[str, str]], haystack: list[str]) -> list[tuple[str, str]]:
    """needles: (id, text). Returns (id, matched haystack text) for any hit."""
    hits = []
    haystack_norm = [normalize(h) for h in haystack]
    for id_, text in needles:
        n = normalize(text)
        for h_orig, h_norm in zip(haystack, haystack_norm):
            if n == h_norm or near_dup(n, h_norm):
                hits.append((id_, h_orig))
                break
    return hits


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def main() -> int:
    eval_prompts = load_jsonl(Path("data/eval/eval_set_v1/eval_prompts.jsonl"))
    attack_probes = load_jsonl(Path("data/eval/eval_set_v1/attack_probes.jsonl"))
    style_ref = load_jsonl(Path("data/eval/eval_set_v1/style_reference.jsonl"))

    cpt_text = (
        Path("data/ko/cpt_corpus.txt").read_text()
        if Path("data/ko/cpt_corpus.txt").exists()
        else ""
    )
    cpt_paragraphs = [p for p in cpt_text.split("\n\n") if p.strip()]

    sft_pairs = load_jsonl(Path("data/ko/sft_3k.jsonl"))
    sft_texts = [f"{p['user']} {p['reply']}" for p in sft_pairs]

    # the CLEANED pool (build_corpora.py's dedup + eval-overlap filter),
    # not data/ko/raw/dpo_prompt_pool.jsonl -- this check is what the
    # cleaning step is verified against.
    dpo_prompts = load_jsonl(Path("data/ko/dpo_prompt_pool.jsonl"))

    violations = []

    eval_needles = [(e["id"], e["prompt_ko"]) for e in eval_prompts]
    probe_needles = [(a["id"], a["prompt_ko"]) for a in attack_probes]
    style_needles = [(f"style_{i}", s["line"]) for i, s in enumerate(style_ref)]

    for name, needles, haystack in [
        ("eval_prompts -> CPT corpus", eval_needles, cpt_paragraphs),
        ("eval_prompts -> SFT set", eval_needles, sft_texts),
        ("attack_probes -> CPT corpus", probe_needles, cpt_paragraphs),
        ("attack_probes -> SFT set", probe_needles, sft_texts),
        ("style_reference -> CPT corpus", style_needles, cpt_paragraphs),
        ("style_reference -> SFT set", style_needles, sft_texts),
        ("eval_prompts -> DPO prompt pool", eval_needles, [d["prompt"] for d in dpo_prompts]),
    ]:
        hits = find_overlaps(needles, haystack)
        if hits:
            violations.append((name, hits))

    if violations:
        print("LEAKAGE CHECK FAILED:")
        for name, hits in violations:
            print(f"  {name}: {len(hits)} overlap(s)")
            for id_, matched in hits[:5]:
                print(f"    {id_}: {matched[:80]!r}")
        return 1

    print(
        "leakage check passed: no overlap found "
        f"(checked {len(eval_prompts)} eval prompts, {len(attack_probes)} attack probes, "
        f"{len(style_ref)} style-reference lines against "
        f"{len(cpt_paragraphs)} CPT paragraphs, {len(sft_pairs)} SFT pairs, "
        f"{len(dpo_prompts)} DPO prompts)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
