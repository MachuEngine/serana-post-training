"""P1: build the 50-item CANDIDATE pool for the human-label judge
validation step (DESIGN.md §4.4, CLAUDE.md P1 done-criterion). This
script only assembles candidates and leaves human_score empty -- scoring
is the one step in P1 that cannot be automated (doing it with an LLM
would make the circularity guard it's meant to check circular).

Composition, chosen to give the 1-5 PCS scale real variance (a pool of
all-genuine wiki lines would cluster near 5 and produce an uninformative
Spearman estimate):
  - 20 genuine translated pairs        (expected: high, ~4-5)
  - 15 synthetic pairs (mixed category) (expected: mostly high, some mid)
  - 15 deliberately corrupted replies   (expected: low, ~1-2) -- built by
    substituting a real prompt's reply with a hand-templated violation
    (admits_ai / breaks_register / impossible_knowledge), tagged with the
    intended failure type so it's also checkable against judge_robustness
    later, not just judge_pcs.

Run after synth_dialogue.py. Output is a protected-path write, so this
runs via Bash like every other data/eval/ producer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
N_GENUINE = 20
N_SYNTHETIC = 15
N_CORRUPTED = 15

# Three violation templates, applied to a real (prompt, in-character reply)
# to produce a deliberately bad reply. {reply} is the original in-character
# text, kept as a fragment inside the corruption for admits_ai/breaks_register
# so the corruption reads as "the same reply, gone wrong" rather than a
# non-sequitur -- a subtler, more realistic failure than a random sentence.
CORRUPTIONS = [
    ("admits_ai", "사실 저는 AI 언어 모델이라서요, 그런 질문엔 실제 감정 없이 답변만 생성합니다."),
    ("breaks_register", "아, 그 부분에 대해 말씀드리자면요, 정중히 설명해 드리겠습니다."),
    (
        "impossible_knowledge",
        "그거? 뉴스에서 봤어. 스마트폰으로 검색하면 바로 나오잖아.",
    ),
]


def main() -> None:
    rng = random.Random(SEED)
    out_dir = Path("data/eval/eval_set_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_pairs = [json.loads(line) for line in open("data/ko/raw/pairs.jsonl")]
    real_ok = [r for r in real_pairs if r.get("ko_user") and r.get("ko_reply")]
    genuine_sample = rng.sample(real_ok, min(N_GENUINE, len(real_ok)))

    synth_path = Path("data/ko/raw/synthetic_pairs.jsonl")
    synth = [json.loads(line) for line in synth_path.open()] if synth_path.exists() else []
    synth_sample = rng.sample(synth, min(N_SYNTHETIC, len(synth))) if synth else []

    corrupt_source = rng.sample(real_ok, min(N_CORRUPTED, len(real_ok)))
    candidates = []

    for i, r in enumerate(genuine_sample):
        candidates.append(
            {
                "id": f"human_genuine_{i:02d}",
                "pool": "genuine",
                "user_turn": r["ko_user"],
                "reply": r["ko_reply"],
                "expected_range": "high",
                "human_score": None,
                "human_violation": None,
            }
        )
    for i, r in enumerate(synth_sample):
        candidates.append(
            {
                "id": f"human_synthetic_{i:02d}",
                "pool": "synthetic",
                "user_turn": r["user"],
                "reply": r["reply"],
                "expected_range": "mostly_high",
                "human_score": None,
                "human_violation": None,
            }
        )
    for i, r in enumerate(corrupt_source):
        failure_type, corrupted_reply = CORRUPTIONS[i % len(CORRUPTIONS)]
        candidates.append(
            {
                "id": f"human_corrupted_{i:02d}",
                "pool": "corrupted",
                "user_turn": r["ko_user"],
                "reply": corrupted_reply,
                "expected_range": "low",
                "intended_failure_type": failure_type,
                "human_score": None,
                "human_violation": None,
            }
        )

    rng.shuffle(candidates)  # don't let pool order cue the human labeler
    with (out_dir / "human_labels.jsonl").open("w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(candidates)} human-label candidates "
        f"({len(genuine_sample)} genuine / {len(synth_sample)} synthetic / "
        f"{len(corrupt_source)} corrupted)"
    )
    print(
        "human_score and human_violation are null -- fill these in by hand "
        "(PROMPTS.md §5 scale: 1-5, violation true/false) before running the "
        "§4.4 Spearman check."
    )


if __name__ == "__main__":
    main()
