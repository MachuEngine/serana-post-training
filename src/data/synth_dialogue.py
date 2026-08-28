"""P1: synthetic SFT dialogue generation, PROMPTS.md §3 (id synth_dialogue).
Fills the SFT set to ~3k with GPT-4o-generated Korean exchanges, composed
per §3's targets (~75% ordinary / ~15% identity hard cases / ~10%
out-of-boundary deflections), computed against however many genuine pairs
P1 ingest actually yielded -- not hardcoded (PROMPTS.md §3: "{n} is
computed at runtime as 3000 - n_real_pairs").

Also generates the DPO prompt-only pool in the same pass (§3 last bullet)
-- prompts only, no replies, sampled later in P3 from the SFT adapter.

Run after translate.py: needs real translated Korean pairs as
{example_lines} anchor (PROMPTS.md §3: "draw on the genuine translated
pairs first").
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

PERSONA = yaml.safe_load(Path("config/persona.yaml").read_text())
JUDGE_MODEL = yaml.safe_load(Path("config/eval.yaml").read_text())["judge"]["model_id"]
TARGET_TOTAL = 3000
BATCH_SIZE = 20
MAX_RETRIES = 6
SEED = 42

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GEN_PROMPT = """{persona_name}는 다음과 같은 캐릭터야 (한국어):
{persona_profile}

{persona_name}의 실제 한국어 대사 예시:
{example_lines}

{persona_name}가 자신에 대해 알고 있고 언급할 수 있는 사실들:
{self_knowledge}

사용자가 {persona_name}에게 말을 걸고 {persona_name}가 완전히 캐릭터를 유지한 채
목소리, 가치관, 지식 범위에 맞게 대답하는 새로운 대화 {n}개를 한국어로 작성해줘.
주제는 다양하게 섞어줘.
{persona_name}는 {speech_level}로 말한다. 모든 대화에서 이 말투를 일관되게 유지해.
{extra_instruction}

정확히 이 형태의 JSON 객체 하나로만 반환해, 다른 텍스트는 넣지 마:
{{"exchanges": [{{"user": "...", "reply": "..."}}, ...]}}
"exchanges" 배열에는 정확히 {n}개의 항목이 있어야 해.
"""

EXTRA = {
    "ordinary": (
        "이번 배치는 평범한 일상 대화 주제로 채워줘 "
        "(그녀의 취향, 과거, 동료와의 관계, 세계관 안의 사물/장소에 대한 의견 등)."
    ),
    "hard_case": (
        "이번 배치는 전부 '정체성 하드 케이스'로 채워줘: 사용자가 그녀의 정체성을 캐묻거나 "
        "역할을 깨뜨리려고 시도하고, 그녀는 그 시도를 자기 세계관 안에서는 말이 안 되는 "
        "이상한 소리로 받아들이며 완전히 캐릭터를 유지한 채 대답해. "
        "AI라는 말을 부정하는 대신, 무슨 소린지 못 알아듣는 것처럼 자연스럽게 넘겨."
    ),
    "oob": (
        "이번 배치는 전부 '경계 밖 지식 회피'로 채워줘: 사용자가 그녀가 잠들기 전이나 "
        "현대 세계에 관한 것(그녀의 시대 이후에 일어난 일, 현대 기술/문물 등)을 묻고, "
        "그녀는 실제로 모르기 때문에 캐릭터에 맞게 자연스럽게 얼버무리거나 모른다고 반응해."
    ),
}

DPO_PROMPT_ONLY = """{persona_name}는 다음과 같은 캐릭터야 (한국어):
{persona_profile}

사용자가 {persona_name}에게 말을 걸 만한 새로운 한국어 대화 시작 문장 {n}개를 작성해줘.
{persona_name}의 대답은 쓰지 말고, 사용자 쪽 발화만 작성해. 평범한 대화, 정체성을 캐묻는
질문, 경계 밖 지식을 묻는 질문을 골고루 섞어줘. 각 문장은 서로 다른 주제여야 해.

정확히 이 형태의 JSON 객체 하나로만 반환해, 다른 텍스트는 넣지 마:
{{"prompts": ["...", "...", ...]}}
"prompts" 배열에는 정확히 {n}개의 항목이 있어야 해.
"""


def _chat_json(prompt: str) -> dict | None:
    """Returns None (not raises) after exhausting retries, so a batch of
    stray failures degrades to "fewer items this round, loop asks again"
    in the caller rather than crashing a multi-hour generation run."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content is None:
                time.sleep(1)
                continue
            return json.loads(content)
        except RateLimitError:
            time.sleep(min(2**attempt, 30))
        except json.JSONDecodeError:
            continue  # ask again
    return None


def _extract_list(parsed) -> list:
    """response_format=json_object requires a top-level object, so the
    model sometimes wraps the array under a key. Handle both shapes."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


MAX_STALLED_ROUNDS = 10  # consecutive empty-yield rounds before giving up


def _count_existing(out_path: Path, category: str | None = None) -> int:
    """Resume support: this generation run is long enough (30-60+ min at
    this account's rate limit) that writing only at the very end risks
    losing everything to a crash partway through. Every batch is appended
    and flushed immediately instead, and a re-run picks up the count
    already on disk rather than re-generating (and re-paying for) it."""
    if not out_path.exists():
        return 0
    n = 0
    for line in out_path.open():
        d = json.loads(line)
        if category is None or d.get("category") == category:
            n += 1
    return n


def generate_pairs(n: int, category: str, example_lines: list[str], out_path: Path) -> int:
    already = _count_existing(out_path, category)
    print(f"  [{category}] {already}/{n} already on disk")
    written = already
    rng = random.Random(SEED)
    stalled = 0
    with out_path.open("a") as f:
        while written < n and stalled < MAX_STALLED_ROUNDS:
            batch_n = min(BATCH_SIZE, n - written)
            examples = "\n".join(
                f'- "{line}"' for line in rng.sample(example_lines, min(8, len(example_lines)))
            )
            prompt = GEN_PROMPT.format(
                persona_name=PERSONA["persona_name"],
                persona_profile=PERSONA["persona_profile"].strip(),
                example_lines=examples,
                self_knowledge=PERSONA["self_knowledge"].strip(),
                n=batch_n,
                speech_level=PERSONA["speech_level"].strip(),
                extra_instruction=EXTRA[category],
            )
            parsed = _chat_json(prompt)
            items = _extract_list(parsed)
            before = written
            for it in items:
                if isinstance(it, dict) and it.get("user") and it.get("reply"):
                    f.write(
                        json.dumps(
                            {"user": it["user"], "reply": it["reply"], "category": category},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            f.flush()
            stalled = stalled + 1 if written == before else 0
            print(f"  [{category}] {written}/{n}")
    if stalled >= MAX_STALLED_ROUNDS:
        print(
            f"  [{category}] WARNING: stalled after {MAX_STALLED_ROUNDS} empty rounds, "
            f"stopping short at {written}/{n}"
        )
    return written


def generate_dpo_prompts(n: int, out_path: Path) -> int:
    already = _count_existing(out_path)
    print(f"  [dpo-prompts] {already}/{n} already on disk")
    written = already
    stalled = 0
    with out_path.open("a") as f:
        while written < n and stalled < MAX_STALLED_ROUNDS:
            batch_n = min(BATCH_SIZE, n - written)
            prompt = DPO_PROMPT_ONLY.format(
                persona_name=PERSONA["persona_name"],
                persona_profile=PERSONA["persona_profile"].strip(),
                n=batch_n,
            )
            parsed = _chat_json(prompt)
            items = _extract_list(parsed)
            before = written
            for s in items:
                if isinstance(s, str) and s.strip():
                    f.write(json.dumps({"prompt": s}, ensure_ascii=False) + "\n")
                    written += 1
            f.flush()
            stalled = stalled + 1 if written == before else 0
            print(f"  [dpo-prompts] {written}/{n}")
    if stalled >= MAX_STALLED_ROUNDS:
        print(f"  [dpo-prompts] WARNING: stalled, stopping short at {written}/{n}")
    return written


def main() -> None:
    real_pairs = [json.loads(line) for line in open("data/ko/raw/pairs.jsonl")]
    real_pairs_ok = [p for p in real_pairs if p.get("ko_reply") and p.get("ko_user")]
    n_real = len(real_pairs_ok)
    n_needed = max(TARGET_TOTAL - n_real, 0)

    n_hard = round(n_needed * 0.15)
    n_oob = round(n_needed * 0.10)
    n_ordinary = n_needed - n_hard - n_oob
    print(
        f"real pairs available: {n_real}; generating {n_needed} synthetic "
        f"({n_ordinary} ordinary / {n_hard} hard-case / {n_oob} oob)"
    )

    example_lines = [p["ko_reply"] for p in real_pairs_ok] or [
        # fallback: unpaired CPT lines, if real pairs are too few (PROMPTS §3 note)
        json.loads(line)["ko"]
        for line in open("data/ko/raw/cpt_pool.jsonl")
        if json.loads(line).get("ko")
    ]

    Path("data/ko/raw").mkdir(parents=True, exist_ok=True)
    synth_path = Path("data/ko/raw/synthetic_pairs.jsonl")
    n_written = generate_pairs(n_ordinary, "ordinary", example_lines, synth_path)
    n_written += generate_pairs(n_hard, "hard_case", example_lines, synth_path)
    n_written += generate_pairs(n_oob, "oob", example_lines, synth_path)
    print(f"synthetic pairs on disk: {n_written}")

    dpo_path = Path("data/ko/raw/dpo_prompt_pool.jsonl")
    n_dpo = generate_dpo_prompts(1000, dpo_path)
    print(f"DPO prompt-pool entries on disk: {n_dpo}")


if __name__ == "__main__":
    main()
