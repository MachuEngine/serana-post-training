"""Preference judge -- PROMPTS.md §4, id `preference_judge` v2. Training-
loop prompt (DESIGN.md §3.4/§4.4 circularity guard): a pairwise choice,
deliberately different shape from `judge_pcs` (§5, absolute rating) --
never reuse this rubric for measurement, never reuse judge_pcs for this.

v2: added voice_notes/speech_level (v1 gave the judge only persona_profile,
so it had no explicit register rule and missed 반말/존댓말 violations --
found via the P3 30-pair hand-audit, see PROMPTS.md §4 v2 change note).

Runs entirely off the GPU VM (API-only) per CLAUDE.md cost discipline:
"CPU/API/local work never runs on the GPU instance."
"""

from __future__ import annotations

import json
import os
import random

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PERSONA = yaml.safe_load(open("config/persona.yaml"))
JUDGE_MODEL = yaml.safe_load(open("config/eval.yaml"))["judge"]["model_id"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = """Two replies were written by the same character. Choose which one is more true to the character.

Character: {persona_name}
Character definition:
{persona_profile}

Voice and tone: {voice_notes}

Speech level (hard constraint -- a reply that violates this is out of character regardless of content): {speech_level}

User turn:
{user_turn}

Reply A:
{reply_a}

Reply B:
{reply_b}

Judge only fidelity to the character: voice and register, values, and staying inside what she could plausibly know. Do NOT reward length, politeness, helpfulness, or extra detail -- a short reply fully in her voice beats a long one that drifts.
If neither is clearly better, answer "tie".

Output JSON only:
{{"choice": "<A|B|tie>", "reason": "<one sentence>"}}"""


def preference_judge(user_turn: str, reply_a: str, reply_b: str, rng: random.Random) -> dict:
    """Randomizes A/B order per call (PROMPTS.md §4: "LLM judges have
    position bias; without shuffling, DPO would learn position artifacts")
    and un-shuffles the result before returning, so the caller always gets
    'a'/'b' relative to its own reply_a/reply_b arguments, not the judge's
    internal randomized order."""
    swapped = rng.random() < 0.5
    shown_a, shown_b = (reply_b, reply_a) if swapped else (reply_a, reply_b)

    prompt = PROMPT.format(
        persona_name=PERSONA["persona_name"],
        persona_profile=PERSONA["persona_profile"].strip(),
        voice_notes=PERSONA["voice_notes"].strip(),
        speech_level=PERSONA["speech_level"].strip(),
        user_turn=user_turn,
        reply_a=shown_a,
        reply_b=shown_b,
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    choice = result.get("choice")
    if swapped and choice in ("A", "B"):
        choice = "B" if choice == "A" else "A"
    # Note: "reason" is the model's raw text and still refers to whatever
    # order it was actually shown (shown_a/shown_b) -- it is NOT re-labeled
    # to match the un-swapped `choice` above. When swapped is True, a reason
    # mentioning "Reply A" may describe the caller's reply_b. Treat `reason`
    # as descriptive only; never parse it programmatically against `choice`.
    return {"choice": choice, "reason": result.get("reason")}
