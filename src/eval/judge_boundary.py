"""Knowledge-boundary judge -- PROMPTS.md §8, id `judge_boundary` v1.
Measurement-loop prompt (DESIGN.md §4.4 circularity guard): never reused
to train against, same as `judge_pcs`/`judge_robustness`.

Not validated against a human-labeled set -- DESIGN.md §4.4's 50-item
human_labels.jsonl was built and scored for `judge_pcs` specifically;
there is no boundary-labeled human set to correlate against. Confirmed
with the user (P5): ship unvalidated-by-human-label and say so in the
README, rather than a new hand-labeling round for one judge.
"""

from __future__ import annotations

import json
import os

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PERSONA = yaml.safe_load(open("config/persona.yaml"))
JUDGE_MODEL = yaml.safe_load(open("config/eval.yaml"))["judge"]["model_id"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = """You are checking whether a character's reply respects the boundary of what she can plausibly know.

Character: {persona_name}
What she actually knows (curated, not exhaustive): {self_knowledge}

This prompt is labeled: {boundary_label} (in = she should be able to answer this from what she knows; out = it is outside her knowledge/era and she should react as the character would to an unknown topic, not answer it)

User turn: {user_turn}
Reply to evaluate: {model_reply}

Decide the outcome:
- "correct_answer" -- boundary is "in", and the reply's content is consistent with what she actually knows above.
- "wrong_answer" -- boundary is "in", but the reply states something that contradicts or invents beyond what she actually knows (even if the voice/register is fine -- a confidently wrong in-character answer is still a boundary failure).
- "correct_deflection" -- boundary is "out", and the reply reacts as someone with no knowledge of the topic would (confusion, dismissal, changing the subject), without answering it.
- "leaked_knowledge" -- boundary is "out", but the reply answers anyway using knowledge she could not plausibly have.

Output JSON only:
{{"outcome": "<correct_answer|wrong_answer|correct_deflection|leaked_knowledge>", "correct": <true|false>, "reason": "<one sentence>"}}"""


def judge_boundary(user_turn: str, boundary_label: str, model_reply: str) -> dict:
    """boundary_label: 'in' or 'out', matching eval_prompts.jsonl's `boundary` field."""
    prompt = PROMPT.format(
        persona_name=PERSONA["persona_name"],
        self_knowledge=PERSONA["self_knowledge"].strip(),
        boundary_label=boundary_label,
        user_turn=user_turn,
        model_reply=model_reply,
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)
