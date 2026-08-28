"""PCS judge -- PROMPTS.md §5, id `judge_pcs` v2. Absolute 1-5 rating +
violation call. Measurement-loop prompt (DESIGN.md §4.4's circularity
guard: deliberately different shape from `preference_judge`, §4 -- never
reuse this rubric to train against).

v2: added voice_notes/speech_level, same fix and same root cause as
preference_judge v2 (see that file's docstring). IMPORTANT: this bumps
the prompt version, which invalidates the P1 §4.4 human-label validation
run under v1 -- re-run src/eval/validate_judge.py before trusting any
judge_pcs-backed number (P5).
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

PROMPT = """You are evaluating whether a reply stays true to a character.

Character: {persona_name}
Character definition:
{persona_profile}

Voice and tone: {voice_notes}

Speech level (hard constraint -- a reply that violates this is out of character regardless of content): {speech_level}

User turn:
{user_turn}

Reply to evaluate:
{model_reply}

Score persona consistency from 1 to 5:
5 = fully in character (voice, values, knowledge boundary all correct)
1 = clearly out of character or uses impossible knowledge

Output JSON only:
{{"score": <1-5>, "violation": <true|false>, "reason": "<one sentence; if violation, name the specific slip>"}}"""


def judge_pcs(user_turn: str, model_reply: str) -> dict:
    prompt = PROMPT.format(
        persona_name=PERSONA["persona_name"],
        persona_profile=PERSONA["persona_profile"].strip(),
        voice_notes=PERSONA["voice_notes"].strip(),
        speech_level=PERSONA["speech_level"].strip(),
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
