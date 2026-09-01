"""Preference judge -- PROMPTS.md §4, id `preference_judge` v3. Training-
loop prompt (DESIGN.md §3.4/§4.4 circularity guard): given N candidate
replies to one prompt, picks the single best and the single worst.
Deliberately different shape from `judge_pcs` (§5, an absolute rating) --
never reuse this rubric for measurement.

Background: P4's DPO run was a null (artifacts/runs/p4_postmortem.md).
P3 had sampled both candidates from serana-sft at temperature 0.9, so the
two replies were nearly identical and DPO got almost no gradient. v3
samples N replies (not 2) from the SFT model and takes the best-vs-worst
pair -- the widest-separated pair the model actually produced, rather than
two random draws.

Three things the P3 code review flagged, fixed here:
- A valid-JSON but wrong-shape response (json_object mode guarantees an
  object, not the nested field types) raises JudgeSchemaError, which the
  caller drops as one bad group instead of letting an AttributeError
  abort the whole run.
- The speech-level check is returned as data (`register_breaks`), so
  build_preferences.py can drop a pair whose "best" reply breaks 반말
  (the exact v1 failure), rather than trusting prompt text alone.
- `reason` is required to contrast "the best reply" with "the worst
  reply" without naming their positions, so it stays correct after the
  shown-order is mapped back to the caller's order.

Runs off the GPU VM -- API-only (CLAUDE.md cost discipline).
"""

from __future__ import annotations

import json
import os
import random
import re

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PERSONA = yaml.safe_load(open("config/persona.yaml"))
JUDGE_MODEL = yaml.safe_load(open("config/eval.yaml"))["judge"]["model_id"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class JudgeSchemaError(ValueError):
    """The judge returned valid JSON whose shape does not match the contract."""


PROMPT = """{n} replies were written by the same character, {persona_name}, answering the same user turn. Judge which one reply is the most true to the character and which one is the least.

Character: {persona_name}, from {source_title}
Who she is:
{persona_profile}

Voice and tone: {voice_notes}

Speech level (HARD CONSTRAINT): {speech_level}

User turn:
{user_turn}

{numbered_replies}

Judge in this order.

1. Speech-level check. List the numbers of every reply that breaks the speech level above (해요체/하십시오체, or any non-반말 ending). A reply that breaks it is out of character no matter how good its content is, and should normally be the worst pick.

2. Compare the replies on: voice (dry, guarded, a little wry -- not warm, not eager, not a lore-dump); values (deflects or stays vague on hard topics instead of lying or over-explaining; slow to trust); knowledge boundary (stays inside what a roughly 4000-year-old vampire sealed away for centuries could know; meets unknown or modern topics with confusion, without inventing specifics). Do NOT reward length, politeness, helpfulness, or extra detail.

3. Pick the single best reply and the single worst reply (they must be different numbers). "confidence" is how clear the gap between those two is: "high" if the best is clearly better than the worst on at least one axis, "medium" for a mild gap, "low" if the best and worst are close.

Output JSON only:
{{"register_breaks": [<reply numbers, or empty>], "best": <reply number>, "worst": <reply number>, "reason": "<one sentence contrasting the best reply with the worst reply, without naming their numbers>", "confidence": "<high|medium|low>"}}"""


def _idx(value: object, n: int, field: str) -> int:
    """1-based reply number from the judge -> validated 0-based position
    in the shown order. Tolerates a string digit ("2"), which the model
    returns often enough that dropping the group over it is wasteful."""
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= n):
        raise JudgeSchemaError(f"{field}={value!r} is not an integer in 1..{n}")
    return value - 1


def preference_judge(user_turn: str, replies: list[str], rng: random.Random) -> dict:
    """Shows the N replies in a random order (LLM judges have position
    bias) and maps the judge's picks back to the caller's indices.

    Returns: `best`/`worst` as 0-based indices into `replies`,
    `register_breaks` as a set of 0-based indices, `confidence`
    (high|medium|low|None), and `reason`. The prompt asks for a
    position-free `reason`, but the model is not fully bound by that, so
    a reason that still names a reply number ("Reply 2 is ...") is
    dropped rather than stored next to the caller-order fields where it
    would mislead the auditor.

    Raises JudgeSchemaError on a valid-JSON / wrong-shape response.
    """
    n = len(replies)
    if n < 2:
        # a group with < 2 replies is unjudgeable -- surface it the same way
        # as a bad judge response so the caller drops the one group
        raise JudgeSchemaError(f"group has {n} replies, need at least 2")

    order = list(range(n))  # order[shown_position] = caller_index
    rng.shuffle(order)
    shown = [replies[caller_i] for caller_i in order]
    numbered = "\n\n".join(f"Reply {i + 1}:\n{r}" for i, r in enumerate(shown))

    prompt = PROMPT.format(
        n=n,
        persona_name=PERSONA["persona_name"],
        source_title=PERSONA["source_title"],
        persona_profile=PERSONA["persona_profile"].strip(),
        voice_notes=PERSONA["voice_notes"].strip(),
        speech_level=PERSONA["speech_level"].strip(),
        user_turn=user_turn,
        numbered_replies=numbered,
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:  # e.g. finish_reason == "content_filter" -> content is None
        raise JudgeSchemaError("empty response content")
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:  # json_object mode should preclude this
        raise JudgeSchemaError(f"non-JSON response: {e}") from e
    if not isinstance(result, dict):
        raise JudgeSchemaError(f"top-level JSON is {type(result).__name__}, not an object")

    best = order[_idx(result.get("best"), n, "best")]
    worst = order[_idx(result.get("worst"), n, "worst")]

    breaks = set()
    raw_breaks = result.get("register_breaks")
    if isinstance(raw_breaks, list):
        for v in raw_breaks:
            if isinstance(v, str) and v.isdigit():  # model sometimes returns "1" not 1
                v = int(v)
            if not isinstance(v, bool) and isinstance(v, int) and 1 <= v <= n:
                breaks.add(order[v - 1])

    conf = result.get("confidence")
    if conf not in ("high", "medium", "low"):
        conf = None

    reason = result.get("reason")
    # A reason that names a reply by its shown-order position would mislead
    # once best/worst are mapped back to caller order -- drop it. Kept narrow
    # (position word next to "답변"/"reply", or an ordinal with the 째 suffix)
    # so it does not fire on "4번 화제" and similar. The audit rows also carry
    # the resolved chosen/rejected text, so a missed case is still checkable.
    _POS = (
        r"reply\s*#?\s*\d"
        r"|답변\s*#?\s*\d"
        r"|\d\s*번(?:째)?\s*답변"
        r"|\d\s*번째"
        r"|(?:첫|두|세|네|다섯|여섯|일곱|여덟)\s*번째"
    )
    if isinstance(reason, str) and re.search(_POS, reason, re.IGNORECASE):
        reason = None

    return {
        "best": best,
        "worst": worst,
        "register_breaks": breaks,
        "confidence": conf,
        "reason": reason,
    }
