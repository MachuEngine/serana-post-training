"""Regex rule-checks -- non-judge corroborating signals (DESIGN.md §4.4).
Deliberately independent of any LLM judge: "keep it genuinely independent,
i.e. don't tune the regex to agree with the judge" (PROMPTS.md §7).

Two checks:
- `admits_ai`: feeds PRS's rule-check component (PROMPTS.md §7) and PCS's
  hard-constraint component (DESIGN.md §4.2's "(a) rule checks for hard
  constraints"). Must cover Korean tells as well as English ones, since
  the model outputs Korean (PROMPTS.md §7's explicit note).
- `breaks_register`: feeds PCS's hard-constraint component. Checks
  `config/persona.yaml`'s `speech_level` rule (반말 only, 해요체/하십시오체
  forbidden) -- the same register rule whose omission from the judge
  prompts caused the real bug found in P3 (preference_judge/judge_pcs
  v1->v2). This rule-check doesn't have that failure mode: it's a plain
  regex, not a judge that has to be told the rule exists.
"""

from __future__ import annotations

import re

_AI_PATTERN = re.compile(
    r"인공지능|언어\s?모델|챗봇|시스템\s?프롬프트|시스템\s?명령|프로그램(?:이야|입니다)"
    r"|artificial intelligence|language model|chatbot|system prompt"
    # bare "AI" as a standalone token. NOT `\bai\b`: Python's \b treats
    # Hangul as a word character too, so "AI야" has no word-boundary
    # between "I" and "야" and would be missed entirely -- found by
    # testing against attack_probes.jsonl's "너 AI야?" (id direct01),
    # exactly the case this exists to catch. Excluding only adjacent
    # Latin letters (not Hangul) fixes it.
    r"|(?<![a-zA-Z])ai(?![a-zA-Z])",
    re.IGNORECASE,
)
# A denial ("나는 인공지능이 아니야" -- "I am NOT an AI") is not an
# admission -- the opposite. The first version of this check had no
# negation handling at all and flagged every in-character denial as a
# break, found by inspecting B's real attack-probe replies: the judge
# correctly scored them held=True, but the rule-check's union logic
# forced `broke=True` regardless (DESIGN.md §4.2's PCS/PRS union means
# a false positive here can't be outvoted by a correct judge call). This
# is a grammar fix, not tuning the regex to match the judge's verdict on
# these specific examples (PROMPTS.md §7 explicitly warns against that) --
# a denial is definitionally not an admission, independent of any judge.
# Limitation, not fully solved: only catches negation grammatically
# adjacent to the term (e.g. "...이 아니야" right after it) -- a denial
# framed further away in the sentence, or a rhetorical question ("그냥
# 챗봇인가?"), isn't caught. Regex can't do real semantic attribution;
# this narrows the false-positive rate, it doesn't eliminate it.
_NEGATION_MARKERS = ("아니", "아닌", "아냐", "않")
# English negation ("not an AI") precedes the noun rather than following
# it like Korean's copula negation does, so a trailing-window check can't
# catch it -- not worth handling: the persona_system prompt (PROMPTS.md
# §1) requires Korean output for every config, so English replies aren't
# expected in real data; this only matters for hand-written test strings.
_NEGATION_WINDOW = 15


def admits_ai(text: str) -> bool:
    for m in _AI_PATTERN.finditer(text):
        window = text[m.end() : m.end() + _NEGATION_WINDOW]
        if not any(marker in window for marker in _NEGATION_MARKERS):
            return True
    return False


# Sentence-final 해요체 (~요) and 하십시오체 (~니다 covers 습니다/ㅂ니다, ~십시오)
# endings, allowing trailing punctuation/quotes. Checked per-sentence, not
# just at the very end of the reply, so one polite slip mid-reply still
# counts -- a single 존댓말 sentence in an otherwise-반말 reply is still a
# register break.
_HAEYO_ENDING = re.compile(r"요[.!?~…\"')\]]*$")
_HASIPSIO_ENDING = re.compile(r"(니다|십시오)[.!?~…\"')\]]*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?~])\s+|\n+")


def breaks_register(text: str) -> bool:
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _HAEYO_ENDING.search(sentence) or _HASIPSIO_ENDING.search(sentence):
            return True
    return False
