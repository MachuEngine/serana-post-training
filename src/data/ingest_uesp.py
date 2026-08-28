"""P1 ingest: parse Serana's wiki dialogue into paired / unpaired lines
(DESIGN.md §3.1). Sources: UESP (primary -- clean one-marker-per-line
wikitext) and Fandom Quotes (supplementary, deduplicated against UESP).

Fandom's Dialogue/Conversations sections combine player+reply on one line
with nested {{Hide|...}} branches -- parsing them reliably is high-risk for
low marginal value (their content overlaps UESP's Follower Dialogue almost
verbatim), so they are deliberately skipped. Note this in the P1 run log.

Output: data/raw/pairs.jsonl (player_line, reply, source, page) and
data/raw/unpaired.jsonl (line, source, page) -- English, pre-translation,
pre-horizon-filter. Downstream steps (horizon filter, translation, dedup)
are separate scripts so each stage is independently inspectable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW = Path("data/raw")

TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
TAG_RE = re.compile(r"<[^>]+>")
QUOTE_RE = re.compile(r"''\"(.*?)\"''", re.S)
BOLD_RE = re.compile(r"'''(.*?)'''", re.S)
# Multi-character scene format: "'''Harkon:''' ''"line"''" -- a speaker tag,
# not a player choice. Distinguished from a player-choice bold (a question
# or statement) by being just a name plus a colon. The wiki uses both
# "'''Name:'''" (colon inside the bold) and "'''Name''':" (colon outside)
# -- both seen in practice, both must be caught or a non-Serana line slips
# through the generic bold/quote path and gets mislabeled as her voice.
SPEAKER_TAG_RES = [
    re.compile(r"^:*'''([A-Za-z][A-Za-z '.-]{1,25}):'''\s*(.*)$"),
    re.compile(r"^:*'''([A-Za-z][A-Za-z '.-]{1,25})''':\s*(.*)$"),
]
PERSONA_NAME = "Serana"


def match_speaker_tag(line: str) -> re.Match | None:
    for pattern in SPEAKER_TAG_RES:
        m = pattern.match(line)
        if m:
            return m
    return None


def clean(text: str) -> str:
    """Strip wikitext markup down to plain prose. Order matters: templates
    and comments first (they can contain junk that looks like other
    markup), links resolved to display text, then leftover formatting."""
    text = COMMENT_RE.sub("", text)

    # Collapse templates with no readable payload (Audio, intnote, vn, Hide-
    # without-plain-alt) but keep a template's last |-piece when there is
    # one, since {{FC|#DD0000|actual text}} carries real dialogue text.
    def _template(m: re.Match) -> str:
        inner = m.group(0)[2:-2]
        parts = inner.split("|")
        return parts[-1] if len(parts) > 1 else ""

    prev = None
    while prev != text:
        prev = text
        text = TEMPLATE_RE.sub(_template, text)
    text = LINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = TAG_RE.sub("", text)
    text = text.replace("''", "").replace("'''", "")
    return " ".join(text.split()).strip()


def parse_topic_pairs(wikitext: str, source: str, page: str) -> tuple[list[dict], list[dict]]:
    """Walk a UESP dialogue-tree section line by line. A bold '''...'''
    line (not itself a quote) sets the current player line; the next
    quoted ''"..."'' line pairs with it. Table/branch structure means one
    player line can pair with several reply variants -- all are kept as
    separate genuine pairs, since each is a real recorded exchange.

    A quote with no player line set yet (e.g. an opening line the wiki
    shows before the first player choice) is kept as unpaired rather than
    dropped. A plain narrative sentence (the wiki editor's own prose,
    introducing a new dialogue option) resets the current player line, so
    a stale bold line from an earlier block can't wrongly claim a later
    quote separated by a paragraph of description.

    Some quest sections switch into a scripted multi-character cutscene
    format, "'''SpeakerName:''' ''"line"''" (Serana talking to Harkon,
    Valerica, Isran...). Those are not player exchanges at all: a
    Serana-tagged line is a standalone voice line (unpaired -> CPT), and
    any other character's tagged line is discarded outright -- it is not
    her dialogue and must not be mislabeled as if it were."""
    pairs: list[dict] = []
    dangling: list[dict] = []
    current_player: str | None = None
    for raw_line in wikitext.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("{|", "|-", "|}", "!")):
            continue
        tag_m = match_speaker_tag(line)
        if tag_m:
            speaker, rest = tag_m.group(1).strip(), tag_m.group(2)
            current_player = None  # cutscene mode: no player choice active
            if speaker == PERSONA_NAME:
                q = QUOTE_RE.search(rest)
                if q:
                    text = clean(q.group(1))
                    if text:
                        dangling.append({"line": text, "source": source, "page": page})
            continue  # any other speaker's tagged line is discarded
        quote_m = QUOTE_RE.search(line)
        bold_m = BOLD_RE.search(line) if not quote_m else None
        if quote_m:
            reply = clean(quote_m.group(1))
            if not reply:
                continue
            if current_player:
                pairs.append(
                    {"player_line": current_player, "reply": reply, "source": source, "page": page}
                )
            else:
                dangling.append({"line": reply, "source": source, "page": page})
        elif bold_m:
            player = clean(bold_m.group(1))
            if player:
                current_player = player
        elif not line.startswith(":"):
            # Plain narrative prose -- the wiki editor introducing a new
            # dialogue option. Stale player-line context no longer applies.
            current_player = None
    return pairs, dangling


def parse_all_quotes(wikitext: str, source: str, page: str) -> list[dict]:
    """Every ''"..."'' quote in a section, order-preserved, no pairing --
    for ambient/unpaired barks (Combat Dialogue, Follower Dialogue, Quotes)."""
    out = []
    for m in QUOTE_RE.finditer(wikitext):
        text = clean(m.group(1))
        if text:
            out.append({"line": text, "source": source, "page": page})
    return out


def main() -> None:
    main_page = json.load(open(RAW / "uesp" / "serana_main.json"))
    full_wt = main_page["parse"]["wikitext"]["*"]

    quest_i = full_wt.find("==Quest-Related Events==")
    dialogue_i = full_wt.find("==Dialogue==")
    quest_wt = full_wt[quest_i:dialogue_i]

    uesp_dialogue = (RAW / "uesp" / "dialogue_section.wikitext").read_text()
    combat_i = uesp_dialogue.find("===Combat Dialogue===")
    follower_i = uesp_dialogue.find("===Follower Dialogue===")

    topics_wt = uesp_dialogue[:combat_i]
    combat_wt = uesp_dialogue[combat_i:follower_i]
    follower_wt = uesp_dialogue[follower_i:]

    quest_pairs, quest_dangling = parse_topic_pairs(
        quest_wt, "UESP", "Skyrim:Serana#Quest-Related_Events"
    )
    topic_pairs, topic_dangling = parse_topic_pairs(topics_wt, "UESP", "Skyrim:Serana#Dialogue")
    pairs = quest_pairs + topic_pairs
    unpaired = (
        quest_dangling
        + topic_dangling
        + parse_all_quotes(combat_wt, "UESP", "Skyrim:Serana#Combat_Dialogue")
        + parse_all_quotes(follower_wt, "UESP", "Skyrim:Serana#Follower_Dialogue")
    )

    fandom_wt = (RAW / "fandom" / "dialogue_conv_quotes.wikitext").read_text()
    quotes_i = fandom_wt.find("==Quotes==")
    fandom_quotes_wt = fandom_wt[quotes_i:]
    unpaired += parse_all_quotes(fandom_quotes_wt, "Fandom", "Serana#Quotes")

    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "pairs.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (RAW / "unpaired.jsonl").open("w") as f:
        for u in unpaired:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    print(f"pairs: {len(pairs)}")
    print(f"unpaired (pre-dedup): {len(unpaired)}")


if __name__ == "__main__":
    main()
