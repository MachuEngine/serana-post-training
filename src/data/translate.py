"""P1 translation: English -> Korean via GPT-4o, using PROMPTS.md §2a
(CPT corpus text) and §2b (dialogue pairs) verbatim. Provider fixed in
config/eval.yaml per DESIGN.md §9.4.

Inputs: data/raw/curated/{cpt_pool_en,style_reference_en,pairs}.jsonl
Outputs: data/ko/raw/{cpt_pool,style_reference,pairs}.jsonl -- still
intermediate (one record per source line/pair); src/data/build_corpora.py
assembles these into the final training files.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

PERSONA = yaml.safe_load(Path("config/persona.yaml").read_text())
JUDGE_MODEL = yaml.safe_load(Path("config/eval.yaml").read_text())["judge"]["model_id"]
# This account's org caps gpt-4o at 30k TPM (found empirically -- the
# first full-concurrency run at 8 workers mostly 429'd). ~3 workers x
# ~500 tokens/request stays under that with room for retries.
MAX_WORKERS = 3
MAX_RETRIES = 6

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _with_retry(create_fn, model: str, prompt: str, response_format: dict | None):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    if response_format:
        kwargs["response_format"] = response_format
    for attempt in range(MAX_RETRIES):
        try:
            return create_fn(**kwargs)
        except RateLimitError:
            wait = min(2**attempt, 30)
            time.sleep(wait)
    return create_fn(**kwargs)  # last attempt: let it raise if it still fails


GLOSSARY_STR = ", ".join(f"{en} -> {ko}" for en, ko in PERSONA["glossary"].items())

CORPUS_PROMPT = """Translate the following {source_title} text into natural Korean.
This text will be used as reading material to teach a model {persona_name}'s voice, so the Korean must read as if originally written in Korean.

Requirements:
- Write in fluent, natural Korean prose. Avoid translationese: no English word order, no mechanical rendering of pronouns or articles, no stiff connectives.
- Preserve {persona_name}'s register: {voice_notes}
- Preserve paragraph and sentence boundaries exactly as in the source. Do not merge or split them.
- Keep proper nouns consistent with this glossary: {glossary}
- Return only the Korean text, with no notes, headers, or commentary.

Text:
{source_text}"""

DIALOGUE_PROMPT = """Translate this exchange from {source_title} into natural Korean.
Speaker roles must be preserved exactly: the player's line stays the player's, {persona_name}'s reply stays hers.

Requirements:
- Natural spoken Korean, not literary or written style. These are lines people say out loud.
- Preserve {persona_name}'s register in her reply: {voice_notes}
- Choose a consistent speech level for {persona_name} and keep it identical across every exchange: {speech_level}
- Keep proper nouns consistent with this glossary: {glossary}
- Return JSON only: {{"user": "<Korean player line>", "reply": "<Korean Serana line>"}}

Player: {player_line}
{persona_name}: {persona_line}"""


def translate_corpus_line(text: str) -> str:
    prompt = CORPUS_PROMPT.format(
        source_title=PERSONA["source_title"],
        persona_name=PERSONA["persona_name"],
        voice_notes=PERSONA["voice_notes"].strip(),
        glossary=GLOSSARY_STR,
        source_text=text,
    )
    resp = _with_retry(
        client.chat.completions.create,
        JUDGE_MODEL,
        prompt,
        None,
    )
    return resp.choices[0].message.content.strip()


def translate_pair(player_line: str, reply: str) -> dict:
    prompt = DIALOGUE_PROMPT.format(
        source_title=PERSONA["source_title"],
        persona_name=PERSONA["persona_name"],
        voice_notes=PERSONA["voice_notes"].strip(),
        speech_level=PERSONA["speech_level"].strip(),
        glossary=GLOSSARY_STR,
        player_line=player_line,
        persona_line=reply,
    )
    resp = _with_retry(
        client.chat.completions.create,
        JUDGE_MODEL,
        prompt,
        {"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _load_existing(out_path: Path, key_fn, ok_fn) -> dict:
    """Resume support: previously-successful results, keyed by source
    content, so a re-run only pays for items that failed or are new."""
    if not out_path.exists():
        return {}
    existing = {}
    for line in out_path.open():
        d = json.loads(line)
        if ok_fn(d):
            existing[key_fn(d)] = d
    return existing


def run_batch_corpus(items: list[dict], out_path: Path) -> None:
    done_map = _load_existing(out_path, lambda d: d["line"], lambda d: d.get("ko"))
    results: list[dict | None] = [None] * len(items)
    todo = []
    for i, it in enumerate(items):
        if it["line"] in done_map:
            results[i] = done_map[it["line"]]
        else:
            todo.append(i)
    print(f"  resuming: {len(items) - len(todo)} already done, {len(todo)} to go")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(translate_corpus_line, items[i]["line"]): i for i in todo}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                ko_text = fut.result()
                results[i] = {**items[i], "ko": ko_text}
            except Exception as e:
                print(f"  ERROR on item {i}: {e}")
                results[i] = {**items[i], "ko": None, "error": str(e)}
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
    with out_path.open("w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in results if r and r.get("ko"))
    print(f"{out_path}: {n_ok}/{len(items)} translated")


def run_batch_pairs(items: list[dict], out_path: Path) -> None:
    done_map = _load_existing(
        out_path,
        lambda d: (d["player_line"], d["reply"]),
        lambda d: d.get("ko_reply"),
    )
    results: list[dict | None] = [None] * len(items)
    todo = []
    for i, it in enumerate(items):
        key = (it["player_line"], it["reply"])
        if key in done_map:
            results[i] = done_map[key]
        else:
            todo.append(i)
    print(f"  resuming: {len(items) - len(todo)} already done, {len(todo)} to go")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(translate_pair, items[i]["player_line"], items[i]["reply"]): i for i in todo
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                ko = fut.result()
                results[i] = {**items[i], "ko_user": ko.get("user"), "ko_reply": ko.get("reply")}
            except Exception as e:
                print(f"  ERROR on pair {i}: {e}")
                results[i] = {**items[i], "ko_user": None, "ko_reply": None, "error": str(e)}
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
    with out_path.open("w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in results if r and r.get("ko_reply"))
    print(f"{out_path}: {n_ok}/{len(items)} translated")


def main() -> None:
    Path("data/ko/raw").mkdir(parents=True, exist_ok=True)
    raw = Path("data/raw/curated")

    cpt_pool = [json.loads(line) for line in (raw / "cpt_pool_en.jsonl").open()]
    style_ref = [json.loads(line) for line in (raw / "style_reference_en.jsonl").open()]
    pairs = [json.loads(line) for line in (raw / "pairs.jsonl").open()]

    print(f"translating CPT pool ({len(cpt_pool)})...")
    run_batch_corpus(cpt_pool, Path("data/ko/raw/cpt_pool.jsonl"))

    print(f"translating style reference ({len(style_ref)})...")
    run_batch_corpus(style_ref, Path("data/ko/raw/style_reference.jsonl"))

    print(f"translating real pairs ({len(pairs)})...")
    run_batch_pairs(pairs, Path("data/ko/raw/pairs.jsonl"))


if __name__ == "__main__":
    main()
