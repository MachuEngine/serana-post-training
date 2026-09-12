"""Score the 50 human-labeled items with candidate (cheaper) judge models and
report how each one correlates with the human labels -- the gate for swapping
the judge before building eval-set v2 (DESIGN.md §4.4's `correlation_floor`).

Non-destructive by design: `src/eval/validate_judge.py` rewrites
`human_labels.jsonl` and `judge_validation_report.json`, which are the frozen
v1 record backing the shipped results tables. This script only *reads* them,
reuses `judge_pcs`'s exact prompt (so the comparison is apples-to-apples), and
writes its own report to `artifacts/runs/judge_candidates.json`.

Also records real token usage per candidate so the per-config eval cost can be
projected instead of guessed.

Usage: `uv run scripts/judge_candidate_check.py --models gpt-4o-mini,gpt-5-mini`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.judge_pcs import PERSONA, PROMPT
from src.eval.validate_judge import spearman

load_dotenv()

LABELS = Path("data/eval/eval_set_v1/human_labels.jsonl")
SHIPPED_REPORT = Path("data/eval/eval_set_v1/judge_validation_report.json")
OUT = Path("artifacts/runs/judge_candidates.json")
FLOOR = yaml.safe_load(open("config/eval.yaml"))["judge"]["correlation_floor"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def score_one(model: str, item: dict) -> dict:
    prompt = PROMPT.format(
        persona_name=PERSONA["persona_name"],
        persona_profile=PERSONA["persona_profile"].strip(),
        voice_notes=PERSONA["voice_notes"].strip(),
        speech_level=PERSONA["speech_level"].strip(),
        user_turn=item["user_turn"],
        model_reply=item["reply"],
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    try:
        resp = client.chat.completions.create(temperature=0.0, **kwargs)
    except Exception as e:  # reasoning models reject temperature -- retry without
        if "temperature" not in str(e).lower():
            raise
        resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError(f"{model} returned empty content (finish={resp.choices[0].finish_reason})")
    parsed = json.loads(content)
    return {
        "score": parsed.get("score"),
        "violation": parsed.get("violation"),
        "in_tok": resp.usage.prompt_tokens,
        "out_tok": resp.usage.completion_tokens,
    }


def evaluate(model: str, items: list[dict]) -> dict:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda it: score_one(model, it), items))

    human = [it["human_score"] for it in items]
    judge = [r["score"] for r in results]
    bad = [i for i, s in enumerate(judge) if not isinstance(s, (int, float))]
    if bad:
        return {"model": model, "error": f"{len(bad)} items returned a non-numeric score"}

    rho = spearman(human, judge)
    n = len(items)
    hv = [it["human_violation"] for it in items]
    jv = [r["violation"] for r in results]
    return {
        "model": model,
        "n": n,
        "spearman": round(rho, 4),
        "exact_agreement": round(sum(h == j for h, j in zip(human, judge)) / n, 4),
        "within_1_agreement": round(sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / n, 4),
        "violation_flag_agreement": round(sum(h == j for h, j in zip(hv, jv)) / n, 4),
        "passes_floor": rho >= FLOOR,
        "mean_in_tok": round(sum(r["in_tok"] for r in results) / n),
        "mean_out_tok": round(sum(r["out_tok"] for r in results) / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="gpt-4o-mini,gpt-4.1-mini,gpt-5-mini",
        help="comma-separated candidate judge model ids",
    )
    args = ap.parse_args()

    items = [json.loads(line) for line in LABELS.open()]
    if any(it["human_score"] is None for it in items):
        sys.exit("human labels incomplete")

    shipped = json.loads(SHIPPED_REPORT.read_text())
    print(
        f"baseline: {yaml.safe_load(open('config/eval.yaml'))['judge']['model_id']} "
        f"spearman={shipped['spearman']} (floor {FLOOR}), n={len(items)}\n"
    )

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"scoring {len(items)} items with {model}...")
        try:
            row = evaluate(model, items)
        except Exception as e:
            row = {"model": model, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print(f"  {json.dumps(row, ensure_ascii=False)}\n")

    report = {"baseline": shipped, "floor": FLOOR, "candidates": rows}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
