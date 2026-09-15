"""Multi-turn length check (DESIGN.md §8) -- GPU half. DRAFT.

Replays each fixed conversation in data/multiturn/multiturn_probes_v1.jsonl
through the real serving path (src/serve/pipeline.generate), feeding each
turn's reply back as history for the next -- the same genuine-multi-turn
approach generate_eval_replies.py uses for the escalating attack probes,
just longer. Captures every turn's reply so the scoring half can compare
persona quality at turn 1 vs 3 vs 10 (does she drift as context grows?).

Judging is NOT here (CLAUDE.md cost discipline) -- scripts/score_multiturn.py
does that locally, API-only, after this file is pulled off the VM.

DRAFT status: the probe set lives at data/multiturn/ rather than the
frozen data/eval/, and is not yet wired into src/data/check_leakage.py.
Training is already frozen so there's no live leakage risk, but graduating
this from a draft means: move the probes under data/eval/ with a version
bump + user review, and add them to the leakage guard.

Usage (on the serving VM, `python3 scripts/serve_up.py` already running):
    python3 scripts/generate_multiturn_replies.py --config config/experiments/sft.yaml
Writes: artifacts/runs/raw_multiturn_<config name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.serve.pipeline import generate

PROBES = Path("data/multiturn/multiturn_probes_v1.jsonl")


def run_conversation(config: dict, turns: list[str]) -> list[dict]:
    history: list[dict[str, str]] = []
    rows = []
    for depth, user_turn in enumerate(turns, start=1):
        r = generate(config, user_turn, history=history)
        rows.append(
            {
                "depth": depth,
                "user_turn": user_turn,
                "model_reply": r["text"],
                "reply_token_count": r["completion_tokens"],
                "latency_s": r["latency_s"],
            }
        )
        history.append({"role": "user", "content": user_turn})
        history.append({"role": "assistant", "content": r["text"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. config/experiments/sft.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = Path(args.config).stem
    conversations = [json.loads(line) for line in PROBES.open()]

    print(f"multi-turn generation: config={config_name}, {len(conversations)} conversations")
    out_conversations = []
    for conv in conversations:
        print(f"  {conv['id']} ({len(conv['turns'])} turns)")
        out_conversations.append(
            {
                "id": conv["id"],
                "topic": conv["topic"],
                "turns": run_conversation(config, conv["turns"]),
            }
        )

    out = {
        "config": config_name,
        "base_model": config["model"]["base_id"],
        "probe_set": "multiturn_probes_v1",
        "conversations": out_conversations,
    }
    out_path = Path("artifacts/runs") / f"raw_multiturn_{config_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
