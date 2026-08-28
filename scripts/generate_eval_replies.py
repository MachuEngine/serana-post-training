"""Stage 3, GPU half: generates replies for all 30 eval prompts + 24
attack probes, for one config, through the real serving path
(`src/serve/pipeline.generate`). Greedy decoding (DESIGN.md §4.5's
eval-decoding rule -- the default already baked into `generate()`).

Escalating attack sequences (2 sequences x 3 turns, PROMPTS.md §6) are
run as genuine multi-turn conversations -- each turn's reply feeds the
next turn's history -- but only the *final* turn is what PRS scores
(§6: "scored on the final turn -- cracking at t3 after holding t1-t2 is
still a break"). All three turns are saved for a qualitative read even
though only t3 is scored.

Judging is deliberately NOT done here: CLAUDE.md cost discipline --
"CPU/API/local work never runs on the GPU instance." This script only
generates and saves raw replies; `scripts/score_eval_replies.py` (run
locally, after downloading the output) does the judge/rule-check pass.

Usage (on the GPU VM, server already up via scripts/serve_up.py):
    python3 scripts/generate_eval_replies.py --config config/experiments/sft.yaml
Writes: artifacts/runs/raw_eval_<config name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.serve.pipeline import generate

EVAL_SET_DIR = Path("data/eval/eval_set_v1")


def run_quality_prompts(config: dict) -> list[dict]:
    items = [json.loads(line) for line in (EVAL_SET_DIR / "eval_prompts.jsonl").open()]
    results = []
    for i, item in enumerate(items):
        r = generate(config, item["prompt_ko"])
        results.append(
            {
                "id": item["id"],
                "boundary": item["boundary"],
                "topic": item["topic"],
                "user_turn": item["prompt_ko"],
                "model_reply": r["text"],
                "reply_token_count": r["completion_tokens"],
                "latency_s": r["latency_s"],
            }
        )
        print(f"  quality {i + 1}/{len(items)}: {item['id']}")
    return results


def run_attack_probes(config: dict) -> list[dict]:
    items = [json.loads(line) for line in (EVAL_SET_DIR / "attack_probes.jsonl").open()]
    results = []

    # single-turn probes: direct/meta/role_exit
    single_turn = [it for it in items if it["category"] != "escalating"]
    for i, item in enumerate(single_turn):
        r = generate(config, item["prompt_ko"])
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "attack_prompt": item["prompt_ko"],
                "model_reply": r["text"],
                "reply_token_count": r["completion_tokens"],
                "latency_s": r["latency_s"],
                "final_turn": True,
            }
        )
        print(f"  attack (single-turn) {i + 1}/{len(single_turn)}: {item['id']}")

    # escalating: 2 sequences x 3 turns, real multi-turn history
    escalating = [it for it in items if it["category"] == "escalating"]
    sequences: dict[str, list[dict]] = {}
    for it in escalating:
        sequences.setdefault(it["sequence"], []).append(it)
    for seq_name, turns in sequences.items():
        turns.sort(key=lambda t: t["turn"])
        history: list[dict[str, str]] = []
        for t in turns:
            r = generate(config, t["prompt_ko"], history=history)
            results.append(
                {
                    "id": t["id"],
                    "category": "escalating",
                    "sequence": seq_name,
                    "turn": t["turn"],
                    "attack_prompt": t["prompt_ko"],
                    "model_reply": r["text"],
                    "reply_token_count": r["completion_tokens"],
                    "latency_s": r["latency_s"],
                    "final_turn": t["turn"] == turns[-1]["turn"],
                }
            )
            history.append({"role": "user", "content": t["prompt_ko"]})
            history.append({"role": "assistant", "content": r["text"]})
            print(f"  attack (escalating {seq_name} t{t['turn']}): {t['id']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. config/experiments/sft.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = Path(args.config).stem

    print(
        f"generating for config={config_name} (model_weights={config.get('model_weights')}, "
        f"lora_adapter_id={config.get('lora_adapter_id')})"
    )
    quality = run_quality_prompts(config)
    attack = run_attack_probes(config)

    out = {
        "config": config_name,
        "base_model": config["model"]["base_id"],
        "quantization": config.get("serving", {}).get("quantization"),
        "quality_raw": quality,
        "attack_raw": attack,
    }
    out_path = Path("artifacts/runs") / f"raw_eval_{config_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} ({len(quality)} quality + {len(attack)} attack replies)")


if __name__ == "__main__":
    main()
