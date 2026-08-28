"""§7.4 throughput-vs-concurrency sweep: tokens/sec and TTFT p50/p95 at
concurrency 1/2/4/8/16 on one config, to find the knee that justifies
`serving.max_num_seqs` (base.yaml).

TTFT needs real token streaming (`stream=True`) -- `src/serve/
pipeline.generate()` is non-streaming (waits for the full completion,
matching what batch eval generation needs), so this script is a
separate, small streaming client rather than a second inference path
wedged into `generate()` for one caller's benefit. Reuses
`resolve_model_name`/`build_system_prompt` from pipeline.py -- same
prompt assembly, no divergence in what's actually being measured.

Usage (on the GPU VM, server already up):
    python3 scripts/throughput_sweep.py --config config/experiments/sft.yaml
Writes: artifacts/runs/hardware_throughput_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.serve.pipeline import _client, build_system_prompt, resolve_model_name

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
REQUESTS_PER_LEVEL = 20  # enough for a real p50/p95 read without over-spending
EVAL_PROMPTS_PATH = Path("data/eval/eval_set_v1/eval_prompts.jsonl")


def load_prompts(n: int) -> list[str]:
    """Cycles the 30 real eval prompts to fill out `n` requests -- same
    prompt shape every sweep uses, not synthetic filler text."""
    items = [json.loads(line)["prompt_ko"] for line in EVAL_PROMPTS_PATH.open()]
    return [items[i % len(items)] for i in range(n)]


def stream_one(config: dict, user_turn: str) -> dict:
    model = resolve_model_name(config)
    gen_cfg = config.get("generation", {})
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_turn},
    ]
    client = _client(config)

    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 512),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            completion_tokens += 1  # chunk count as a token-count proxy
    total_s = time.perf_counter() - start
    return {"ttft_s": ttft, "total_s": total_s, "completion_tokens": completion_tokens}


def sweep_one_level(config: dict, concurrency: int) -> dict:
    prompts = load_prompts(REQUESTS_PER_LEVEL)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda p: stream_one(config, p), prompts))
    wall_s = time.perf_counter() - start

    ttfts = sorted(r["ttft_s"] for r in results if r["ttft_s"] is not None)
    total_tokens = sum(r["completion_tokens"] for r in results)

    def pctile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return float("nan")
        idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
        return sorted_vals[idx]

    return {
        "concurrency": concurrency,
        "n_requests": len(prompts),
        "wall_s": wall_s,
        "throughput_tokens_per_s": total_tokens / wall_s if wall_s > 0 else None,
        "ttft_p50_s": pctile(ttfts, 0.50),
        "ttft_p95_s": pctile(ttfts, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. config/experiments/sft.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = []
    for c in CONCURRENCY_LEVELS:
        print(f"concurrency={c}: running {REQUESTS_PER_LEVEL} requests...")
        r = sweep_one_level(config, c)
        print(
            f"  throughput={r['throughput_tokens_per_s']:.1f} tok/s  "
            f"TTFT p50={r['ttft_p50_s']:.3f}s  p95={r['ttft_p95_s']:.3f}s"
        )
        results.append(r)

    out_path = Path("artifacts/runs/hardware_throughput_sweep.json")
    out_path.write_text(json.dumps({"config": Path(args.config).stem, "sweep": results}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
