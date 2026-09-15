"""Drives traffic through the FastAPI gateway so the dashboard has
something to show.

Deliberately *not* a measurement. `scripts/throughput_sweep.py` is the
measurement -- it talks to vLLM directly, one config at a time, at
controlled concurrency, and its numbers are what the hardware table
quotes. This one spreads requests across every config at low concurrency
for one purpose: to make the monitoring panels non-empty and prove they
are wired to something real.

Saying which is which matters. Latency observed here is worse than P5's
isolated numbers, because three configs are interleaved and B generates
replies six times longer than SFT's. Reading these as performance figures
would be a mistake, so the script prints the warning itself.

    kubectl port-forward svc/serana-gateway 8080:8080
    uv run scripts/gateway_load.py --requests 120 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

PROMPTS_FILE = Path("data/eval/eval_set_v1/eval_prompts.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080/chat")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="capped so B's long replies do not dominate the wall clock",
    )
    parser.add_argument("--configs", default="sft,dpo,b")
    args = parser.parse_args()

    prompts = [json.loads(line)["prompt_ko"] for line in PROMPTS_FILE.open() if line.strip()]
    configs = args.configs.split(",")
    rng = random.Random(0)  # same prompt sequence run to run

    def one(i: int) -> str:
        cfg = configs[i % len(configs)]
        try:
            r = httpx.post(
                args.url,
                json={
                    "turn": rng.choice(prompts),
                    "config": cfg,
                    "max_tokens": args.max_tokens,
                },
                timeout=120,
            )
            return str(r.status_code)
        except Exception as exc:
            return type(exc).__name__

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one, range(args.requests)))
    elapsed = time.perf_counter() - start

    counts = Counter(results)
    print(
        f"{args.requests} requests across {configs} at concurrency "
        f"{args.concurrency} in {elapsed:.1f}s ({args.requests / elapsed:.2f} req/s)"
    )
    for code, n in counts.most_common():
        print(f"  {code}: {n}")
    print(
        "\nNOTE: this is traffic to populate the dashboard, not a benchmark. "
        "scripts/throughput_sweep.py is the measurement; its numbers are the "
        "ones in the hardware table."
    )


if __name__ == "__main__":
    main()
