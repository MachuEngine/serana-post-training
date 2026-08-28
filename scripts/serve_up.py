"""Launches vLLM's OpenAI-compatible server (`vllm serve`) with the base
model and every trained LoRA adapter registered -- DESIGN.md §9.3:
"scripts/ owns VM lifecycle. One command up, one down." `src/serve/
pipeline.py` is the client side; this is the "up" half.

Registers every distinct `lora_adapter_id` found across
`config/experiments/*.yaml` whose adapter directory actually exists
under `artifacts/lora/` -- rather than hardcoding "serana-sft"/
"serana-dpo" here too, so this stays in sync with whatever configs exist
without a second place to edit. Diagnostics adapters are excluded for
free (CLAUDE.md: "diagnostic runs never produce a shipped adapter") --
they were never written under `artifacts/lora/` in the first place.

Usage:
    uv run scripts/serve_up.py                       # bf16, base.yaml's other serving knobs
    uv run scripts/serve_up.py --quantization awq --model artifacts/merged/serana-sft-awq
        # Stage 4's AWQ comparison run, against the merged+quantized model
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.config import load_config

BASE_CFG = yaml.safe_load(open("config/base.yaml"))


def discover_adapters() -> dict[str, str]:
    """{adapter_id: local_path}, one entry per distinct lora_adapter_id
    named across config/experiments/*.yaml whose artifacts/lora/<id>
    directory exists."""
    adapters = {}
    for exp_path in sorted(glob.glob("config/experiments/*.yaml")):
        cfg = load_config(exp_path)
        adapter_id = cfg.get("lora_adapter_id")
        if not adapter_id:
            continue
        local_path = Path("artifacts/lora") / adapter_id
        if local_path.exists():
            adapters[adapter_id] = str(local_path)
    return adapters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization",
        default=None,
        help="override base.yaml's serving.quantization (e.g. 'awq'); omit for bf16",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="override base.yaml's model.base_id (e.g. an AWQ-merged model dir for Stage 4)",
    )
    args = parser.parse_args()

    serving = BASE_CFG["serving"]
    model = args.model or BASE_CFG["model"]["base_id"]
    adapters = discover_adapters()

    cmd = [
        "vllm",
        "serve",
        model,
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if adapters:
        cmd += ["--enable-lora", "--lora-modules"]
        cmd += [f"{name}={path}" for name, path in adapters.items()]

    print("launching:", " ".join(cmd))
    print(f"registered adapters: {adapters or '(none found)'}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
