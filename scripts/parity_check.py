"""Training/serving parity: the same adapter, the same prompts, greedy
decoding, two runtimes.

The question this answers is narrow and worth stating precisely. Everything
measured in P5 came out of vLLM. Nothing has ever checked that the adapter
*as served* behaves like the adapter *as trained* -- i.e. that the numbers
describe the artifact rather than the serving stack. This compares plain
PyTorch (`transformers` + `peft` on the laptop's MPS backend, the same code
shape `src/finetune/train.py` runs) against vLLM behind the cluster
Service, on identical inputs.

**Predicted before running, per CLAUDE.md's predict-then-measure rule:
exact match will NOT be 100%, and that is not a defect.** Greedy decoding
picks the top-probability token at every step, so in principle both
runtimes should emit the same text. In practice they diverge, because
kernel implementations, reduction order, the attention backend (sdpa here
vs vLLM's paged attention), and batch composition all perturb the logits
in the last few decimal places -- and greedy decoding amplifies exactly
that perturbation wherever the top two candidates are close. One flipped
token then changes the whole continuation.

So the pass criterion is not equality. It is:
  1. divergence starts late rather than at token 0 (an immediate split
     means a real mismatch -- wrong adapter, wrong chat template, wrong
     dtype -- not floating-point noise), and
  2. the replies still mean the same thing, which the existing persona
     metrics measure and this script deliberately does not re-implement.

A caveat kept in the output rather than hidden: the server's reply is
re-tokenized locally to compare token ids, because the OpenAI-compatible
API returns text, not the ids it actually emitted. Re-tokenizing is a
faithful proxy for where two strings diverge, not a transcript of the
server's sampling.

Usage (server reachable on config's serving.base_url, e.g. via
`kubectl port-forward svc/serana-vllm 8000:8000`):

    uv run scripts/parity_check.py --config config/experiments/sft.yaml
    uv run scripts/parity_check.py --config config/experiments/b.yaml --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.serve.pipeline import build_system_prompt, generate

ADAPTER_ROOT = Path("artifacts/lora")


def resolve_device() -> str:
    """MPS on the M5, CPU elsewhere. Never CUDA: this side is deliberately
    the laptop path, and a local run on the same GPU would not be an
    independent check of anything."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_local(config: dict, device: str):
    """Base weights plus the config's adapter, if it has one.
    `model_weights: base` is a real path here too (CLAUDE.md §single
    pipeline), so B can be parity-checked exactly like SFT/DPO."""
    base_id = config["model"]["base_id"]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    # bfloat16 both sides: vLLM serves bf16, and matching the dtype removes
    # the one confound that would otherwise dominate the comparison.
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
    if config.get("model_weights") == "lora":
        adapter_dir = ADAPTER_ROOT / config["lora_adapter_id"]
        if not adapter_dir.exists():
            raise SystemExit(f"adapter not found locally: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model.to(device), tokenizer


def local_reply(model, tokenizer, device: str, user_turn: str, max_tokens: int) -> dict:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_turn},
    ]
    # enable_thinking=False mirrors src/serve/pipeline.py's extra_body --
    # without it Qwen3 spends the whole budget on an English reasoning
    # trace and the two sides are not comparable at all.
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy, matching generation.temperature: 0.0
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_s = time.perf_counter() - start
    completion_ids = out[0][inputs["input_ids"].shape[1] :].tolist()
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids,
        "latency_s": latency_s,
    }


def compare(local_text: str, server_text: str, tokenizer) -> dict:
    """Token-level divergence between two replies, using one tokenizer for
    both so the comparison is apples to apples."""
    a = tokenizer(local_text, add_special_tokens=False)["input_ids"]
    b = tokenizer(server_text, add_special_tokens=False)["input_ids"]
    first_divergence = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            first_divergence = i
            break
    if first_divergence is None and len(a) != len(b):
        first_divergence = min(len(a), len(b))
    return {
        "exact_match": local_text == server_text,
        "first_divergence_token": first_divergence,
        "local_tokens": len(a),
        "server_tokens": len(b),
        "shared_prefix_ratio": (
            1.0 if first_divergence is None else first_divergence / max(len(a), len(b), 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiments/sft.yaml")
    parser.add_argument("--prompts", default="data/eval/eval_set_v1/eval_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="first N prompts only")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="capped below base.yaml's 512: the laptop side generates one token at a "
        "time on MPS, and divergence, if it happens, happens early",
    )
    parser.add_argument("--out", default="artifacts/runs/parity_sft.json")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = [json.loads(line) for line in Path(args.prompts).open() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    device = resolve_device()
    print(f"local device: {device} | config: {config['name']} | prompts: {len(prompts)}")
    model, tokenizer = load_local(config, device)

    rows = []
    for i, row in enumerate(prompts, 1):
        # `prompt_ko` is the eval set's field name, matching
        # scripts/generate_eval_replies.py -- the prompts are Korean
        # (DESIGN.md §3 language policy).
        user_turn = row["prompt_ko"]
        server = generate(config, user_turn, max_tokens=args.max_tokens)
        local = local_reply(model, tokenizer, device, user_turn, args.max_tokens)
        result = compare(local["text"], server["text"], tokenizer)
        result.update(
            {
                "id": row.get("id", f"p{i}"),
                "prompt": user_turn,
                "local_text": local["text"],
                "server_text": server["text"],
                "local_latency_s": round(local["latency_s"], 2),
                "server_latency_s": round(server["latency_s"], 3),
            }
        )
        rows.append(result)
        mark = "=" if result["exact_match"] else f"~{result['first_divergence_token']}"
        print(f"  [{i}/{len(prompts)}] {result['id']}: {mark}")

    matched = sum(r["exact_match"] for r in rows)
    summary = {
        "config": config["name"],
        "model": config["model"]["base_id"],
        "device_local": device,
        "device_server": "vLLM (see run notes)",
        "max_tokens": args.max_tokens,
        "n_prompts": len(rows),
        "exact_match_rate": round(matched / len(rows), 3) if rows else None,
        "mean_shared_prefix_ratio": (
            round(sum(r["shared_prefix_ratio"] for r in rows) / len(rows), 3) if rows else None
        ),
        "mean_local_latency_s": (
            round(sum(r["local_latency_s"] for r in rows) / len(rows), 2) if rows else None
        ),
        "mean_server_latency_s": (
            round(sum(r["server_latency_s"] for r in rows) / len(rows), 3) if rows else None
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print(
        "NOTE: local latency is NOT a GPU result (CLAUDE.md: local timings are never "
        "reported as GPU results) -- it is here only to show the run happened."
    )


if __name__ == "__main__":
    main()
