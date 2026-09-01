"""P4 redo (artifacts/runs/p4_dpo_redo_plan.md): sample N replies per
prompt from the SFT adapter, so the preference judge has a spread to pick
the best and worst from. Supersedes the P3 2-reply script
(`generate_replies.py`, removed): `--n 2 --temperature 0.9` reproduces
P3's sampling (the default here is N=4 at temperature 1.0 for more spread).

P3 sampled only 2 replies per prompt and both came out nearly identical
(same narrow SFT distribution), which left DPO with almost no gradient
(artifacts/runs/p4_postmortem.md). Sampling N=4 at a higher temperature
and taking best-vs-worst gives a wider-separated training pair while
staying on-policy (both replies still come from the model being trained).

Input:  data/ko/dpo_prompt_pool.jsonl (P1, already excludes eval prompts / attack probes)
Output: data/ko/raw/reply_groups_v3.jsonl -- {"prompt": ..., "replies": [r1, r2, r3, r4]}

Then: python3 -m src.data.build_preferences \
        --in data/ko/raw/reply_groups_v3.jsonl \
        --out data/ko/prefs_v3.jsonl \
        --report artifacts/runs/p4redo_preference_report.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA = yaml.safe_load(open("config/persona.yaml"))
BASE_ID = yaml.safe_load(open("config/base.yaml"))["model"]["base_id"]

SYSTEM_PROMPT = """You are {persona_name}, a character from {source_title}.
Always reply in natural Korean, in {persona_name}'s voice.

Personality and voice:
{persona_profile}

Rules:
- Stay fully in character. Speak in {persona_name}'s voice, register, and worldview at all times.
- Only use knowledge {persona_name} could plausibly have. If asked about something outside that world or era, react as the character would to an unknown topic -- do not break character to answer.
- Do not mention being an AI, a model, or a language system.""".format(
    persona_name=PERSONA["persona_name"],
    source_title=PERSONA["source_title"],
    persona_profile=PERSONA["persona_profile"].strip(),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/lora/serana-sft")
    parser.add_argument("--n", type=int, default=4, help="replies to sample per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    # generate() expands to batch_size * n concurrent decode streams. Default
    # 4 * 4 = 16 matches P3's 2-reply / batch-8 job (measured 18.7GB); raising
    # either without re-checking peak VRAM risks OOM on the 24GB L4.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap prompt count for a timing test"
    )
    parser.add_argument("--out", default="data/ko/raw/reply_groups_v3.jsonl")
    args = parser.parse_args()

    prompts = [json.loads(line)["prompt"] for line in open("data/ko/dpo_prompt_pool.jsonl")]
    if args.limit:
        prompts = prompts[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    results = []
    start = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for p in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.n,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = out[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for j, p in enumerate(batch):
            replies = [decoded[args.n * j + k] for k in range(args.n)]
            results.append({"prompt": p, "replies": replies})
        elapsed = time.time() - start
        rate = len(results) / elapsed
        print(f"  {len(results)}/{len(prompts)}  ({elapsed:.1f}s, {rate:.2f} prompts/s)")

    total_s = time.time() - start
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "n_prompts": len(prompts),
        "adapter": args.adapter,
        "replies_per_prompt": args.n,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "wall_clock_s": round(total_s, 1),
        "prompts_per_s": round(len(prompts) / total_s, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if device == "cuda"
        else None,
    }
    print(json.dumps(report, indent=2))
    with open("artifacts/runs/p4redo_generation_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
