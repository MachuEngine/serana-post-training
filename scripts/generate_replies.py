"""P3 step 1 (DESIGN.md §3.4): sample two replies per prompt from the SFT
adapter at temperature≈0.9 -- the preference judge picks between them
next. Input: data/ko/dpo_prompt_pool.jsonl (893 prompts, built in P1,
already excludes eval prompts/attack probes per the leakage check).
Output: data/ko/raw/reply_pairs.jsonl (prompt, reply_a, reply_b).

Batched generation (not one-by-one) for throughput -- DESIGN.md's P3
done-criterion explicitly wants batch-inference throughput recorded.
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap prompt count, for a timing test")
    parser.add_argument("--out", default="data/ko/raw/reply_pairs.jsonl")
    args = parser.parse_args()

    prompts = [json.loads(line)["prompt"] for line in open("data/ko/dpo_prompt_pool.jsonl")]
    if args.limit:
        prompts = prompts[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
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
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
            for p in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                num_return_sequences=2,  # the two candidate replies, DESIGN.md §3.4
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for j, p in enumerate(batch):
            reply_a, reply_b = decoded[2 * j], decoded[2 * j + 1]
            results.append({"prompt": p, "reply_a": reply_a, "reply_b": reply_b})
        elapsed = time.time() - start
        done = len(results)
        print(f"  {done}/{len(prompts)}  ({elapsed:.1f}s, {done/elapsed:.2f} prompts/s)")

    total_s = time.time() - start
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "n_prompts": len(prompts),
        "batch_size": args.batch_size,
        "wall_clock_s": round(total_s, 1),
        "prompts_per_s": round(len(prompts) / total_s, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device == "cuda" else None,
    }
    print(json.dumps(report, indent=2))
    with open("artifacts/runs/p3_generation_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
