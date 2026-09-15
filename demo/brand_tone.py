"""Brand-tone demo (DESIGN.md §8) -- prompt-only voice transfer, ~zero GPU.

Runs one base model (config/base.yaml's model.base_id, no adapter) against
several brand-voice system prompts from demo/brand_prompts.yaml, on the
same set of domain-neutral test prompts. Shows what a system prompt alone
buys -- a recognizable, in-session-consistent tone -- as a foil to what
the Serana post-training actually bought (multi-turn persistence + an
adversarially-robust knowledge boundary; see the repo's PRS/PCS tables).
Qualitative only, no metrics -- exactly the §8 scope.

Loading mirrors scripts/smoke_test.py (plain transformers + greedy
decoding + enable_thinking=False for Qwen3), so it runs on the M5 with a
small model for a smoke check and on the L4 with the real 8B for the
write-up.

Usage:
    # local smoke (M5, small model, fast):
    uv run python3 demo/brand_tone.py --base-id Qwen/Qwen3-0.6B --max-new-tokens 80

    # real run for the blog appendix (on a GPU box):
    uv run python3 demo/brand_tone.py --markdown > artifacts/diagnostics/brand_tone.md

    # secondary curiosity: put a brand voice on top of the persona adapter
    # (expected: the Serana training fights the brand prompt) --
    uv run python3 demo/brand_tone.py --adapter artifacts/lora/serana-sft
"""

from __future__ import annotations

import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = yaml.safe_load(open("demo/brand_prompts.yaml"))
BASE_ID = yaml.safe_load(open("config/base.yaml"))["model"]["base_id"]


def generate(model, tokenizer, system: str, user: str, device: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,  # Qwen3 thinking mode eats the token budget; see smoke_test.py
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -- one deterministic reply per (voice, prompt)
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def render_markdown(base_id: str, adapter: str | None, results: dict) -> str:
    voices = PROMPTS["voices"]
    lines = [
        f"# Brand-tone demo -- `{base_id}`"
        + (f" + adapter `{adapter}`" if adapter else " (no adapter, prompt only)"),
        "",
        "Same base model and same greedy decoding for every cell. Only the "
        "system prompt changes between columns.",
        "",
    ]
    for user in PROMPTS["test_prompts"]:
        lines.append(f"### {user}")
        lines.append("")
        lines.append("| voice | reply |")
        lines.append("|---|---|")
        for key, v in voices.items():
            reply = results[(key, user)].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {v['label']} | {reply} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-id", default=BASE_ID)
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    results: dict[tuple[str, str], str] = {}
    for key, voice in PROMPTS["voices"].items():
        for user in PROMPTS["test_prompts"]:
            reply = generate(model, tokenizer, voice["system"], user, device, args.max_new_tokens)
            results[(key, user)] = reply
            if not args.markdown:
                print(f"=== [{voice['label']}] {user}")
                print(reply)
                print()

    if args.markdown:
        print(render_markdown(args.base_id, args.adapter, results))


if __name__ == "__main__":
    main()
