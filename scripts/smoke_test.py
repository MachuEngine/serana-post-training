"""P2 done-criterion: in-persona Korean smoke test. Loads the base model
+ a LoRA adapter and generates a few replies to eyeball -- coherent
Korean (CPT catastrophic-forgetting check) and in-character as Serana
(SFT check). Not the real serving path: P5 wires vLLM into
src/serve/pipeline.py; this is a lightweight plain-transformers check for
"did the adapter come out working," run right after training.

Usage: python3 scripts/smoke_test.py --adapter artifacts/lora/serana-sft
"""

from __future__ import annotations

import argparse

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

# Deliberately NOT drawn from data/eval/eval_set_v1/ -- that set stays
# untouched for the real P5 eval. These are fresh, ad hoc prompts, one of
# each kind the smoke test cares about.
TEST_PROMPTS = [
    ("in-character", "너는 누구야?"),
    ("out-of-boundary", "요즘 유행하는 게임이 뭐야?"),
    (
        "general Korean fluency (CPT forgetting check, unrelated to persona)",
        "파스타 레시피를 간단히 알려줘.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to the LoRA adapter dir; omit to test the base model alone (config/experiments/b.yaml equivalent)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    for label, prompt in TEST_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,  # Qwen3 thinking mode eats the token budget on
            # reasoning instead of the answer -- off for a
            # clean, comparable smoke-test reply.
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,  # greedy, matches eval decoding (config/base.yaml generation.temperature: 0.0)
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"=== [{label}] ===")
        print(f"Q: {prompt}")
        print(f"A: {reply}")
        print()


if __name__ == "__main__":
    main()
