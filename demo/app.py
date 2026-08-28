"""HF Spaces demo (P6, ZeroGPU) -- shows the project's actual
deliverable directly: one input, three responses (B / SFT / DPO) from
the same base model, generated the same way results were measured
(greedy decoding, persona_system prompt byte-identical across configs,
`enable_thinking=False` -- same fix `scripts/smoke_test.py` and
`src/serve/pipeline.py` already needed for Qwen3).

Not the real serving path (that's vLLM, P5) -- plain
`transformers`+`peft`, mirroring `scripts/smoke_test.py`'s already-proven
loading pattern, because a Space should be simple and robust, not
re-fight the vLLM/CUDA-version fragility this project hit repeatedly on
the GCP VM. One base model loaded once; SFT/DPO are adapters switched
via `model.set_adapter()`, "B" via `model.disable_adapter()` -- the
same single-pipeline principle as everywhere else in this repo, just
with PEFT's adapter-switching instead of vLLM's `--lora-modules`.

IP note: "Serana" and The Elder Scrolls are property of Bethesda/ZeniMax.
This demo is a non-commercial engineering portfolio artifact.
"""

from __future__ import annotations

import gradio as gr
import spaces
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA = yaml.safe_load(open("config/persona.yaml"))
BASE_ID = yaml.safe_load(open("config/base.yaml"))["model"]["base_id"]
SFT_ADAPTER_REPO = "machu8/serana-sft"
DPO_ADAPTER_REPO = "machu8/serana-dpo"
MAX_NEW_TOKENS = 200  # shorter than eval's 512 -- interactive UX, not a benchmark run

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

print(f"loading {BASE_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)
print(f"loading adapters {SFT_ADAPTER_REPO}, {DPO_ADAPTER_REPO}...")
model = PeftModel.from_pretrained(model, SFT_ADAPTER_REPO, adapter_name="sft")
model.load_adapter(DPO_ADAPTER_REPO, adapter_name="dpo")
model.eval()
print("ready.")


def _generate(user_turn: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy -- matches eval decoding, reflects the reported numbers
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@spaces.GPU(duration=90)
def compare(user_turn: str) -> tuple[str, str, str]:
    if not user_turn.strip():
        return "", "", ""
    model.to("cuda")

    with model.disable_adapter():
        b_reply = _generate(user_turn)

    model.set_adapter("sft")
    sft_reply = _generate(user_turn)

    model.set_adapter("dpo")
    dpo_reply = _generate(user_turn)

    return b_reply, sft_reply, dpo_reply


with gr.Blocks(title="Serana post-training demo") as demo:
    gr.Markdown(
        "# Serana post-training demo\n"
        '> "Serana" and The Elder Scrolls are property of Bethesda/ZeniMax. '
        "This is a non-commercial engineering portfolio artifact, not an "
        "official product.\n\n"
        "One Korean message in, three responses out -- the same base model "
        "(`Qwen/Qwen3-8B`), three post-training stages (**B**: prompt only, "
        "**SFT**, **DPO**). Full pipeline, eval design, and hardware numbers: "
        "[github.com/MachuEngine/serana-post-training]"
        "(https://github.com/MachuEngine/serana-post-training). "
        "In this project's own evaluation, DPO showed no measurable quality "
        "gain over SFT -- see the repo's results tables."
    )
    user_input = gr.Textbox(label="Message (Korean)", placeholder="너는 누구야?")
    submit = gr.Button("Compare")
    with gr.Row():
        b_out = gr.Textbox(label="B (base + prompt)", interactive=False)
        sft_out = gr.Textbox(label="SFT", interactive=False)
        dpo_out = gr.Textbox(label="DPO", interactive=False)

    submit.click(compare, inputs=user_input, outputs=[b_out, sft_out, dpo_out])
    user_input.submit(compare, inputs=user_input, outputs=[b_out, sft_out, dpo_out])

if __name__ == "__main__":
    demo.launch()
