"""§7.4 quantization comparison, prep step: merges the SFT LoRA adapter
into the base weights, then AWQ-quantizes the merged model. Chosen
target: SFT (DESIGN.md §7.4 says "serve *the* merged model", singular --
SFT is the metric-primary trained artifact; P4 already flagged DPO's
gain as weak, so featuring DPO here would optimize the hardware section
around a shaky win -- confirmed in the P5 plan).

Calibration data: a sample of our own SFT set (`data/ko/sft_3k.jsonl`),
not a generic English corpus -- AWQ's salient-channel calibration is
domain-sensitive, and our domain is short Korean persona dialogue, not
whatever a stock English calibration set contains. Also avoids an
external download.

Usage (on the GPU VM):
    python3 scripts/merge_and_quantize_awq.py
Writes: artifacts/merged/serana-sft-merged/ (bf16, for reference)
        artifacts/merged/serana-sft-awq/ (AWQ 4-bit, what Stage 4 serves)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from awq import AutoAWQForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_CFG = yaml.safe_load(open("config/base.yaml"))
BASE_ID = BASE_CFG["model"]["base_id"]
ADAPTER_PATH = "artifacts/lora/serana-sft"
MERGED_PATH = "artifacts/merged/serana-sft-merged"
AWQ_PATH = "artifacts/merged/serana-sft-awq"
N_CALIB_SAMPLES = 128


def merge() -> None:
    print(f"loading base {BASE_ID} + adapter {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("merging adapter into base weights...")
    model = model.merge_and_unload()
    Path(MERGED_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"wrote merged bf16 model to {MERGED_PATH}")


def load_calibration_texts(n: int) -> list[str]:
    lines = [json.loads(line) for line in open("data/ko/sft_3k.jsonl")]
    texts = []
    for item in lines[:n]:
        # same shape as real inference: user turn + her reply, so
        # calibration activations reflect what the model actually sees.
        texts.append(f"{item['user']}\n{item['reply']}")
    return texts


def quantize() -> None:
    print(f"loading {MERGED_PATH} for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(MERGED_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)

    calib_data = load_calibration_texts(N_CALIB_SAMPLES)
    print(f"quantizing with {len(calib_data)} domain calibration samples...")
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    Path(AWQ_PATH).mkdir(parents=True, exist_ok=True)
    model.save_quantized(AWQ_PATH)
    tokenizer.save_pretrained(AWQ_PATH)
    print(f"wrote AWQ-quantized model to {AWQ_PATH}")


if __name__ == "__main__":
    merge()
    quantize()
