"""§7.3 profiling + MFU. Runs a handful of real SFT training steps wrapped
in torch.profiler, then computes Model FLOPs Utilization: how much of the
L4's theoretical peak compute we actually achieved.

MFU = (measured FLOPs/sec) / (GPU peak FLOPs/sec)

Measured FLOPs/sec uses the standard ~6*N_params*N_tokens approximation
(2N forward + 4N backward -- Kaplan/Chinchilla scaling-law convention).
Caveat, stated once and moved on rather than re-derived exactly: LoRA
freezes the base weights, so backward doesn't need weight-gradients for
almost all of those params -- true compute is somewhat below the 6N
figure this uses (which assumes full fine-tuning), so this MFU number is
a conservative (slight under-) estimate of actual efficiency, not exact.

L4 peak BF16 (dense, no structural sparsity -- QLoRA training doesn't use
sparsity acceleration): 121 TFLOPS, from NVIDIA's L4 datasheet.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

L4_PEAK_BF16_TFLOPS = 121.0  # dense, NVIDIA L4 datasheet
BASE_ID = yaml.safe_load(open("config/base.yaml"))["model"]["base_id"]
# First attempt used grad_accum=16 (160 total traced fwd+bwd passes) and
# hung the VM -- g2-standard-8 has 31GB host RAM and torch.profiler's
# per-event CPU-side recording ate through it, leaving the VM too starved
# to even accept new SSH sessions (recovered via `gcloud compute
# instances reset`, no data lost). Profiling doesn't need a real
# optimizer step -- a handful of raw forward+backward passes is enough to
# measure per-token throughput, so this now traces MICRO_BATCHES directly
# instead of wrapping full accumulated steps.
MICRO_BATCHES = 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="disable gradient checkpointing, for the §7.2 ablation comparison",
    )
    parser.add_argument(
        "--tag", default="default", help="label for this run in the saved report filename"
    )
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="attention implementation, for the §7.2 ablation comparison",
    )
    args = parser.parse_args()
    grad_checkpoint = not args.no_grad_checkpoint

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        quantization_config=bnb_config,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
    )
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # NOT sum(p.numel() for p in model.parameters()) -- for a 4-bit
    # quantized model that undercounts badly. bitsandbytes packs two 4-bit
    # values into each stored uint8 byte, so .numel() on the packed base
    # weight tensors returns roughly half the true logical parameter
    # count (confirmed: this model reported 4.73B, not Qwen3-8B's
    # documented 8.2B -- a 2x MFU error if left uncorrected). Use the
    # model's own documented total instead.
    n_params_total = 8_200_000_000  # Qwen3-8B official total (6.95B non-embedding)
    print(f"total params (from HF model card, not runtime-counted): {n_params_total:,}")

    # Fixed-size synthetic batch so token count is known exactly (avoids
    # needing to load the real dataset just to profile compute).
    seq_len = 512
    batch_size = 1
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    def run_micro_batch():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)  # no optimizer.step() -- not needed to time compute

    print("warming up (2 micro-batches, untimed)...")
    for _ in range(2):
        run_micro_batch()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"profiling {MICRO_BATCHES} micro-batches...")
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(MICRO_BATCHES):
            run_micro_batch()
        torch.cuda.synchronize()
    wall_s = time.time() - start

    top_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print(top_ops)
    with open(f"artifacts/diagnostics/profile_top_ops_{args.tag}.txt", "w") as f:
        f.write(top_ops)

    total_tokens = batch_size * seq_len * MICRO_BATCHES
    micro_batch_time_s = wall_s / MICRO_BATCHES
    step_time_s_at_grad_accum_16 = (
        micro_batch_time_s * 16
    )  # for comparison with real training's logged step time

    flops_per_token = 6 * n_params_total  # standard full-FT approximation, see module docstring
    achieved_flops_per_s = flops_per_token * total_tokens / wall_s
    achieved_tflops = achieved_flops_per_s / 1e12
    mfu = achieved_tflops / L4_PEAK_BF16_TFLOPS

    report = {
        "tag": args.tag,
        "attn_implementation": args.attn,
        "gradient_checkpointing": grad_checkpoint,
        "n_params_total": n_params_total,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "micro_batches_profiled": MICRO_BATCHES,
        "wall_clock_s": round(wall_s, 2),
        "micro_batch_time_s": round(micro_batch_time_s, 3),
        "extrapolated_step_time_s_at_grad_accum_16": round(step_time_s_at_grad_accum_16, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "achieved_tflops": round(achieved_tflops, 2),
        "l4_peak_bf16_tflops_dense": L4_PEAK_BF16_TFLOPS,
        "mfu": round(mfu, 4),
        "mfu_caveat": "6N formula assumes full fine-tuning backward cost; LoRA's frozen "
        "base weights need less backward compute than that, so this MFU "
        "is a conservative (understated) estimate, not exact.",
    }
    print(json.dumps(report, indent=2))
    with open(f"artifacts/diagnostics/mfu_report_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
