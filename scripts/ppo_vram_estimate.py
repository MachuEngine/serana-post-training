"""§7.1 PPO counterfactual -- the arithmetic behind "DPO, not PPO".

CLAUDE.md excludes PPO/reward-model RLHF on the grounds that policy +
reference + reward model + value head resident at once does not fit 8B on
24GB. DESIGN.md §7.1 asks for that claim as a *number*, not an assertion:
"PPO needed N GB and I had 24" is the deliverable.

Method, in the order the output prints:

1. Parameter counts are derived from the real model config (layer shapes),
   not quoted. The derivation is checked against the two figures P2 already
   established the hard way (6.95B non-embedding / 8.2B total -- see
   scripts/profile_train.py on why .numel() undercounts 4-bit weights).
2. Static terms (quantized weights, LoRA params, gradients, optimizer
   state) are computed from those counts.
3. The dynamic term (activations + allocator slack) is *calibrated* off
   P2's measured SFT peak rather than modelled -- DESIGN.md §7.1's
   "measure once and carry the number" rule applied to the one term that
   does not survive a paper estimate.
4. That calibrated model is then checked against a second, independent
   measured point (P4's DPO peak) before being trusted for PPO.

No GPU, no API. Usage: `uv run scripts/ppo_vram_estimate.py`
"""

from __future__ import annotations

import yaml
from transformers import AutoConfig

BASE_CFG = yaml.safe_load(open("config/base.yaml"))
BASE_ID = BASE_CFG["model"]["base_id"]
LORA_R = BASE_CFG["train"]["lora_r"]
TOLERANCE = BASE_CFG["gpu"]["vram_estimate_tolerance"]

GIB = 1024**3
L4_TOTAL_GIB = 22.5  # nvidia-smi reports ~23GB usable on the L4; leave the driver its slack

# Measured peaks to calibrate against, both from this project's own run logs.
MEASURED_SFT_GIB = 9.64  # artifacts/runs/p2_progress.md, torch.cuda.max_memory_allocated()
MEASURED_DPO_GIB = 15.0  # artifacts/runs/p4_progress.md, nvidia-smi during the real DPO run

# NF4 + double quant: 4 bits per weight plus ~0.127 bits/param of (already
# quantized) absmax constants -- bitsandbytes' documented double-quant overhead.
NF4_BYTES_PER_PARAM = (4 + 0.127) / 8
BF16 = 2
ADAMW_8BIT_BYTES_PER_TRAINABLE = 2  # two optimizer states, 1 byte each (paged_adamw_8bit)


def param_counts(cfg) -> dict[str, float]:
    """Derive parameter counts from layer shapes -- see module docstring point 1."""
    h, ffn, layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim

    attn = h * q_out + 2 * (h * kv_out) + q_out * h  # q, k, v, o
    mlp = 3 * (h * ffn)  # gate, up, down
    linear = (attn + mlp) * layers  # what bitsandbytes actually quantizes

    embed = cfg.vocab_size * h
    lm_head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h
    return {"linear": linear, "embed_and_head": embed + lm_head, "total": linear + embed + lm_head}


def lora_params(cfg) -> float:
    """LoRA A+B over the 7 projections QLoRA targets, all layers."""
    h, ffn = cfg.hidden_size, cfg.intermediate_size
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    per_layer = sum(
        LORA_R * (d_in + d_out)
        for d_in, d_out in [
            (h, q_out),
            (h, kv_out),
            (h, kv_out),
            (q_out, h),  # q, k, v, o
            (h, ffn),
            (h, ffn),
            (ffn, h),  # gate, up, down
        ]
    )
    return per_layer * cfg.num_hidden_layers


def frozen_4bit_gib(counts: dict[str, float]) -> float:
    """One resident 8B model, quantized, no gradients: weights only."""
    quantized = counts["linear"] * NF4_BYTES_PER_PARAM
    unquantized = counts["embed_and_head"] * BF16  # bnb skips embeddings and lm_head
    return (quantized + unquantized) / GIB


def trainable_extra_gib(n_lora: float) -> float:
    """What making a resident model trainable costs on top of its weights."""
    return n_lora * (BF16 + BF16 + ADAMW_8BIT_BYTES_PER_TRAINABLE) / GIB  # params, grads, optim


def kv_cache_gib(cfg, batch: int, seq: int) -> float:
    """PPO generates rollouts; generation needs a KV cache that training does not."""
    per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * BF16
    return per_token * batch * seq / GIB


def main() -> None:
    cfg = AutoConfig.from_pretrained(BASE_ID)
    counts = param_counts(cfg)
    n_lora = lora_params(cfg)

    print(f"model: {BASE_ID}")
    print(f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} ffn={cfg.intermediate_size}")
    print(
        f"  attn_heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.head_dim} vocab={cfg.vocab_size} "
        f"tied_embeddings={cfg.tie_word_embeddings}"
    )
    print("\n-- 1. parameter counts, derived from the shapes above --")
    print(f"  transformer linears (what bnb quantizes): {counts['linear'] / 1e9:.3f}B")
    print(f"  embeddings + lm_head (bnb skips these):   {counts['embed_and_head'] / 1e9:.3f}B")
    print(f"  total:                                    {counts['total'] / 1e9:.3f}B")
    print("  cross-check vs P2's hardcoded figures: 6.95B non-embedding / 8.2B total")
    print(
        f"  LoRA r={LORA_R} trainable params: {n_lora / 1e6:.1f}M "
        f"({100 * n_lora / counts['total']:.2f}% of total)"
    )

    weights = frozen_4bit_gib(counts)
    trainable = trainable_extra_gib(n_lora)
    print("\n-- 2. static terms per resident model --")
    print(f"  4-bit weights (NF4 + double quant):   {weights:.2f} GiB")
    print(f"  + LoRA params, grads, 8-bit optimizer: {trainable:.2f} GiB (if trainable)")

    # Point 3: the one term paper estimates get wrong, taken from measurement.
    dynamic = MEASURED_SFT_GIB - (weights + trainable)
    print("\n-- 3. dynamic term, calibrated not modelled --")
    print(f"  measured SFT peak (P2):               {MEASURED_SFT_GIB:.2f} GiB")
    print(f"  minus static terms above:             {weights + trainable:.2f} GiB")
    print(f"  => activations + allocator slack:     {dynamic:.2f} GiB per trained forward/backward")

    # Point 4: does the calibrated model reproduce an independent measurement?
    # DPO adds no second model (adapter toggle) but scores chosen+rejected and
    # runs a reference forward, so it pays the dynamic term ~3x over.
    dpo_pred = weights + trainable + 3 * dynamic
    gap = abs(dpo_pred - MEASURED_DPO_GIB) / MEASURED_DPO_GIB
    verdict = "within" if gap <= TOLERANCE else "OUTSIDE"
    print("\n-- 4. validating that model against a second measured point --")
    print(f"  predicted DPO (1 base, ref by adapter toggle, 3x dynamic): {dpo_pred:.2f} GiB")
    print(f"  measured DPO (P4):                                        {MEASURED_DPO_GIB:.2f} GiB")
    print(f"  gap {100 * gap:.1f}% -- {verdict} the {100 * TOLERANCE:.0f}% tolerance")

    # PPO. Give it every trick this project already uses -- 4-bit everywhere,
    # LoRA, reference for free -- so the verdict is conservative. TRL's
    # PPOTrainer takes policy, ref_policy, reward_model and value_model as
    # four separate objects; only ref_policy collapses into the policy via the
    # PEFT adapter toggle, so three sets of weights are unavoidably resident.
    seq = BASE_CFG["train"]["max_seq_len"]
    rollout_kv = kv_cache_gib(cfg, batch=8, seq=seq)
    resident = [
        ("policy (trainable, 4-bit + LoRA)", weights + trainable),
        ("reference (adapter toggle -- free, same trick DPO uses)", 0.0),
        ("reward model (separate 8B, frozen, 4-bit)", weights),
        ("value model (separate 8B, trainable, 4-bit + LoRA)", weights + trainable),
        (f"rollout KV cache (8 seqs x {seq} tokens)", rollout_kv),
    ]
    resident_total = sum(v for _, v in resident)

    print("\n-- 5. PPO: what must be resident at once --")
    for label, val in resident:
        print(f"  {val:6.2f} GiB  {label}")
    print(f"  {'-' * 6}\n  {resident_total:6.2f} GiB  subtotal, before any activations")

    # Two bounds rather than one number. The dynamic term above was calibrated
    # on a forward+BACKWARD step, so charging it to the frozen reward and
    # reference models -- which only ever run inference -- would inflate the
    # answer in the direction of the conclusion. Bound the honest range instead.
    lower = resident_total + dynamic  # only ONE trained backward is ever live
    upper = resident_total + 2 * dynamic  # policy and value both train

    print("\n-- 6. verdict, bounded rather than asserted --")
    for label, val in [
        ("most generous to PPO (one trained backward live at a time)", lower),
        ("realistic (policy and value both backward)", upper),
    ]:
        over = val - L4_TOTAL_GIB
        room = f"over by {over:.2f} GiB" if over > 0 else f"only {-over:.2f} GiB to spare"
        print(f"  {val:6.2f} GiB  {label} -- {100 * val / L4_TOTAL_GIB:.0f}% of the card, {room}")

    print(f"\n  L4 usable: {L4_TOTAL_GIB:.2f} GiB")
    print(
        f"  Even the generous bound leaves no room for the optimizer step, "
        f"fragmentation,\n  or any sequence longer than {seq} -- and the realistic bound is "
        f"{upper / L4_TOTAL_GIB:.1f}x the card."
    )
    print(
        f"  DPO, measured, on the same hardware: {MEASURED_DPO_GIB:.2f} GiB "
        f"({100 * MEASURED_DPO_GIB / L4_TOTAL_GIB:.0f}% of the card)."
    )


if __name__ == "__main__":
    main()
