"""§7.4 KV-cache arithmetic -- predict before the serving VM boots.
Per-token cache cost: 2 (K and V) x layers x kv_heads x head_dim x
dtype_bytes. Qwen3 uses grouped-query attention (kv_heads < attention
heads), which is exactly why this has to be computed from the model
config rather than assumed from attention-head count (DESIGN.md §7.4).

Also estimates base-weight footprint for both serving quantizations
under comparison (bf16 vs AWQ, DESIGN.md §7.4) so the KV-cache budget is
checked against what's actually left over on a 24GB L4 after weights --
not just computed in isolation.

Verification step (not this script): once the vLLM server is up,
compare `total_kv_cache_bytes` here against vLLM's own reported
allocated KV blocks x block size. A gap beyond
`gpu.vram_estimate_tolerance` (base.yaml, 0.20) needs a written
explanation, same convention as every VRAM prediction in P2/P4.

Usage: `uv run scripts/kv_cache_estimate.py`
"""

from __future__ import annotations

import yaml
from transformers import AutoConfig

BASE_CFG = yaml.safe_load(open("config/base.yaml"))
BASE_ID = BASE_CFG["model"]["base_id"]
MAX_MODEL_LEN = BASE_CFG["serving"]["max_model_len"]
MAX_NUM_SEQS = BASE_CFG["serving"]["max_num_seqs"]
GPU_MEM_UTIL = BASE_CFG["serving"]["gpu_memory_utilization"]

L4_TOTAL_VRAM_GB = 24  # nvidia-smi reports ~23GB usable, see P2/P4 run logs
KV_DTYPE_BYTES = 2  # bf16/fp16 KV cache (vLLM default; not using fp8 KV cache)
# 8.2B total params, matching the hardcoded figure used in P2's MFU calc
# (scripts/profile_train.py) -- bitsandbytes/AWQ pack multiple values per
# stored byte, so runtime-counting .numel() on quantized weights silently
# undercounts, as P2 already found the hard way. Use the documented total.
TOTAL_PARAMS = 8.2e9


def main() -> None:
    cfg = AutoConfig.from_pretrained(BASE_ID)
    layers = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads
    attn_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim

    per_token_bytes = 2 * layers * kv_heads * head_dim * KV_DTYPE_BYTES
    total_kv_bytes = per_token_bytes * MAX_MODEL_LEN * MAX_NUM_SEQS
    total_kv_gb = total_kv_bytes / 1024**3

    print(f"model: {BASE_ID}")
    print(
        f"layers={layers}  attention_heads={attn_heads}  kv_heads={kv_heads}  head_dim={head_dim}"
    )
    print(f"  (GQA ratio: {attn_heads // kv_heads}x fewer kv_heads than attention heads)")
    print(f"per-token KV bytes: {per_token_bytes} ({per_token_bytes / 1024:.1f} KiB)")
    print(f"max_model_len={MAX_MODEL_LEN}  max_num_seqs={MAX_NUM_SEQS}")
    print(f"predicted worst-case total KV cache: {total_kv_gb:.2f} GB")
    print(
        f"  (all {MAX_NUM_SEQS} sequences simultaneously at the full {MAX_MODEL_LEN}-token "
        f"length -- vLLM's PagedAttention allocates adaptively, so this is the theoretical "
        f"ceiling the config must fit under, not a fixed reservation)"
    )

    usable_vram_gb = L4_TOTAL_VRAM_GB * GPU_MEM_UTIL
    print(f"\nusable VRAM at gpu_memory_utilization={GPU_MEM_UTIL}: {usable_vram_gb:.2f} GB")

    for label, bytes_per_param in [("bf16", 2.0), ("AWQ (4-bit)", 0.5)]:
        weight_gb = TOTAL_PARAMS * bytes_per_param / 1024**3
        remaining_gb = usable_vram_gb - weight_gb
        headroom_gb = remaining_gb - total_kv_gb
        verdict = "OK" if headroom_gb > 0 else "DOES NOT FIT as configured"
        print(
            f"  {label}: weights ~{weight_gb:.2f}GB -> {remaining_gb:.2f}GB left for "
            f"KV+activations+context -> {headroom_gb:.2f}GB headroom after the {total_kv_gb:.2f}GB "
            f"KV budget -- {verdict}"
        )


if __name__ == "__main__":
    main()
