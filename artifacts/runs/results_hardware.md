| stage/config | predicted VRAM | measured peak VRAM | step time / TTFT p50,p95 | GPU util % | MFU % | throughput | cost |
|---|---|---|---|---|---|---|---|
| sft-lora-bf16 (adapter, concurrency=8) | - | - | p50=0.213s p95=0.422s | - | - | 80.6 tok/s | - |
| sft-merged-bf16 (no adapter, concurrency=8) | - | - | p50=0.247s p95=0.839s | - | - | 75.4 tok/s | - |
| serving KV-cache (max_model_len=4096, max_num_seqs=8, bf16) | 4.50 GB (worst-case, all 8 seqs @ 4096 tok) | 3.17 GB default alloc / 4.87 GB to fully utilize (vLLM's own report) | - | - | - | - | - |
| sft-merged-bf16 (concurrency=8) | 15.27 GB weights (predicted) | 15.36 GiB weights / ~19.5GB total @ util=0.9 | p50=0.247s p95=0.839s | - | - | 75.4 tok/s | - |
| sft-merged-awq (concurrency=8) | 3.82 GB weights (predicted) | 5.8 GiB weights / ~19.3GB total @ util=0.9 | p50=0.091s p95=0.529s | - | - | 183.8 tok/s | - |
| sft-lora-bf16 (concurrency=1) | - | - | p50=0.147s p95=0.911s | - | - | 13.2 tok/s | - |
| sft-lora-bf16 (concurrency=2) | - | - | p50=0.203s p95=0.398s | - | - | 24.4 tok/s | - |
| sft-lora-bf16 (concurrency=4) | - | - | p50=0.207s p95=0.406s | - | - | 44.6 tok/s | - |
| sft-lora-bf16 (concurrency=8) | - | - | p50=0.213s p95=0.422s | - | - | 80.6 tok/s | - |
| sft-lora-bf16 (concurrency=16) | - | - | p50=2.268s p95=2.911s | - | - | 80.9 tok/s | - |
| CPT (training, L4, spot) | - | 9.64 GB | 36.2s wall clock (tiny corpus, 3 steps) | - | - | - | ~$0 (few sec) |
| SFT (training, L4, spot, r=16, lr=2e-4) | - | 9.64 GB | wall clock: predicted 20-40min, measured 4h12m (grad_accum_steps override bug) | - | 13.5% | - | ~$5 (incl. reruns/diagnostics) |
| DPO (training, L4, resumed after 1 preemption) | 13-16 GB | ~15 GB (nvidia-smi) | predicted 34-38s/step, measured ~26-27s/step | - | - | - | predicted ~$0.30-0.35, measured ~$0.25 |
