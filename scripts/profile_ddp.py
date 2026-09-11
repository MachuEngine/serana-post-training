"""DDP 통신 비용 측정. torchrun으로 2-GPU에서 실행하며, torch.profiler로
NCCL all-reduce(GPU 간 기울기 합산)가 전체 GPU 시간의 몇 %를 먹는지 잰다.

확장 효율이 2.0x가 아닌 이유를 추측이 아니라 측정으로 답하기 위한 스크립트.
scripts/profile_train.py(1-GPU, MFU)의 DDP판이며, 모델 로딩 방식은 동일하게 맞췄다.

    torchrun --nproc_per_node=2 scripts/profile_ddp.py
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_ID = yaml.safe_load(open("config/base.yaml"))["model"]["base_id"]
MICRO_BATCHES = 6  # profile_train.py와 동일 -- 프로파일러 CPU 측 기록이 호스트 RAM을 먹는다


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/diagnostics/ddp_profile.json")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

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
        device_map={"": local_rank},  # 4-bit는 로드 시점에 GPU를 지정해야 한다
        dtype=torch.bfloat16,
    )
    # use_reentrant=False -- 재진입 방식 체크포인팅은 DDP와 충돌한다
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model = DistributedDataParallel(model, device_ids=[local_rank])

    seq_len = 512
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    labels = input_ids.clone()

    def run_micro_batch():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(2):  # 워밍업 -- NCCL 첫 호출은 채널 셋업 비용이 붙는다
        run_micro_batch()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(MICRO_BATCHES):
            run_micro_batch()
        torch.cuda.synchronize()

    # NCCL 커널만 골라내 GPU 총 시간 대비 비율을 낸다.
    total_cuda_us = 0.0
    nccl_us = 0.0
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or 0
        total_cuda_us += cuda_us
        name = evt.key.lower()
        if "nccl" in name or "allreduce" in name or "all_reduce" in name:
            nccl_us += cuda_us

    if local_rank == 0:
        report = {
            "world_size": world_size,
            "micro_batches": MICRO_BATCHES,
            "seq_len": seq_len,
            "total_cuda_ms": round(total_cuda_us / 1000, 2),
            "nccl_allreduce_ms": round(nccl_us / 1000, 2),
            "nccl_share_pct": round(nccl_us / total_cuda_us * 100, 2) if total_cuda_us else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print("\n--- GPU 시간 상위 커널 ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
