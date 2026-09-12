#!/usr/bin/env bash
# 2x L4 DDP 확장성 측정. 1-GPU와 2-GPU를 같은 장비에서 연달아 돌려
# GPU 개수만 변수로 남긴다. 전체 배치 크기는 16으로 고정(1-GPU는 누적 16회,
# 2-GPU는 랭크당 8회) -- "같은 학습을 더 빨리"임을 손실 곡선으로 확인하기 위함.
#
# bf16으로 도는 이유: bitsandbytes 4비트 + DDP 조합이 이 스택(torch 2.13 /
# bitsandbytes 0.50.2 / peft 0.20)에서 첫 all-reduce에 멈춘다. NCCL 초기화와
# P2P 채널 확보는 정상이고, Trainer 없이 4비트+LoRA+DDP만 남긴 최소 재현
# (scripts/profile_ddp.py)에서도 동일하게 멈추는 것을 확인했다.
#
#   bash scripts/ddp_bench.sh [스텝수]
set -u
cd "$(dirname "$0")/.." || exit 1
STEPS="${1:-40}"
MODEL="${2:-}"   # 로컬 모델 디렉터리. 비우면 config의 base_id(HF)를 쓴다.
R=/workspace/results
mkdir -p "$R" artifacts/diagnostics

# FlashAttention-2는 이 이미지에 없다. 두 실행 모두 sdpa로 맞춰야 비교가 성립한다.
COMMON=(--config config/train_runs/sft.yaml
        --set train.load_in_4bit=false
        --set train.attn_implementation=sdpa
        --set "train.max_steps=$STEPS")
# HuggingFace가 RunPod IP 대역에 429를 걸어 가중치를 받을 수 없을 때가 있다.
# 그럴 땐 ModelScope에서 받은 로컬 디렉터리를 두 번째 인자로 넘긴다.
[ -n "$MODEL" ] && COMMON+=(--set "model.base_id=$MODEL")

# 2-GPU를 먼저 돌린다. DDP가 깨지면 2분 안에 드러나므로, 긴 1-GPU 기준선에
# 시간을 쓰기 전에 중단할 수 있다(반대 순서면 실패를 50분 뒤에야 안다).
echo "=== 2-GPU bf16 DDP, $STEPS steps ==="
S=$(date +%s)
.venv/bin/torchrun --nproc_per_node=2 scripts/train.py "${COMMON[@]}" \
  --set train.grad_accum_steps=8 \
  --set train.output_adapter=artifacts/diagnostics/bf16_2gpu \
  > "$R/bf16_2gpu.log" 2>&1 || { echo "2-GPU FAILED"; tail -25 "$R/bf16_2gpu.log"; exit 1; }
E2=$(( $(date +%s) - S ))
echo "2-GPU total ${E2}s"

echo "=== 1-GPU bf16, $STEPS steps ==="
S=$(date +%s)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py "${COMMON[@]}" \
  --set train.output_adapter=artifacts/diagnostics/bf16_1gpu \
  > "$R/bf16_1gpu.log" 2>&1 || { echo "1-GPU FAILED"; tail -25 "$R/bf16_1gpu.log"; exit 1; }
E1=$(( $(date +%s) - S ))
echo "1-GPU total ${E1}s"

# 손실 곡선의 정식 출처는 터미널 로그가 아니라 run_report.json의 log_history다.
# 터미널 로그에는 tqdm 진행 표시줄이 캐리지 리턴으로 섞여 들어와 기록이
# 덮어써진다(첫 측정에서 2-GPU의 스텝 5~25 구간을 이렇게 잃었다).
for T in 1gpu 2gpu; do
  cp "artifacts/diagnostics/bf16_$T/run_report.json" "$R/report_$T.json" 2>/dev/null \
    || echo "WARN: bf16_$T의 run_report.json 없음 (완주 전에 중단된 실행)"
done

python3 - "$E1" "$E2" "$STEPS" <<'PY' | tee "$R/ddp_summary.txt"
import sys
e1, e2, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(f"1-GPU  {e1/n:.2f} s/step   ({e1}s / {n})")
print(f"2-GPU  {e2/n:.2f} s/step   ({e2}s / {n})")
print(f"speedup     {e1/e2:.2f}x   (ideal 2.00x)")
print(f"efficiency  {e1/e2/2*100:.1f}%")
PY
