#!/usr/bin/env bash
# bf16 경로 스모크 테스트. torchrun 없이 단일 프로세스로 2스텝만 돌려
# 예외를 화면에 그대로 띄운다 -- torchrun은 자식 프로세스의 traceback을
# 삼키고 ChildFailedError만 남기므로 원인 파악에 쓸 수 없다.
#
#   bash scripts/bf16_smoke.sh
set -u
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/train.py \
  --config config/train_runs/sft.yaml \
  --set train.load_in_4bit=false \
  --set train.attn_implementation=sdpa \
  --set train.max_steps=2 \
  --set train.output_adapter=artifacts/diagnostics/bf16_smoke 2>&1 | grep -v 429 | tail -30
