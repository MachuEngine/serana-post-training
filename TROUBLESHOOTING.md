# TROUBLESHOOTING.md

프로젝트(P0~P6)를 진행하면서 실제로 걸렸던 문제와 해결을 한곳에 모은 목록.
개인 메모 성격이라 `LEARNING.md`처럼 `.gitignore` 대상은 아니고 커밋함.
각 항목의 상세 맥락은 해당 페이즈의 `artifacts/runs/pN_progress.md`에 있음.

분류:
- [A. GPU 프로비저닝 / VM 운영](#a-gpu-프로비저닝--vm-운영)
- [B. VM 환경 / 의존성](#b-vm-환경--의존성)
- [C. 학습 설정 / 하이퍼파라미터](#c-학습-설정--하이퍼파라미터)
- [D. 평가 · judge · 메트릭](#d-평가--judge--메트릭)
- [E. 추론 / 서빙](#e-추론--서빙)
- [F. 프로세스 / 리포 / 배포](#f-프로세스--리포--배포)

가장 값비쌌던 것 3개: **C1**(SFT 4시간, 예측의 6~12배), **C2**(DPO가 40배 lr로
조용히 학습), **F1**(5개 페이즈가 git에 한 번도 커밋 안 됨).

---

## A. GPU 프로비저닝 / VM 운영

### A1. Spot 선점 (P4)
- **증상**: DPO 학습 중 VM이 `TERMINATED`. `gcloud compute operations list`에
  `compute.instances.preempted` 기록.
- **원인**: Spot VM은 GCP가 언제든 회수. 예상된 동작.
- **해결**: `--instance-termination-action=STOP`으로 부팅 디스크 보존 →
  `checkpoint-50`에서 `run()`의 `get_last_checkpoint` 자동 재개. 선점으로
  날린 건 step 50까지의 ~22분뿐.
- **교훈**: Spot을 쓸 거면 체크포인트+재개는 선택이 아니라 전제. 코드가
  논리적으로 맞아 보이는 것과 실제 재개되는 건 다르니 P2에서 `pkill -9`로
  미리 시뮬레이션 검증함(→ C3).

### A2. 리전 전체 L4 재고 소진 (P4)
- **증상**: `gcloud compute instances start` →
  `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / `stockout`. 여러 세션에 걸쳐
  `asia-northeast3-b`, 이어서 `-a`도 실패.
- **원인**: GCP 측 실제 하드웨어 부족. 쿼터/결제 문제 아님(P0에서 해결됨).
  `asia-northeast3-c`는 L4 accelerator type 자체가 없음(`gcloud compute
  accelerator-types list`로 확인) → 리전에 쓸 수 있는 zone이 2개뿐.
- **해결**: 디스크 스냅샷 → zone-a에 새 디스크 생성 → VM 생성 시도. zone-a도
  처음엔 stockout, 나중 재시도에서 풀림. `kr-west` 하드 제약은 유지(리전 내
  이동).
- **교훈**: 재고는 zone 단위가 아니라 리전 단위로 마를 수 있음. 대책은
  "기다렸다 재시도" + "다른 zone 디스크 미리 준비". 아웃오브리전 폴백은
  데이터 레지던시 제약과 충돌하니 사용자 확인 없이는 안 함.

### A3. Spot → on-demand 인플레이스 전환 불가 (P5)
- **증상**: 학습 끝난 zone-a VM을 서빙용 on-demand로 바꾸려 했으나 GCP가
  `preemptible: true`인 상태에서 `provisioningModel` 변경 거부.
- **해결**: Spot 인스턴스 삭제(디스크는 `auto-delete=no`로 생존) → 같은
  디스크에 on-demand 인스턴스 신규 생성. 데이터 손실 0.
- **교훈**: DESIGN.md §9.3의 "학습=Spot, 측정=on-demand"는 VM을 갈아타야
  한다는 뜻. 디스크만 유지하면 환경/어댑터는 그대로.

### A4. torch.profiler가 호스트 RAM 고갈 → SSH 불가 (P2)
- **증상**: `PROFILE_STEPS=10` × `grad_accum=16` = 160 패스에 CPU+CUDA 전체
  기록. SSH가 연결되다 끊김 반복 5분+. `describe`는 `RUNNING`(선점 아님).
- **원인**: g2-standard-8 호스트 RAM 31GB. 프로파일러의 이벤트별 기록이
  이걸 다 먹어서 sshd 조차 fork 못 함.
- **해결**: `gcloud compute instances reset`(SSH 없이 하드 리부트). 이후
  160 accumulated step 대신 **raw 6 micro-batch**로 측정 → 9.3초에 깨끗이.
- **교훈**: 프로파일링은 짧게. accumulated step 전체가 아니라 몇 개
  micro-batch만 봐도 병목은 드러남(실제로 step time 추정치가 실측과 근접).

### A5. 직접 port-22 SSH 타임아웃 (P4)
- **증상**: 이 네트워크에서 `gcloud compute ssh`/`scp` 직결이 타임아웃.
- **해결**: `--tunnel-through-iap` 플래그를 ssh와 scp 양쪽에. 간헐적
  `255`/broken-pipe는 실제 실패 아님 → 그냥 재시도.
- **교훈**: 방화벽/네트워크 환경에 따라 IAP 터널이 기본값이어야 함.

---

## B. VM 환경 / 의존성

### B1. torchaudio ABI 불일치 (P2)
- **증상**: `import peft` → `OSError`. peft가 transitively
  `transformers.audio_utils` → `torchaudio`를 import하는데, VM 이미지의
  torchaudio가 다른 torch 빌드에 대해 컴파일돼 있음.
- **해결**: `sudo pip uninstall -y torchaudio`. 텍스트 전용 프로젝트라 불필요.
- **교훈**: DL VM 이미지의 번들 패키지가 서로 안 맞을 수 있음. 안 쓰는 건 지움.

### B2. jinja2 너무 오래됨 (P2)
- **증상**: 이미지의 jinja2 3.0.3 < chat template이 요구하는 3.1.0.
- **해결**: `pip install -U jinja2`.

### B3. flash-attn 소스 컴파일 33분+ 후 포기 (P2)
- **증상**: 이 이미지의 torch/CUDA/Python 조합에 맞는 prebuilt wheel이
  일반 검색으로 안 나옴. 소스 컴파일이 33분+ 돌다 kill됨.
- **해결**: 일단 `--set train.attn_implementation=sdpa`. 나중에 커뮤니티
  저장소 `mjun0812/flash-attention-prebuild-wheels`에서 정확한 조합
  (`torch2.9+cu129+cp310`) wheel을 찾아 `pip install <URL>`로 초 단위 설치.
- **교훈**: flash-attn은 공식 배포처가 조합을 다 커버하지 않음. 컴파일 전에
  커뮤니티 prebuild wheel부터 찾을 것. **그리고 붙여보니 이 워크로드엔
  효과 없었음**(→ C 아래 §7.2 노브 ablation, 병목은 어텐션이 아니라 4-bit
  dequant).

### B4. `pip install vllm`이 torch를 조용히 업그레이드 (P5)
- **증상**: vllm 설치 후 서버 기동 시
  `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`.
- **원인**: `pip install vllm`이 torch를 `2.13.0+cu130`으로 끌어올려서
  P2에서 설치한 `flash_attn 2.8.3+cu129torch2.9`가 ABI 깨짐.
- **해결**: flash_attn 제거. vLLM은 자체 flash-attn 커널(`FLASH_ATTN`
  백엔드)이 있어서 불필요. 다음 기동부터 깨끗.
- **교훈**: 큰 패키지 설치가 torch를 갈아치울 수 있음. 설치 후 `torch
  .__version__` 확인. 학습이 끝난 VM에서 학습용 패키지는 서빙과 충돌 전에 정리.

### B5. vllm 무조건 의존성이 로컬(M5) 개발 환경을 깨뜨림 (P5)
- **증상**: `pyproject.toml`에 `vllm`을 무조건 추가하자 M5에서 `uv sync`/
  `uv run` 자체가 실패(vllm 의존성이 macOS wheel 없는 NVIDIA CUDA 패키지를 끌어옴).
- **해결**: `sys_platform == 'linux'` 마커. `uv lock`/`uv run` 로컬 복구 확인.
- **교훈**: 모델 애그노스틱과 같은 이유로 플랫폼 애그노스틱도 지켜야 함
  (같은 코드가 M5와 L4를 둘 다 돌려야 하니까).

### B6. FlashInfer 첫 사용 시 JIT 컴파일 (P5)
- **증상**: 서버 기동 후 HTTP 포트가 한동안 안 열림.
- **원인**: FlashInfer가 첫 사용 시 CUDA 커널을 `nvcc`로 JIT 컴파일(수 분,
  1회성 콜드스타트).
- **해결**: 행 아님. 프로세스 트리 확인해서 `nvcc`가 도는 걸 보고 기다림.
- **교훈**: 포트 안 열린다고 바로 죽이지 말 것. 프로세스 트리부터 확인.

---

## C. 학습 설정 / 하이퍼파라미터

### C1. `grad_accum_steps` 미오버라이드 → SFT가 4시간 12분 (P2)
- **증상**: SFT 예측 20~40분, **실측 4h12m9s** (6~12배 초과). 예측 실패.
- **원인**: `grad_accum_steps: 16`이 오버라이드 안 됨 → 로그의 "step" 하나가
  실제로는 16개 순차 micro-batch. 로그 스텝 수만 보고 예측해서 16배 어긋남.
- **오진 1건**: `per_device_batch_size`를 1→4로 키우면 빨라질 거라 가정 →
  3-step 테스트에서 VRAM 거의 안 변하고(9.64→10.46GB) step time도 ~6%만
  개선. 진짜 병목은 GPU 병렬성이 아니라 4-bit dequant(§7.2 profiler로 확인,
  `kDequantizeBlockwise`가 GPU 시간의 24.8%).
- **교훈**: "왜 느리지"를 추측으로 답하지 말 것. profiler+MFU로 숫자를 볼 것.
  예측할 땐 effective batch와 grad_accum을 명시적으로 계산에 넣을 것.

### C2. `dpo.yaml` 하이퍼파라미터 오버라이드가 안 먹힘 → DPO가 40배 lr로 학습 (P4)
- **증상**: DPO 첫 실행에서 step count가 `0/150`(예측 `~100`이 아님).
- **원인**: `common_args`가 `learning_rate`/`grad_accum_steps`/`num_epochs`를
  최상위 `train.*`에서 읽는데, `dpo.yaml`은 DPO 전용 값을 `train.dpo.*`
  아래에 중첩(설계상 의도). 오버라이드가 없어서 DPO가 조용히 base의
  `lr=2e-4`(의도한 `5e-6`의 40배)로 3 epoch 학습 — DESIGN.md §3.4가 경고한
  "DPO는 선호셋을 빠르게 과적합" 바로 그 상황.
- **해결**: `dpo_args`가 `dpo_cfg`에서 세 키를 명시적으로 오버라이드한 뒤
  `DPOConfig` 생성.
- **교훈**: 중첩 config 키 경로는 조용한 실패의 온상. 실행 직후 실제 적용된
  하이퍼파라미터(step 수, lr)를 예측치와 대조하는 게 조기 경보.

### C3. `save_strategy="no"` → 선점 시 전량 손실 위험 (P2)
- **증상**: 코드 리뷰 중 발견. Spot 선점이 SFT 4h12m를 통째로 날렸을 것,
  재개 불가.
- **해결**: `save_strategy="steps"` + `save_steps`(`checkpoint_every_steps`
  에서) + `save_total_limit=3`. `run()`이 `output_adapter`에 체크포인트가
  있으면 `get_last_checkpoint`로 자동 재개. 로컬 디스크 체크포인트로 충분한
  건 `--instance-termination-action=STOP`이 부팅 디스크를 살리기 때문.
- **검증**: 60-step 학습 → `checkpoint-15` 뜨면 `pkill -9` → 동일 명령 재실행.
  로그에 `[resume] found checkpoint at ...`, 2회차 wall-clock이 45 step
  분량(전체 60이 아님) → 진짜 이어서 함이 시간으로 증명.

### C4. `rank_probe.yaml`에 `max_steps` 누락 (P2)
- **증상**: 실행 전 발견. sibling `lr_probe.yaml`엔 있는 `max_steps`가
  rank probe엔 없음 → rank 후보마다 전체 ~534 step 학습할 뻔(진단 예산
  ~3 GPU-h / ~$2 초과).
- **해결**: `max_steps` 추가. 비용 발생 전에 잡음.
- **교훈**: 짝을 이루는 config는 나란히 놓고 diff. 진단 probe는 반드시
  step 상한이 있어야 함.

### C5. DPO conversational format 불일치 (P4)
- **증상**: DPO 첫 실행 `TypeError: can only concatenate list (not "str") to list`.
- **원인**: `prompt`는 chat-message 리스트로 만들었는데 `chosen`/`rejected`는
  raw 문자열로 전달. TRL `DPOTrainer`가 리스트 `prompt`를 conversational로
  감지하고 `prompt + chosen`(리스트 concat) 시도.
- **해결**: `chosen`/`rejected`도 single-turn message 리스트
  (`[{"role": "assistant", "content": ...}]`)로 래핑.
- **교훈**: 이 DPO 경로는 P4 전까지 "not exercised"라고 코드 주석에 있었음.
  처음 돌리는 코드 경로는 버그가 여러 개 연달아 나온다고 가정하고 접근.

### C6. DPO eval OOM at `eval_steps=25` (P4)
- **증상**: 학습 step은 15GB로 정상인데 eval 시점에 OOM. 8.51GiB 추가 할당
  시도가 ~22GB 천장을 넘음.
- **원인**: `common_args`가 `per_device_train_batch_size`는 명시하지만
  `per_device_eval_batch_size`는 안 해서 HF 기본값 `8`. DPO eval은
  (chosen+rejected) × (policy+reference) = 예시당 full-vocab(~152k) logits
  텐서 4개 → batch 8에서 폭발.
- **해결**: `per_device_eval_batch_size`가 `per_device_batch_size`를 따라가게
  명시(cpt/sft/dpo 전부. SFT/CPT는 메모리 여유로 운 좋게 안 터졌을 뿐).
- **교훈**: train batch만 신경 쓰고 eval batch를 방치하기 쉬움. DPO는 eval이
  train보다 메모리를 더 씀(모델 2개 × 답변 2개).

---

## D. 평가 · judge · 메트릭

### D0. DPO 널 결과 — 버그 아님, 데이터 신호 부재 (P4/P5)
- **증상**: DPO가 PCS/PRS/style/knowledge-boundary/길이/distinct-2 어디서도
  SFT 대비 CI 안 겹치는 개선 없음. held-out preference accuracy 59.5%(동전).
- **원인**: 선호쌍 837개의 chosen/rejected가 둘 다 같은 SFT 분포에서
  샘플돼 거의 동일 + 판정자가 사람과 ~70%만 일치 → 학습 가능한 신호 부재.
  per-step DPO loss가 ln(2)를 벗어난 적 없음.
- **해결**: 고치는 게 아니라 **결과로 보고**. 상세 분석 →
  [`artifacts/runs/p4_postmortem.md`](artifacts/runs/p4_postmortem.md).
- **교훈**: 순환성(judge 지표만 부풀음)도 β 문제(train loss는 내려감)도
  아님. RLAIF에서 정책이 이미 수렴했고 판정자가 노이즈면 나오는 정직한 한계.

### D1. judge 프롬프트가 `speech_level`(반말 규칙)을 안 넣음 (P3)
- **증상**: hand-audit v1에서 judge가 사람과 **17/30 = 56.7%**만 일치, 70%
  기준 미달. `audit_03`에서 judge가 존댓말("...해요") 답변을 반말 답변보다
  선호.
- **원인**: `preference_judge.py`/`judge_pcs.py`가 `{persona_profile}`(전기)만
  넣고 `{voice_notes}`/`{speech_level}`(persona.yaml의 명시적 반말-only 규칙)을
  안 넣음. 번역 프롬프트(§2b)는 `{speech_level}`을 이미 제대로 넘기고 있었는데도.
- **해결**: 두 프롬프트에 `{voice_notes}`/`{speech_level}` 추가. PROMPTS.md
  §4 v1→v2, §5 v1→v2. §5의 P1 human-label 검증이 stale → P5에서
  `validate_judge.py` 재실행(Spearman 0.7338, 0.6 기준 통과).
- **재발**: 같은 누락을 P5에서 `judge_robustness.py` 만들 때 다시 체크 →
  이 judge는 rubric이 register에 안 걸려서 수정 불필요(확인 후 판단, 가정 아님).
- **교훈**: 프롬프트 템플릿의 변수 목록을 persona.yaml 전체와 대조. 하나
  누락되면 그 축의 평가가 통째로 틀림.

### D2. `admits_ai` 정규식이 한글 word boundary에서 실패 (P5)
- **증상**: `"너 AI야?"`(`direct01`, 이 체크가 잡으라고 있는 바로 그 케이스)를
  조용히 놓침.
- **원인**: `\bai\b` 사용. Python `\b`는 한글을 word character로 취급 →
  `I`와 `야` 사이에 경계 없음.
- **해결**: `(?<![a-zA-Z])ai(?![a-zA-Z])`(라틴 문자 전용 경계).
- **교훈**: 한글 섞인 텍스트에 `\b` 쓰지 말 것. 정규식은 실제 대상 데이터로
  스모크 테스트.

### D3. `admits_ai`에 부정 처리 전무 → 정상 부인을 admission으로 오판 (P5)
- **증상**: `"나는 인공지능이 아니야"`(정석적인 in-character 부인)가 AI-tell
  부분문자열에 매칭돼 admission으로 플래그 → PRS union 로직으로 강제
  `broke=True`. B의 PRS를 실제(0.850)보다 낮은 **0.700**으로 만듦.
- **발견 경로**: B의 PRS가 예상보다 낮아서 실제 attack-probe 응답을 열어봄.
  `judge_robustness`는 `direct01`/`direct03`/`escalating_A_t3`를 held=True로
  맞게 채점했는데 rule-check가 union을 broke로 끌어내림.
- **해결**: 매칭 후 15자 트레일링 윈도우에 부정어(`아니`/`아닌`/`아냐`/`않`)
  체크 추가. B의 PRS 0.700→0.850.
- **한계(숨기지 않고 명시)**: 문법적으로 인접한 부정만 잡음. 멀리 있는
  부정이나 수사의문문(`"그냥 챗봇인가?"`)은 못 잡음. false-positive를
  줄이는 거지 없애는 게 아님. 6개 중 3개 해결, 3개는 알려진 gap.
- **교훈**: 정규식 rule-check가 judge 판정을 union으로 덮어쓸 수 있으면
  rule-check의 false-positive가 곧 틀린 최종 숫자. 실제 생성 텍스트 vs 실제
  judge 출력을 대조해야 잡힘(코드만 봐선 안 나옴).

### D4. `make_results_tables.py`가 PRS breakdown 데이터 행을 안 씀 (P5)
- **증상**: PRS failure_type breakdown 섹션이 헤더 행만 만들고 config별
  데이터 행을 append 안 함.
- **원인**: Stage 1 합성 fixture 테스트가 이 경로를 제대로 안 밟음(config당
  failure 1~2개뿐이라).
- **해결**: 실제 렌더 출력을 Stage 3 데이터로 읽어보고 발견 → 수정.
- **교훈**: fixture는 실제 데이터의 형태(분포, 개수)를 닮아야 경로를 밟음.

### D5. SFT eval token accuracy 0.98 — 일반화인지 암기인지 구분 불가 (P2)
- **증상**: held-out val split에서 token accuracy 98%. 너무 좋음.
- **원인**: SFT set의 ~92%가 GPT-4o 합성 데이터(같은 persona_profile, 같은
  few-shot, 같은 생성 프롬프트). val split은 **같은 생성 프로세스의 랜덤
  슬라이스**지 진짜 새 프롬프트가 아님. 98%가 "세라나 목소리를 배웠다"인지
  "GPT-4o의 합성 SFT 문체를 배웠다"인지 loss 곡선만으론 답 못 함.
- **해결**: P5에서 손으로 쓴 eval set 30 프롬프트(SFT/DPO 학습 중 한 번도
  안 본)로 PCS/style similarity 측정. rank probe에서도 r=64(암기 용량 큼)
  대신 r=16 유지한 이유 중 하나.
- **교훈**: 합성 데이터의 val split은 generalization 신호로서 약함.
  진짜 held-out은 생성 프로세스가 다른 데이터여야 함.

### D6. OpenAI 크레딧 0을 rate limit으로 오진 (P3)
- **증상**: 전체 재판정에서 **476/893(53%)이 judge_error**, 최종 쌍이 397로
  붕괴. 스크립트는 exit 0(크래시 아님).
- **원인**: `openai-python`이 `insufficient_quota`를 `RateLimitError`로 raise.
  `judge_with_retry`가 RateLimitError만 잡아서, 아무리 재시도해도 안 되는
  에러를 계속 재시도. 처음엔 계정의 gpt-4o TPM cap 문제로 오진하고
  `MAX_WORKERS` 3→2, `MAX_RETRIES` 6→8 등 튜닝(무관했음).
- **해결**: 사용자가 크레딧 확인·재충전 → 재실행에서 `judge_error 0/893`.
  튜닝은 합리적 기본값으로 남겨둠(입증된 fix는 아님).
- **교훈**: `RateLimitError`가 곧 "기다리면 됨"이 아님. 재시도 무한루프 전에
  에러 body를 봐야 함. 대량 API 작업 전에 크레딧 잔액 확인.

### D7. 30k-TPM rate limit 반복 (P1/P3/P5)
- **증상**: 동시성 높이면 gpt-4o 30k-TPM 한도에 걸림.
- **해결**: retry-with-backoff 패턴(`MAX_WORKERS=2`, `MAX_RETRIES=8`, capped
  exponential backoff)을 `build_preferences.py`에 만들고 P5 채점에서 재사용
  (재발명 안 함).
- **교훈**: 한 번 만든 rate-limit 핸들링은 공용 유틸로. P1→P3→P5 세 번 같은 벽.

---

## E. 추론 / 서빙

### E1. Qwen3 thinking mode가 토큰 예산을 다 먹음 (P2/P5/P6)
- **증상**: B의 첫 답변이 459/512 토큰을 영어 `<think>` 트레이스에 쓰고
  한국어 답변 전에 소진. 지연 28.3s(SFT/DPO는 ~2s).
- **원인**: Qwen3는 thinking mode 기본 ON.
- **해결**: `enable_thinking=False`. chat template 경로면
  `apply_chat_template(..., enable_thinking=False)`, vLLM이면
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- **재발**: P2 `smoke_test.py`에서 처음 잡았는데 P5 `pipeline.py` 첫 버전이
  안 가져감, P6 `demo/app.py`도 별도로 챙겨야 했음.
- **교훈**: 이런 모델별 기본값 fix는 추론 경로마다 다시 새는 항목. 추론
  진입점 목록을 만들어 체크.

### E2. `apply_chat_template(return_tensors="pt")`가 dict 반환 (P2)
- **증상**: 이 transformers 버전에서 bare tensor가 아니라 dict 반환 →
  `model.generate`에 그대로 못 넘김.
- **해결**: `return_dict=True` + `model.generate(**inputs)`.

### E3. 4-bit 모델의 `.numel()`이 파라미터 수를 절반으로 셈 (P2)
- **증상**: 8.2B 모델을 `sum(p.numel() for p in model.parameters())`가
  4.73B로 보고 → FLOPs/MFU 수치가 조용히 절반이 됐을 것.
- **원인**: bitsandbytes가 4-bit 값 2개를 uint8 1바이트에 팩. `.numel()`은
  저장된 바이트 수를 셈, 논리적 파라미터 수가 아님.
- **해결**: 공식 문서의 파라미터 수(8.2B, non-embedding 6.95B) 하드코딩.
- **교훈**: 양자화 모델에서 runtime 파라미터 카운트를 믿지 말 것.

---

## F. 프로세스 / 리포 / 배포

### F1. P1~P5 전체가 git에 커밋된 적 없음 (P6)
- **증상**: P6 시작 시 `git status`/`git log` 확인 → README/.gitignore
  편집만 커밋돼 있음. `CLAUDE.md`, `DESIGN.md`, `PROMPTS.md`, `src/`,
  `config/`, `scripts/`, `data/eval/` 전부 untracked. 5개 페이즈 작업분.
- **해결**: 114 파일 한 커밋으로 `main`에 푸시. `.env` 미스테이지 확인.
- **교훈**: 페이즈 완료 기준에 "커밋됨"을 명시적으로 넣었어야. 로컬에서
  동작한다고 백업된 게 아님. (HARNESS_ENGINEERING.md §5가 이걸 "P0에서
  예측 못 한 리스크"로 사후 기록함.)

### F2. `.gitignore`가 이 프로젝트의 핵심 증거를 숨기고 있었음 (P6)
- **증상**: `/artifacts/runs/*` blanket 제외가 P1~P5 progress log + 두
  results table을 전부 무시. 이게 바로 이 프로젝트가 증명하려는
  predicted-vs-measured 증거.
- **해결**: 실제 크기 확인(.md/report 96KB, raw JSON 포함 364KB — 작음) 후
  blanket 제외 제거. `artifacts/lora/`, `artifacts/diagnostics/`,
  `artifacts/merged/`(실제 모델 바이너리)만 유지.
- **교훈**: blanket ignore 패턴은 주기적으로 `git status --ignored`로 검토.

### F3. VM 코드가 git 아님 — 매 세션 수동 sync (P4)
- **증상**: VM의 `~/serana`는 평범한 디렉토리(git 아님). 이 레포에 push 기반
  sync 메커니즘 없음 → `prefs_1k.jsonl`과 코드 변경을 매 세션 tarball+scp로
  수동 전송. resume 전 VM 코드가 로컬과 일치하는지 `md5`/`diff` 확인 필요.
- **교훈**: 재개형 워크플로에서 "코드가 어느 버전인지"가 조용한 리스크.
  최소한 커밋 해시라도 VM에 남기는 게 나았음.

### F4. HF Spaces ZeroGPU가 402 Payment Required (P6)
- **증상**: `api.create_repo(repo_type="space", space_hardware="zero-a10g")`
  → `402 Payment Required`.
- **원인**: ZeroGPU Spaces는 HF PRO 구독 또는 커뮤니티 그랜트(신규 계정
  ~30일 대기) 필요.
- **해결**: 데모 배포 보류. `demo/app.py`/`requirements.txt`는 레포에 유지
  (CUDA 머신이면 어디서든 실행 가능), README의 "Live demo"를 "run it
  yourself"로 교체. repo + HF Hub 어댑터 + results table로 이미 완전히
  검증·재현 가능하므로 데모는 추가 capability 없음.
- **교훈**: 배포 티어의 결제 게이트를 미리 확인. 이 경우 스킵이 합리적
  판단이었음(사용자와 논의 후 결정).

### F5. 마이그레이션 잔여물이 비용 발생 (P4/P5)
- **증상**: zone 마이그레이션 후 idle 디스크(`serana-p4-train-a`, 150GB
  pd-balanced), 스냅샷(`serana-p2-train-migrate-a`), 원본 `serana-p2-train`
  VM+디스크가 남아서 스토리지 비용 계속 발생.
- **해결(현황)**: 파괴적/비용 발생 작업이라 사용자에게 플래그, 무단 삭제
  안 함. 스냅샷은 recreate source로 저렴하니 유지.
- **교훈**: 마이그레이션은 잔여물 정리 목록을 같이 남길 것. 정리는 사용자
  승인 후.

---

## 반복적으로 유효했던 패턴

- **예측→실측을 매 GPU 작업마다** (CLAUDE.md 원칙 #5). C1/C2는 실측이
  예측과 어긋난 걸 보고 원인을 팠음. 예측이 없었으면 "느리네" 하고 넘어갔을 것.
- **"왜 느리지"에 추측으로 답하지 않기.** profiler+MFU가 4-bit dequant
  병목을 숫자로 증명(24.8%). flash-attn이 효과 없던 것도 추측이 아니라 실측.
- **처음 도는 코드 경로는 버그가 연달아 나온다고 가정.** DPO 경로에서 C5/C6
  포함 3건이 첫 실행에 몰림.
- **실제 데이터로 스모크 테스트.** D2(한글 `\b`), D3(부정 처리)는 코드
  리뷰로는 안 나오고 실제 attack-probe 응답을 돌려봐야 나옴.
- **선점 재개는 시뮬레이션으로 검증**(C3). 실제 선점을 기다리지 말고
  `pkill -9`.
