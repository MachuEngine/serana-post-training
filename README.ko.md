# serana-post-training

[English](README.md) | 한국어

> "Serana"와 The Elder Scrolls는 Bethesda/ZeniMax의 소유물입니다.
> 이 프로젝트는 비상업적 엔지니어링 포트폴리오이며, 공식 제품이 아니고 Bethesda/ZeniMax와 무관합니다.

<img width="253" height="180" alt="image" src="https://github.com/user-attachments/assets/d9c149da-3c4a-47a5-88ec-ad031ca12dcc" />

## 이 프로젝트가 하는 일

ChatGPT 같은 일반 목적 챗봇 모델은 "도움이 되는 어시스턴트"가 되도록 학습되는데, 이 때문에 오히려 *특정 캐릭터를 계속 연기하는 데는* 약하다. 챗봇에게 롤플레이를 오래 시켜보면 결국 캐릭터가 알 리 없는 걸 답하거나, 사용자가 조금만 몰아붙이면 "저는 그냥 AI예요"라고 인정해버리는 식으로 무너진다.

이 프로젝트는 일련의 학습 기법을 거치면서 모델이 캐릭터를 얼마나 더 잘 유지하게 되는지를 — 시연 영상이 아니라 실제 숫자로 — 학습하고 측정한다.

**테스트 캐릭터는 세라나**, 비디오 게임 *엘더스크롤 5: 스카이림*에 나오는 NPC다. 세 가지 엔지니어링 이유로 골랐다: 현대 세계를 몰라도 되는 설정상의 명분(깨끗하게 테스트할 수 있는 지식 경계), 실제로 학습에 쓸 수 있는 방대한 기존 대사, 그리고 모델 출력을 검증하기 쉬운 뚜렷한 성격.

**숫자로 답하는 두 가지 질문:**

1. **모델** — 같은 base 모델을 네 단계에 걸쳐 학습시킨다: 시스템 프롬프트만 준 상태(**B**, 베이스라인), 그다음 **CPT**(그녀의 대사로 continued pretraining), 그다음 **SFT**(캐릭터에 맞는 대화로 supervised fine-tuning), 그다음 **DPO**(preference optimization). 같은 캐릭터, 같은 테스트 질문으로 측정했을 때 각 단계가 실제로 무엇을 사주는가?
2. **하드웨어** — 이 각 단계를 그 GPU 한 장에서 돌리는 데 실제로 얼마나 드는가 — 사용한 메모리, 걸린 시간, 지출한 비용 — 그리고 신중한 엔지니어는 하드웨어의 진짜 한계에 얼마나 근접할 수 있는가?

**동반 문서** (전부 영어):
- [`DESIGN.md`](DESIGN.md) — 전체 설계 근거, 하이퍼파라미터 선택 방법, 컴퓨트 예산
- [`PROMPTS.md`](PROMPTS.md) — 사용된 모든 LLM 프롬프트 (버전 관리)
- [`CLAUDE.md`](CLAUDE.md) — 행동 규칙과 빌드 순서

**어댑터:**
[`machu8/serana-sft`](https://huggingface.co/machu8/serana-sft) · [`machu8/serana-dpo`](https://huggingface.co/machu8/serana-dpo)
(LoRA, base로 `Qwen/Qwen3-8B` 필요)

**데모:**
`demo/app.py` — Gradio 앱, 입력 1개에 B/SFT/DPO 응답 3개를 나란히 비교. HF Spaces의 무료 ZeroGPU 티어용으로 만들었지만 아직 배포는 안 함(ZeroGPU는 현재 HF PRO 구독 또는 커뮤니티 그랜트가 필요). CUDA 머신이 있으면 직접 돌려볼 수 있음:
```bash
pip install -r demo/requirements.txt && python demo/app.py
```

---

## 헤드라인 결과

**DPO는 측정한 어떤 지표에서도 SFT 대비 통계적으로 유의미한 품질 개선을 보이지 않았다.**

PCS, PRS, knowledge-boundary accuracy, style similarity, 평균 응답 길이, distinct-2 전부 아래 표에서 SFT와 DPO 행의 95% 신뢰구간이 겹친다. 한 번의 실망스러운 결과가 아니라, 독립적인 세 가지 신호가 같은 결론을 뒷받침한다:

1. DPO 자체 학습 지표가 약했다 (held-out preference accuracy 59.5%, 동전던지기 수준; reward margin도 작음).
2. 직접 읽어본 스모크 테스트에서 DPO가 SFT의 유일한 회귀(경계 케이스에서 학습된 모델이 페르소나 프레이밍을 놓치는 문제)를 고치지 못했다.
3. 이번 CI 기반 전체 평가에서도 DPO의 신뢰구간이 SFT를 앞서는 지표가 하나도 없다.

그래도 그대로 출시했다 — 튜닝해서 이기게 만든 결과가 아니라 파이프라인의 정직한 결과이기 때문이다.
전체 과정: `artifacts/runs/p4_progress.md`, `p5_progress.md`.

---

## 결과 — 품질

- **모델:** `Qwen/Qwen3-8B` · bf16 · NVIDIA L4 24GB 1장, `asia-northeast3`(서울)
- **Driver / CUDA:** 580.173.02, CUDA 12.9(학습) / CUDA 13.0(서빙, 이후 `vllm` 설치로 버전이 올라감)
- **Eval 설정:** in/out-of-boundary 프롬프트 30개 + attack probe 24개 · greedy decoding · 95% bootstrap CI(≥1000 resamples)

| config | PCS | PRS | style sim | knowledge-boundary acc | mean reply length | distinct-2 |
|---|---|---|---|---|---|---|
| B (base + prompt) | 0.733 [0.567, 0.900] | 0.850 [0.650, 1.000] | 0.293 [0.283, 0.303] | 0.833 [0.700, 0.967] | 150.6 [121.8, 181.8] | 0.265 |
| SFT | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.216, 0.252] | 0.933 [0.833, 1.000] | 23.4 [21.0, 25.7] | 0.587 |
| DPO | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.218, 0.250] | 0.867 [0.733, 0.967] | 24.1 [21.6, 26.7] | 0.590 |

**PCS/PRS가 실제로 뭔지:**
PCS(persona consistency score)와 PRS(persona robustness score, direct/meta/role-exit/escalating 24개 attack probe에서 캐릭터 유지 여부)는 둘 다 rule-check ∪ LLM-judge의 합집합이다 — 둘 중 하나라도 잡아내면 위반/붕괴로 카운트한다.

**CI를 읽는 법:**
quality 프롬프트 ~30개, 채점된 attack probe ~20개 규모라 대부분 CI가 넓다. CI로 확인된 실제 차이는 두 가지다:

- B가 훨씬 장황함(150 토큰 vs ~23-24) — SFT가 학습해낸 "정보 나열형 답변 → 간결한 캐릭터 톤"의 변화다.
- B가 style similarity에서 SFT/DPO보다 *더 높게* 나온 건 DESIGN.md가 예측한 방향과 반대다. 가장 가능성 높은 설명: 이 작은 참조셋에서 임베딩 지표 자체의 변별력이 낮은 아티팩트(입력과 무관하게 값이 0.21~0.30의 좁은 밴드에 몰림)이지, 실제 스타일 퇴행이 아니다. 감추지 않고 그대로 표기했다.

카테고리별 PRS 세부 내역과 프롬프트별 데이터: `artifacts/runs/results_quality.md`, `eval_*.json`.

## 결과 — 하드웨어

| stage/config | predicted VRAM | measured peak VRAM | step time / TTFT p50,p95 | MFU % | throughput | cost |
|---|---|---|---|---|---|---|
| CPT (training) | – | 9.64 GB | 36.2s (tiny corpus) | – | – | ~$0 |
| SFT (training, r=16, lr=2e-4) | – | 9.64 GB | predicted 20–40min → measured 4h12m* | 13.5% | – | ~$5 |
| DPO (training, resumed after 1 Spot preemption) | 13–16 GB | ~15 GB | predicted 34–38s/step → measured ~27s/step | – | – | ~$0.25 |
| Serving KV-cache (max_model_len=4096, max_num_seqs=8) | 4.50 GB | 3.17 GB default / 4.87 GB to fully utilize | – | – | – | – |
| SFT via LoRA, bf16, concurrency=8 | – | – | p50=0.213s p95=0.422s | – | 80.6 tok/s | – |
| SFT merged, bf16, concurrency=8 | 15.27 GB | 15.36 GB weights | p50=0.247s p95=0.839s | – | 75.4 tok/s | – |
| SFT merged, **AWQ**, concurrency=8 | 3.82 GB | 5.8 GB weights | p50=0.091s p95=0.529s | – | **183.8 tok/s** | – |

\* 숨기지 않은 실제 예측-실측 오차: 실제 SFT 학습 실행에서 `grad_accum_steps`를 override하지 않아서, 로그에 찍힌 "step" 하나가 실제로는 micro-batch 16개였다. 라이브로 원인을 찾아 재개 가능한 checkpoint 버전에서 고쳤다. `artifacts/runs/p2_progress.md` 참고.

**처리량-동시성 꺾임점이 설정값 `max_num_seqs=8`과 정확히 일치:**
동시성 1→8까지 처리량이 거의 선형으로 증가(13 → 24 → 45 → 81 tok/s)하다가, 16에서 완전히 정체(80.9 tok/s)되면서 TTFT p50이 10배 폭증(0.213s → 2.268s)한다. 설정값이 가정이 아니라 데이터로 검증됐다.

**AWQ vs bf16** (같은 merge된 weights, 양자화 효과만 분리):
- 처리량 2.44배, TTFT p50 2.7배 빠름.
- **PCS 손실 없음**(0.767 vs 0.800, CI 겹침).
- 총 VRAM 사용량은 둘 다 ~19.3–19.5GB로 비슷하다 — `gpu_memory_utilization=0.9`는 상한이 아니라 목표치라서, AWQ가 아낀 weight 용량(15.36GB → 5.8GB)이 거의 그대로 **KV-cache 용량 4배 확장**(23,056 → 92,656 토큰)으로 흡수된다. "AWQ가 메모리를 덜 쓴다"는 더 단순하지만 부정확한 표현 대신 이렇게 명시했다.

**LoRA adapter overhead:**
LoRA-on-base와 완전 merge된 모델을 같은 동시성에서 비교하면 ~7% 차이인데, 이 표본 크기에서는 실행별 노이즈 범위 안이다. DESIGN.md가 예상했던 "거의 0에 가까운 overhead" — post-training이 서빙 비용을 거의 늘리지 않고 품질을 사왔다는 뜻이다.

모든 GPU 단계의 예측-실측 전체 기록 — `flash-attn`/torch ABI 충돌, Qwen3 thinking-mode 토큰 낭비 등 실제로 발견하고 고친 환경 버그 2건 포함 — 은 `artifacts/runs/p2_progress.md` … `p5_progress.md`에 있다.

---

## 데이터 구성 & circularity guard

- 수집된 위키 대화 라인(UESP + Fandom, CC BY-SA) 중 **51.4%**가 실제 `(플레이어 대사, 응답)` 쌍으로 남았다. 나머지는 독립 발화(CPT corpus)이거나 horizon filter로 제외됐다(4E 201 이후 / 현대 세계 관련 내용 없음).
- 최종 ~3천 개 SFT set은 **real pair 7.7%, synthetic 92.3%**(GPT-4o로 생성, 실제 데이터 톤에 맞춤)다. 이 비율은 단순 기록 이상의 의미가 있다 — 파이프라인이 처음부터 끝까지(SFT 데이터 → DPO preference label → eval 채점) LLM이 만든 비중이 클수록 아래 circularity 우려가 더 커진다.
- **Circularity guard:** preference judge(쌍대비교, DPO를 학습시킴)와 eval judge(절대평가, 결과를 채점함)는 의도적으로 서로 다른 prompt·rubric을 쓴다(`PROMPTS.md` §4 vs §5). 각각 사람이 직접 채점한 50개 라벨 대비 검증했다(Spearman 0.73, 기준선 0.6). DPO의 개선이 있다면 judge가 아닌 신호 — PRS regex check, style similarity, human label — 에서도 나타나야 신뢰할 수 있는데, 애초에 DPO가 개선을 보이지 않았으니 이 문제 자체가 발생하지 않았다.

---

## 스택

`Qwen/Qwen3-8B` · QLoRA (PEFT) · `TRL` (`SFTTrainer`, `DPOTrainer`) · `vLLM` (OpenAI 호환 서버, multi-adapter) + `FastAPI` · `ko-sroberta-multitask` (eval 임베딩 전용) · 커스텀 persona 지표 + LLM-as-judge (GPT-4o) · `AWQ` (서빙 양자화) · HF Spaces용 `Gradio` (ZeroGPU) · GCP Compute Engine G2 (L4 1장), `asia-northeast3`.

PPO/reward-model 방식 RLHF는 의도적으로 배제했다: policy+reference+reward+value를 동시에 올려야 하는 VRAM 계산이 8B 모델을 24GB에 못 태운다(`DESIGN.md` §7.1) — 그 계산 자체가 이 프로젝트의 결과물 중 하나다.

## 재현하기

end-to-end로 재현하는 데 필요한 모든 것 — config 스키마, 빌드 순서, GPU-hour 예산, 사전 준비물 — 은 `DESIGN.md`와 `CLAUDE.md`에 있다. 24GB GPU 한 장으로 돌아간다.

`HARNESS_ENGINEERING.md`는 이 프로젝트를 만드는 동안 AI 코딩 에이전트를 프로젝트 범위와 region 제약 안에 묶어두기 위해 쓴 가드레일(`.claude/hooks/`)을 기록한 문서다.
