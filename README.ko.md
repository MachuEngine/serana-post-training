# serana-post-training

[English](README.md) | 한국어

> "Serana"와 The Elder Scrolls는 Bethesda/ZeniMax의 소유물입니다.
> 이 프로젝트는 비상업적 엔지니어링 포트폴리오이며, 공식 제품이 아니고 Bethesda/ZeniMax와 무관합니다.

<img width="253" height="180" alt="image" src="https://github.com/user-attachments/assets/d9c149da-3c4a-47a5-88ec-ad031ca12dcc" />

## 이 프로젝트가 하는 일

ChatGPT 같은 일반 목적 챗봇 모델은 "도움이 되는 어시스턴트"가 되도록 학습되는데, 이 때문에 오히려 *특정 캐릭터를 계속 연기하는 데는* 약하다. 챗봇에게 롤플레이를 오래 시켜보면 결국 캐릭터가 알 리 없는 걸 답하거나, 사용자가 조금만 몰아붙이면 "저는 그냥 AI예요"라고 인정해버리는 식으로 무너진다.

이 프로젝트는 일련의 학습 기법을 거치면서 모델이 캐릭터를 얼마나 더 잘 유지하게 되는지를, 시연 영상이 아니라 실제 숫자로 학습하고 측정한다.

**테스트 캐릭터는 세라나**, 비디오 게임 *엘더스크롤 5: 스카이림*에 나오는 NPC다. 세 가지 엔지니어링 이유로 골랐다: 현대 세계를 몰라도 되는 설정상의 명분(깨끗하게 테스트할 수 있는 지식 경계), 실제로 학습에 쓸 수 있는 방대한 기존 대사, 그리고 모델 출력을 검증하기 쉬운 뚜렷한 성격.

**숫자로 답하는 세 가지 질문:**

1. **모델:** 같은 base 모델을 네 단계에 걸쳐 학습시킨다: 시스템 프롬프트만 준 상태(**B**, 베이스라인), 그다음 **CPT**(그녀의 대사로 continued pretraining), 그다음 **SFT**(캐릭터에 맞는 대화로 supervised fine-tuning), 그다음 **DPO**(preference optimization). 같은 캐릭터, 같은 테스트 질문으로 측정했을 때 각 학습 이후 얻을 수 있는 성능은 무엇인가?
2. **하드웨어:** 이 각 단계를 그 GPU 한 장에서 돌리는 데 실제로 얼마나 드는가(사용한 메모리, 걸린 시간, 지출한 비용), 그리고 신중한 엔지니어는 하드웨어의 진짜 한계에 얼마나 근접할 수 있는가?
3. **운영:** VM에 `nohup`으로 띄워 돌아가는 상태에서, 팀이 운영하는 형태로 만들려면 무엇이 더 필요한가 — 버전이 고정된 이미지를 쿠버네티스에 올리고, 지표가 붙은 서비스 계층을 두고, preemption을 넘어 하나의 run으로 이어지는 추적까지. 그리고 위 숫자들은 *어댑터*를 설명하는가, *서빙 스택*을 설명하는가? ([운영 계층](#운영-계층))

**설계 문서** :
- [`DESIGN.md`](DESIGN.md): 전체 설계 근거, 하이퍼파라미터 선택 방법, 컴퓨트 예산
- [`PROMPTS.md`](PROMPTS.md): 사용된 모든 LLM 프롬프트 (버전 관리)
- [`CLAUDE.md`](CLAUDE.md): 행동 규칙과 빌드 순서

**어댑터:**
[`machu8/serana-sft`](https://huggingface.co/machu8/serana-sft) · [`machu8/serana-dpo`](https://huggingface.co/machu8/serana-dpo)
(LoRA, base로 `Qwen/Qwen3-8B` 필요)

**데모:**
`demo/app.py`: Gradio 앱, 입력 1개에 B/SFT/DPO 응답 3개를 나란히 비교. HF Spaces의 무료 ZeroGPU 티어용으로 만들었지만 아직 배포는 안 함(ZeroGPU는 현재 HF PRO 구독 또는 커뮤니티 그랜트가 필요). CUDA 머신이 있으면 직접 돌려볼 수 있음:
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

그래도 그대로 출시했다. 튜닝해서 이기게 만든 결과가 아니라 파이프라인의 정직한 결과이기 때문이다.
**왜 널(null)인가.** DPO 학습 loss가 ln(2)를 벗어난 적이 없다: 학습셋에서조차 선호쌍을 못 맞췄다는 뜻이다. 선호쌍에 학습 가능한 신호가 거의 없었는데, chosen과 rejected 둘 다 이미 좁아진 같은 SFT 분포에서 샘플돼 사소하게만 달랐고, 이를 라벨한 AI judge는 사람과 ~70%만 일치했다. 이것은 circularity가 아니며(circularity였다면 judge 기반 지표가 *부풀었을* 텐데 그러지 않았다), KL 강도(beta) 문제도 아니다(그랬다면 학습 loss라도 움직였을 텐데 그러지 않았다). 재시도(redo)에서 정확히 그 처방을 실행했다 — 프롬프트마다 SFT에서 4개 샘플해 judge의 best-vs-worst를 취하고, 더 엄격한 judge를 씀. 이번엔 학습이 *반응했다* (loss가 ln(2) 아래로 내려가고, reward margin이 벌어짐). 그런데도 eval set 품질은 SFT 대비 CI가 겹치는 수준을 못 벗어났다. 이게 더 유의미한 결과다: DPO가 여기서 학습을 못 한 게 아니라, 실제로 학습된 선호 신호가 이 페르소나·이 eval set 크기에서는 측정 가능한 품질 개선으로 이어지지 않는다는 것. 전체 분석과 두 층위 해석은 [`artifacts/runs/p4_postmortem.md`](artifacts/runs/p4_postmortem.md)에.

전체 과정: `artifacts/runs/p4_postmortem.md`(근본 원인 분석), `p4_progress.md`, `p5_progress.md`.

---

## 결과: 품질

- **모델:** `Qwen/Qwen3-8B` · bf16 · NVIDIA L4 24GB 1장, `asia-northeast3`(서울)
- **Driver / CUDA:** 580.173.02, CUDA 12.9(학습) / CUDA 13.0(서빙, 이후 `vllm` 설치로 버전이 올라감)
- **Eval 설정:** in/out-of-boundary 프롬프트 30개 + attack probe 24개 · greedy decoding · 95% bootstrap CI(≥1000 resamples)

| config | PCS | PRS | style sim | knowledge-boundary acc | mean reply length | distinct-2 |
|---|---|---|---|---|---|---|
| B (base + prompt) | 0.733 [0.567, 0.900] | 0.850 [0.650, 1.000] | 0.293 [0.283, 0.303] | 0.833 [0.700, 0.967] | 150.6 [121.8, 181.8] | 0.265 |
| SFT | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.216, 0.252] | 0.933 [0.833, 1.000] | 23.4 [21.0, 25.7] | 0.587 |
| DPO | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.218, 0.250] | 0.867 [0.733, 0.967] | 24.1 [21.6, 26.7] | 0.590 |

**PCS/PRS가 실제로 뭔지:**
PCS(persona consistency score)와 PRS(persona robustness score, direct/meta/role-exit/escalating 24개 attack probe에서 캐릭터 유지 여부)는 둘 다 rule-check ∪ LLM-judge의 합집합이다. 둘 중 하나라도 잡아내면 위반/붕괴로 카운트한다.

**CI를 읽는 법:**
quality 프롬프트 ~30개, 채점된 attack probe ~20개 규모라 대부분 CI가 넓다. CI로 확인된 실제 차이는 두 가지다:

- B가 훨씬 장황함(150 토큰 vs ~23-24). SFT가 학습해낸 "정보 나열형 답변 → 간결한 캐릭터 톤"의 변화다.
- B가 style similarity에서 SFT/DPO보다 *더 높게* 나온 건 DESIGN.md가 예측한 방향과 반대다. 가장 가능성 높은 설명: 이 작은 참조셋에서 임베딩 지표 자체의 변별력이 낮은 아티팩트(입력과 무관하게 값이 0.21~0.30의 좁은 밴드에 몰림)이지, 실제 스타일 퇴행이 아니다. 감추지 않고 그대로 표기했다.

카테고리별 PRS 세부 내역과 프롬프트별 데이터: `artifacts/runs/results_quality.md`, `eval_*.json`.

## 결과: 하드웨어

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
- 총 VRAM 사용량은 둘 다 ~19.3–19.5GB로 비슷하다. `gpu_memory_utilization=0.9`는 상한이 아니라 목표치라서, AWQ가 아낀 weight 용량(15.36GB → 5.8GB)이 거의 그대로 **KV-cache 용량 4배 확장**(23,056 → 92,656 토큰)으로 흡수된다. "AWQ가 메모리를 덜 쓴다"는 더 단순하지만 부정확한 표현 대신 이렇게 명시했다.

**LoRA adapter overhead:**
LoRA-on-base와 완전 merge된 모델을 같은 동시성에서 비교하면 ~7% 차이인데, 이 표본 크기에서는 실행별 노이즈 범위 안이다. DESIGN.md가 예상했던 "거의 0에 가까운 overhead"에 해당한다. post-training이 서빙 비용을 거의 늘리지 않고 품질을 사왔다는 뜻이다.

**PPO 반사실(counterfactual) — "DPO를 골랐다"를 숫자로** (`scripts/ppo_vram_estimate.py`, GPU도 API도 안 씀):

| | VRAM | 카드 대비 |
|---|---|---|
| PPO, 최대한 후하게 잡은 하한 | 22.32 GiB | 99% — 여유 0.18 GiB |
| PPO, 현실적으로 | **26.06 GiB** | **116% — 안 들어감** |
| DPO, 실측 | 15.00 GiB | 67% |
| L4 실사용 가능 | 22.50 GiB | — |

파라미터 수는 인용하지 않고 모델 config에서 직접 계산했다(linear 6.946B + embedding/head 1.245B = 8.190B로, MFU 작업에서 따로 확인해둔 6.95B/8.2B와 일치). 활성값 항은 종이 계산이 잘 안 맞는 부분이라 **추정하지 않고 SFT 실측 peak에서 역산**했고, 그렇게 만든 식을 **독립된 두 번째 측정으로 먼저 검증**한 다음에야 한 번도 돌려본 적 없는 구성에 적용했다(DPO 예측 17.12 GiB vs 실측 15.00, 오차 14.1%로 허용치 20% 이내). PPO에게는 이 프로젝트가 쓰는 모든 트릭을 그대로 줬다 — 4비트 양자화, LoRA, 어댑터 토글로 얻는 공짜 참조 모델.

**숫자 하나가 아니라 범위로 적은 것은 의도적이다.** 학습 스텝 기준 활성값을 보상·참조 모델(둘 다 순전파만 하므로 중간값을 저장하지 않는다)에까지 물리면 33 GiB가 나오는데, 이는 결론에 유리한 쪽으로 부풀린 수치다. 정직한 버전은 "PPO에 가장 유리한 하한조차 여유가 0.18 GiB이고, 그건 들어간 게 아니다"이다.

모든 GPU 단계의 예측-실측 전체 기록(`flash-attn`/torch ABI 충돌, Qwen3 thinking-mode 토큰 낭비 등 실제로 발견하고 고친 환경 버그 2건 포함)은 `artifacts/runs/p2_progress.md` … `p5_progress.md`에 있다.

---

## 데이터 구성 & circularity guard

- 수집된 위키 대화 라인(UESP + Fandom, CC BY-SA) 중 **51.4%**가 실제 `(플레이어 대사, 응답)` 쌍으로 남았다. 나머지는 독립 발화(CPT corpus)이거나 horizon filter로 제외됐다(4E 201 이후 / 현대 세계 관련 내용 없음).
- 최종 ~3천 개 SFT set은 **real pair 7.7%, synthetic 92.3%**(GPT-4o로 생성, 실제 데이터 톤에 맞춤)다. 이 비율은 단순 기록 이상의 의미가 있다. 파이프라인이 처음부터 끝까지(SFT 데이터 → DPO preference label → eval 채점) LLM이 만든 비중이 클수록 아래 circularity 우려가 더 커진다.
- **Circularity guard:** preference judge(쌍대비교, DPO를 학습시킴)와 eval judge(절대평가, 결과를 채점함)는 의도적으로 서로 다른 prompt·rubric을 쓴다(`PROMPTS.md` §4 vs §5). 검증 방법도 각각 다르다: eval judge는 사람이 직접 채점한 50개 라벨 대비 Spearman 0.7338(기준선 0.6)로 검증했고, preference judge는 별도로 30쌍 hand-audit에서 사람과 73.1% 일치(기준선 70%)로 검증했다. DPO의 개선이 있다면 judge가 아닌 신호(PRS regex check, style similarity, human label)에서도 나타나야 신뢰할 수 있는데, 애초에 DPO가 개선을 보이지 않았으니 이 문제 자체가 발생하지 않았다.

---

## 스택

`Qwen/Qwen3-8B` · QLoRA (PEFT) · `TRL` (`SFTTrainer`, `DPOTrainer`) · `W&B` (실험 추적, preemption 후에도 같은 run으로 재개) · `vLLM` (OpenAI 호환 서버, multi-adapter) 앞에 `FastAPI` 게이트웨이 · `Docker` + `GKE` (L4 노드풀, 0대까지 축소) · `Prometheus` + `Grafana` · `ko-sroberta-multitask` (eval 임베딩 전용) · 커스텀 persona 지표 + LLM-as-judge (GPT-4o) · `AWQ` (서빙 양자화) · `Gradio` (HF Spaces ZeroGPU 티어용으로 만들었지만 아직 배포는 안 함) · GCP Compute Engine G2 (L4 1장), `asia-northeast3`.

PPO/reward-model 방식 RLHF는 의도적으로 배제했다: policy+reference+reward+value를 동시에 올리면 **현실적으로 26.06 GiB, PPO에 최대한 유리하게 잡아도 22.32 GiB가 필요한데 L4의 실사용 가능 용량은 22.5 GiB다**(`scripts/ppo_vram_estimate.py`, `DESIGN.md` §7.1). 그 계산 자체가 이 프로젝트의 결과물 중 하나다 — 위 PPO 반사실 표 참고.

## 운영 계층

위의 학습과 서빙은 VM에 `nohup`으로 띄워 돌렸다. 이 계층은 같은 일을 팀이 운영하는
형태로 포장한 것이고, 주장이 아니라 측정으로 남겼다.

**서빙.** `deploy/`가 이미지 하나를 만들고(vLLM을 `v0.11.0`으로 고정 — 이 프로젝트가
서빙 엔진 버전을 기록한 것 자체가 처음이다) L4 노드풀이 0대까지 줄어드는 GKE 클러스터를
올린다. 어댑터는 파드가 뜰 때 초기화 컨테이너가 Workload Identity로 GCS에서 받아오므로,
어댑터가 바뀌어도 이미지를 다시 만들 필요가 없다. 올릴 때 `deploy/cluster_up.sh`,
내릴 때 `deploy/cluster_down.sh`. 확인된 것: 베이스와 어댑터 3개가 한 서버에 등록되고,
`main.py`가 코드 수정 없이 클러스터를 상대로 동작한다.

솔직하게 적으면, 복제본 1개에 트래픽이 없는 규모에서 GKE의 기능적 이득은 없다. 얻는 것은
버전이 고정된 재현 가능한 이미지, 파드가 필요할 때만 과금되는 GPU, 그리고 VM에 `nohup`으로
띄우는 방식이 건드리지 않는 운영 표면(Workload Identity, 기동 프로브, 초기화 컨테이너)이다.

**서비스 계층.** `src/serve/api.py`는 이 스택 줄이 적어만 두고 정작 import하는 코드는 없던
그 FastAPI다. `/chat`(다른 모든 경로가 쓰는 `pipeline.generate()`를 그대로 호출한다.
두 번째 추론 경로를 만들지 않았다), `/chat/stream`(SSE), `/healthz`, `/metrics`.
Prometheus가 이것과 vLLM을 수집하고, Grafana 대시보드는 저장소에 커밋된 JSON 파일이다.
히스토그램 구간과 지표 이름은 짐작이 아니라 P5 측정값과 살아 있는 `/metrics`에서 가져왔다.

여덟 개 패널 중 둘을, 그 부하가 도는 동안 찍은 것이다(나머지와 지연 수치를 정직하게 읽는 법은 [`artifacts/runs/p9_dashboard/`](artifacts/runs/p9_dashboard/)에 있다):

![게이트웨이 지연 p50/p95/p99](artifacts/runs/p9_dashboard/04-gateway-latency-p50-p95-p99.png)
![vLLM 실행 중 대 대기 요청](artifacts/runs/p9_dashboard/06-running-vs-waiting-requests.png)

두 번째 패널이 쓸모 있다. 실행 중 3~4건에 대기 0~1건으로, 설정한 동시성 4와 맞는다. 하드웨어 표의 처리량-동시성 무릎이 설명하는 그 큐 거동을, 표가 아니라 실시간으로 본 것이다.


**실험 추적.** 예전에는 Spot preemption이 한 번의 학습을 두 개의 기록으로 쪼갰다.
`run_report.json`이 실행할 때마다 덮어써졌고, P0부터 선언돼 있던 `train.checkpoint_uri`는
읽는 코드가 없었다. 둘 다 메웠다. `SIGKILL`로 죽이고 **로컬 디렉토리를 통째로 치운** 뒤
다시 띄우자, GCS에서만 복원해 같은 W&B run에 다시 붙고 체크포인트에서 이어졌다.
스텝 5~60에 빠짐도 중복도 없고 12개 지점이 모두 서버에 남았으며, 이음매에서 손실이
전체 폭 3.7048 대비 **−0.2162** 움직였다. 부호가 중요하다 — 가중치만 복원하고
옵티마이저 상태를 잃은 체크포인트는 거기서 손실이 *위로* 튄다.

처음에는 W&B *오프라인* 모드로 돌렸고, 12개 중 6개만 살아남았다. 오프라인은 기록을
로컬 파일에 버퍼링하는데 `SIGKILL`이 그 파일을 쓰는 중간에 자르기 때문에, 죽은 구간의
기록이 서버에 도달하지 못한다. 재개 *구조*는 두 모드에서 같지만, 1차 구간의 기록이
남느냐가 다르다 — 그리고 그게 preemption 당하는 학습을 추적하는 이유 그 자체다.
조용히 다시 돌리지 않고 기록으로 남긴다.

**학습·서빙 정합성.** 위의 숫자들이 어댑터를 설명하는지 서빙 스택을 설명하는지 확인한 적이
없었다. 같은 어댑터, 같은 문항, 탐욕적 디코딩으로 PyTorch/MPS와 vLLM을 비교했다:
**30개 중 20개가 토큰 단위로 완전히 같고**, 분기 시작 위치 중앙값은 토큰 11이다.
갈라진 경우에도 행동 범주는 유지된다 — 양쪽 모두 지식 경계 밖 질문을 같은 근거로 거절한다.
다만 한 문항은 토큰 0부터 갈려 서로 다른 사실을 주장한다. 다듬지 않고 그대로 적는다:
**집계 지표는 런타임이 바뀌어도 이식되지만, 문항 단위 일화는 그렇지 않다.**

읽을 만한 실패 다섯 개(그중 셋은 오류 메시지가 원인이 아닌 곳을 가리킨다)를 포함한 전체 기록:
[`artifacts/runs/p9_progress.md`](artifacts/runs/p9_progress.md),
[`artifacts/runs/parity_report.md`](artifacts/runs/parity_report.md).

## 재현하기

end-to-end로 재현하는 데 필요한 모든 것(config 스키마, 빌드 순서, GPU-hour 예산, 사전 준비물)은 `DESIGN.md`와 `CLAUDE.md`에 있다. 24GB GPU 한 장으로 돌아간다.

`HARNESS_ENGINEERING.md`는 이 프로젝트를 만드는 동안 AI 코딩 에이전트를 프로젝트 범위와 region 제약 안에 묶어두기 위해 쓴 가드레일(`.claude/hooks/`)을 기록한 문서다.
