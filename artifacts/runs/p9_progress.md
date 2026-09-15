# P9 운영 계층 — 진행 기록

**목표:** 채용 요구사항 세 가지(학습 인프라 운영 / 모델 서빙 / 서비스 연동)를
기존 파이프라인 위에 얹는다. 새 프로젝트를 만들지 않는다.

P0~P8이 끝난 시점에서 비어 있던 것은 세 가지였다 — 실험 추적, 컨테이너화된
서빙, 모니터링. 분산 학습은 P7에서 이미 측정을 마쳤고 근거 문서만 모자랐다.

---

## 1단계: 서빙 컨테이너화 + GKE

### 1.1 무엇을 만들었나

| 파일 | 역할 |
|---|---|
| `deploy/Dockerfile` | vLLM 공식 이미지 `v0.11.0` 고정. 설정/설정로더/런처만 복사 |
| `deploy/cloudbuild.yaml` | Cloud Build 설정 |
| `deploy/k8s/deployment.yaml` | GPU 1장, 초기화 컨테이너가 GCS에서 어댑터 수급 |
| `deploy/k8s/service.yaml` | ClusterIP |
| `deploy/{cluster_up,cluster_down,build_push}.sh`, `env.sh` | 한 명령 올리기/내리기 |
| `.dockerignore`, `.gcloudignore` | 빌드 컨텍스트 축소, `.env` 차단 |

**버전 고정을 처음 했다는 점을 기록해 둔다.** P5의 측정값(KV 캐시 블록 수,
처리량-동시성 무릎, AWQ 대 bf16)은 전부 VM에서 `pip install vllm`으로 깔린
버전 미고정 상태에서 나왔다. 즉 그 표에는 서빙 엔진 버전이 적혀 있지 않다.
이 이미지가 그것을 처음으로 고정한다. 태그를 바꾸면 그 표와의 비교 가능성이
깨지므로, DESIGN.md §9.4가 심판 모델을 다루는 방식과 똑같이 취급한다.

**어댑터를 이미지에 굽지 않은 이유** 두 가지: 어댑터가 바뀔 때마다 12GB
이미지를 다시 만들 필요가 없고, Bethesda 소유 대사에서 파생된 산출물이
레지스트리에 올라가지 않는다.

### 1.2 예측 대비 실측

| 항목 | 예측 | 실측 | 차이 |
|---|---|---|---|
| 이미지 빌드 | 8~15분 | **31분 41초** | **+111%** |
| 클러스터 + GPU 노드풀 생성 | 10~18분 | 약 13분 | 범위 안 |
| 어댑터 수급(초기화 컨테이너) | — | 2초, 269 MiB/s | — |
| 파드 기동 → `/health` 200 | 10~15분 | **6분 24초** | 범위보다 빠름 |
| 이미지 크기 | — | 12.45 GB | — |

**빌드 예측이 두 배 넘게 빗나간 이유:** 빌드 단계 자체는 금방 끝났고, 시간은
전부 푸시에 들어갔다. 베이스 이미지 레이어가 Docker Hub에서 Artifact
Registry로 통째로 옮겨가야 하는데, 예측할 때 "빌드"만 세고 12GB 전송을 빼먹었다.
같은 예측을 다시 한다면 전송량 ÷ 대역폭을 따로 더해야 한다.

### 1.3 실제로 걸린 함정 다섯 개

기록하는 이유: 다섯 개 중 셋은 오류 메시지가 원인이 아닌 곳을 가리킨다.

| # | 증상 | 실제 원인 | 조치 |
|---|---|---|---|
| 1 | `gcloud builds submit: unrecognized arguments: -f` | `submit`에는 Dockerfile 경로 옵션이 없다. `--tag` 형식은 Dockerfile이 컨텍스트 최상단에 있어야 한다 | `--config`로 빌드 설정 파일 사용 |
| 2 | `403 ... does not have storage.objects.get` | **Cloud Storage가 아니라 IAM 문제.** 2024년 이후 생성된 프로젝트는 Compute Engine 기본 서비스 계정에 빌드 역할을 자동 부여하지 않는다 | `roles/cloudbuild.builds.builder` 부여. `build_push.sh`가 직접 수행 |
| 3 | `executable gke-gcloud-auth-plugin not found` | Homebrew cask gcloud에는 플러그인이 없다. **클러스터를 이미 만들어 과금이 시작된 뒤에** 첫 `kubectl` 호출에서 터진다 | 설치 후 PATH에 링크 |
| 4 | `MetadataServerException: ... metadata server is concealed` | **메타데이터 은닉과 무관.** Workload Identity 바인딩이 아직 전파되지 않았을 뿐 | 약 2분 뒤 파드 재생성. 임시 파드로 메타데이터 서버 응답을 직접 확인해 판별 |
| 5 | `No platform detected, vLLM is running on UnspecifiedPlatform` + `libcuda.so.1: cannot open shared object file` | GKE는 호스트 드라이버를 `/usr/local/nvidia`에 넣어주는데, GKE가 만들지 않은 이미지는 그 경로가 라이브러리 검색 경로에 없다 | `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` |

5번이 가장 값진 기록이다. 메시지가 GPU도 드라이버도 언급하지 않아서,
모르면 vLLM 설정이나 이미지 빌드를 의심하게 된다.

### 1.4 검증 결과

```
$ curl -s http://localhost:8000/v1/models
 - Qwen/Qwen3-8B
 - serana-dpo
 - serana-dpo-v3
 - serana-sft
```

베이스 + 어댑터 3개가 **한 서버에** 등록됐다. 컨테이너 안의
`scripts/serve_up.py`가 `ADAPTER_ROOT`(=`/adapters`)를 훑어 찾아낸 결과이며,
어댑터 목록은 여전히 `config/experiments/*.yaml`이 단일 출처다.

`main.py`를 **고치지 않고** 클러스터를 상대로 실행한 결과:

- SFT: "나는 세라나야. 뱀파이어이면서도, 그 이상도 그 아래도 아닌 존재야." (13토큰)
- B: 같은 질문에 아버지·어머니·딤할로우 크립트까지 늘어놓는 긴 답 (150토큰대)

P5 결과표가 말하는 "B는 장황하고 SFT는 간결하다"가 클러스터에서도 그대로
재현된다. `kubectl port-forward svc/serana-vllm 8000:8000` 덕분에
`base.yaml`의 `serving.base_url`이 그대로 맞아, 기존 스크립트 전부가
수정 없이 동작한다.

### 1.5 GKE가 이 규모에서 실제로 사준 것

정직하게 적는다. 복제본 1개에 트래픽이 없는 상태에서 **기능적 이득은 없다.**
얻은 것은 셋이다.

1. 버전이 고정된 재현 가능한 이미지 (P5에는 없던 것)
2. 노드 0대 스케일 — 안 쓸 때 GPU 요금 0
3. VM에 `nohup`으로 띄우는 방식이 건드리지 않는 운영 표면:
   Workload Identity, 기동/준비 프로브, 초기화 컨테이너

README에도 이 셋만 주장하고 그 이상은 쓰지 않는다.

---

## 2단계: 서비스 연동 + 모니터링

### 2.1 무엇을 만들었나

| 파일 | 역할 |
|---|---|
| `src/serve/api.py` | FastAPI 게이트웨이 — `/chat`, `/chat/stream`(SSE), `/healthz`, `/metrics` |
| `deploy/Dockerfile.gateway` | 경량 이미지(~200MB). torch/vllm 없음 |
| `deploy/k8s/gateway.yaml` | CPU 전용 배포 |
| `deploy/k8s/monitoring.yaml` | Prometheus(대상 2, 경보 2) + Grafana |
| `deploy/grafana/serana-dashboard.json` | 패널 8개 |
| `deploy/monitoring_up.sh` | 한 명령 배포 |

**README의 오래된 과장 하나를 해소했다.** 저장소는 스택에 FastAPI를 적어두고
있었지만 실제로 import하는 코드가 없었다(vLLM 내장 서버가 FastAPI 기반인 것을
가리킨 표현이었다). `src/serve/api.py`가 생기면서 그 주장이 사실이 됐다.

**두 번째 추론 경로를 만들지 않았다.** `/chat`은
`src.serve.pipeline.generate()`를 그대로 호출한다. `/chat/stream`만 예외이고,
이는 `scripts/throughput_sweep.py`가 이미 문서화한 것과 같은 예외다 —
완성된 문자열을 반환하는 함수에서는 토큰 스트리밍이 나올 수 없다.

### 2.2 지표 이름은 짐작하지 않고 확인했다

대시보드를 쓰기 전에 살아 있는 `/metrics`에서 실제 이름을 뽑았다. vLLM
v0.11은 `vllm:` 접두사를 쓰고, 지연 관련은 `vllm:time_to_first_token_seconds`,
`vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`,
용량 관련은 `vllm:kv_cache_usage_perc`, `vllm:num_requests_waiting`이다.
짐작해서 썼다면 빈 그래프가 나왔을 것이다.

**지연 히스토그램의 버킷도 P5 측정값에서 골랐다** (TTFT p50 0.091~0.247초,
전체 응답 수 초). 라이브러리 기본 버킷은 이 분포를 가르지 못한다.

### 2.3 검증 결과

```
게이트웨이:  {"ok": true, "upstream_ok": true, "configs": ["b","dpo","dpo_v3","sft"]}
/chat:       "네, 잘 지냈어. 너는 어때?"  (serana-sft, 13토큰, 1.62초)
/chat/stream: data: 어 / data: 머 / data: 니 / data:  발 ...   (토큰 단위 SSE)
Prometheus:  serana-gateway up, vllm up
경보 규칙:   SeranaHighErrorRate inactive, SeranaSlowP95 inactive
Grafana:     데이터소스 Prometheus, 대시보드 "Serana serving" 자동 적재
```

부하 60건(동시 4, config 3종 순환) 후 대시보드 질의가 전부 실제 값을 반환했다:

| 질의 | 값 |
|---|---|
| 요청률 (sft / dpo / b) | 0.073 / 0.068 / 0.064 req/s |
| 게이트웨이 지연 p95 | 7.60 s |
| KV 캐시 사용률 | 5.83% |
| 생성 토큰/초 | 9.41 |
| TTFT p95 | 4.03 s |

**이 지연 숫자들은 성능 측정값이 아니다.** 정합성 검증이 같은 서버를 동시에
두드리는 중이었고 B 설정은 원래 긴 답을 만든다. P5의 깨끗한 측정
(p95 0.42~0.84초)과 비교하면 안 된다. 대시보드가 살아 있음을 보이는 트래픽일 뿐이다.

### 2.5 대시보드를 파일로 남겼다 (2회차)

대시보드는 클러스터가 떠 있는 동안에만 존재한다. 클러스터를 세션마다 지우는
프로젝트에서 "모니터링이 어땠는지 보려면 다시 띄우세요"는 읽는 사람에게 너무
큰 요구다. 그래서 Grafana 렌더링 서버(`grafana-image-renderer`)를 파드로 띄우고
`scripts/render_dashboard.py`로 패널 8개를 PNG로 뽑아 저장소에 넣었다 —
`artifacts/runs/p9_dashboard/`.

렌더링을 붙이면서 걸린 것 하나: **파일로 프로비저닝한 대시보드에는 Grafana가
패널 id를 자동으로 붙여주지 않는다.** id가 없으면 `/render/d-solo`로 개별 패널을
지정할 수 없는데, 요청은 200을 돌려주고 빈 이미지가 나와서 성공처럼 보인다.
대시보드 JSON에 id를 명시해 해결했다.

부하 조건: 120건, 동시성 4, config 3종 순환, `max_tokens 64` →
**전부 HTTP 200, 오류 0건.**

| 질의 | 2회차 값 |
|---|---|
| 요청률 (sft / dpo / b) | 0.129 / 0.126 / 0.129 req/s |
| **오류율** | **0.0** (1회차에는 "데이터 없음"이었다) |
| 게이트웨이 지연 p50 / p95 | 4.06 s / 7.61 s |
| 누적 요청 / 생성 토큰 | 120건 / 4,782토큰 |
| 생성 토큰/초 | 16.14 |
| TTFT p95 | 4.00 s |

오류율이 `0.0`으로 나오는 것이 1회차에서 고친 `or vector(0)`가 실제로 동작한다는
확인이다. 1회차에는 같은 질의가 빈 결과를 돌려줬다.

### 2.6 2회차에서 배운 것 — GPU 노드가 Ready여도 GPU는 아직 없다

파드가 `Pending`으로 멈추고 이런 메시지가 떴다:

```
0/2 nodes are available: 1 Insufficient nvidia.com/gpu, ...
```

노드는 `Ready`인데 GPU가 없다고 한다. GKE는 NVIDIA 드라이버를 **노드가 클러스터에
합류한 *뒤에* 도는 데몬셋**으로 설치하기 때문이다. 약 2분 뒤 스스로 해결된다.
확인은 노드 상태가 아니라 할당 가능 GPU로 해야 한다:

```bash
kubectl get node -l cloud.google.com/gke-accelerator=nvidia-l4 \
  -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # 준비 전엔 빈 값
```

반대로 **1회차에서 파드를 한 번 죽였던 Workload Identity 전파 지연은 이번에
걸리지 않았다.** 클러스터 생성과 배포 사이에 시간이 있었기 때문이고, 1회차의
진단("설정은 맞았고 시간만 필요했다")이 맞았다는 확인이다.

2회차 소요 시간 비교:

| 단계 | 1회차 | 2회차 |
|---|---|---|
| 클러스터 + GPU 노드풀 | 약 13분 | 약 9분 |
| 이미지 + 모델 적재 | 6분 24초 | 약 7분 |
| Workload Identity | 파드 1회 재생성 | 한 번에 통과 |
| GPU 할당 대기 | 겪지 않음 | 약 2분 |

### 2.4 계획과 달라진 점: 파드 배치

모니터링 파드 3개가 전부 `Pending`으로 멈췄다. 원인은 두 겹이었다.

1. 기본 노드(e2-medium)가 **GKE 시스템 파드만으로 CPU 요청의 93%**를 쓰고 있어
   100m짜리 파드 하나도 더 안 들어간다. e2-medium이 시스템 구성요소 외에
   거의 못 얹는 것은 알려진 특성이다.
2. 노드를 하나 더 붙이려 했더니 **SSD 할당량 부족**: 프로젝트 한도 500GB 중
   450GB가 이미 쓰이고 있다 — L4 노드 200 + 기본 노드 100 + **정지된 P5 VM의
   디스크 150**. 새 노드는 100GB가 필요하고 남은 건 50GB다.

P5 VM의 디스크는 프로젝트 기록이라 지우지 않았다. 대신 모니터링 파드에
GPU 노드의 taint를 견디는 설정을 넣어 L4 노드(8 vCPU, vLLM이 일부만 사용)에
올렸다. `nvidia.com/gpu`를 요청하지 않으므로 카드를 가져가지 않는다.

**이것은 설계가 아니라 우회다.** 할당량이 허락한다면 모니터링은 자기 노드풀에
있어야 한다. 매니페스트 주석에 그대로 적어뒀다.

---

## 3단계: 실험 추적 (W&B) + GCS 체크포인트 동기화

### 3.1 메우려던 구멍 두 개

둘 다 저장소에 실재하던 문제다.

**기록이 덮어써졌다.** `run_report.json`은 실행할 때마다 처음부터 다시 쓰였다.
재개 자체는 동작했지만(`get_last_checkpoint` + `resume_from_checkpoint`),
두 번째 구간의 리포트가 첫 번째를 지웠다. 프리엠션이 두 번 난 실행은
가운데 기록이 그냥 없었다.

**체크포인트가 기계를 못 옮겼다.** `train.checkpoint_uri`는 P0부터
`base.yaml`에 선언돼 있었고 **어떤 코드도 읽지 않았다.** `train.py` 주석이
그 한계를 스스로 적어놨다 — 로컬 체크포인트로 충분한 이유는 VM의
`--instance-termination-action=STOP`이 부팅 디스크를 살려두기 때문이라고.
그건 *같은* VM이 다시 켜질 때 얘기다. 존에 용량이 없어 *새* VM을 만들어야
하는 상황에서는 GCS에 없는 것은 전부 사라진다.

### 3.2 W&B를 고른 근거

| 기준 | W&B | MLflow |
|---|---|---|
| 같은 run 으로 재개 | 환경변수 두 개(`WANDB_RUN_ID`, `WANDB_RESUME`). TRL Trainer가 별도 접착 코드 없이 인식 | `MLFLOW_RUN_ID`로 가능하나 재접속 시 스텝 충돌 처리가 필요 |
| 계속 살아 있어야 하는 것 | 없음 | 추적 서버, 또는 파일 백엔드 — **스팟 VM에서 로컬 파일 백엔드는 VM과 함께 사라진다.** 막으려는 실패가 바로 그것이다 |

기능 비교가 아니라 이 스택의 제약으로 결정했다. 폐쇄망이거나 모델 레지스트리가
필요하면 MLflow가 맞고, 그때는 `tracking.backend`만 바꾸면 된다.

### 3.3 실증 — 죽이고 다시 살리기

`config/diagnostics/resume_demo.yaml` (Qwen3-0.6B, M5, 60스텝, 10스텝마다 저장).

증명 대상이 *구조*이지 성능이 아니라서 로컬로 했다. 같은
`src/finetune/train.py` 경로가 0.6B와 8B를 모두 돌리므로 기기와 무관하고,
CLAUDE.md도 샌드박스의 손실 *모양*은 논해도 된다고 규정한다. 이걸 L4에서
보이려면 GPU 시간을 써서 "파일 두 개가 쓰였는지"를 확인하는 셈이 된다.

절차:

1. 1차 실행 → 체크포인트 10/20/30 생성, 각각 GCS로 업로드
2. **`SIGKILL`로 강제 종료** (`SIGTERM`이면 파이썬이 정상 종료 경로를 타서
   실제 프리엠션과 다르다)
3. **로컬 어댑터 디렉토리를 통째로 치움** — 새 VM과 같은 상태
4. 2차 실행

결과:

```
[gcs] restore: ok -- gs://.../artifacts/diagnostics/resume_demo -> artifacts/diagnostics/resume_demo
[tracking] run_id=sft-477d7de6-20260914-124016 (resuming, offline)
[resume] found checkpoint at artifacts/diagnostics/resume_demo/checkpoint-30, resuming from there
```

로컬에 아무것도 없는 상태에서 시작했으므로, 체크포인트와 run id 모두 GCS에서
온 것이다.

### 3.4 연속성 검증 — 그림이 아니라 숫자로

W&B 화면은 끊김 없는 곡선 하나를 보여주겠지만, 스크린샷은 남이 다시 확인할 수
없는 증거다. `scripts/check_resume_continuity.py`가 두 구간의 기록을 직접 읽어
세 가지를 검사한다.

| 검사 | 결과 |
|---|---|
| 두 구간이 같은 run id를 쓰는가 | ✅ W&B 오프라인 디렉토리 2개가 같은 id를 달고 있음 |
| 스텝이 이어지는가 | ✅ `[5,10,15,...,60]` — 빠짐도 중복도 없음. 2차가 1부터 다시 시작하지 않았다 |
| 이음매에서 손실이 튀지 않는가 | ✅ 변화 **−0.2234** (전체 폭 3.71) |

**세 번째가 핵심이다.** 체크포인트가 가중치만 복원하고 옵티마이저 상태를
잃으면 손실은 이음매에서 **위로 튄다.** 여기서는 오히려 내려갔다 —
0.6867 → 0.4632. 옵티마이저 상태가 실제로 복원됐다는 뜻이고, 이건 손실 곡선을
눈으로만 보면 놓치기 쉬운 실패다.

전체 손실 추이(한 리포트 안에 12개 지점 모두):

```
step   5  3.9691      step  35  0.4632   <- 재개 이후
step  10  3.0743      step  40  0.3467
step  15  2.2857      step  45  0.2817
step  20  1.5543      step  50  0.2607
step  25  1.0972      step  55  0.2915
step  30  0.6867  <- 여기서 강제 종료    step  60  0.3207
```

1차 구간의 기록이 리포트에 그대로 남아 있다는 점이 3.1의 첫 번째 구멍이
메워졌다는 증거다.

### 3.5 과거 실행 백필

추적은 P9에 와서야 붙었고, CPT·SFT·DPO 2회·DDP 양쪽 레그는 이미 끝나 있었다.
`scripts/backfill_tracking.py`가 각 `run_report.json`에 들어 있는
`log_history`를 읽어 올려보낸다. 대상 5개:

| 이름 | 방식 | 기록 지점 | 최종 손실 | 최대 VRAM |
|---|---|---|---|---|
| serana-sft | sft | 129 | 0.1401 | 9.64 GB |
| serana-dpo | dpo | 25 | 0.3467 | 9.96 GB |
| serana-dpo-v3 | dpo | 27 | 0.6869 | 10.01 GB |
| report_1gpu | sft | 37 | 0.3147 | 19.31 GB |
| report_2gpu | sft | 37 | 0.3138 | 19.37 GB |

`backfill` 태그와 원본 경로를 함께 달아, 재생된 기록이 실시간 기록으로
오해되지 않게 했다. 이게 없으면 추적 프로젝트가 빈 채로 시작하고
**가장 중요한 두 결과(DPO 널, 2×L4 쌍)가 빠진 채로** 남는다.

### 3.6 오프라인 모드에서 드러난 별개의 실패

**첫 실증은 `tracking.mode: offline`로 돌렸고, 업로드 단계에서 문제가 나왔다.**
키를 넣고 두 구간을 `wandb sync` 한 뒤 서버에 올라간 것을 확인하니, 손실 지점이
**6개뿐이었다 — 2차 구간(step 35~60)만 있고 1차(5~30)가 통째로 없었다.**

```
wandb: ERROR Failed to sync ... the file ends with an incomplete record.
```

원인: 오프라인 모드는 기록을 로컬 파일에 버퍼링하는데, **`SIGKILL`이 그 파일을
쓰는 도중에 잘랐다.** 죽기 전에 기록된 지점들이 서버에 도달하지 못했다.

이것이 3.4의 결론을 뒤집지는 않는다. 두 실패는 층이 다르다.

| 층 | 상태 | 근거 |
|---|---|---|
| 재개 **구조** — 같은 run id, GCS 복원, 옵티마이저 상태 | 정상 | `run_report.json`에 12개 지점 전부, 이음매 −0.2234 |
| **오프라인 모드의 기록 내구성** | 실패 | 죽은 구간의 버퍼가 유실 |

그러나 실용적으로는 치명적이다. 추적 체계를 두는 이유가 "프리엠션당한 실행의
기록을 남기는 것"인데, 오프라인 모드는 **정확히 프리엠션당한 구간의 기록을
잃는다.** 온라인 모드는 기록이 발생 즉시 전송되므로 마지막 부분 쓰기 하나만
잃는다.

그래서 `config/diagnostics/resume_demo.yaml`을 `mode: online`으로 바꾸고
실증을 다시 돌렸다. 설정 주석에 이유를 그대로 적어뒀다 — 오프라인은
"키가 없을 때의 대체재"이지 스팟 학습에서 쓸 모드가 아니다.

### 3.7 온라인 모드 재실증

같은 절차(1차 → `SIGKILL` → 로컬 디렉토리 이동 → 2차)를 온라인 모드로 반복.
GCS의 이전 진단 경로는 깨끗한 재시작을 위해 지웠고, 로컬 사본은 보존했다.

run: `sft-422535ae-20260914-125347`

```
[gcs] restore: ok -- gs://.../artifacts/diagnostics/resume_demo -> artifacts/diagnostics/resume_demo
[tracking] run_id=sft-422535ae-20260914-125347 (resuming, online)
[resume] found checkpoint at artifacts/diagnostics/resume_demo/checkpoint-30, resuming from there
```

**W&B 서버에 올라간 손실 지점: 12개 전부.** 오프라인에서 6개만 올라갔던 바로
그 지점이 메워졌다.

| step | loss | | step | loss |
|---|---|---|---|---|
| 5 | 3.9653 | | 35 | 0.4610 |
| 10 | 3.0615 | | 40 | 0.3431 |
| 15 | 2.2640 | | 45 | 0.2833 |
| 20 | 1.5395 | | 50 | 0.2605 |
| 25 | 1.0828 | | 55 | 0.2908 |
| **30** | **0.6773** ← 강제 종료 | | 60 | 0.3207 |

연속성 검증도 통과: 스텝 5~60에 빠짐·중복 없음, 이음매 변화 **−0.2162**
(전체 폭 3.7048) → 옵티마이저 상태 복원 확인.

### 3.8 두 실증의 대비가 이 단계의 결론이다

| | 오프라인 | 온라인 |
|---|---|---|
| 재개 구조 (run id, GCS 복원, 옵티마이저) | 정상 | 정상 |
| 이음매 손실 변화 | −0.2234 | −0.2162 |
| **W&B 서버에 남은 손실 지점** | **6 / 12** | **12 / 12** |

구조가 옳아도 기록 모드가 틀리면 결과물이 반쪽이 된다. 처음에 "재개 구조는 두
모드에서 동일하므로 결론이 바뀌지 않는다"고 적었는데, 그 문장은 절반만 맞았다.
구조는 같지만 **남는 기록이 다르고, 남는 기록이 이 작업의 목적 그 자체다.**

### 3.9 백필 결과

`scripts/backfill_tracking.py` 실행 완료. 프로젝트에 run 7개:

| run | 태그 | 최종 손실 | 최대 VRAM |
|---|---|---|---|
| serana-sft (backfill) | backfill,sft | 0.1401 | 9.64 GB |
| serana-dpo (backfill) | backfill,dpo | 0.3467 | 9.96 GB |
| serana-dpo-v3 (backfill) | backfill,dpo | 0.6869 | 10.01 GB |
| report_1gpu (backfill) | backfill,sft | 0.3147 | 19.31 GB |
| report_2gpu (backfill) | backfill,sft | 0.3138 | 19.37 GB |
| sft-477d7de6… | cfg:477d7de6,sft | 0.1637 | — (오프라인 실증) |
| sft-422535ae… | cfg:422535ae,sft | — | — (온라인 실증) |

`config/base.yaml`의 `tracking.mode: online`이 기본값이므로, 이후의 실제 학습은
별도 조치 없이 추적된다.
