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
