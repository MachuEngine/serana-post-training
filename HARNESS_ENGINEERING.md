# HARNESS_ENGINEERING.md

> 이 문서는 **개발 프로세스**에 관한 것이다 — 이 레포를 만드는 동안 Claude Code를 어떻게
> 제약했는지를 다루며, post-training 파이프라인 자체를 다루지 않는다. 파이프라인 설계는
> DESIGN.md, 에이전트 행동 규칙은 CLAUDE.md에 있다. 이 파일은 `.claude/` 스캐폴딩이
> 왜 지금 모습인지를 설명한다 — 나중에 파일 자체가 바뀌어도 근거는 남아있도록.
>
> P0(스캐폴드) 단계에서 작성했다. 아직 파이프라인 코드가 하나도 없는 시점이라, 사후
> 회고와 달리 이 문서는 **관측된 사고가 아니라 예측된 리스크**를 기록한다. 실제
> 사고가 발생하면 5절을 갱신할 것.

---

## 1. 두 층위의 강제

| | CLAUDE.md (프롬프트 레이어) | `.claude/hooks/` (시스템 레이어) |
|---|---|---|
| 작동 방식 | 모델이 읽고 따르길 기대하는 텍스트 | 매칭되는 모든 도구 호출마다 실행되는 스크립트 — 모델의 협조 여부와 무관 |
| 무시됐을 때 | 조용히 무시됨 — 컨텍스트가 길어지거나 그럴듯한 정당화가 끼어들면 규칙이 마모될 수 있음 | `exit 2` — 도구 호출 자체가 거부되고, 사유가 모델에게 반환됨 |
| 추가 비용 | 사실상 공짜(그냥 산문) | 정상적인 작업까지 오탐하지 않을 만큼 정밀해야 함 |

CLAUDE.md는 의도를 진술한다. 훅은 그 의도의 특정 부분을 협상 불가능하게 만든다. 어떤
규칙이 CLAUDE.md의 산문에서 훅으로 승격되는 건 (a) 이미 CLAUDE.md/DESIGN.md 어딘가에
하드 제약으로 명시돼 있고, (b) 도구 호출의 `file_path`나 `command` 문자열만으로 기계적으로
판별 가능할 때뿐이다. 판단이 필요한 규칙(예: CLAUDE.md의 scope-discipline 표 — "PPO/RAG/그리드
서치를 제안하지 말 것")은 산문으로 남는다 — "제안"이라는 행위 자체를 매칭할 패턴이 없기 때문이다.

## 2. 무엇을 강제하는가, 그리고 각 훅이 강제하는 정확한 문장

| 파일 | 강제하는 것 | 근거 |
|---|---|---|
| `.claude/hooks/block_secrets.py` | API 키로 보이는 문자열을 어떤 파일·명령에도 쓰지 못하게 함, 시크릿 파일을 `cat`/`curl` 등으로 직접 읽거나 전송하는 것 차단 | CLAUDE.md §Repo conventions: "Secrets via `.env` (never committed)" |
| `.claude/hooks/block_protected_paths.py` | `data/eval/`·`data/raw/`를 Write/Edit로 직접 건드릴 수 없게 함(반드시 `scripts/` 실행 결과로만 생성) | DESIGN.md §4.6 leakage guard: eval/attack-probe/style-reference 데이터가 학습 데이터에 절대 섞이면 안 된다는 보장은, eval set 자체를 아무렇지 않게 수정할 수 있으면 성립하지 않는다 |
| `.claude/hooks/guard_region.py` | `gcloud`/`gsutil` 명령이 `asia-northeast3` 이외의 region/zone을 타겟하지 못하게 함 | CLAUDE.md: "Hard constraint: compute and data stay in kr-west" |
| `settings.json` → `permissions.deny` | `Read(./.env)`, `rm -rf*`, `git push --force*` 차단 | 이 프로젝트의 특정 리스크와 무관한 기본 위생 수칙 |

모든 훅은 이미 진술된 제약을 근거로 작성됐다 — 새로운 정책을 만들어낸 훅은 하나도 없다.
이건 의도적이다: P0 시점엔 실제로 뭘 잘못했는지에 대한 기록이 전혀 없으므로, 지금 강제할
수 있는 정당한 대상은 설계 문서가 이미 확약한 것뿐이다.

## 3. `guard_region.py` — 이 프로젝트에만 있는 훅

`block_secrets.py`와 `block_protected_paths.py`는 이전 프로젝트(cs-assistant)에서 쓴
패턴을 거의 그대로 이식한 것이다 — 시크릿, 그리고 frozen/eval-integrity 디렉터리는 흔한
형태이기 때문이다. `guard_region.py`는 새로 만든 것이다: 이 프로젝트는 컴퓨트와 데이터를
`asia-northeast3`로 한정하는 하드 데이터 레지던시 제약이 있는데, GPU도 안 쓰고 리전
개념도 없는 프로젝트에는 대응물이 없다. `gcloud`/`gsutil` 명령의 `--region`/`--zone`/`-l`
플래그 값에 대한 좁은 정규식 매칭이라, **`gcloud config`의 기본 리전에 의존하면서 플래그를
아예 안 쓴 명령은 못 잡는다** — 즉 리스크를 보장하는 게 아니라 줄여줄 뿐이다. GCE 작업
전에는 `gcloud config get-value compute/region`을 별도로 확인할 것.

## 4. 검증

세 훅 모두 `settings.json`에 연결하기 전에 단독으로(JSON payload를 stdin으로 넣고 exit
code·stderr 확인) 실행해봤다 — 코드를 읽고 그럴듯하다고 판단한 게 아니라 실제로 돌려봤다:

| 케이스 | 훅 | 기대 결과 | 결과 |
|---|---|---|---|
| 쓰는 내용에 API 키 모양 문자열 | `block_secrets.py` | 차단 | ✅ |
| 정상적인 `os.environ[...]` 사용 | `block_secrets.py` | 통과 | ✅ |
| 시크릿 파일에 `cat` | `block_secrets.py` | 차단 | ✅ |
| `.env.example`에 `cat` | `block_secrets.py` | 통과 | ✅ |
| `data/eval/*`에 쓰기 | `block_protected_paths.py` | 차단 | ✅ |
| `src/data/*`에 쓰기 | `block_protected_paths.py` | 통과 | ✅ |
| 시크릿 파일에 쓰기 | `block_protected_paths.py` | 차단 | ✅ |
| `gcloud ... --zone=us-central1-a` | `guard_region.py` | 차단 | ✅ |
| `gcloud ... --zone=asia-northeast3-a` | `guard_region.py` | 통과 | ✅ |
| GCP와 무관한 명령 | `guard_region.py` | 통과 | ✅ |

10/10. 테스트한 정상 케이스에서 오탐 없음.

## 5. 아직 안 만든 것들

실제 사용 전에 방어 인프라부터 짓는 건, 리스크가 이미 구체적으로 명시돼 있지 않은 이상
이 프로젝트 스스로의 "단순함 우선" 원칙 위반이다. 아래는 이름은 붙어 있지만 아직 훅으로
만들지 않은 리스크와 그 이유다:

| 리스크 | 어디에 명시돼 있나 | 왜 아직 훅이 없나 |
|---|---|---|
| GPU 시간/비용 초과 (§9.3: diagnostics는 ~3시간/~$2로 캡) | DESIGN.md §9.3 | 매칭할 대상인 GPU 라이프사이클 관련 `scripts/` 코드가 아직 하나도 없음. 아무것도 없는 상태에서 비용 훅을 쓰면 명령 모양을 추측하는 것밖에 안 됨 |
| `if config_name == "dpo"` 식으로 config 이름에 분기 | CLAUDE.md single-pipeline 강제 | `src/` 코드가 아직 없음. 코드베이스가 생긴 뒤에 grep 기반 훅을 쓰는 게 오탐 걱정 없이 쉬움 — 지금은 시기상조 |
| judge/preference-label의 주관적 채점 (circularity guard, §4.4) | DESIGN.md §4.4 | 이건 예전 프로젝트에서 "진단"과 "수정"을 서로 다른 에이전트 역할로 분리했던 것과 같은 종류의 주관적 판단 영역(`eval-reviewer` 서브에이전트 패턴) — 다만 이건 실제로 점수를 내는 judge 파이프라인이 생겨서 그 점수에 이견이 생길 수 있을 때만 그 오버헤드를 들일 가치가 있음 |

**승격 규칙** (자매 프로젝트의 컨벤션을 그대로 가져옴): 위 항목이든, 여기 없는 다른 것이든
**같은 실수가 두 번 반복되면**, 세 번째부터는 CLAUDE.md에 세 번째 리마인더를 적는 대신
그 자리에서 훅으로 만든다. "기억됨"과 "물리적으로 막힘"은 다른 보장이다 — 반복된 실패는
후자를 받을 자격이 있다.

## 6. 알려진 한계

`settings.json`의 훅 명령은 절대경로를 쓴다
(`/Users/jongmin/Project/serana-post-training/.claude/hooks/...`). 여기선 `.claude/`가
gitignore 대상이 아니라서 커밋될 텐데, 이 말은 다른 경로로 이 레포를 clone하는 사람에게는
훅이 조용히 아무 작동도 안 한다는 뜻이다(P6의 "낯선 사람이 레포에서 재현 가능해야 한다"는
기준은 이 파일에 의존하지 않으니 출시를 막는 문제는 아니지만, `.claude/`를 개인 도구가 아니라
공유 가능한 것으로 취급하기 전에 다시 짚어볼 지점이다).

## 7. CLAUDE.md 무게

P0 시점에 CLAUDE.md에는 DESIGN.md가 이미 소유한 내용을 그대로 반복하는 구간이 3곳
있었다(데이터 라우팅 표, 언어정책 문장이 거의 그대로 중복, B/SFT/DPO 실험 표) — CLAUDE.md
스스로 "각 영역은 companion doc이 단일 출처"라고 정해둔 규칙에 비춰보면 드리프트 위험이었다.
DESIGN.md를 가리키는 한 줄 포인터로 줄였고, 행동 규칙에 해당하는 Enforcement 불릿(설계
설명이 아니라 지시인 부분)은 그대로 남겼다. 189줄 → 164줄.
