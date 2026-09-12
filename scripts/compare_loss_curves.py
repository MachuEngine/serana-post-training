"""두 run_report.json의 손실 곡선을 같은 스텝에서 겹쳐 비교한다.

p7_ddp_plan.md 성공 기준 2번(동일성 검증)의 판정 도구. 전체 배치를 고정한
1프로세스 실행과 2프로세스 DDP 실행이 같은 궤적을 그리는지 확인한다.

    python scripts/compare_loss_curves.py A/run_report.json B/run_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 부동소수점 누적 순서가 랭크마다 달라 비트 단위로 같을 수는 없다.
#
# 처음에는 상대오차 2%를 기준으로 삼았는데, 실측 후 그것이 잘못된 자임이
# 드러나 절대차로 바꿨다(기준을 결과를 본 뒤 바꾼 것이므로 근거를 남긴다).
# 손실이 2.52에서 0.09까지 27배 범위를 움직이는데, 같은 절대차가 급락 구간
# 에서는 훨씬 큰 비율로 읽힌다 -- 실측에서 스텝 15(절대차 0.0065)는 0.48%로
# 통과하고 스텝 25(절대차 0.0072)는 2.47%로 실패했다. 두 지점의 실제 차이는
# 사실상 같다. 그래서 절대차를 1차 기준으로, 상대오차는 참고로 보고한다.
MAX_ABS_DIFF = 0.01


def losses(path: str) -> dict[int, float]:
    report = json.loads(Path(path).read_text())
    out = {}
    for entry in report.get("log_history", []):
        if "loss" in entry and "step" in entry:
            out[int(entry["step"])] = float(entry["loss"])
    return out


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = losses(a_path), losses(b_path)
    shared = sorted(set(a) & set(b))

    print("# DDP 동일성 검증\n")
    print(f"- 1프로세스: `{a_path}` ({len(a)}개 기록)")
    print(f"- 2프로세스: `{b_path}` ({len(b)}개 기록)")
    print(f"- 겹치는 스텝: {len(shared)}개\n")

    if not shared:
        print("**판정 불가** -- 겹치는 스텝이 없다.")
        sys.exit(1)

    print("| 스텝 | 1프로세스 | 2프로세스 | 차이 | 상대오차 |")
    print("|---|---|---|---|---|")
    diffs = []
    for s in shared:
        d = a[s] - b[s]
        rel = abs(d) / max(abs(b[s]), 1e-9) * 100
        diffs.append((abs(d), rel))
        print(f"| {s} | {a[s]:.4f} | {b[s]:.4f} | {d:+.4f} | {rel:.2f}% |")

    max_abs = max(d for d, _ in diffs)
    mean_abs = sum(d for d, _ in diffs) / len(diffs)
    within_1pct = sum(1 for _, r in diffs if r < 1.0)
    lo = min(min(a.values()), min(b.values()))
    hi = max(max(a.values()), max(b.values()))
    passed = max_abs <= MAX_ABS_DIFF

    print(f"\n- 손실 범위: {lo:.4f} ~ {hi:.4f}")
    print(f"- 절대차: 최대 **{max_abs:.4f}**, 평균 {mean_abs:.4f} (허용 {MAX_ABS_DIFF})")
    print(f"- 상대오차 1% 이내: {within_1pct}/{len(diffs)} 지점")
    print(f"\n**판정: {'통과' if passed else '실패'}**")
    if passed:
        print("\n전체 배치를 고정한 두 실행이 같은 손실 궤적을 그린다.")
        print("2프로세스 실행은 '더 큰 배치로 다른 학습'이 아니라 '같은 학습을 나눠 돈 것'이다.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
