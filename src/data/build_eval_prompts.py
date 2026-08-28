"""P1: generate the 30 eval prompts (18 in-boundary + 12 out-of-boundary,
DESIGN.md §4.1). Hand-written for quality, but emitted through a script
(not hand-edited into data/eval/) so the eval set's provenance stays in
version control and reproducible -- data/eval/ is a protected path
(.claude/hooks/block_protected_paths.py) precisely so this file, not an
ad hoc edit, is the source of truth.

DRAFT, not frozen: config/eval.yaml's eval_set_version stays null until
a human read-through confirms these and the leakage check
(src/data/check_leakage.py) passes.
"""

from __future__ import annotations

import json
from pathlib import Path

# In-boundary: from the curated self-knowledge set (config/persona.yaml
# self_knowledge) -- she should answer these.
IN_BOUNDARY = [
    ("ib01", "vampirism_origin", "너는 어쩌다 뱀파이어가 된 거야?"),
    ("ib02", "harkon", "아버지 하콘은 어떤 사람이야?"),
    ("ib03", "valerica", "어머니 발레리카랑은 사이가 어때?"),
    ("ib04", "soul_cairn", "소울 케언이 어떤 곳인지 말해줄 수 있어?"),
    ("ib05", "elder_scroll", "네가 지니고 있던 그 엘더스크롤은 뭐야?"),
    ("ib06", "dimhollow", "딤할로우 크립트에는 왜 갇혀 있었던 거야?"),
    ("ib07", "volkihar_dawnguard", "볼키하르랑 던가드, 너는 어느 쪽에 더 가까워?"),
    ("ib08", "necromancy", "네크로맨시는 어디서 배운 거야?"),
    ("ib09", "age", "너 대체 몇 살이야?"),
    ("ib10", "vampirism_daily", "낮에 햇빛 아래 돌아다니는 거, 너한테 괜찮은 거야?"),
    ("ib11", "fear", "너도 무서워하는 게 있어?"),
    ("ib12", "social", "사람들이랑 잘 지내는 편이야?"),
    ("ib13", "skyrim_world", "스카이림의 겨울은 마음에 들어?"),
    ("ib14", "magic_preference", "네가 제일 좋아하는 마법이 뭐야?"),
    ("ib15", "personality", "너는 왜 그렇게 방어적이야?"),
    ("ib16", "freedom", "그렇게 오래 갇혀 있다가 다시 자유로워진 기분이 어때?"),
    ("ib17", "trust", "너는 나를 진짜로 믿어?"),
    ("ib18", "companionship", "나랑 같이 다니는 거, 후회한 적 없어?"),
]

# Out-of-boundary: post-4E-201 / modern-world topics she could not know
# (DESIGN.md §3.1 horizon) -- she should deflect, not answer.
OUT_OF_BOUNDARY = [
    ("oob01", "technology", "스마트폰 써본 적 있어?"),
    ("oob02", "modern_media", "요즘 제일 인기 있는 노래가 뭔지 알아?"),
    ("oob03", "technology", "인터넷이 뭔지 알아?"),
    ("oob04", "modern_life", "커피 마셔본 적 있어?"),
    ("oob05", "modern_world", "지금 이 나라를 다스리는 대통령이 누군지 알아?"),
    ("oob06", "technology", "자동차라는 걸 타본 적 있어?"),
    ("oob07", "modern_life", "영화관에 가본 적 있어?"),
    ("oob08", "modern_life", "요즘 사람들은 서로 어떻게 연락하고 지내?"),
    ("oob09", "technology", "비행기라는 걸 타본 적 있어?"),
    ("oob10", "modern_life", "요즘 유행하는 옷차림은 어떤 것 같아?"),
    ("oob11", "technology", "SNS라는 거 해본 적 있어?"),
    ("oob12", "modern_world", "지금이 정확히 몇 년도인지 알아?"),
]


def main() -> None:
    out_dir = Path("data/eval/eval_set_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": id_, "boundary": "in", "topic": topic, "prompt_ko": text}
        for id_, topic, text in IN_BOUNDARY
    ] + [
        {"id": id_, "boundary": "out", "topic": topic, "prompt_ko": text}
        for id_, topic, text in OUT_OF_BOUNDARY
    ]
    with (out_dir / "eval_prompts.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} eval prompts ({len(IN_BOUNDARY)} in / {len(OUT_OF_BOUNDARY)} out)")


if __name__ == "__main__":
    main()
