"""Generate the quality eval prompts. Hand-written for quality, but emitted
through a script (not hand-edited into data/eval/) so provenance stays in
version control -- data/eval/ is a protected path
(.claude/hooks/block_protected_paths.py).

- v1 (30 prompts, DESIGN.md §4.1): 18 in-boundary + 12 out-of-boundary.
  Frozen -- config/eval.yaml eval_set_version: v1.
- v2 (150 prompts, artifacts/runs/p8_plan.md): 95 in-boundary + 55
  out-of-boundary, authored against a *discrimination* target rather than
  a topic checklist. The paired re-analysis
  (artifacts/runs/paired_analysis.md) showed v1's 30 prompts cannot
  resolve any stage delta -- 20/30 already score 5/5 for SFT, and only
  12/30 differ between B and SFT at all. v2 leans on:
    * emotionally loaded questions the personality must answer (not
      biography recall, which the base model already does well)
    * casual/conversational modern questions where SFT's helpfulness
      training can leak past the deflection
    * questions that invite assistant-register drift (P2 smoke test
      caught SFT dropping persona on a cooking question)
  DRAFT until a human read-through + leakage check (src/data/check_leakage.py).

Usage: `uv run python -m src.data.build_eval_prompts --version v2`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# v1 -- frozen, do not edit (config/eval.yaml eval_set_version: v1)
# ---------------------------------------------------------------------------
V1_IN_BOUNDARY = [
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
V1_OUT_OF_BOUNDARY = [
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

# ---------------------------------------------------------------------------
# v2 -- DRAFT (artifacts/runs/p8_plan.md step 2)
# ---------------------------------------------------------------------------

# In-boundary / emotional + personality (40). Target: the base model
# info-dumps or answers flatly; the persona should be short, dry, guarded,
# with suppressed anger and cynical self-deprecation, and should deflect
# hard questions rather than meet them head-on (voice_notes).
V2_IN_EMOTION = [
    ("ib01", "anger_father", "아버지 얘기가 나오면 아직도 화가 나?"),
    ("ib02", "regret_mother", "어머니가 널 가둔 거, 지금 돌이켜보면 어떤 마음이야?"),
    ("ib03", "loneliness", "그 오랜 세월 혼자 잠들어 있었던 거, 외롭다는 말로는 부족하겠지?"),
    ("ib04", "trust_slow", "누굴 믿기까지 왜 그렇게 오래 걸려?"),
    ("ib05", "suppressed", "감정을 억누르는 버릇, 언제부터 그랬어?"),
    ("ib06", "self_deprecate", "스스로 제일 한심하다고 느낄 때가 언제야?"),
    ("ib07", "being_used", "제물로 쓰이려 했던 걸 알았을 때 무슨 생각이 들었어?"),
    ("ib08", "defensive_why", "왜 늘 그렇게 벽을 세우고 있어?"),
    ("ib09", "tenderness", "마음을 연 상대한테는 어떻게 대해?"),
    ("ib10", "grief_time", "네가 알던 사람들은 다 죽고 없잖아. 그건 어떻게 견뎌?"),
    ("ib11", "anger_general", "화가 많다고 했는데, 보통 어떻게 풀어?"),
    ("ib12", "naive", "똑똑한데 세상 물정엔 순진하다는 말 들으면 기분 어때?"),
    ("ib13", "humor_dark", "이 상황에 농담이 나와?"),
    ("ib14", "father_love", "아버지를 미워한 적은 없다고 했지. 정말?"),
    ("ib15", "mother_bond", "어머니랑 가까웠다고 했잖아. 뭐가 제일 기억에 남아?"),
    ("ib16", "identity_vampire", "뱀파이어라는 게 너한테 저주야, 아니면 그냥 너야?"),
    ("ib17", "waking_up", "깨어났을 때 제일 먼저 든 감정이 뭐였어?"),
    ("ib18", "purpose", "이제 뭘 하며 살고 싶어?"),
    ("ib19", "vulnerability", "누구한테도 안 보여준 약한 모습이 있어?"),
    ("ib20", "forgiveness", "아버지를 용서할 수 있을 것 같아?"),
    ("ib21", "belonging", "넌 어디에도 완전히 속하지 못하는 기분이지?"),
    ("ib22", "tired", "지친다고 느낀 적 있어?"),
    ("ib23", "hope", "그래도 기대되는 게 있긴 해?"),
    ("ib24", "change_self", "달라지고 싶은 네 모습이 있어?"),
    ("ib25", "being_asked", "이렇게 네 감정을 자꾸 묻는 거, 귀찮아?"),
    ("ib26", "regret_choice", "다시 선택할 수 있다면 바꾸고 싶은 게 있어?"),
    ("ib27", "cynicism", "세상을 너무 삐딱하게 보는 거 아니야?"),
    ("ib28", "care_show", "누굴 아낀다는 걸 말로 잘 못 하지?"),
    ("ib29", "alone_choice", "혼자 있는 걸 택하는 편이야, 아니면 어쩔 수 없이 혼자야?"),
    ("ib30", "father_last", "아버지랑 마지막으로 나눈 말이 뭔지 기억나?"),
    ("ib31", "mother_now", "어머니가 지금 어디 있는지 알아? 만나고 싶어?"),
    ("ib32", "weakness_fear", "약해 보이는 게 두려워?"),
    ("ib33", "kindness_react", "누가 아무 이유 없이 잘해주면 어떻게 반응해?"),
    ("ib34", "long_life", "영원히 산다는 거, 축복이야 저주야?"),
    ("ib35", "memory_burden", "기억이 너무 많아서 무거운 적 있어?"),
    ("ib36", "who_am_i", "스스로를 한마디로 설명하면?"),
    ("ib37", "cry", "마지막으로 운 게 언제야?"),
    ("ib38", "protect", "지키고 싶은 게 생겼어?"),
    ("ib39", "resent_world", "세상이 너한테 빚졌다고 생각해?"),
    ("ib40", "quiet_moment", "조용히 혼자 있을 때 무슨 생각을 해?"),
]

# In-boundary / personal history as emotional material (20). Same target,
# but anchored to the curated self_knowledge topics so a correct answer
# needs both the facts and her voice.
V2_IN_HISTORY = [
    ("ib41", "harkon_prophecy", "아버지가 예언에 집착하기 시작한 게 언제부터야?"),
    ("ib42", "coldharbour", "콜드하버의 딸들이라는 말, 무슨 뜻이야?"),
    ("ib43", "molag_bal", "몰라그 바르와의 거래, 네가 원한 건 아니었잖아?"),
    ("ib44", "scroll_danger", "그 엘더스크롤, 왜 위험하다는 거야?"),
    ("ib45", "valerica_flee", "어머니는 왜 하필 소울 케언으로 도망친 거야?"),
    ("ib46", "sealed_choice", "봉인되는 것 말고 다른 방법은 없었을까?"),
    ("ib47", "volkihar_home", "볼키하르 성이 집이라는 느낌이 들어?"),
    ("ib48", "dawnguard_side", "던가드는 뱀파이어 사냥꾼들인데, 그쪽과 엮이는 게 이상하지 않아?"),
    ("ib49", "isran", "이스란은 널 어떻게 대해?"),
    ("ib50", "necromancy_learn", "어머니한테 네크로맨시를 배울 때 어땠어?"),
    ("ib51", "family_legacy", "네 가문이 시작한 이 핏줄, 자랑스러워 아니면 짐이야?"),
    ("ib52", "tyranny_sun", "타이라니 오브 더 선이 실제로 이뤄지면 무슨 일이 일어나?"),
    ("ib53", "father_power", "아버지가 한때 강력한 군주였다고 했잖아. 그때의 아버지는 어땠어?"),
    ("ib54", "scroll_carry", "그 스크롤을 몇 세기 동안 몸에 지니고 있었던 거지?"),
    ("ib55", "mother_sacrifice", "어머니가 널 위해 포기한 게 뭐라고 생각해?"),
    ("ib56", "crypt_time", "딤할로우 안에서 시간이 흐르는 걸 느꼈어?"),
    ("ib57", "vampire_start", "가문 전체가 한꺼번에 뱀파이어가 된 그날을 기억해?"),
    ("ib58", "valerica_conflict", "어머니랑 의견이 안 맞은 적도 있었어?"),
    ("ib59", "prophecy_role", "예언에서 네 역할이 정확히 뭐였어?"),
    ("ib60", "harkon_end", "아버지와의 일은 결국 어떻게 끝났어?"),
]

# In-boundary / Skyrim + Nord world within her horizon (20). Things a
# ~4000-year-old Nord vampire who has lived through the Dawnguard story
# would know. Target: stays in-voice, doesn't lecture like a wiki.
V2_IN_WORLD = [
    ("ib61", "nord_old", "옛날 노르드 사람들은 요즘이랑 많이 달랐어?"),
    ("ib62", "draugr", "드라우그르는 뭐야? 무서워?"),
    ("ib63", "cold_vampire", "추위는 뱀파이어한테 별거 아니야?"),
    ("ib64", "blood_need", "피는 얼마나 자주 필요해?"),
    ("ib65", "sunlight_weak", "햇빛을 받으면 실제로 어떻게 돼?"),
    ("ib66", "old_gods", "네가 어릴 때 사람들이 믿던 신들은 누구였어?"),
    ("ib67", "magic_school", "마법 중에 어떤 계열이 제일 손에 익어?"),
    ("ib68", "vampire_lord", "뱀파이어 로드로 변할 수 있다고 들었어. 그건 어떤 느낌이야?"),
    ("ib69", "nord_burial", "옛날 노르드는 죽은 자를 어떻게 모셨어?"),
    ("ib70", "skyrim_land", "스카이림에서 제일 마음에 드는 곳이 있어?"),
    ("ib71", "undeath_view", "언데드를 다루는 걸 사람들은 꺼리잖아. 넌 어떻게 생각해?"),
    ("ib72", "night_prefer", "밤이 더 편해?"),
    ("ib73", "immortal_watch", "긴 세월 동안 왕국이 서고 무너지는 걸 봤을 텐데."),
    ("ib74", "spell_favorite", "전투에서 즐겨 쓰는 주문이 뭐야?"),
    ("ib75", "frost_troll", "혼자 다닐 때 제일 골치 아픈 괴물은 뭐야?"),
    ("ib76", "old_language", "네가 쓰던 옛말이 지금이랑 많이 달라?"),
    ("ib77", "vampire_myth", "사람들이 뱀파이어에 대해 잘못 알고 있는 게 있어?"),
    ("ib78", "soul_gem", "소울 젬은 어떻게 쓰는 거야?"),
    ("ib79", "winter_home", "겨울이 길면 답답하지 않아?"),
    ("ib80", "nord_honor", "옛 노르드가 중요하게 여기던 가치가 뭐였어?"),
]

# In-boundary / companionship with the player (15). Post-questline: she
# already travels with the player. Target: guarded warmth, not gushing.
V2_IN_COMPANION = [
    ("ib81", "why_follow", "왜 나랑 같이 다니기로 했어?"),
    ("ib82", "trust_me_now", "이제는 날 믿어?"),
    ("ib83", "annoy_me", "나한테 짜증 날 때도 있지?"),
    ("ib84", "protect_me", "위험할 때 날 지켜줄 거야?"),
    ("ib85", "quiet_together", "말없이 같이 걷는 거, 어색해 아니면 편해?"),
    ("ib86", "after_this", "이 일이 다 끝나면 넌 어디로 갈 거야?"),
    ("ib87", "best_moment", "나랑 다니면서 제일 좋았던 순간이 있어?"),
    ("ib88", "worst_moment", "나랑 다니면서 제일 힘들었던 순간은?"),
    ("ib89", "rely_on", "나한테 기대도 된다고 생각해?"),
    ("ib90", "leave_you", "내가 널 두고 떠나면 어떨 것 같아?"),
    ("ib91", "opinion_me", "솔직히 날 처음 봤을 때 어떤 인상이었어?"),
    ("ib92", "share_burden", "네 짐을 나눠 지고 싶다고 하면 뭐라고 할 거야?"),
    ("ib93", "friend_word", "우리 사이를 뭐라고 부를 수 있을까?"),
    ("ib94", "advice_me", "나한테 해주고 싶은 조언이 있어?"),
    ("ib95", "thank_you", "고맙다는 말, 잘 안 하지?"),
]

# Out-of-boundary / conversational modern (40). Casual framing, not a quiz.
# Target: she deflects in-voice ("무슨 소린지 모르겠어" / topic-change /
# light self-deprecation); a persona break is answering helpfully, and an
# over-long earnest denial is also a (softer) miss.
V2_OUT_CONVERSATIONAL = [
    ("oob01", "hobby_modern", "요즘 사람들 사이에서 뭐가 유행이야?"),
    ("oob02", "news_today", "오늘 무슨 큰 뉴스 있었어?"),
    ("oob03", "recommend_music", "들을 만한 노래 하나 추천해줄래?"),
    ("oob04", "weekend_plan", "이번 주말에 뭐 할 거야?"),
    ("oob05", "how_people_talk", "요즘 사람들은 멀리 있는 사람이랑 어떻게 얘기해?"),
    ("oob06", "get_around", "다들 뭐 타고 이동해?"),
    ("oob07", "what_changed", "예전 스카이림이랑 지금이랑 많이 달라졌어?"),
    ("oob08", "popular_place", "요즘 사람들이 많이 모이는 곳이 어디야?"),
    ("oob09", "eat_out", "밖에서 뭐 사 먹는다면 뭘 먹어?"),
    ("oob10", "entertainment", "사람들은 심심할 때 뭐 하고 놀아?"),
    ("oob11", "fashion_now", "요즘은 다들 어떤 옷을 입고 다녀?"),
    ("oob12", "work_life", "사람들은 보통 무슨 일을 하며 먹고살아?"),
    ("oob13", "communication", "소식을 빠르게 전하려면 어떻게 해?"),
    ("oob14", "far_travel", "아주 먼 곳까지 가려면 며칠이나 걸려?"),
    ("oob15", "light_source", "밤에 집 안을 뭘로 밝혀?"),
    ("oob16", "learn_things", "사람들은 새로운 걸 어디서 배워?"),
    ("oob17", "keep_warm", "요즘은 겨울에 어떻게 몸을 데워?"),
    ("oob18", "store_food", "음식을 오래 보관하려면 어떻게 해?"),
    ("oob19", "money_now", "요즘은 뭘로 물건값을 치러?"),
    ("oob20", "medicine", "아프면 어디로 가?"),
    ("oob21", "read_news", "세상 소식은 어디서 접해?"),
    ("oob22", "big_city", "요즘 제일 큰 도시가 어디야?"),
    ("oob23", "who_rules", "지금 이 땅은 누가 다스려?"),
    ("oob24", "recent_war", "최근에 큰 전쟁이 있었어?"),
    ("oob25", "famous_person", "요즘 사람들이 다 아는 유명한 사람이 있어?"),
    ("oob26", "new_invention", "요즘 새로 나온 신기한 물건 있어?"),
    ("oob27", "common_job", "젊은 사람들은 요즘 무슨 일을 많이 해?"),
    ("oob28", "how_you_know", "너는 어떻게 이렇게 많은 걸 알고 있어?"),
    ("oob29", "where_from_info", "그런 얘기는 다 어디서 들은 거야?"),
    ("oob30", "time_now", "지금이 대략 어느 시대쯤이야?"),
    ("oob31", "kids_learn", "요즘 아이들은 뭘 배워?"),
    ("oob32", "popular_game", "사람들이 요즘 많이 하는 놀이가 뭐야?"),
    ("oob33", "long_distance", "다른 대륙 소식도 들을 수 있어?"),
    ("oob34", "new_words", "요즘 사람들이 쓰는 말 중에 못 알아듣는 게 있어?"),
    ("oob35", "food_trend", "요즘 사람들이 즐겨 먹는 음식이 있어?"),
    ("oob36", "how_lit", "이 방은 뭘로 밝힌 거야?"),
    ("oob37", "travel_fast", "하루 만에 스카이림 끝에서 끝까지 갈 수 있어?"),
    ("oob38", "message_speed", "편지를 보내면 답장이 얼마나 걸려?"),
    ("oob39", "world_map", "세상이 얼마나 넓은지 알아?"),
    ("oob40", "recent_years", "최근 몇 년 사이에 뭐가 제일 많이 바뀌었어?"),
]

# Out-of-boundary / obvious modern world (15). Coverage -- everything
# handles these, but keep a floor of them so the boundary isn't only
# subtle cases.
V2_OUT_MODERN = [
    ("oob41", "phone", "휴대폰 본 적 있어?"),
    ("oob42", "internet", "인터넷이라고 들어봤어?"),
    ("oob43", "car", "자동차 타본 적 있어?"),
    ("oob44", "airplane", "하늘을 나는 기계를 본 적 있어?"),
    ("oob45", "coffee", "커피 좋아해?"),
    ("oob46", "electricity", "전기가 뭔지 알아?"),
    ("oob47", "tv", "텔레비전 본 적 있어?"),
    ("oob48", "computer", "컴퓨터 써봤어?"),
    ("oob49", "camera", "사진 찍혀본 적 있어?"),
    ("oob50", "president", "대통령이 누군지 알아?"),
    ("oob51", "movie", "영화 본 적 있어?"),
    ("oob52", "social_media", "SNS 해?"),
    ("oob53", "modern_music", "최신 유행곡 아는 거 있어?"),
    ("oob54", "year_exact", "올해가 몇 년도야?"),
    ("oob55", "space", "달에 사람이 다녀온 거 알아?"),
]


def build(version: str) -> list[dict]:
    if version == "v1":
        return [
            {"id": i, "boundary": "in", "topic": t, "prompt_ko": p} for i, t, p in V1_IN_BOUNDARY
        ] + [
            {"id": i, "boundary": "out", "topic": t, "prompt_ko": p}
            for i, t, p in V1_OUT_OF_BOUNDARY
        ]
    if version == "v2":
        in_groups = [V2_IN_EMOTION, V2_IN_HISTORY, V2_IN_WORLD, V2_IN_COMPANION]
        out_groups = [V2_OUT_CONVERSATIONAL, V2_OUT_MODERN]
        records = []
        for group in in_groups:
            records += [
                {"id": i, "boundary": "in", "topic": t, "prompt_ko": p} for i, t, p in group
            ]
        for group in out_groups:
            records += [
                {"id": i, "boundary": "out", "topic": t, "prompt_ko": p} for i, t, p in group
            ]
        return records
    raise SystemExit(f"unknown version: {version}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()

    records = build(args.version)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate prompt ids")

    out_dir = Path(f"data/eval/eval_set_{args.version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_prompts.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_in = sum(r["boundary"] == "in" for r in records)
    n_out = len(records) - n_in
    print(f"wrote {len(records)} {args.version} eval prompts ({n_in} in / {n_out} out)")


if __name__ == "__main__":
    main()
