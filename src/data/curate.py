"""P1 curate: dedup + horizon filter on data/raw/{pairs,unpaired}.jsonl
(DESIGN.md §3.1). Runs after ingest_uesp.py, before translation. Output is
still English -- translation is a separate, API-costing step so this stage
stays free to re-run.

Horizon filter (CLAUDE.md, DESIGN.md §3.1): exclude anything post-4E 201
or modern/meta-world. Serana's actual game dialogue is Nirn/Skyrim-era by
construction, so this is mostly a sanity net for stray meta lines (fourth-
wall breaks, dev jokes) rather than an expected high-volume cut -- flagged
lines are written out for a manual look rather than silently dropped.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

RAW = Path("data/raw")

# Sanity net, not a real filter -- Skyrim-era dialogue shouldn't trip these.
META_KEYWORDS = [
    "xbox",
    "playstation",
    "controller",
    "achievement",
    "internet",
    "computer",
    "download",
    "patch",
    "dlc",
    "bethesda",
    "skyrim special edition",
    "quest marker",
    "load a save",
    "console command",
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?\"'")


def is_flagged(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in META_KEYWORDS)


def dedup_unpaired(items: list[dict]) -> tuple[list[dict], int]:
    """Exact-normalized dedup first (cheap, catches the Fandom/UESP
    Follower-Dialogue overlap), then a near-duplicate pass (difflib ratio
    > 0.92) since Fandom's Quotes table sometimes has minor punctuation
    drift. O(n^2) is fine at this volume (a few hundred lines)."""
    seen: dict[str, dict] = {}
    for item in items:
        key = normalize(item["line"])
        if key not in seen:
            seen[key] = item
    exact_deduped = list(seen.values())

    kept: list[dict] = []
    dropped_near = 0
    for item in exact_deduped:
        key = normalize(item["line"])
        is_near_dup = any(
            SequenceMatcher(None, key, normalize(k["line"])).ratio() > 0.92 for k in kept
        )
        if is_near_dup:
            dropped_near += 1
        else:
            kept.append(item)
    return kept, (len(items) - len(exact_deduped)) + dropped_near


def dedup_pairs(items: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    kept = []
    for item in items:
        key = (normalize(item["player_line"]), normalize(item["reply"]))
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept, len(items) - len(kept)


def main() -> None:
    pairs = [json.loads(line) for line in (RAW / "pairs.jsonl").open()]
    unpaired = [json.loads(line) for line in (RAW / "unpaired.jsonl").open()]

    pairs, pairs_dropped_dup = dedup_pairs(pairs)
    unpaired, unpaired_dropped_dup = dedup_unpaired(unpaired)

    flagged_pairs = [p for p in pairs if is_flagged(p["player_line"]) or is_flagged(p["reply"])]
    flagged_unpaired = [u for u in unpaired if is_flagged(u["line"])]
    pairs = [p for p in pairs if p not in flagged_pairs]
    unpaired = [u for u in unpaired if u not in flagged_unpaired]

    Path("data/raw/curated").mkdir(parents=True, exist_ok=True)
    with open("data/raw/curated/pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open("data/raw/curated/unpaired.jsonl", "w") as f:
        for u in unpaired:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open("data/raw/curated/horizon_flagged.jsonl", "w") as f:
        for item in flagged_pairs + flagged_unpaired:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_lines = len(pairs) + len(unpaired)
    real_pair_ratio = len(pairs) / total_lines if total_lines else 0.0

    report = {
        "pairs_kept": len(pairs),
        "pairs_dropped_exact_dup": pairs_dropped_dup,
        "unpaired_kept": len(unpaired),
        "unpaired_dropped_dup_or_near_dup": unpaired_dropped_dup,
        "horizon_flagged_count": len(flagged_pairs) + len(flagged_unpaired),
        "real_pair_ratio_of_ingested_lines": round(real_pair_ratio, 4),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("data/raw/curated/p1_ingest_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
