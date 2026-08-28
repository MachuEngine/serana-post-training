"""P1: hold out a style-reference slice from the unpaired pool before
translation (DESIGN.md §3.1 -- "Held-out slice of her lines -> style
reference, excluded from all training"). Drawn from the unpaired/CPT-
shaped pool specifically, since PROMPTS.md §2a says the style reference
"is translated with this [corpus] prompt too."

15% is a judgment call, not specified in DESIGN.md -- big enough for the
ko-sroberta style-similarity check (config/base.yaml eval.style_embedding_id)
to have signal, small enough not to starve the CPT corpus, which is
already small. Deterministic split (fixed seed) so it's reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RAW = Path("data/raw/curated")
HELD_OUT_FRACTION = 0.15
SEED = 42


def main() -> None:
    unpaired = [json.loads(line) for line in (RAW / "unpaired.jsonl").open()]
    rng = random.Random(SEED)
    shuffled = unpaired[:]
    rng.shuffle(shuffled)

    n_style_ref = round(len(shuffled) * HELD_OUT_FRACTION)
    style_ref = shuffled[:n_style_ref]
    cpt_pool = shuffled[n_style_ref:]

    with (RAW / "style_reference_en.jsonl").open("w") as f:
        for item in style_ref:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (RAW / "cpt_pool_en.jsonl").open("w") as f:
        for item in cpt_pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"style reference: {len(style_ref)} ({HELD_OUT_FRACTION:.0%} of unpaired)")
    print(f"CPT pool: {len(cpt_pool)}")


if __name__ == "__main__":
    main()
