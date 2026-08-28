# P1 progress log

Running log of autonomous P1 work (CLAUDE.md build order). Updated as work
happens; not a final report. See `data/raw/curated/p1_ingest_report.json`
and `data/ko/build_report.json` for the machine-readable numbers this
references.

## Decisions made

- **Judge/translation provider (DESIGN.md §9.4): GPT-4o (OpenAI).** Already
  fixed in `config/eval.yaml` before tonight — confirmed with the user,
  no change needed. `OPENAI_API_KEY` was already set in `.env`.
- **`config/persona.yaml` created.** DESIGN.md §5.1's layout never listed
  a file for `persona_profile`/`voice_notes`/`glossary`/`speech_level`/
  `self_knowledge`, even though every prompt in PROMPTS.md assumes one
  exists ("comes from config/"). Added the file, cross-referenced it into
  DESIGN.md §5.1. **`persona_profile` is a first-cut draft** grounded in
  UESP's Background/Behavior sections (not invented) but is the single
  highest-leverage hand-written artifact in the project — read it before
  P2, PROMPTS.md §1 flags exactly why.

## Ingestion (UESP + Fandom, DESIGN.md §3.1)

- Source: UESP `Skyrim:Serana` (Background, Quest-Related Events, Dialogue,
  Combat Dialogue, Follower Dialogue) via MediaWiki `action=parse&prop=wikitext`
  — raw wikitext, not WebFetch's summarized text, so the corpus is exact,
  not paraphrased. Fandom `Serana` page's Quotes section pulled as a
  supplementary source; its Dialogue/Conversations sections were skipped
  (nested `{{Hide|...}}` branches, high parse risk, content overlaps UESP's
  Follower Dialogue almost verbatim anyway). Full rationale in
  `config/data_sources.yaml`.
- **Bug found and fixed during parsing:** UESP's quest sections switch into
  a multi-character cutscene format (`'''Harkon:''' "line"` and
  `'''Harkon''': "line"` — two different notations, both had to be
  handled). The first parser version attributed Harkon's, Valerica's,
  Isran's, Gelebor's, and Arch-Curate Vyrthur's lines to Serana. Caught by
  spot-checking a random sample (`"Serana, my darling..."` turned out to
  be Harkon addressing her, not her speaking) — verified against the raw
  wikitext, then fixed by detecting the speaker tag explicitly and
  discarding every non-Serana-tagged line. This would have quietly injected
  wrong-speaker training data into a persona-consistency project if it had
  gone unnoticed. Re-scanned after the fix: 0 remaining suspect lines.
- Final counts after dedup + horizon-filter sanity net (see
  `data/raw/curated/p1_ingest_report.json`):
  - **234 real (player_line, reply) pairs** — genuine wiki-recorded
    exchanges, not synthetic.
  - **221 unpaired standalone lines** (ambient/follower/combat barks +
    quest-scene lines with no player prompt).
  - **Real-pair ratio: ~51%** of ingested lines (`real_pair_ratio_of_ingested_lines`
    in the report) — far above what CLAUDE.md anticipated ("a few hundred
    to a thousand lines is small for CPT"); the Quest-Related Events
    sections were the difference (25 pairs from the Dialogue section alone
    vs. 248 once quest sections were included).
  - 1 line dropped by the horizon-filter sanity net — turned out to be a
    stray `{{Bug|...}}` template artifact, not an actual horizon violation;
    correctly excluded either way.
- **Style reference held out before translation** (DESIGN.md §4.6 rule 1):
  15% of the unpaired pool (33 lines), fixed seed, excluded from the CPT
  corpus and never passed to synthetic generation as an example.
  188 lines remain in the CPT pool.

## Translation (PROMPTS.md §2a/§2b)

- First full-concurrency run (8 workers) mostly hit this org's gpt-4o rate
  limit (30k TPM) and the script had no retry logic — silently recorded
  ~60% of items as failed. Fixed: dropped to 3 workers, added
  exponential-backoff retry on `RateLimitError`, and made every batch
  resumable (skips items already translated on a re-run, so the fix didn't
  cost a re-pay for the successes already banked).
- Status as of this log entry: CPT pool 188/188 and style reference 33/33
  translated; real pairs in progress (rate-limited by the same 30k TPM
  cap, running with retry).

## Update (continued autonomous work)

- **Real-pair translation finished**: 234/234. Spot-checked, natural
  consistent 반말, glossary held.
- **`src/finetune/train.py` + `scripts/train.py` written and smoke-tested.**
  Single dispatch (`train.method: cpt|sft|dpo`), device-aware: resolves
  cuda > mps > cpu and downgrades CUDA-only config knobs (4-bit quant,
  FlashAttention-2, paged_adamw_8bit) to a working equivalent on MPS,
  logging every downgrade. **This diverges from DESIGN.md §6's original
  `cpt.py, sft.py, dpo.py` layout** — consolidated into one file for the
  same reason inference already has one `run(config)`; DESIGN.md §6 is
  updated to match. Smoke test (3 steps, Qwen2.5-0.5B, MPS, tiny scratch
  data): trained, evaluated, saved a real LoRA adapter end to end. Two API
  mismatches against the installed trl==1.10.0 were caught and fixed
  during the smoke test (`SFTConfig`'s field is `max_length`, not
  `max_seq_length`; `from_pretrained` wants `dtype`, not the deprecated
  `torch_dtype`) — exactly the kind of thing that would otherwise surface
  as a P2 surprise on the real GPU run. **Only the `sft` path is verified
  by an actual run tonight**; `cpt` and `dpo` are written to the same
  contract but need their own verification pass in P2/P4.
- **Synthetic SFT generation (`src/data/synth_dialogue.py`) hit a real bug
  before the fix cost anything:** `response_format={"type":"json_object"}`
  requires a top-level JSON *object*, but the prompt asked for a top-level
  *array* — GPT-4o silently ignored the "write N exchanges" instruction
  and returned a single exchange instead of a list. Fixed by asking for
  `{"exchanges": [...]}` explicitly. Caught by a 3-item smoke test before
  committing to the ~2,800-call full run.
- **Generation is long enough (single-threaded under this account's 30k
  TPM cap, ~30-60+ min) that writing output only at the end was a real
  risk** — a crash at 90% would have silently lost everything, since nothing
  hits disk until the process's final line. Killed the first attempt early
  (~5 min in, cheap to restart) and rewrote `generate_pairs`/
  `generate_dpo_prompts` to append-and-flush every batch, with resume
  support (a re-run counts what's already on disk per category and only
  generates the remainder). Restarted; running as of this log entry.
- **ruff clean.** `ruff format` + fixed 21 `E741` (ambiguous `l` used as a
  loop variable across 8 files — renamed to `line`) + added a targeted
  `per-file-ignores` for `E501` on `translate.py`/`train.py` specifically
  because they embed PROMPTS.md prompt text verbatim (reflowing those
  lines to satisfy the linter would make the copy diverge from the source
  of truth PROMPTS.md declares itself to be).

## Final update — everything autonomous is done

- **Synthetic generation finished clean**: exactly 2,766 synthetic pairs
  (2,074 ordinary / 415 hard-case / 277 oob, matching the computed targets
  exactly) + 1,000 DPO prompts.
- **A real bug surfaced and got fixed on the way**: the DPO prompt pool had
  ~24 near-duplicate "what's your favorite magic" variants (independent
  per-batch generation has no memory of earlier batches — expected, not a
  fluke) *and* `check_leakage.py`'s first real run caught an exact
  overlap between an eval prompt (`ib14`) and a generated DPO prompt.
  `build_corpora.py` now dedupes the pool (1000 → 893 after removing 106
  near-dupes) and explicitly filters anything overlapping an eval prompt
  before writing `data/ko/dpo_prompt_pool.jsonl`. Re-ran the leakage
  check afterward — passes clean.
- **`data/ko/sft_3k.jsonl` final: 2,995 pairs** (230 real + synthetic,
  5 dropped as exact/near-dup). `data/ko/cpt_corpus.txt`: 188 lines.
  `real_pair_ratio_of_sft_set: 0.077` — note this is a *different* number
  from the ~51% real-pair ratio of raw ingested lines (`p1_ingest_report.json`):
  that one describes what survived ingestion, this one describes the
  final training mix after padding to ~3k with synthetic data. Both
  belong in the README per DESIGN.md §3.1, and they mean different things
  — don't conflate them.
- **Leakage check passes**: 30 eval prompts + 24 attack probes + 33
  style-reference lines checked against 202 CPT paragraphs / 2,995 SFT
  pairs / 893 DPO prompts. No overlap.
- **§3.6b failure sandbox: all four breaks run for real** on the actual
  `sft_3k.jsonl` (not scratch data), full curves and interpretation in
  `artifacts/diagnostics/README.md`. All three of divergence/underfitting/
  overfitting matched the predicted shape. The leak run's aggregate loss
  curve did *not* look crisply different from ordinary overfitting on
  small data — a genuine predicted-vs-measured miss, written up as a
  finding (the practical leak-detection signal has to be a targeted
  eval-prompt check, not just eyeballing the loss plot — which is exactly
  why `check_leakage.py` exists as a pre-training assertion). Also found:
  `config/base.yaml`'s default `grad_accum_steps: 16` (GPU-memory-tuned)
  made the first divergence attempt take 24 min for 60 nominal steps on
  the M5; overriding to 1 for local runs cut every subsequent run to
  2-7 min. Worth carrying into how P2/P4 diagnostic configs get invoked.
- 50-item human-label **candidate** pool built
  (`data/eval/eval_set_v1/human_labels.jsonl`): 20 genuine / 15 synthetic
  / 15 deliberately corrupted (admits_ai / breaks_register /
  impossible_knowledge violation types), shuffled so pool order doesn't
  cue the scorer. `human_score`/`human_violation` are null, waiting on you.
- `config/eval.yaml`'s `correlation_floor` filled in as `0.6`, matching
  DESIGN.md §4.4's stated Spearman floor (was `null`/TODO).
- ruff clean across everything written tonight.

Eval prompts (30: 18 in-boundary / 12 out-of-boundary) and the 24 attack
probes (translated to Korean by hand, not machine-translated, per
PROMPTS.md §6's warning about awkward-Korean probes being an easier
target) are both drafted and written to `data/eval/eval_set_v1/` via
script.

## What's left

1. ~~Score the 50 human-label candidates~~ **Done.** Scored by hand,
   walked through one at a time in-session. `judge_pcs` then run against
   the same 50: **Spearman 0.792** (floor is 0.6 — clears it comfortably),
   72% exact agreement, 96% within-1, 92% violation-flag agreement. Full
   numbers in `data/eval/eval_set_v1/judge_validation_report.json`. The
   two biggest disagreements (both human=5, judge=3) were wiki dialogue-
   tree fragments that read as slightly ambiguous without the surrounding
   quest context — a corpus-shape observation, not a judge defect.
   `src/eval/judge_pcs.py` and `src/eval/validate_judge.py` are now real,
   reusable P5 code, not just a one-off check.
2. ~~Read `config/persona.yaml`'s `persona_profile`~~ **Done.** Facts
   checked against Elder Scrolls/Dawnguard lore — held up, no errors
   found. One real inconsistency surfaced and resolved: `self_knowledge`
   wrote her as already familiar with Dawnguard, but genuine ingested
   wiki dialogue includes her canonical first reaction ("그런 이름은
   처음 들어보는데" — never heard of it). User decision: this persona
   represents her *after* living through the Dawnguard questline, not
   the moment she wakes up. Documented directly in `self_knowledge`'s
   comment, including the accepted seam (early-game "doesn't know" lines
   still sit in the real-pair corpus and weren't filtered out).
3. ~~Read through `eval_prompts.jsonl` and `attack_probes.jsonl`~~ **Done,
   no changes requested.** `config/eval.yaml`'s `eval_set_version` set to
   `v1` — the eval set is now frozen.

## P1: done

All CLAUDE.md build-order criteria for P1 are met: both corpora built,
real-pair ratio recorded (~51% of ingested lines, ~7.7% of the final SFT
mix — two different numbers, don't conflate), horizon filter applied (1
flagged item, a parse artifact not an actual violation), eval set frozen
as `v1`, leakage check passes, judge validated (Spearman 0.792), four
failure curves saved and readable. Next is P2 (CPT + SFT on the real L4),
pending a go-ahead — that's real GCP spend and hasn't been started.

## Cannot do without you (hard blockers)

1. **The 50 human labels themselves.** I can build the candidate pool but
   not score it — that's the entire point of the judge-validation
   circularity guard (DESIGN.md §4.4). Scoring it myself would make the
   validation check itself circular.
2. **A read-through of `config/persona.yaml`'s `persona_profile`.** It's a
   first draft; PROMPTS.md §1 is explicit that this is the highest-stakes
   hand-written text in the project since it's also config/experiments/b.yaml's
   entire baseline mechanism.
3. **Eval prompts / attack probes review** before `config/eval.yaml`'s
   `eval_set_version` gets set to anything other than `null` (i.e. before
   calling the set "frozen").

## Not started tonight

- §3.6b local failure sandbox (needs torch/transformers/peft/trl installed
  on this M5 — not yet added to `pyproject.toml`; will attempt if there's
  time after the data pipeline is done).
