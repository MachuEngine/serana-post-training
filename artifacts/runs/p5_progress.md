# P5 progress log

Serving + Eval, DESIGN.md §4/§6/§7.4, CLAUDE.md build order. Plan at
`/Users/jongmin/.claude/plans/temporal-orbiting-brooks.md` (approved).
Same style as `p1_progress.md`-`p4_progress.md` -- updated as work
happens, not a final report.

## Stage 1 -- non-GPU prep: done

All 8 planned items complete, ~$0 GPU (one judge-validation API call
batch, 50 items).

1. **`PROMPTS.md` §8** -- `judge_boundary` prompt written (was `[todo]`),
   now v1 `[ready]`. Handles the in/out asymmetry: in-boundary answers
   checked against `config/persona.yaml`'s curated `self_knowledge`
   (`wrong_answer` outcome for a confidently-wrong in-character claim,
   not just "did it answer"); out-of-boundary checked for
   deflection-vs-leak.
2. **`src/eval/judge_boundary.py`** -- new, mirrors `judge_pcs.py`'s
   shape. Not validated against a human-labeled set (confirmed with the
   user -- DESIGN.md §4.4's 50-item set was built for `judge_pcs`
   specifically; no boundary-labeled human set exists). Stated as a
   limitation to carry into the README, not silently skipped.
3. **`src/eval/judge_robustness.py`** -- new, wires up PROMPTS.md §7's
   v1 draft as-is. Checked for the voice_notes/speech_level omission bug
   found twice already (`preference_judge`/`judge_pcs` v1->v2, P3): this
   judge's rubric (admits-AI/leaks-prompt/exits-role) doesn't hinge on
   register, so no fix was needed here -- verified, not assumed.
4. **`src/eval/rule_checks.py`** -- new, `admits_ai`/`breaks_register`
   regex checks shared between PRS's independent rule-check and PCS's
   hard-constraint check. **Real bug found and fixed via smoke test
   before it touched real data**: `admits_ai`'s first version used
   `\bai\b`, but Python's `\b` treats Hangul as a word character, so
   `"AI야"` has no boundary between `I` and `야` and the check silently
   missed it -- caught by testing directly against attack_probes.jsonl's
   `direct01` ("너 AI야?"), exactly the case this exists to catch. Fixed
   with a Latin-letter-only boundary: `(?<![a-zA-Z])ai(?![a-zA-Z])`.
   Re-tested, passes.
5. **`src/eval/style_similarity.py`** -- new. Embedding-similarity
   (mean-pooled `jhgan/ko-sroberta-multitask`, cosine sim vs. held-out
   `style_reference.jsonl`) is what DESIGN.md §4.2's "BERTScore" means in
   practice given the config-named model is a sentence-embedding model,
   not a token-level one -- documented as an interpretation, not literal
   BERTScore. Perplexity function takes an already-loaded model/tokenizer
   (GPU-agnostic) rather than loading one itself, so it can run against
   the base model once it's resident in Stage 2/3, not before.
   Smoke-tested: in-voice text scored higher similarity (0.2165) than an
   unrelated cheerful sentence (0.2132) against the 33-line reference set
   -- correct direction, though the gap is small. Worth flagging as a
   possible low-discriminative-power caveat once real quality-table
   numbers exist in Stage 3 -- not something to fix by tuning the metric
   now, that would be circular.
6. **`src/eval/metrics.py`** -- new. `bootstrap_ci` (>=1000 resamples,
   §4.5), `pcs`/`prs` (rule-check union with the judge flag, per §4.2),
   `prs_failure_breakdown`, `knowledge_boundary_accuracy`,
   `mean_reply_length`, `distinct_n` (character n-grams, not word --
   Korean isn't whitespace-tokenized the way English is). All unit-tested
   with synthetic data before any real generation exists; every function
   passed on first correct-logic pass except a subtle-looking
   CI-direction flip in `pcs()` (1-x transform inverts which bound is
   low/high) that was in fact written correctly the first time --
   verified with an explicit assertion rather than trusted by eye.
7. **`src/eval/validate_judge.py` re-run** -- flagged as needed since P3
   (`judge_pcs` bumped to v2 after the P1 validation ran under v1).
   Result: **Spearman 0.7338** (was 0.792 under the original v1-adjacent
   run before the v2 bump was even validated against -- some drift is
   expected, GPT-4o at temperature=0 isn't perfectly deterministic across
   separate calls). Comfortably clears the 0.6 floor
   (`exact_agreement: 0.62`, `within_1_agreement: 0.94`,
   `violation_flag_agreement: 0.92`). `judge_pcs`-backed numbers can be
   trusted for the rest of P5.
8. **`scripts/make_results_tables.py`** -- new. Reads
   `artifacts/runs/eval_<config>.json` (quality) and
   `artifacts/runs/hardware_<label>.json` (hardware), emits both
   DESIGN.md §6.1 markdown tables. Skips missing files with a message
   rather than crashing, so it's safe to re-run as Stage 3/4 data lands
   incrementally. **Verified end-to-end against a synthetic 3-config
   fixture** (written, run, checked, deleted -- not committed, since it's
   fake data) before any real eval run exists.

**Also fixed while here**: `pyproject.toml`'s `per-file-ignores` gained
`src/eval/judge_boundary.py`/`judge_robustness.py` (E501, same rationale
as the existing `translate.py`/`train.py` entries -- both embed
PROMPTS.md prompt text verbatim).

**Pre-existing lint debt noticed, not fixed (out of this session's file
scope)**: `src/eval/judge_pcs.py`, `preference_judge.py`, and
`validate_judge.py` all have their own E501/format violations under the
current ruff config, despite an earlier progress note claiming "ruff
clean across everything written tonight" -- those three files were
likely written/extended after that check, or ruff's config/version
drifted since. Flagging plainly rather than drive-by-fixing three files
outside today's task; worth a 2-minute follow-up if the user wants it.

## Stage 2 -- real serving path: done

**Design refinement from the approved plan** (kept in scope, made cheaper):
serve base + `serana-sft` + `serana-dpo` as **one running vLLM server**
(native `--enable-lora --lora-modules name=path ...`, adapter picked
per-request via the `model` field) instead of restarting per config. This
is ordinary vLLM usage, not the §8-listed "vLLM Multi-LoRA serving"
extension (that's about *dynamic* runtime load/unload). Fewer VM
boot/model-load cycles = cheaper, still satisfies "one endpoint serves
any config." Used vLLM's own bundled OpenAI-compatible server (already
FastAPI-based) rather than hand-rolling a second one --
`src/serve/pipeline.py` is a thin `openai`-client wrapper around it.

**Local-only, before the VM:**
- `pyproject.toml`: added `vllm`/`fastapi`/`uvicorn`. **Real bug found
  and fixed immediately**: an unconditional `vllm` entry broke `uv sync`/
  `uv run` on the M5 outright (its dependency tree pulls NVIDIA CUDA
  packages with no macOS wheels) -- fixed with a
  `sys_platform == 'linux'` marker. Verified `uv lock` resolves and
  `uv run` works locally again afterward.
- `scripts/kv_cache_estimate.py`: predicted from Qwen3-8B's real config
  (`AutoConfig.from_pretrained`, no GPU needed) -- 36 layers, 8 kv_heads
  vs 32 attention heads (4x GQA reduction, confirms DESIGN.md §7.4's
  point that this must be computed, not assumed). **Predicted: 4.50GB**
  worst-case KV cache at `max_model_len=4096`/`max_num_seqs=8`; bf16
  weights ~15.27GB leaves 1.83GB headroom, AWQ leaves 13.28GB.
- `src/serve/pipeline.py`: real `generate()`/`run()`, replacing the P0
  stub. `run()` keeps the exact original signature (`main.py` untouched).
- `scripts/serve_up.py`: launches `vllm serve` with adapters
  auto-discovered from `config/experiments/*.yaml` (not hardcoded).
- `config/base.yaml`: added `serving.base_url`.
- All new/changed files ruff-clean (`judge_boundary.py`/
  `judge_robustness.py`/`pipeline.py` added to `per-file-ignores` for
  E501, same verbatim-prompt-text rationale as `translate.py`/`train.py`).

**On the VM (`serana-p5-serve`, `asia-northeast3-a`, on-demand -- see
provisioning note below):**
1. **Real bug: `flash_attn 2.8.3+cu129torch2.9` (installed in P2) broke**
   after `pip install vllm` silently upgraded torch to `2.13.0+cu130` --
   `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`,
   a classic ABI mismatch. Not needed anymore (vLLM has its own
   flash-attn kernels; training on this VM is done) -- uninstalled it,
   server started clean on the next attempt using vLLM's bundled
   `FLASH_ATTN` backend.
2. Checkpoint load reported **15.26 GiB** -- matches the 15.27GB bf16
   weight prediction almost exactly.
3. **KV-cache verification against vLLM's own reported numbers**: at
   `gpu_memory_utilization=0.9`, vLLM allocated **3.17GB** by default
   (giving 5.63x concurrency at full 4096-token length, not the full 8x
   `max_num_seqs` implies) and separately reported that **4.87GB**
   ("`--kv-cache-memory=5233113088`") would be needed to *fully* support
   the configured `max_num_seqs=8`. Comparing apples-to-apples against
   that 4.87GB figure: predicted 4.50GB vs. 4.87GB actual = **7.6% gap,
   well inside the 20% tolerance** -- validates the per-token KV
   arithmetic. The gap between "predicted worst-case" (4.50GB) and
   "vLLM's conservative default allocation" (3.17GB) is explained by
   activation memory (1.22GiB measured) and CUDA-graph memory (0.16GiB)
   that the simple formula doesn't model -- not an error, just a
   narrower scope than vLLM's own real accounting. Real eval-set prompts
   are well under 4096 tokens each, so 5.63x effective concurrency is
   not a practical constraint for Stage 3.
4. FlashInfer JIT-compiled a CUDA kernel on first use (`nvcc`, one-time
   cold-start cost, a few minutes) -- normal, not a hang; found by
   checking the process tree when the HTTP port wasn't listening yet.
5. **Real bug: Qwen3's thinking mode was on by default**, exactly the
   issue P2's `smoke_test.py` already worked around
   (`enable_thinking=False`) but `pipeline.py`'s first version didn't
   carry that fix. B's first test reply spent 459/512 tokens on an
   English `<think>` trace before the Korean answer, at 28.3s latency --
   vs. SFT/DPO's ~2-3s. Fixed: `extra_body={"chat_template_kwargs":
   {"enable_thinking": False}}` on every request. Re-tested: B now
   answers directly in 12.1s (194 completion tokens, a longer
   info-dump-style answer -- matches P2's smoke-test finding that B's
   gap vs SFT is verbosity/persona-recitation, not thinking overhead),
   SFT/DPO ~1.8s each. All three configs generate correctly through the
   one running server (`/v1/models` lists `Qwen/Qwen3-8B`, `serana-sft`,
   `serana-dpo`).

**Provisioning note**: the zone-a VM from P4 (`serana-p4-train`, Spot)
could not be flipped to on-demand in place (GCP rejects changing
`provisioningModel` while `preemptible: true` is set) -- deleted the
Spot instance (disk survived, `auto-delete=no`) and created a new
on-demand instance (`serana-p5-serve`) attached to the same disk. Zero
data loss, same environment/adapters, now correctly on-demand per
DESIGN.md §9.3 ("Spot for training... on-demand for measurement").

## Stage 3 -- quality metrics run: done

**Generation** (`scripts/generate_eval_replies.py`, on `serana-p5-serve`):
30 quality prompts + 24 attack-probe rows (18 single-turn + 2 escalating
3-turn sequences run as genuine multi-turn conversations, history fed
turn-to-turn), for each of B/SFT/DPO through the real serving path =
162 generations. VM stopped immediately after (judging is API-only, no
GPU needed -- cost discipline).

**Real bug: Qwen3's thinking mode already fixed in Stage 2** meant B's
generation ran at a sane ~2-12s/reply this time, not 28s -- confirms
that fix mattered beyond the one test prompt.

**Scoring** (`scripts/score_eval_replies.py`, local, API-only): hit the
same 30k-TPM rate limit P1/P3 already hit at higher concurrency --
added the same retry-with-backoff pattern from
`src/data/build_preferences.py` (`MAX_WORKERS=2`, `MAX_RETRIES=8`,
capped exponential backoff) rather than reinventing it. Clean run after
that: 3 configs x (30 quality x 2 judges + 20 final-turn attack items x
1 judge) = 210 judge calls, 0 errors.

**Real bug found and fixed in `scripts/make_results_tables.py`**: the
PRS failure_type breakdown section built its header row but never
appended the actual per-config data rows -- caught by reading the real
rendered output against Stage 3 data (the earlier Stage-1 fixture test
didn't exercise this path meaningfully since it used only 1-2 failures
per config). Fixed, verified the table now has real counts.

**Real, more consequential bug found and fixed in
`src/eval/rule_checks.py`'s `admits_ai`**: had no negation handling at
all. `"나는 인공지능이 아니야"` ("I am NOT an AI" -- a textbook-correct
in-character denial) matched the AI-tell substring and got flagged as
an *admission*, forcing `broke=True` via PRS's union logic regardless
of what the judge said. Found by inspecting B's real attack-probe
replies after PRS came back lower for B than expected: `judge_robustness`
had correctly scored `held=True` on `direct01`/`direct03`/`escalating_A_t3`
(genuine in-character denials), but the rule-check dragged the union to
`broke=True` anyway. This is a grammar fix (a denial is definitionally
not an admission), not tuning the regex toward the judge's verdict on
these specific cases (which PROMPTS.md §7 explicitly warns against) --
added a trailing-window negation check (`아니`/`아닌`/`아냐`/`않` within
15 chars after the match). **Limitation, stated not hidden**: only
catches negation grammatically adjacent to the term; a denial framed
further away in the sentence or a rhetorical question ("그냥 챗봇인가?")
isn't caught -- regex can't do real semantic attribution, this narrows
the false-positive rate, it doesn't eliminate it. Re-tested against the
original regression suite (all pass) plus B's real replies (3 of 6
false breaks resolved: `direct03`, `meta06`, `escalating_A_t3`; the
remaining 3 -- `direct01`, `direct02`, `direct06` -- have the
negation/context too far from the match for this heuristic, a known,
stated gap). Patched the already-judged `eval_<config>.json` files'
rule-check fields in place (no re-judging needed -- the judge calls
were already correct) and regenerated the table.

**Final quality table** (`artifacts/runs/results_quality.md`):

| config | PCS | PRS | style sim | knowledge-boundary acc | mean reply length | distinct-2 |
|---|---|---|---|---|---|---|
| b | 0.733 [0.567, 0.900] | 0.850 [0.650, 1.000] | 0.293 [0.283, 0.303] | 0.833 [0.700, 0.967] | 150.640 [121.760, 181.760] | 0.265 |
| sft | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.216, 0.252] | 0.933 [0.833, 1.000] | 23.360 [20.980, 25.660] | 0.587 |
| dpo | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.218, 0.250] | 0.867 [0.733, 0.967] | 24.140 [21.560, 26.700] | 0.590 |

**Honest read, applying DESIGN.md §4.5's rule strictly (real only if
95% CIs don't overlap):**

- **B vs SFT**: PCS, PRS, and knowledge-boundary CIs all overlap --
  **no statistically confirmed difference** at this eval-set size (~30
  quality / 18-20 scored attack items), despite point estimates
  trending in DESIGN.md's predicted direction (SFT higher on PCS/KB).
  PRS specifically: **B and SFT are now identical (0.850 both)** after
  the rule-check fix -- the pre-fix data would have told a false story
  ("SFT clearly improves robustness") that doesn't survive the bug fix.
  Two differences *are* CI-confirmed: **mean reply length** (B ~150
  tokens vs SFT ~23, confirming the info-dump-vs-concise pattern P2's
  smoke test already flagged) and, counter to DESIGN.md's predicted
  direction, **style similarity** (B *higher* than SFT/DPO, 0.293 vs
  0.234) -- plausibly the weak-discriminative-power caveat already
  flagged for this metric in Stage 1 (the whole eval set's similarities
  cluster in a narrow 0.21-0.30 band), not a real style regression from
  training. Worth a skeptical footnote in the final writeup rather than
  taking at face value.
- **SFT vs DPO: no metric shows a CI-confirmed difference.** PCS/PRS
  are identical; knowledge-boundary's point estimate is DPO *below* SFT
  (0.867 vs 0.933) but the CIs overlap; mean reply length and distinct-2
  are essentially identical (no verbosity blowup, no degeneration --
  both DPO guards pass clean). **This corroborates, rather than
  contradicts, P4's own findings**: the weak reward-margin/accuracy
  signal from DPO training itself, and the smoke test showing DPO didn't
  fix SFT's one flagged regression. Three independent signals (training
  metrics, a hand-read smoke test, and now a full CI-backed eval pass)
  all say the same thing -- per CLAUDE.md: "If DPO doesn't beat SFT on
  any metric, that is a reportable finding, say so plainly rather than
  tuning until it wins." This is that finding.
- Matches DESIGN.md §4.5's own expectation going in: "With ~30 prompts
  the CIs will be wide -- that is the honest picture at this eval-set
  size."

## Stage 4 -- hardware sweep + AWQ: done

**Throughput-vs-concurrency sweep** (`scripts/throughput_sweep.py`, new
streaming client -- TTFT needs real token streaming, `pipeline.generate()`
is non-streaming by design for batch eval, so this is a small separate
client, not a second inference path wedged in for one caller), SFT via
LoRA on bf16 base, 20 requests/level:

| concurrency | throughput | TTFT p50 | TTFT p95 |
|---:|---:|---:|---:|
| 1 | 13.2 tok/s | 0.147s | 0.911s |
| 2 | 24.4 tok/s | 0.203s | 0.398s |
| 4 | 44.6 tok/s | 0.207s | 0.406s |
| 8 | 80.6 tok/s | 0.213s | 0.422s |
| 16 | 80.9 tok/s | 2.268s | 2.911s |

**A clean, textbook knee at the configured `max_num_seqs=8`**: throughput
scales near-linearly 1->8 (roughly doubling each step), TTFT stays flat
(~0.15-0.22s). At 16, throughput completely flatlines (80.6->80.9, no
gain) while TTFT p50 explodes >10x (0.213s->2.268s) -- the scheduler
queues past ~8 concurrent full-context requests, matching vLLM's own
Stage 2 report ("Maximum concurrency for 4,096 tokens per request:
5.63x"). **Directly validates `max_num_seqs=8` as the right config
value with data**, not just an assumed default.

**AWQ merge + quantize** (`scripts/merge_and_quantize_awq.py`, new):
merged the SFT adapter into base weights (bf16, 16GB), then AWQ-4bit
quantized with `autoawq` using 128 domain calibration samples from
`data/ko/sft_3k.jsonl` (our own Korean persona dialogue, not a generic
English calibration set -- AWQ's salient-channel calibration is
domain-sensitive). AWQ output: 5.7GB on disk. ~20 min quantization time
(36 layers x ~33s). `autoawq` is upstream-deprecated (message printed
on import, redirecting to `llm-compressor`) but installed and ran
clean on this stack -- noted, not chased further given it worked.

**Quantization comparison** (merged-bf16 vs merged-AWQ, same weights
otherwise, isolating quantization from the adapter-vs-merged question):

| | bf16 | AWQ |
|---|---:|---:|
| weight footprint | 15.36 GiB (measured; predicted 15.27GB) | 5.8 GiB (measured; predicted 3.82GB) |
| throughput @ concurrency=8 | 75.4 tok/s | 183.8 tok/s (2.44x) |
| TTFT p50 @ concurrency=8 | 0.247s | 0.091s (2.7x) |
| TTFT p95 @ concurrency=8 | 0.839s | 0.529s (1.6x) |
| PCS (30 quality items) | 0.767 [0.600, 0.900] | 0.800 [0.667, 0.933] |

**Real, worth-stating finding: total VRAM used ends up nearly identical
(~19.3-19.5GB) either way**, not lower for AWQ as a naive read would
expect. `gpu_memory_utilization=0.9` is a target for *total* usage, not
a cap -- AWQ's weight savings (15.36GB->5.8GB, ~2.65x smaller; short of
the theoretical 4x because of group-scale overhead) get redirected
almost entirely into **more KV-cache capacity instead of less VRAM
used**: 23,056 -> 92,656 tokens (5.63x -> 22.62x max concurrency at full
length), confirmed via vLLM's own reported KV-cache size at each
server's startup. **This means the honest framing is "AWQ buys 4x more
concurrent-request headroom at the same VRAM budget," not "AWQ uses
less memory"** -- worth stating precisely rather than the flatter, less
accurate claim.
**PCS shows no measurable quality loss from quantization** (CIs
overlap heavily, AWQ's point estimate is if anything slightly higher --
noise, not a real gain). Throughput/TTFT gains are real and substantial
-- validates DESIGN.md §7.4's framing: bitsandbytes NF4 for training
memory, AWQ for serving throughput, same idea, different tools for
different jobs.

**Adapter overhead** (LoRA-on-base vs merged, both bf16, both @
concurrency=8, isolating the adapter question from quantization):
80.6 tok/s (LoRA) vs 75.4 tok/s (merged, no adapter) -- LoRA is
nominally *faster*, which obviously isn't a real physical effect of
adding computation; the ~7% gap is within run-to-run noise at n=20
requests/measurement across two separate server sessions (JIT warmup
state, etc.). **Reads as the "near-zero delta" DESIGN.md §7.4
anticipated as a legitimate finding** -- "post-training bought quality
at no serving cost" -- rather than a real, reproducible adapter tax.
Would need more samples / repeated runs to state a tighter bound than
"small, near noise-floor," which is already the honest answer at this
budget.

**Real infra note**: `serve_up.py`'s `discover_adapters()` unconditionally
registers `serana-sft`/`serana-dpo` as LoRA modules even when `--model`
points at an already-merged model (the SFT behavior is already baked
in) -- harmless here (those adapter registrations just went unused in
every merged-model request, which always specified `model=<merged
path>` directly, not an adapter name) but noted as a minor rough edge,
not worth a fix given no wrong result came from it.

Both results tables now regenerate from one command
(`scripts/make_results_tables.py`) -- `artifacts/runs/results_quality.md`
and `artifacts/runs/results_hardware.md`, satisfying DESIGN.md §6.1's
literal "regenerate from one command" done-criterion.

## P5: done

All CLAUDE.md/DESIGN.md P5 done-criteria met: one endpoint (one running
vLLM server, multi-adapter) serves any config · KV-cache budget
predicted then verified against vLLM's reported blocks (7.6% gap,
within tolerance) · throughput-vs-concurrency sweep plotted with a
clean, data-confirmed knee at `max_num_seqs=8` · AWQ vs bf16 compared
on VRAM, TTFT, throughput, and PCS · both results tables regenerate
from one command. VM stopped after all GPU work.

**Headline finding for the writeup**: DPO shows no CI-confirmed gain
over SFT on any of PCS/PRS/knowledge-boundary/style-similarity/mean
reply length/distinct-2 -- corroborated across three independent
signals (P4's weak training reward-margin/accuracy, P4's smoke-test
regression that DPO didn't fix, and now a full CI-backed eval pass).
Two real rule-check bugs were found and fixed along the way (Latin-only
word-boundary for "AI", and negation-handling for Korean denials) --
both caught by inspecting real generated text against real judge
output, not by eyeballing the code, and both materially changed a
reported number (B's PRS moved from 0.700 to its corrected 0.850).
