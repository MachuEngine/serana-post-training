# P6 progress log

Ship: git, HF Hub, Gradio demo, README. Plan at
`/Users/jongmin/.claude/plans/temporal-orbiting-brooks.md` (approved).
Same style as p1-p5_progress.md.

## Stage 1 -- get the whole project into git: done

**Real discovery**: despite P1-P5 being fully done, nothing but
README/.gitignore edits had ever been committed -- confirmed via
`git status`/`git log` at the start of this stage. `CLAUDE.md`,
`DESIGN.md`, `PROMPTS.md`, all of `src/`, `config/`, `scripts/`,
`data/eval/` were still untracked after five phases of work.

Confirmed with the user: commit directly to `main` (no PR, solo
portfolio repo). Include `.claude/` + `HARNESS_ENGINEERING.md`;
exclude `LEARNING.md` (personal notes) via `.gitignore`.

**Also fixed while here**: `.gitignore`'s blanket `/artifacts/runs/*`
exclusion was hiding all P1-P5 progress logs and both results tables --
exactly the predicted-vs-measured evidence this whole project is about.
Checked actual size first (96KB for the .md/report files, 364KB
including the raw eval-generation JSON dumps -- genuinely small, no
"repo bloat" concern) and removed the blanket exclusion, keeping only
`artifacts/lora/`, `artifacts/diagnostics/`, `artifacts/merged/`
ignored (those hold real model binaries, correctly excluded -- adapters
ship via HF Hub, not git).

114 files, one commit, pushed to `origin/main`
(`ee16968..1181e5c`). `.env` confirmed not staged (still gitignored).

## Stage 2 -- back up serana-sft: done

`serana-sft` existed in exactly one place before this: the stopped
`serana-p5-serve` VM's disk (not GCS -- checked, only `serana-dpo` was
there; not local; nowhere else). Real data-loss risk if that disk were
ever deleted.

Started the VM, `gcloud compute scp --recurse` down to
`artifacts/lora/serana-sft/` locally (41MB), `gcloud storage cp` to
`gs://serana-post-training-ann10266/artifacts/lora/serana-sft/` for a
second copy matching where `serana-dpo` already lives, stopped the VM.
No GPU compute used -- disk I/O only. `serana-sft` now exists in three
places: VM disk, local, GCS.

## Stage 3 -- HF Hub adapter upload: done

`scripts/upload_to_hub.py` -- new. Username resolved from the token via
`whoami()` (`machu8`), not hardcoded or asked for separately. Real model
cards written (replacing PEFT's auto-generated "[More Information
Needed]" stub -- checked the downloaded `serana-sft/README.md` first
and confirmed it was exactly that stub), each with the IP/licensing
note, usage snippet, and a pointer to the repo's results tables.
`serana-dpo`'s card explicitly states the no-measurable-gain finding --
shipped as the honest result, not hidden because it "lost."

`pyproject.toml` gained `gradio`/`huggingface-hub` as explicit
dependencies (huggingface-hub was already available transitively via
transformers; gradio genuinely new, needed for Stage 4 too). `uv lock`
+ `uv sync` both clean locally.

Uploaded: `adapter_config.json`, `adapter_model.safetensors`,
`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`,
`run_report.json` (kept -- documents the real predicted-vs-measured
training numbers, genuinely useful on a model card), `README.md`.
Deliberately excluded `training_args.bin` (pickled Trainer object, not
human-legible, not needed to load/use the adapter).

Verified without loading full model weights (lighter than a real
`PeftModel.from_pretrained` smoke test, which would need the 8B base
downloaded): `list_repo_files` confirms all expected files present on
both `machu8/serana-sft` and `machu8/serana-dpo`;
`adapter_config.json`'s `base_model_name_or_path` correctly resolves to
`Qwen/Qwen3-8B` on both.

- https://huggingface.co/machu8/serana-sft
- https://huggingface.co/machu8/serana-dpo

## Stage 4 -- Gradio demo (`demo/app.py`): code done, not yet deployed

Design: ZeroGPU Space, plain `transformers`+`peft` (not vLLM/AWQ) --
mirrors `scripts/smoke_test.py`'s already-proven pattern rather than
re-fighting the vLLM/CUDA-version fragility this project hit
repeatedly on the GCP VM (flash-attn ABI breaks, torch version
upgrades from `pip install vllm`, etc. -- P5 Stage 2's log). One base
model loaded once; `serana-sft`/`serana-dpo` loaded as two named PEFT
adapters (`PeftModel.from_pretrained(..., adapter_name="sft")` then
`model.load_adapter(..., adapter_name="dpo")`), switched via
`model.set_adapter()` -- "B" via `model.disable_adapter()`. Same
single-pipeline principle as P5's vLLM multi-adapter server, just
PEFT's version of it.

UI: one input, three labeled outputs (B/SFT/DPO) side by side, not a
config-selector chatbot -- the project's actual deliverable is the
comparison itself, matched directly rather than buried behind a
dropdown. Greedy decoding (matches eval convention, so the demo
reflects the numbers in `results_quality.md`, not different behavior).
`enable_thinking=False` carried over from the same real bug P5 Stage 2
found (Qwen3's thinking mode burns the token budget otherwise).

`demo/requirements.txt` -- new, plain pip file (Spaces doesn't read
`pyproject.toml`).

**Verified so far** (can't test the `@spaces.GPU`-decorated path or
real generation locally -- no GPU here, and `spaces` only works inside
an actual Space): ruff clean; config loading + system-prompt
construction produces the exact same string as
`scripts/smoke_test.py`'s already-proven version (tested in isolation).
Real end-to-end verification happens in Stage 6 after deployment.

## Stage 6 -- deploy to HF Spaces: skipped (real blocker, not a bug)

`api.create_repo(repo_type="space", space_hardware="zero-a10g")` failed
with a real, informative `402 Payment Required`: ZeroGPU Spaces require
either a HF PRO subscription or a community grant (new accounts wait
~30 days before they're eligible to apply). Not something to route
around silently.

Talked through it with the user: a Spaces demo only serves other
people trying it without setup -- it adds no capability neither of us
already has (repo + HF Hub adapters + results tables already make the
work fully inspectable and runnable). Decided: skip the paid/wait path
for now. `demo/app.py`/`demo/requirements.txt` stay in the repo as-is
(genuinely useful, runs on any CUDA machine), README's "Live demo"
link replaced with "run it yourself" instructions. Revisit if a real
need for a zero-setup demo comes up later (e.g. showing it to someone
directly).

## Stage 5 -- README: done

Rewrote the stub. Both results tables embedded directly (not just
linked) -- quality table as-is from `results_quality.md`, hardware
table extended beyond what Stage 4's scripts wrote: added
`hardware_training_*.json` entries for CPT/SFT/DPO (predicted-vs-measured
VRAM/step-time/cost pulled from `p2_progress.md`/`p4_progress.md`) since
DESIGN.md §6.1 wants "per training stage *and* per serving config" and
the auto-generated table only had serving rows from Stage 4. Regenerated
via `scripts/make_results_tables.py` -- now genuinely complete.

Headline finding stated as its own section, not buried in a table.
Real-pair ratios (51.4% / 7.7%, kept distinct per `p1_progress.md`'s own
note not to conflate them). Circularity guard explained in prose, not
just cited. Hardware section keeps the "more KV-cache headroom, not
less VRAM used" precision from `p5_progress.md` rather than the flatter,
less accurate claim.
