"""P6: pushes the shipped LoRA adapters (`serana-sft`, `serana-dpo`) to
HF Hub, each with a real model card replacing PEFT's auto-generated
"[More Information Needed]" stub. Username is read from the token via
`whoami()`, not hardcoded or asked for separately.

Only the two shipped adapters -- not `serana-cpt-intermediate`
(CPT-only, folded into SFT, never a served config per CLAUDE.md) and
not the P5 Stage-4 merged/AWQ models (hardware-comparison artifacts,
not part of the B/SFT/DPO triplet).

Usage: `uv run python3 scripts/upload_to_hub.py`
Requires `HF_TOKEN` in `.env` (write access).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# Files that make an adapter loadable + a couple of docs worth keeping;
# training_args.bin (a pickled Trainer object, not human-legible and not
# needed to load/use the adapter) is deliberately left out.
UPLOAD_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "run_report.json",
)

ADAPTERS = {
    "serana-sft": {
        "stage": "SFT",
        "parent": None,
        "description": (
            "Continues the base model with QLoRA SFT on ~3k Korean in-character "
            "exchanges (real wiki-recorded pairs + synthetic continuations, "
            "7.7% real / 92.3% synthetic in the final mix -- see the repo's "
            "results tables for the full breakdown)."
        ),
    },
    "serana-dpo": {
        "stage": "DPO",
        "parent": "serana-sft",
        "description": (
            "Continues `serana-sft` with DPO on 837 RLAIF preference pairs "
            "(judge picks between two SFT-sampled replies per prompt). "
            "**Note**: in this project's own evaluation, DPO showed no "
            "CI-confirmed quality gain over SFT on any metric -- see the repo's "
            "results tables. Shipped anyway, as the honest result of the "
            "pipeline, not because it won."
        ),
    },
}

REPO_URL = "https://github.com/MachuEngine/serana-post-training"


def build_model_card(adapter_id: str, meta: dict, base_model: str) -> str:
    parent_line = ""
    if meta["parent"]:
        parent_url = f"https://huggingface.co/{{username}}/{meta['parent']}"
        parent_line = (
            f"\nBuilt on top of [`{{username}}/{meta['parent']}`]({parent_url}), "
            "not the base model directly.\n"
        )
    return f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- persona
- korean
license: other
---

# {adapter_id}

> "Serana" and The Elder Scrolls are property of Bethesda/ZeniMax. This
> adapter is a non-commercial engineering portfolio artifact, not an
> official product, and is not affiliated with Bethesda/ZeniMax.

Stage: **{meta["stage"]}**, part of a CPT -> SFT -> DPO post-training
pipeline for persona consistency, built end-to-end on one 24GB GPU
(NVIDIA L4). Full pipeline, data sourcing, evaluation design, and GPU
engineering: [{REPO_URL}]({REPO_URL})

{meta["description"]}
{parent_line}
## Usage

This is a **LoRA adapter**, not a standalone model -- it requires the
base model (`{base_model}`) to be loaded first.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{{username}}/{adapter_id}")
```

See the repo's `PROMPTS.md` §1 for the exact persona system prompt this
was trained/evaluated with -- results are only comparable when it's
reused as-is.

## Results

Both the quality table (PCS / PRS / style similarity / knowledge-boundary
accuracy / mean reply length, all with 95% CIs) and the hardware table
(VRAM, KV-cache, throughput, AWQ vs bf16) are in the repo's
`artifacts/runs/results_quality.md` and `results_hardware.md`, generated
directly from this adapter's real eval runs.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    print(f"authenticated as: {username}")

    for adapter_id, meta in ADAPTERS.items():
        local_dir = Path("artifacts/lora") / adapter_id
        if not local_dir.exists():
            raise SystemExit(f"{local_dir} not found -- run Stage 2 (backup) first")

        adapter_config = json.loads((local_dir / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]

        repo_id = f"{username}/{adapter_id}"
        print(f"creating/updating {repo_id} (base: {base_model})...")
        api.create_repo(repo_id, repo_type="model", exist_ok=True)

        card = build_model_card(adapter_id, meta, base_model).format(username=username)
        (local_dir / "README.md").write_text(card)

        for filename in (*UPLOAD_FILES, "README.md"):
            path = local_dir / filename
            if not path.exists():
                print(f"  skip {filename} (not present)")
                continue
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  uploaded {filename}")

        print(f"done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
