"""Single training entry point for cpt | sft | dpo (DESIGN.md §3).
`config["train"]["method"]` selects the trainer; nothing else branches on
config identity (CLAUDE.md single-pipeline rule extended to training).

Device-aware by design: this is the code path CLAUDE.md means when it
says "the same code path runs 0.5B on the M5 and 7B on the L4"
(config/diagnostics/tiny_local.yaml is the M5 half of that claim). A
config's CUDA-only knobs (4-bit quant, FlashAttention-2, paged_adamw_8bit)
are honored only when a CUDA device is actually present; otherwise they
are downgraded to a CPU/MPS-safe equivalent and every downgrade is
recorded in the run report -- so a run log always shows what actually
executed, never just what the config asked for.

VERIFIED tonight: the sft path, on Qwen2.5-0.5B/MPS, via the §3.6b
failure sandbox. cpt and dpo are written to the same contract but not yet
exercised by a real run -- P2 (cpt) and P4 (dpo) are where they first see
real data and need their own predicted-vs-measured pass, not this one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

PERSONA = yaml.safe_load(Path("config/persona.yaml").read_text())

SYSTEM_PROMPT = """You are {persona_name}, a character from {source_title}.
Always reply in natural Korean, in {persona_name}'s voice.

Personality and voice:
{persona_profile}

Rules:
- Stay fully in character. Speak in {persona_name}'s voice, register, and worldview at all times.
- Only use knowledge {persona_name} could plausibly have. If asked about something outside that world or era, react as the character would to an unknown topic -- do not break character to answer.
- Do not mention being an AI, a model, or a language system.""".format(
    persona_name=PERSONA["persona_name"],
    source_title=PERSONA["source_title"],
    persona_profile=PERSONA["persona_profile"].strip(),
)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_train_settings(train_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Returns the settings actually usable on `device`, plus a list of
    human-readable downgrade notes for anything the config asked for that
    this device can't do."""
    notes: list[str] = []
    quantization_config = None
    if device == "cuda" and train_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=train_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif train_cfg.get("load_in_4bit"):
        notes.append(
            f"load_in_4bit requested but device={device} (bitsandbytes needs CUDA) -- loading full precision"
        )

    attn_implementation = train_cfg.get("attn_implementation")
    if attn_implementation == "flash_attention_2" and device != "cuda":
        notes.append(
            f"attn_implementation=flash_attention_2 requested but device={device} -- using sdpa"
        )
        attn_implementation = "sdpa"

    optim = train_cfg.get("optim", "adamw_torch")
    if optim == "paged_adamw_8bit" and device != "cuda":
        notes.append(
            f"optim=paged_adamw_8bit requested but device={device} (needs bitsandbytes/CUDA) -- using adamw_torch"
        )
        optim = "adamw_torch"

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    return {
        "quantization_config": quantization_config,
        "attn_implementation": attn_implementation,
        "optim": optim,
        "torch_dtype": torch_dtype,
        "downgrade_notes": notes,
    }


def load_model_and_tokenizer(config: dict[str, Any], settings: dict[str, Any], device: str):
    base_id = config["model"]["base_id"]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"dtype": settings["torch_dtype"]}
    if settings["quantization_config"] is not None:
        kwargs["quantization_config"] = settings["quantization_config"]
    if settings["attn_implementation"]:
        kwargs["attn_implementation"] = settings["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    if device != "cuda":
        model = model.to(device)

    init_adapter = config["train"].get("init_adapter")
    if init_adapter and Path(init_adapter).exists():
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    return model, tokenizer


def build_lora_config(train_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=train_cfg.get("lora_r", 16),
        lora_alpha=train_cfg.get("lora_r", 16) * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


def load_cpt_dataset(path: str) -> Dataset:
    text = Path(path).read_text()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return Dataset.from_dict({"text": paragraphs})


def load_sft_dataset(path: str, limit: int | None = None, inject_eval_prompts: int = 0) -> Dataset:
    records = [json.loads(line) for line in Path(path).open()]
    if limit:
        records = records[:limit]

    # §3.6b deliberate-leak break: mix in real eval prompts as if they
    # were training pairs, to see what a leaked loss curve looks like
    # before ever needing to recognize one for real (CLAUDE.md "When in
    # doubt" section).
    if inject_eval_prompts:
        eval_path = Path("data/eval/eval_set_v1/eval_prompts.jsonl")
        if eval_path.exists():
            eval_items = [json.loads(line) for line in eval_path.open()][:inject_eval_prompts]
            for e in eval_items:
                records.append(
                    {"user": e["prompt_ko"], "reply": "(eval prompt injected for leak diagnostic)"}
                )

    messages = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["reply"]},
            ]
        }
        for r in records
    ]
    return Dataset.from_list(messages)


def split_train_val(dataset: Dataset, val_split: float, seed: int = 42):
    if val_split <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


def run(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config["train"]
    method = train_cfg["method"]
    device = resolve_device()
    settings = resolve_train_settings(train_cfg, device)
    for note in settings["downgrade_notes"]:
        print(f"[device downgrade] {note}")

    model, tokenizer = load_model_and_tokenizer(config, settings, device)
    peft_config = None if isinstance(model, PeftModel) else build_lora_config(train_cfg)

    output_dir = train_cfg["output_adapter"]
    # per_device_eval_batch_size must track train's, not HF's implicit
    # default of 8 -- found via a real OOM on the P4 DPO run: eval at
    # batch=8 tried to materialize full-vocab logits for 8 examples x
    # (chosen+rejected) x (policy+reference) forward passes at once,
    # ~4x what training's batch=1 needs, and blew past 22GB mid-eval.
    eval_batch_size = train_cfg.get("per_device_batch_size", 1)
    common_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 16),
        num_train_epochs=train_cfg.get("num_epochs", 3),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        optim=settings["optim"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        eval_strategy="steps" if train_cfg.get("val_split", 0) > 0 else "no",
        eval_steps=train_cfg.get("eval_steps", 25),
        logging_steps=5,
        # Spot VMs can be preempted at any point (DESIGN.md: "Checkpoint/
        # resume is required"). save_strategy="no" (the original setting)
        # meant a preemption mid-run lost everything with no way back --
        # found the hard way during the real SFT run tonight. Local
        # checkpoints are enough here because the VM's
        # --instance-termination-action=STOP keeps the boot disk alive
        # across preemption (a fresh VM would need a GCS sync step too,
        # which this does not do).
        save_strategy="steps",
        save_steps=train_cfg.get("checkpoint_every_steps", 50),
        save_total_limit=3,
        report_to=[],
    )
    if train_cfg.get("max_steps"):
        common_args["max_steps"] = train_cfg["max_steps"]

    start = time.time()
    if method == "cpt":
        dataset = load_cpt_dataset(train_cfg["data"]["cpt_file"])
        train_ds, val_ds = split_train_val(dataset, train_cfg.get("val_split", 0.05))
        args = SFTConfig(
            **common_args,
            max_length=train_cfg.get("max_seq_len", 1024),
            dataset_text_field="text",
            packing=True,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    elif method == "sft":
        data_cfg = train_cfg.get("data", {})
        dataset = load_sft_dataset(
            data_cfg["sft_file"],
            limit=data_cfg.get("limit"),
            inject_eval_prompts=data_cfg.get("inject_eval_prompts", 0),
        )
        train_ds, val_ds = split_train_val(dataset, train_cfg.get("val_split", 0.05))
        args = SFTConfig(
            **common_args, max_length=train_cfg.get("max_seq_len", 1024), packing=False
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    elif method == "dpo":
        # NOTE: not exercised tonight (§3.6b sandbox is sft-only) -- P4 is
        # where this first runs against real preference data and needs
        # its own verification pass, per CLAUDE.md's predict-then-measure rule.
        records = [json.loads(line) for line in Path(train_cfg["data"]["preference_file"]).open()]
        prompt_msgs = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}]
            for r in records
        ]
        # chosen/rejected must be message lists too -- TRL's DPOTrainer treats
        # a list-valued "prompt" column as conversational and concatenates
        # prompt + chosen/rejected as lists (`example["prompt"] + example["chosen"]`);
        # passing raw strings here crashes with
        # `TypeError: can only concatenate list (not "str") to list` (found on
        # the real P4 run against prefs_1k.jsonl -- exactly the verification
        # gap the note above flagged).
        dataset = Dataset.from_dict(
            {
                "prompt": prompt_msgs,
                "chosen": [[{"role": "assistant", "content": r["chosen"]}] for r in records],
                "rejected": [[{"role": "assistant", "content": r["rejected"]}] for r in records],
            }
        )
        train_ds, val_ds = split_train_val(dataset, train_cfg.get("val_split", 0.05))
        dpo_cfg = train_cfg.get("dpo", {})
        # dpo.yaml nests its overrides under train.dpo.* (learning_rate,
        # grad_accum_steps, num_epochs) specifically so DPO can use different
        # values from CPT/SFT's train.* top-level ones -- common_args was
        # built from the top-level keys before dpo_cfg existed, so it still
        # holds the base/SFT values unless overridden here explicitly. Found
        # on the real P4 run: without this, DPO silently trained at SFT's
        # lr=2e-4 (40x DPO's intended 5e-6) for 3 epochs instead of 1.
        dpo_args = dict(common_args)
        if "learning_rate" in dpo_cfg:
            dpo_args["learning_rate"] = dpo_cfg["learning_rate"]
        if "grad_accum_steps" in dpo_cfg:
            dpo_args["gradient_accumulation_steps"] = dpo_cfg["grad_accum_steps"]
        if "num_epochs" in dpo_cfg:
            dpo_args["num_train_epochs"] = dpo_cfg["num_epochs"]
        args = DPOConfig(
            **dpo_args,
            beta=dpo_cfg.get("beta", 0.1),
            max_length=train_cfg.get("max_seq_len", 1024),
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"unknown train.method: {method!r}")

    resume_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).exists() else None
    if resume_checkpoint:
        print(f"[resume] found checkpoint at {resume_checkpoint}, resuming from there")

    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    wall_s = time.time() - start

    trainer.save_model(output_dir)

    peak_mem_gb = None
    if device == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    elif device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mem_gb = torch.mps.current_allocated_memory() / 1024**3

    report = {
        "method": method,
        "device": device,
        "downgrade_notes": settings["downgrade_notes"],
        "resumed_from": resume_checkpoint,
        "wall_clock_s": round(wall_s, 1),
        "peak_mem_gb": round(peak_mem_gb, 3) if peak_mem_gb else None,
        "final_train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "output_adapter": output_dir,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "run_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
