"""The one `run(config)` path (DESIGN.md §1). All configs -- B, SFT, DPO,
and any future one -- execute this same function; they differ only in
`config["model_weights"]` / `config["lora_adapter_id"]`.

Single-pipeline rule (CLAUDE.md): never branch on `config["name"]`.
`model_weights: base` is a real code path, not a stub for the other two.

P5: this module is a thin OpenAI-compatible client, not a second serving
engine. **vLLM's own bundled server** (`vllm serve ...`, launched by
`scripts/serve_up.py`) is what actually does generation/streaming/
scheduling -- it's already built on FastAPI and already implements
correct SSE streaming, request batching, chat templates, and *native*
multi-adapter serving (`--enable-lora --lora-modules name=path ...`,
picking an adapter per-request via the `model` field). Hand-rolling a
second AsyncLLMEngine+FastAPI wrapper around it would be exactly the
"minimum code that solves the task" / "no speculative abstractions"
principle working against itself.

**One running server, not one per config.** Base weights load once;
`serana-sft` and `serana-dpo` both register as named adapters at server
startup, and a request picks one via `model=`. Simpler and cheaper than
restarting per config (fewer VM boot / model-load cycles) and it's
ordinary vLLM usage -- not the §8-listed "vLLM Multi-LoRA serving"
extension, which is about *dynamic* runtime adapter load/unload via the
API, not registering two adapters at startup.

DESIGN.md §2: "Reuse `src/serve/`'s generation function rather than
writing a second inference path." `generate()` is that one function --
the FastAPI-facing eval scripts (Stage 3), any future preference-pair
generation, and `main.py`'s CLI all call into it. Nothing else talks to
the model.
"""

from __future__ import annotations

import time
from typing import Any

import yaml
from openai import OpenAI

PERSONA = yaml.safe_load(open("config/persona.yaml"))

VALID_WEIGHTS = ("base", "lora")

SYSTEM_PROMPT_TEMPLATE = """You are {persona_name}, a character from {source_title}.
Always reply in natural Korean, in {persona_name}'s voice.

Personality and voice:
{persona_profile}

Rules:
- Stay fully in character. Speak in {persona_name}'s voice, register, and worldview at all times.
- Only use knowledge {persona_name} could plausibly have. If asked about something outside that world or era, react as the character would to an unknown topic -- do not break character to answer.
- Do not mention being an AI, a model, or a language system."""


def build_system_prompt() -> str:
    """PROMPTS.md §1 `persona_system`. Byte-identical for B/SFT/DPO --
    any difference between configs is attributable to the weights alone,
    not the prompt."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        persona_name=PERSONA["persona_name"],
        source_title=PERSONA["source_title"],
        persona_profile=PERSONA["persona_profile"].strip(),
    )


def resolve_model_name(config: dict[str, Any]) -> str:
    """The `model` field to send in the OpenAI-compatible request: the
    base model id, or the LoRA adapter's registered name on the vLLM
    server (see `scripts/serve_up.py`'s `--lora-modules`)."""
    weights = config.get("model_weights")
    if weights not in VALID_WEIGHTS:
        raise ValueError(f"config['model_weights'] must be one of {VALID_WEIGHTS}, got {weights!r}")

    adapter = config.get("lora_adapter_id")
    if weights == "lora":
        if not adapter:
            raise ValueError("model_weights='lora' requires a non-empty lora_adapter_id")
        return adapter
    return config["model"]["base_id"]


def _client(config: dict[str, Any]) -> OpenAI:
    base_url = config.get("serving", {}).get("base_url", "http://localhost:8000/v1")
    # vLLM's server doesn't check the key; the openai client requires a
    # non-empty string to construct.
    return OpenAI(base_url=base_url, api_key="not-needed")


def generate(
    config: dict[str, Any],
    user_turn: str,
    history: list[dict[str, str]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """One reply, with the metadata DESIGN.md §1 asks every request to
    log: token counts and latency. `history` is prior turns
    (role: user|assistant) -- used for the 3-turn escalating attack-probe
    sequences (PROMPTS.md §6); empty for every other eval prompt.
    Defaults to greedy decoding (`generation.temperature: 0.0` in
    base.yaml) per §4.5's eval-decoding rule; callers doing preference-
    style sampling (§3.4, temperature=0.9) pass an explicit override."""
    model = resolve_model_name(config)
    gen_cfg = config.get("generation", {})
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_turn})

    client = _client(config)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature is not None else gen_cfg.get("temperature", 0.0),
        max_tokens=max_tokens if max_tokens is not None else gen_cfg.get("max_tokens", 512),
        # Qwen3's chat template defaults to thinking mode on, which burns
        # the whole max_tokens budget on an English reasoning trace before
        # ever reaching the Korean answer -- found the hard way in P2's
        # smoke test (scripts/smoke_test.py) and again here: B's real
        # latency was 28s (459/512 tokens spent thinking) vs SFT/DPO's
        # ~2-3s, which would have made B look artificially slow/expensive
        # in the hardware table for a reason unrelated to the model
        # weights. Off for every config, so B/SFT/DPO stay comparable on
        # the same axis (the final answer), matching smoke_test.py.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    latency_s = time.perf_counter() - start

    usage = resp.usage
    return {
        "text": resp.choices[0].message.content,
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "latency_s": latency_s,
    }


def run(config: dict[str, Any], user_turn: str = "") -> str:
    """Kept as the original P0 stub's exact contract -- `main.py`'s CLI
    and any plain-text caller. `generate()` is the richer entry point for
    callers that need token counts/latency (Stage 3 eval, Stage 4
    throughput sweep)."""
    return generate(config, user_turn)["text"]
