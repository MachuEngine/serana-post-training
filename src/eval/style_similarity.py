"""Style similarity -- DESIGN.md §4.2: "BERTScore(generated, held-out
reference) + perplexity under a reference LM."

Two independent signals:

1. **Embedding similarity**, using `jhgan/ko-sroberta-multitask`
   (`base.yaml`'s `eval.style_embedding_id`). That model is a
   sentence-embedding model (mean-pooled sentence vectors), not a
   token-level contextual-embedding model in the original BERTScore
   sense -- so what's implemented here is cosine similarity between the
   generated reply's sentence embedding and the held-out
   `style_reference.jsonl` lines' embeddings, which is the practical
   reading of "BERTScore" given the config-specified model. Runs on
   CPU; the model is ~340MB, fine for the eval set's size.
2. **Perplexity under a reference LM.** DESIGN.md doesn't name a
   separate reference LM, so this uses the base model itself (already
   resident during serving, per the P5 plan) -- kept GPU-agnostic here
   by taking an already-loaded `model`/`tokenizer` rather than loading
   one internally, so it runs against whatever's loaded (Qwen3-0.6B for
   a local smoke test, Qwen3-8B for the real P5 numbers).

Both return raw numbers; direction and thresholds are the caller's
concern (DESIGN.md §4.2: higher embedding-similarity / lower Δppl is
better -- Δppl meaning perplexity relative to a config-vs-config
baseline, not an absolute target).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

STYLE_EMBEDDING_ID = yaml.safe_load(open("config/base.yaml"))["eval"]["style_embedding_id"]

_embed_tokenizer = None
_embed_model = None


def _get_embedder():
    global _embed_tokenizer, _embed_model
    if _embed_model is None:
        _embed_tokenizer = AutoTokenizer.from_pretrained(STYLE_EMBEDDING_ID)
        _embed_model = AutoModel.from_pretrained(STYLE_EMBEDDING_ID)
        _embed_model.eval()
    return _embed_tokenizer, _embed_model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed(texts: list[str]) -> torch.Tensor:
    tokenizer, model = _get_embedder()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


def style_embedding_similarity(text: str, reference_lines: list[str]) -> float:
    """Mean cosine similarity between `text` and every line in
    `reference_lines` (the held-out style_reference.jsonl set). Mean
    rather than max -- a reply that's close to only one reference line
    by chance shouldn't score as "in her style" overall."""
    if not reference_lines:
        raise ValueError("reference_lines must be non-empty")
    text_emb = _embed([text])
    ref_embs = _embed(reference_lines)
    sims = (text_emb @ ref_embs.T).squeeze(0)
    return sims.mean().item()


def perplexity(text: str, model, tokenizer, device: str = "cpu") -> float:
    """Perplexity of `text` under an already-loaded causal LM. Caller
    owns the model/tokenizer lifecycle (see module docstring)."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError("text too short to compute perplexity (need >=2 tokens)")
    with torch.no_grad():
        out = model(**inputs, labels=input_ids)
    return torch.exp(out.loss).item()
