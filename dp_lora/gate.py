"""
Held-out perplexity gate with a degeneracy check.

The gate's job is to make a *silent* failure impossible:
- it reports perplexity on a fixed held-out set, and
- it reports the average lora_B norm and flags ``lora_B_degenerate`` when every
  adapter's B matrix is still zero — the unmistakable signature of a no-op run.

The decision rule that matters is NOT an absolute perplexity threshold (a
degenerate model can sit below any threshold you pick); it is *improvement over
the base model on the identical eval set*. ``compare_to_base`` enforces that.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch


@torch.no_grad()
def perplexity(model, tokenizer, texts: List[str], device: str, max_length: int = 128,
               batch_size: int = 4, max_batches: int = 50) -> float:
    model.eval()
    enc = tokenizer(texts, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]
    vocab = model.config.vocab_size
    total_loss, total_tok = 0.0, 0
    n = min((len(ids_all) + batch_size - 1) // batch_size, max_batches)
    for i in range(n):
        s, e = i * batch_size, min((i + 1) * batch_size, len(ids_all))
        ids = ids_all[s:e].clamp(0, vocab - 1).to(device)
        mask = mask_all[s:e].to(device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        ntok = int((labels != -100).sum().item())
        if ntok:
            total_loss += out.loss.item() * ntok
            total_tok += ntok
    model.train()
    return math.exp(total_loss / max(total_tok, 1)) if total_tok else float("inf")


def lora_diagnostics(model) -> Dict:
    norms = [p.norm().item() for n, p in model.named_parameters() if "lora_b" in n.lower()]
    avg = sum(norms) / len(norms) if norms else 0.0
    zero = sum(1 for x in norms if x < 1e-6)
    return {
        "avg_lora_B_norm": round(avg, 6),
        "zero_lora_B_count": zero,
        "total_lora_layers": len(norms),
        "lora_B_degenerate": len(norms) > 0 and zero == len(norms),
    }


def gate(model, tokenizer, eval_texts: List[str], device: str) -> Dict:
    ppl = perplexity(model, tokenizer, eval_texts, device)
    diag = lora_diagnostics(model)
    return {"perplexity": round(ppl, 4), "n_eval": len(eval_texts), **diag}


def compare_to_base(base_ppl: float, trained: Dict, min_improvement: float = 0.05) -> Dict:
    """
    The honest gate: trained must beat base perplexity by ``min_improvement`` on
    the IDENTICAL eval set, and lora_B must be non-degenerate. Equal perplexity
    is flagged as an anomaly (the no-op signature), not a pass.
    """
    tp = trained["perplexity"]
    delta = (base_ppl - tp) / base_ppl
    anomaly = trained["lora_B_degenerate"] or abs(tp - base_ppl) < 1e-9
    return {
        "base_ppl": round(base_ppl, 4),
        "trained_ppl": round(tp, 4),
        "delta_pct": round(100 * delta, 2),
        "lora_B_norm": trained["avg_lora_B_norm"],
        "lora_B_degenerate": trained["lora_B_degenerate"],
        "anomaly": anomaly,
        "pass": (not anomaly) and delta >= min_improvement,
    }
