"""
FFA-LoRA DP-SGD trainer.

FFA-LoRA (Frozen-A Federated Averaging LoRA): freeze ``lora_A`` at its random
init and train only ``lora_B``. This is not a micro-optimisation — it is a
correctness requirement under DP:

    PEFT zero-initialises lora_B. With B = 0, the gradient w.r.t. A is
    identically zero (dL/dA = Bᵀ·δ·xᵀ = 0). A trainer that updates lora_A
    therefore clips a zero, adds Gaussian noise, and steps the optimiser on
    PURE NOISE — the model never changes. (This is a real bug that silently
    invalidated multiple training runs; see REPORT.md "The no-op".)

Training lora_B (which has a non-zero gradient) is the correct single-matrix
scheme and keeps DP noise applied to exactly one matrix per layer.

Includes a built-in sanity gate that aborts if lora_B stays at zero or the
probe loss is unchanged after the first round — a silent no-op can never
survive a run again.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .accountant import compute_epsilon, noise_per_element


@dataclass
class DPConfig:
    noise_multiplier: float = 0.5      # σ
    clip_norm: float = 0.5             # C (per-sample L2 clip)
    rounds: int = 4
    local_steps: int = 20
    batch_size: int = 48
    lora_rank: int = 4
    lr: float = 1e-3
    delta: float = 1e-5
    seed: int = 42
    target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])
    global_clip: bool = False           # global per-sample clip (recommended for P > n)
    sampling_rate: float = 1.0         # Poisson subsampling rate for ε accounting


class SanityGateError(RuntimeError):
    """Raised when training is detected to be a no-op (the failure mode FFA-LoRA fixes)."""


def _lora_b_norm(model) -> float:
    return sum(p.norm().item() for n, p in model.named_parameters() if "lora_b" in n.lower())


def _probe_loss(model, tokenizer, texts, device, max_length=512) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            labels = enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            out = model(**enc, labels=labels)
            if out.loss is not None and torch.isfinite(out.loss):
                losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def _dp_lora_gradients(model, batch_texts, tokenizer, device, cfg: DPConfig, max_length=512) -> dict:
    """Per-sample gradients on lora_B only; clip to cfg.clip_norm, add Gaussian noise.

    When ``cfg.global_clip`` is False (default), each LoRA parameter tensor is
    clipped independently to ``clip_norm``. This is simpler but can under-state ε
    when the number of LoRA modules (P) exceeds the batch size (n) — see
    SECURITY.md "Per-parameter clipping caveat".

    When ``cfg.global_clip`` is True, the combined L2 norm across ALL lora_B
    parameters is clipped to ``clip_norm`` per sample. This is the standard
    DP-SGD approach and the accountant's ε is correct regardless of P or n.
    """
    deltas: dict = {}
    n = len(batch_texts)
    for text in batch_texts:
        model.zero_grad()
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        return_tensors="pt", padding="max_length").to(device)
        labels = enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        if out.loss is None or not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if "lora_b" in name.lower():
                deltas.setdefault(name, []).append(p.grad.detach().clone().cpu())
    if not deltas:
        return {}

    if cfg.global_clip:
        # Global per-sample clipping: combined L2 norm across all lora_B params
        sample_norms = []
        for i in range(n):
            combined = torch.cat([deltas[name][i].flatten() for name in deltas])
            sample_norms.append(combined.norm(2))
        sample_norms = torch.stack(sample_norms)
        factor = torch.clamp_max(cfg.clip_norm / (sample_norms + 1e-8), 1.0)
        noised: dict = {}
        for name, grads in deltas.items():
            g = torch.stack(grads)
            shape = [g.size(0)] + [1] * (g.ndim - 1)
            clipped = g * factor.view(shape)
            avg = clipped.mean(dim=0)
            noise_std = cfg.noise_multiplier * cfg.clip_norm / n
            noised[name] = (avg + torch.randn_like(avg) * noise_std).to(device)
        return noised
    else:
        # Per-parameter clipping (original behaviour, documented caveat)
        noised: dict = {}
        for name, grads in deltas.items():
            g = torch.stack(grads)
            norms = torch.stack([x.norm(2) for x in g])
            factor = torch.clamp_max(cfg.clip_norm / (norms + 1e-8), 1.0)
            shape = [g.size(0)] + [1] * (g.ndim - 1)
            clipped = g * factor.view(shape)
            avg = clipped.mean(dim=0)
            noise_std = cfg.noise_multiplier * cfg.clip_norm / n
            noised[name] = (avg + torch.randn_like(avg) * noise_std).to(device)
        return noised


def build_model(model_id: str, cfg: DPConfig, device: str):
    """Load a causal-LM, wrap with LoRA, and apply the FFA-LoRA freeze (only lora_B trains)."""
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2,
        target_modules=cfg.target_modules, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
    ))
    # FFA-LoRA: freeze everything except lora_B.
    for name, p in model.named_parameters():
        p.requires_grad = "lora_b" in name.lower()
    model.to(device)
    return model, tok


def train_dp_lora(
    model_id: str,
    train_texts: List[str],
    cfg: Optional[DPConfig] = None,
    device: Optional[str] = None,
    probe_texts: Optional[List[str]] = None,
) -> dict:
    """
    Train one client with FFA-LoRA DP-SGD. Returns a result dict with the
    per-round probe loss, final lora_B norm, and the reported ε for the run.

    Raises SanityGateError if training is a no-op after round 1.
    """
    cfg = cfg or DPConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, tok = build_model(model_id, cfg, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    probe = probe_texts or train_texts[-min(16, len(train_texts)):]
    base_probe = _probe_loss(model, tok, probe, device)
    history = []

    for rnd in range(1, cfg.rounds + 1):
        for _ in range(cfg.local_steps):
            idx = random.sample(range(len(train_texts)), min(cfg.batch_size, len(train_texts)))
            batch = [train_texts[i] for i in idx]
            deltas = _dp_lora_gradients(model, batch, tok, device, cfg)
            optimizer.zero_grad()
            for name, p in model.named_parameters():
                if p.requires_grad and name in deltas:
                    p.grad = deltas[name]
            optimizer.step()

        rp = _probe_loss(model, tok, probe, device)
        bn = _lora_b_norm(model)
        history.append({"round": rnd, "probe_loss": rp, "lora_B_norm": bn})

        if rnd == 1:
            if bn == 0.0:
                raise SanityGateError("lora_B norm is 0 after round 1 — training is a no-op")
            if abs(rp - base_probe) < 1e-6:
                raise SanityGateError("probe loss unchanged after round 1 — training is a no-op")

    steps = cfg.rounds * cfg.local_steps
    eps = compute_epsilon(cfg.noise_multiplier, steps, cfg.delta, sampling_rate=cfg.sampling_rate)
    return {
        "model": model_id,
        "base_probe_loss": base_probe,
        "history": history,
        "final_lora_B_norm": history[-1]["lora_B_norm"],
        "epsilon": eps,
        "noise_per_element": noise_per_element(cfg.noise_multiplier, cfg.clip_norm, cfg.batch_size),
        "config": cfg.__dict__,
        "_model": model,
        "_tokenizer": tok,
    }
