#!/usr/bin/env python3
"""Larger model validation for FFA-LoRA DP-SGD.

Runs FFA-LoRA on larger models (3B, 7B) to validate that the method
scales beyond the 1.5B model used in the main ablation study.

Uses env-based init for 2-node distributed (NOT torchrun, which hangs
on the DGX Spark cluster — see scripts/launch_2node.py).

Usage (single-node, 3B):
  source ~/LISA_FTM/.venv/bin/activate
  python3 scripts/larger_model_validation.py --model-size 3b

Usage (2-node, 7B):
  # On primary:
  source ~/LISA_FTM/.venv/bin/activate && source ~/LISA_FTM/nccl_cluster_env.sh
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.168.100.12 MASTER_PORT=29550 \
    python3 scripts/larger_model_validation.py --model-size 7b

  # On secondary:
  source ~/LISA_FTM/.venv/bin/activate && source ~/LISA_FTM/nccl_cluster_env.sh
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=192.168.100.12 MASTER_PORT=29550 \
    python3 scripts/larger_model_validation.py --model-size 7b
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODELS = {
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

EVAL_SET = os.path.expanduser("~/LISA_FTM/data/hospital_eval_set.jsonl")
DATA_GLOB = "~/LISA_FTM/data/synthea/shards_100/client_*.jsonl"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("larger_model")


def resolve_data_paths(pattern):
    import glob
    return sorted(glob.glob(os.path.expanduser(pattern)))


@torch.no_grad()
def eval_ppl(model, eval_path, tokenizer, device="cuda"):
    samples = []
    with open(os.path.expanduser(eval_path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "") or obj.get("text", "")
            if prompt:
                samples.append(prompt)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in samples:
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(ids, attention_mask=attn, labels=ids)
        n_tok = attn.sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    ppl = math.exp(total_loss / max(total_tokens, 1))
    model.train()
    return ppl


def lora_b_norm(model):
    return sum(p.data.norm().item() for n, p in model.named_parameters() if "lora_B" in n)


def get_lora_targets(model):
    candidates = ["q_proj", "v_proj", "query_key_value", "c_attn", "Wqkv"]
    found = []
    for name, _ in model.named_modules():
        for cand in candidates:
            if cand in name:
                found.append(name)
    modules = set()
    for f in found:
        parts = f.split(".")
        modules.add(parts[-1])
    return list(modules) if modules else ["q_proj", "v_proj"]


def load_texts(paths, max_samples=57000):
    texts = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "") or obj.get("text", "")
                if prompt and len(prompt) > 20:
                    texts.append(prompt)
                    if len(texts) >= max_samples:
                        return texts
    return texts


def dp_gradients(model, batch_texts, tokenizer, device, sigma, clip, mode="ffa", max_length=512):
    """Per-sample gradients with DP noise (FFA-LoRA: only lora_B)."""
    deltas = {}
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
            if p.grad is None or not p.requires_grad:
                continue
            if mode == "ffa" and "lora_B" not in name:
                continue
            deltas.setdefault(name, []).append(p.grad.detach().clone().cpu())
    if not deltas:
        return {}
    noised = {}
    for name, grads in deltas.items():
        g = torch.stack(grads)
        norms = torch.stack([x.norm(2) for x in g])
        factor = torch.clamp_max(clip / (norms + 1e-8), 1.0)
        shape = [g.size(0)] + [1] * (g.ndim - 1)
        clipped = g * factor.view(shape)
        avg = clipped.mean(dim=0)
        noise_std = sigma * clip / n
        noised[name] = (avg + torch.randn_like(avg) * noise_std).to(device)
    return noised


def build_model(model_name, device, mode="ffa", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    targets = get_lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, target_modules=targets,
        lora_dropout=0.0, bias="none",
    ))
    if mode == "ffa":
        for name, p in model.named_parameters():
            p.requires_grad = "lora_B" in name
    model.to(device)
    return model, tok


def all_reduce_params(model):
    """NCCL all-reduce: average trainable params across nodes."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data.div_(dist.get_world_size())


def run_experiment(model_name, train_texts, eval_path, device,
                   sigma, clip, batch_size, rounds, clients, local_steps, seed):
    """Run FFA-LoRA on a single model. Returns result dict."""
    random.seed(seed)
    torch.manual_seed(seed)

    model, tok = build_model(model_name, device, mode="ffa")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    base_ppl = eval_ppl(model, eval_path, tok, device)
    logger.info("Base PPL: %.4f", base_ppl)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    my_clients = max(1, clients // world_size)
    if rank < clients % world_size:
        my_clients += 1

    history = []
    t0 = time.time()

    for rnd in range(1, rounds + 1):
        for _c in range(my_clients):
            for _ in range(local_steps):
                idx = random.sample(range(len(train_texts)), min(batch_size, len(train_texts)))
                batch = [train_texts[i] for i in idx]
                deltas = dp_gradients(model, batch, tok, device, sigma, clip, "ffa")
                optimizer.zero_grad()
                for pname, p in model.named_parameters():
                    if p.requires_grad and pname in deltas:
                        p.grad = deltas[pname]
                optimizer.step()

        if distributed:
            all_reduce_params(model)
            dist.barrier()

        ppl = eval_ppl(model, eval_path, tok, device)
        norm = lora_b_norm(model)
        elapsed = time.time() - t0
        history.append({"round": rnd, "ppl": round(ppl, 4),
                        "lora_B_norm": round(norm, 6), "elapsed_s": round(elapsed, 1)})
        if rank == 0:
            logger.info("Round %d: PPL=%.4f, norm=%.4f, %.0fs", rnd, ppl, norm, elapsed)

    final_ppl = history[-1]["ppl"]
    delta_pct = ((base_ppl - final_ppl) / max(base_ppl, 0.01)) * 100

    # Memory report
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info("GPU memory: allocated=%.1fGB, peak=%.1fGB", mem_alloc, mem_peak)
    else:
        mem_alloc = mem_peak = 0.0

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model": model_name,
        "seed": seed,
        "base_ppl": round(base_ppl, 4),
        "final_ppl": final_ppl,
        "delta_pct": round(delta_pct, 2),
        "sigma": sigma,
        "clip": clip,
        "batch_size": batch_size,
        "rounds": rounds,
        "clients": clients,
        "local_steps": local_steps,
        "distributed": distributed,
        "world_size": world_size,
        "gpu_mem_allocated_gb": round(mem_alloc, 2),
        "gpu_mem_peak_gb": round(mem_peak, 2),
        "history": history,
        "total_time_s": round(time.time() - t0, 1),
    }
    logger.info("RESULT: %s", json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser(description="Larger model FFA-LoRA validation")
    ap.add_argument("--model-size", choices=["1.5b", "3b", "7b"], default="3b")
    ap.add_argument("--sigma", type=float, default=1.18)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--local-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/larger_model_validation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Auto-adjust batch size based on model size
    batch_size = 48
    if args.model_size == "3b":
        batch_size = 32  # 3B uses more memory
    elif args.model_size == "7b":
        batch_size = 16  # 7B uses much more memory
    args.batch_size = batch_size

    # Env-based distributed init (NOT torchrun)
    distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist.init_process_group(backend="nccl")
            distributed = True
            logger.info("Distributed init: rank=%s, world_size=%s",
                        os.environ["RANK"], os.environ["WORLD_SIZE"])

    rank = dist.get_rank() if distributed else 0

    model_name = MODELS[args.model_size]
    logger.info("=" * 60)
    logger.info("LARGER MODEL VALIDATION: %s (%s)", args.model_size, model_name)
    logger.info("Config: sigma=%s, clip=%s, batch=%d, rounds=%d, clients=%d, local_steps=%d",
                args.sigma, args.clip, args.batch_size, args.rounds, args.clients, args.local_steps)
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    paths = resolve_data_paths(DATA_GLOB)
    if not paths:
        logger.error("No training data found at %s", DATA_GLOB)
        sys.exit(1)
    train_texts = load_texts(paths)
    logger.info("Loaded %d training texts from %d files", len(train_texts), len(paths))

    result = run_experiment(
        model_name, train_texts, EVAL_SET, args.device,
        args.sigma, args.clip, args.batch_size, args.rounds,
        args.clients, args.local_steps, args.seed,
    )

    if rank == 0:
        with open(args.output, "w") as f:
            json.dump({"validation": result}, f, indent=2)
        logger.info("Results saved to %s", args.output)

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  Model: {result['model']}")
        print(f"  PPL: {result['base_ppl']:.2f} -> {result['final_ppl']:.2f}  "
              f"(d={result['delta_pct']:+.1f}%)")
        print(f"  GPU memory peak: {result['gpu_mem_peak_gb']:.1f}GB")
        print(f"  Time: {result['total_time_s']:.0f}s")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()