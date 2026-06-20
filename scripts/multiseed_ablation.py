#!/usr/bin/env python3
"""Multi-seed + ablation study for dp-lora-edge.

Three experiments:
  1. Multi-seed variance: 3 seeds x FFA-LoRA at eps=8, report mean +/- std
  2. Standard LoRA DP-SGD baseline: train BOTH lora_A + lora_B at same eps
  3. Full DP-SGD baseline: train ALL weights at same eps (no LoRA)

Usage (2-node):
  torchrun --nproc_per_node=1 --nnodes=2 \
    --rdzv_backend=c10d --rdzv_endpoint=192.168.100.12:29500 \
    scripts/multiseed_ablation.py --experiment all --seeds 3

Usage (single-node):
  python scripts/multiseed_ablation.py --experiment all --seeds 3

Results saved to results/multiseed_ablation.json
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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_SET = os.path.expanduser("~/LISA_FTM/data/hospital_eval_set.jsonl")
DATA_GLOB = "~/LISA_FTM/data/synthea/shards_100/client_*.jsonl"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiseed")


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


def dp_gradients(model, batch_texts, tokenizer, device, sigma, clip, mode, max_length=512):
    """Per-sample gradients with DP noise. mode: 'ffa', 'standard_lora', 'full'."""
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
            if mode == "standard_lora" and "lora_" not in name:
                continue
            # mode == "full": all params
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


def build_model(model_name, device, mode, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if mode in ("ffa", "standard_lora"):
        targets = get_lora_targets(model)
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, target_modules=targets,
            lora_dropout=0.0, bias="none",
        ))
        if mode == "ffa":
            for name, p in model.named_parameters():
                p.requires_grad = "lora_B" in name
        else:
            for name, p in model.named_parameters():
                p.requires_grad = "lora_" in name
    else:
        # Full: train all weights
        for p in model.parameters():
            p.requires_grad = True
    model.to(device)
    return model, tok


def all_reduce_params(model):
    """NCCL all-reduce: average trainable params across nodes."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data.div_(dist.get_world_size())


def run_experiment(name, model_name, train_texts, eval_path, device,
                   sigma, clip, batch_size, rounds, clients, local_steps, seed, mode):
    random.seed(seed)
    torch.manual_seed(seed)

    model, tok = build_model(model_name, device, mode)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    base_ppl = eval_ppl(model, eval_path, tok, device)
    logger.info("[%s seed=%d] Base PPL: %.4f", name, seed, base_ppl)

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
                deltas = dp_gradients(model, batch, tok, device, sigma, clip, mode)
                optimizer.zero_grad()
                for pname, p in model.named_parameters():
                    if p.requires_grad and pname in deltas:
                        p.grad = deltas[pname]
                optimizer.step()

        if distributed:
            all_reduce_params(model)
            dist.barrier()

        ppl = eval_ppl(model, eval_path, tok, device)
        if mode == "ffa":
            norm = lora_b_norm(model)
        else:
            norm = sum(p.data.norm().item() for p in model.parameters() if p.requires_grad)
        elapsed = time.time() - t0
        history.append({"round": rnd, "ppl": round(ppl, 4),
                        "param_norm": round(norm, 6), "elapsed_s": round(elapsed, 1)})
        if rank == 0:
            logger.info("[%s seed=%d] Round %d: PPL=%.4f, norm=%.4f, %.0fs",
                        name, seed, rnd, ppl, norm, elapsed)

    final_ppl = history[-1]["ppl"]
    delta_pct = ((base_ppl - final_ppl) / max(base_ppl, 0.01)) * 100

    # Free memory
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "name": name, "mode": mode, "seed": seed,
        "base_ppl": round(base_ppl, 4), "final_ppl": final_ppl,
        "delta_pct": round(delta_pct, 2),
        "sigma": sigma, "clip": clip, "batch_size": batch_size,
        "rounds": rounds, "clients": clients, "local_steps": local_steps,
        "distributed": distributed, "world_size": world_size,
        "history": history, "total_time_s": round(time.time() - t0, 1),
    }
    logger.info("[%s seed=%d] RESULT: %s", name, seed, json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser(description="Multi-seed + ablation study")
    ap.add_argument("--experiment", choices=["multiseed", "ablation", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--sigma", type=float, default=1.18)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--local-steps", type=int, default=20)
    ap.add_argument("--output", default="results/multiseed_ablation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        distributed = True

    rank = dist.get_rank() if distributed else 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    paths = resolve_data_paths(DATA_GLOB)
    if not paths:
        logger.error("No training data found at %s", DATA_GLOB)
        sys.exit(1)
    train_texts = load_texts(paths)
    logger.info("Loaded %d training texts from %d files", len(train_texts), len(paths))

    results = {"experiments": [], "config": vars(args)}

    # Experiment 1: Multi-seed variance (FFA-LoRA)
    if args.experiment in ("multiseed", "all"):
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Multi-seed variance (FFA-LoRA, %d seeds)", args.seeds)
        logger.info("=" * 60)
        for seed in range(42, 42 + args.seeds):
            r = run_experiment(
                f"ffa_seed{seed}", args.model, train_texts, EVAL_SET, args.device,
                args.sigma, args.clip, args.batch_size, args.rounds,
                args.clients, args.local_steps, seed, mode="ffa",
            )
            results["experiments"].append(r)

        ppls = [e["final_ppl"] for e in results["experiments"] if e["name"].startswith("ffa_seed")]
        deltas = [e["delta_pct"] for e in results["experiments"] if e["name"].startswith("ffa_seed")]
        if len(ppls) >= 2:
            import statistics
            results["multiseed_summary"] = {
                "mean_ppl": round(statistics.mean(ppls), 4),
                "std_ppl": round(statistics.stdev(ppls), 4),
                "mean_delta_pct": round(statistics.mean(deltas), 2),
                "std_delta_pct": round(statistics.stdev(deltas), 2),
                "seeds": len(ppls),
            }
            if rank == 0:
                logger.info("MULTISEED SUMMARY: %s", json.dumps(results["multiseed_summary"]))

    # Experiment 2: Standard LoRA (both A+B)
    if args.experiment in ("ablation", "all"):
        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Standard LoRA DP-SGD (both A+B)")
        logger.info("=" * 60)
        r = run_experiment(
            "standard_lora", args.model, train_texts, EVAL_SET, args.device,
            args.sigma, args.clip, args.batch_size, args.rounds,
            args.clients, args.local_steps, 42, mode="standard_lora",
        )
        results["experiments"].append(r)

    # Experiment 3: Full DP-SGD (no LoRA)
    if args.experiment in ("ablation", "all"):
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: Full DP-SGD (all weights, no LoRA)")
        logger.info("=" * 60)
        r = run_experiment(
            "full_dp_sgd", args.model, train_texts, EVAL_SET, args.device,
            args.sigma, args.clip, args.batch_size, args.rounds,
            args.clients, args.local_steps, 42, mode="full",
        )
        results["experiments"].append(r)

    if rank == 0:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)

        print("\n" + "=" * 60)
        print("ABLATION SUMMARY")
        print("=" * 60)
        for e in results["experiments"]:
            print(f"  {e['name']:20s}  PPL: {e['base_ppl']:.2f} -> {e['final_ppl']:.2f}  "
                  f"d={e['delta_pct']:+.1f}%  time={e['total_time_s']:.0f}s")
        if "multiseed_summary" in results:
            ms = results["multiseed_summary"]
            print(f"\n  Multi-seed (FFA-LoRA): PPL {ms['mean_ppl']:.2f} +/- {ms['std_ppl']:.2f}  "
                  f"d {ms['mean_delta_pct']:.1f} +/- {ms['std_delta_pct']:.1f}%")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
