"""
Quick-start: train FFA-LoRA DP-SGD on wikitext-2 with 5 lines of config.

This is a standalone script — no project data required. It downloads
wikitext-2 from Hugging Face, wraps a small model in LoRA, and trains
with differential privacy using the dp-lora-edge toolkit.

    python examples/train_dp_lora.py --model EleAIEAI/pythia-160m --epsilon 8

Requires: torch, transformers, peft (installed with dp-lora-edge)
Optional: opacus (for cross-checking ε)
"""
import argparse
import random

from dp_lora import DPConfig, train_dp_lora


def load_wikitext_texts(tokenizer, max_samples=200, max_length=512):
    """Load wikitext-2 raw lines as training texts."""
    try:
        from datasets import load_dataset
    except ImportError:
        # Fallback: simple synthetic texts if datasets not installed
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Differential privacy provides mathematical guarantees.",
        ] * max_samples

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 50:  # skip empty/short lines
            texts.append(text[:max_length])
        if len(texts) >= max_samples:
            break
    return texts


def main():
    p = argparse.ArgumentParser(description="FFA-LoRA DP-SGD quick-start on wikitext-2")
    p.add_argument("--model", default="EleAIEAI/pythia-160m",
                   help="HuggingFace model ID (default: pythia-160m)")
    p.add_argument("--epsilon", type=float, default=8.0,
                   help="Target privacy budget ε (default: 8)")
    p.add_argument("--rounds", type=int, default=2,
                   help="Number of DP-SGD rounds (default: 2)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size per step (default: 8)")
    p.add_argument("--local-steps", type=int, default=10,
                   help="Local steps per round (default: 10)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Model dtype (default: float32, bfloat16 ~2x faster)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    # Calibrate noise for the target ε
    from dp_lora.accountant import calibrate_noise_for_epsilon
    steps = args.rounds * args.local_steps
    sigma = calibrate_noise_for_epsilon(args.epsilon, steps, delta=1e-5)
    print(f"Calibrated σ={sigma:.4f} for ε={args.epsilon} at {steps} steps")

    cfg = DPConfig(
        noise_multiplier=sigma,
        clip_norm=0.5,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        lora_rank=4,
        lr=1e-3,
        delta=1e-5,
        seed=args.seed,
        dtype=args.dtype,
    )

    print(f"Loading {args.model} and wikitext-2...")
    # We need a tokenizer to load texts; use a dummy one for the text loader
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = load_wikitext_texts(tok, max_samples=200)
    print(f"Loaded {len(texts)} training texts")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Training on {device} with {cfg}")

    result = train_dp_lora(args.model, texts, cfg, device=device)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Base probe loss:  {result['base_probe_loss']:.4f}")
    for h in result["history"]:
        print(f"  Round {h['round']}: probe_loss={h['probe_loss']:.4f}, "
              f"lora_B_norm={h['lora_B_norm']:.4f}")
    print(f"  Final lora_B norm: {result['final_lora_B_norm']:.4f}")
    print(f"  Reported ε:        {result['epsilon']:.4f}")
    print(f"  Distributed:       {result['distributed']}")


if __name__ == "__main__":
    main()
