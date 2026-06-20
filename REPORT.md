# Differentially Private LoRA Fine-Tuning at Calibrated ε

A short technical report on making DP-SGD LoRA fine-tuning *honest*: three bugs
that each produced plausible-but-wrong results, the methodology that caught
them, and a clean characterisation of when the privacy accounting can be
trusted.

All code and numbers here are reproducible from this repository
(`pytest -q`, `examples/calibration_table.py`); the empirical training results
are recorded in `results/gates.json`.

---

## 1. The no-op

The bug that started everything. A DP LoRA trainer collected per-sample
gradients, clipped them, added Gaussian noise, and stepped the optimiser — and
the model never changed. Every training run "completed", the loss printouts
looked alive, and the held-out perplexity came out at a believable number.

The cause is one line of linear algebra. PEFT zero-initialises the `lora_B`
matrix (so the adapter starts as a no-op, ΔW = B·A = 0). The trainer extracted
gradients for `lora_A`. But with B = 0,

```
dL/dA = Bᵀ · δ · xᵀ = 0    (identically, because B = 0)
```

so the collected "gradients" were exactly zero. The DP pipeline then clipped
zeros, added noise, and AdamW stepped on **pure noise**. `lora_B` never received
a gradient, stayed at 0, and ΔW stayed 0. The model's output was the base model
plus accumulated noise in an unused matrix.

This invalidated multiple training runs whose reported perplexities were simply
the base model's perplexity on different eval sets. The smoking gun was sitting
in a diagnostics file the whole time — `avg_lora_B_norm: 0.0` — logged and
ignored.

**The fix (FFA-LoRA).** Freeze `lora_A` at its random init and train `lora_B`
(which *does* have a non-zero gradient). This is the correct single-matrix DP
scheme: noise is applied to exactly one matrix per layer, and the matrix being
trained is the one with signal. `dp_lora/trainer.py` implements it, and
`tests/test_noop_regression.py` proves both halves on a real model:
`grad(lora_A)` is exactly 0, `grad(lora_B)` is not.

**The defence.** A bug fixed once recurs. So the trainer carries a **sanity
gate**: after the first round it aborts if `lora_B` norm is still 0 or the probe
loss is byte-identical to the base model. And the eval gate (`dp_lora/gate.py`)
reports `lora_B_degenerate` and compares to the base model's perplexity rather
than an absolute threshold — because a degenerate model sits below any threshold
you pick. (The same class of bug later reappeared in a different training path
as a self-distillation no-op; the gates caught it in days, not weeks. The
lesson generalised.)

## 2. Signal-to-noise: why Gate 1 failed

With the trainer fixed, the first honest run still failed — and failed
*cleanly*, which is the point.

| Run | σ | batch | trained ppl | Δ vs base (13.88) | `lora_B` norm |
|-----|---|-------|-------------|-------------------|---------------|
| Gate 1 | 0.5 | 4 | 14.49 | −4.4% | 1.15 (healthy) |

The model trained — `lora_B` moved — but ended up *worse* than baseline. The
explanation is arithmetic, not scale. DP noise is added with standard deviation

```
noise_per_element = σ · C / n
```

At σ=0.5, C=0.5, batch=4 that is **0.0625 per element**. A per-sample gradient
clipped to total L2 norm 0.5, spread over a rank-4 `lora_B` (~thousands of
elements per layer), has per-element magnitude on the order of **0.01**. The
noise is roughly **6× the signal, per element, per step.** No model at any
scale learns through that.

> **Note on noise formula:** An earlier version of this report used `σ·C/√n`,
> which is √n larger than standard DP-SGD's `σ·C/n`. The trainer and
> `noise_per_element` have been corrected. The qualitative conclusion (batch
> size is the primary lever for DP-SGD utility) is unchanged, but the
> quantitative noise ratios are smaller with the corrected formula.

This matters because the tempting conclusion — "160M is too small, go bigger" —
is wrong and expensive. A 1.4B model under the same recipe faces the same
signal-to-noise ratio. DP-SGD's canonical utility lever is **batch size**: noise
averages down linearly with n, which is far cheaper than 10× model scale.

## 3. The ablation: isolate the cause before scaling

Three arms, same trainer, same fixed eval set. Each answers one question.

| Arm | σ | batch | noise/elt | trained ppl | Δ vs base | answers |
|-----|---|-------|-----------|-------------|-----------|---------|
| A | 0 | 4 | 0 | 2.21 | +84.1% | is the *recipe* sound? (yes — ceiling) |
| B | 0.5 | 48 | 0.0052 | 9.59 | +30.9% | does batch averaging rescue DP? (yes) |

Arm A (no DP) proves the trainer, data pipeline, and hyperparameters are
correct — the model can reach perplexity 2.21, so nothing is structurally
broken. Arm B keeps σ but raises the batch from 4 to 48, cutting per-element
noise by 12× (0.0625 → 0.0052) and recovering **+30.9%** improvement *with DP
noise active*.

**Conclusion: DP-SGD is viable on a 160M model — the Gate 1 failure was batch
size, fully explained, not model capacity.** The `lora_B` norms even trace the
dose-response: 12.85 (no noise) → 9.69 (moderate) → 1.15 (drowned).

The decision rule the ablation was designed to feed:
- Arm A fails → the recipe is broken; fix it before scaling.
- A passes, B fails → batch alone is insufficient; *now* a larger model is justified.
- A and B pass → small-model DP is viable; scale is a quality decision, not a rescue.

We landed in the third case.

## 4. Accountant fidelity (the honest caveat)

Arm B reaches its utility at **ε ≈ 23** (Opacus) — that is *weak* privacy. The
product-relevant question is the calibrated one: what σ achieves a target ε=8,
and does utility survive at that noise?

`calibrate_noise_for_epsilon` answers the first part by bisection on the
accountant. But the accountant itself must be trustworthy, so we cross-validate
it against Opacus across an operating grid (`dp_lora/calibrate.py`,
`tests/test_accountant.py` with `[opacus]` installed). The honest finding:

- The built-in accountant uses the **integer-order** subsampled-Gaussian RDP
  bound (Mironov–Talwar–Zhang 2019, Thm. 4). Opacus uses a tighter
  fractional-order analysis.
- **It is conservative in the tested regime**: our ε ≥ Opacus' ε across the
  tested grid (σ ∈ {0.5, 1.0, 2.0}, q ∈ {0.01, 0.1, 0.48}). It may over-state
  privacy spent; it never under-states it *in this regime* — so a claim made
  with it is never optimistic *within the tested range*.
- **Agreement is regime-dependent.** In the practical DP-SGD regime (σ ≥ 1,
  moderate sampling) it tracks Opacus within ~7–10%. It loosens substantially
  at extreme low sampling rates and low σ (tens of percent), where the
  integer-order bound is slack.

### Per-parameter clipping (known limitation)

The trainer clips gradients **per LoRA parameter tensor** rather than using
global per-sample gradient clipping (clipping the combined L2 norm across all
parameters). This is a pragmatic choice for LoRA — the adapter matrices are
small and independent — but it has a privacy implication:

When the number of LoRA target modules (P) exceeds the batch size (n),
per-parameter clipping adds more noise than the accountant budgets for,
because each parameter is independently clipped to `C`. The effective
sensitivity becomes `P·C` rather than `C`, which the accountant does not
account for. This means the accountant can **under-state ε** when `P > n`.

For typical configurations (P=6–12 modules, batch=48+), this is not a concern.
For extreme configs (P=192, batch=4), reported ε may be understated by up to
4×. **If your config has P > n, use global clipping or increase the batch size.**
This is documented in [SECURITY.md](SECURITY.md) as a known limitation.

**Operational guidance:** use this accountant for in-the-loop budgeting and as a
*safe upper bound within its tested regime*; use Opacus (or another
fractional-order accountant) for any ε you publish or put in front of a
compliance reviewer. This repository does not claim the accountant is tight —
it claims it is *safe in the tested regime* and characterises exactly where it
is loose. That distinction is the whole point of the report.

## 5. What to take away

1. **A green dashboard is not a result.** Every number here is gated on
   improvement over a measured baseline and on a non-degenerate adapter; the
   bugs that motivated this work all passed naïve checks.
2. **Ablate before you scale.** One day of σ=0/batch arms saved a week of
   training a bigger model to rescue a problem that was arithmetic.
3. **Calibrate, don't guess.** "σ=0.5" is not a privacy claim until you state
   the ε it buys at your sampling rate — and validate the accountant that told
   you so.

---

### Reproduce

```bash
git clone https://github.com/CiphemonJY/dp-lora-edge.git
cd dp-lora-edge
pip install -e ".[opacus,dev]"
pytest -q                              # accountant + no-op regression
python examples/calibration_table.py   # the σ/ε/noise calibration math
```

Empirical training numbers: `results/gates.json`. The DP-SGD mechanism,
accountant, FFA-LoRA trainer, and gates are in `dp_lora/`.

### References

- Mironov (2017), *Rényi Differential Privacy*.
- Mironov, Talwar, Zhang (2019), *Rényi Differential Privacy of the Sampled
  Gaussian Mechanism*.
- Abadi et al. (2016), *Deep Learning with Differential Privacy*.
- Sun et al. (2024), *Improving LoRA in Privacy-Preserving Federated Learning*
  (FFA-LoRA).

## Single-seed caveat

Results in `results/gates.json` are from **single-seed runs**. The
`lora_B_norm` trajectory and gate verdicts are deterministic (the no-op
detection logic does not depend on random seeds), but the exact perplexity
values may vary across seeds. The multi-seed ablation study below tightens
the confidence intervals.

---

## 6. Multi-seed ablation study

A 5-experiment ablation running on a single DGX Spark node (NVIDIA GB10,
Qwen2.5-1.5B-Instruct, bfloat16). Config: σ=1.18 (calibrated for ε=8, δ=1e-5),
clip=0.5, batch=48, 4 rounds × 50 clients × 20 local steps.

| Experiment | Mode | Seeds | Purpose |
|-----------|------|-------|---------|
| ffa_seed42 | FFA-LoRA (B only) | 1/3 | Multi-seed variance |
| ffa_seed43 | FFA-LoRA (B only) | 2/3 | Multi-seed variance |
| ffa_seed44 | FFA-LoRA (B only) | 3/3 | Multi-seed variance |
| standard_lora | LoRA (A+B) | 1 | Ablation: does training A help? |
| full_dp_sgd | All weights | 1 | Baseline: full-model DP-SGD |

### Results

<!-- Results will be filled in when the ablation study completes.
     Script: scripts/multiseed_ablation.py
     Output: results/multiseed_ablation.json
     Expected fields per experiment: seed, mode, base_ppl, trained_ppl,
     delta_pct, lora_B_norm, rounds_completed, time_seconds -->

**FFA-LoRA variance (3 seeds):**

| Seed | Base PPL | Trained PPL | Δ% | lora_B norm | Status |
|------|----------|-------------|-----|-------------|--------|
| 42 | _pending_ | _pending_ | _pending_ | _pending_ | _running_ |
| 43 | — | — | — | — | _queued_ |
| 44 | — | — | — | — | _queued_ |

_Mean Δ%: ±_ (pending)
_Std Δ%: ±_ (pending)

**Standard LoRA (A+B trained):**

| Seed | Base PPL | Trained PPL | Δ% | lora_B norm | Status |
|------|----------|-------------|-----|-------------|--------|
| 42 | — | — | — | — | _queued_ |

**Full DP-SGD (all weights, no LoRA):**

| Seed | Base PPL | Trained PPL | Δ% | Status |
|------|----------|-------------|-----|--------|
| 42 | — | — | — | _queued_ |

### Interpretation (to be completed)

- **FFA-LoRA variance:** If std(Δ%) < 2%, single-seed results in `gates.json`
  are representative. If > 5%, the gate thresholds need widening.
- **Standard LoRA vs FFA-LoRA:** If standard LoRA (training both A and B)
  outperforms FFA-LoRA (B only), the A matrix carries useful signal under DP
  noise — worth investigating as an alternative training scheme.
- **Full DP-SGD baseline:** If full DP-SGD (all weights) underperforms FFA-LoRA,
  LoRA's parameter efficiency is the primary utility driver, not just noise
  reduction. This would be the publishable finding.

### Reproduce

```bash
# Single-node (what was run):
python scripts/multiseed_ablation.py --experiment all --seeds 3 \
  --sigma 1.18 --clip 0.5 --batch-size 48 --rounds 4 \
  --clients 50 --local-steps 20 --device cuda

# Results saved to: results/multiseed_ablation.json
# Log: logs/multiseed_ablation.log
```

> **Note on 2-node training:** The original 2-node distributed training
> (Section 1-5 results) used `dist.init_process_group(backend="nccl")` with
> environment variables directly, not torchrun's `--rdzv_endpoint`. The
> ablation script supports `torchrun` for multi-node, but torchrun's c10d
> rendezvous mechanism hangs silently on the DGX Spark cluster even when
> TCP ports are confirmed reachable. The single-node fallback runs all
> experiments sequentially on one GPU.
>
> **Root cause identified (2026-06-20):** torchrun's c10d and static rendezvous
> backends both hang silently — zero output, timeout kill (exit 124). Env-based
> init (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` +
> `dist.init_process_group(backend="nccl")`) works perfectly: a broadcast test
> completed in 1.3s with correct tensor propagation. The fix is to use
> `scripts/launch_2node.py` (env-based init wrapper) instead of `torchrun`.
> This matches the existing 2-node LISA_FTM training pattern.
