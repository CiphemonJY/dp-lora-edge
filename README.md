# dp-lora-edge

**Differentially private LoRA fine-tuning at calibrated ε — with a self-contained
RDP accountant and gating that makes silent failure impossible.**

This is a small, honest toolkit distilled from a federated-learning research
project. It exists because three subtle bugs in DP LoRA fine-tuning each
produced *plausible-looking* results that were quietly wrong, and the fixes —
plus the methodology that catches them — turned out to be the reusable part.

---

## The results, first

Fine-tuning **pythia-160m** with DP-SGD on a fixed held-out eval set
(base perplexity **13.88**). Every `trained_ppl` uses the *identical* eval set.

| Run | σ | batch | ε (Opacus) | noise/elt | trained ppl | Δ vs base | verdict |
|-----|---|-------|-----------|-----------|-------------|-----------|---------|
| Gate 1 | 0.5 | 4 | 9.0 | 0.0625 | 14.49 | **−4.4%** | DP noise drowns the signal |
| Arm A | 0 | 4 | — | 0 | 2.21 | **+84.1%** | no-DP ceiling: recipe is sound |
| Arm B | 0.5 | 48 | 23.3 | 0.0052 | 9.59 | **+30.9%** | batch averaging recovers DP utility |

The story in one line: **Gate 1 didn't fail because the model was too small — it
failed because batch=4 puts ~6× more noise per gradient element than signal.**
Raising the batch to 48 averages the noise down by 12× and recovers meaningful
utility. DP-SGD on a 160M model is viable; it just has to be calibrated, not
guessed. Full write-up in **[REPORT.md](REPORT.md)**.

> **Note on noise formula:** An earlier version of this repo used `σ·C/√n` for
> per-element noise, which is √n larger than standard DP-SGD's `σ·C/n`. The
> trainer and `noise_per_element` have been corrected to use the standard
> formula. The qualitative findings (batch size matters, calibrate don't guess)
> are unchanged. See REPORT.md "Accountant fidelity" for details.

## What's in here

- **`dp_lora/accountant.py`** — a dependency-free Rényi-DP accountant for the
  *subsampled* Gaussian mechanism (Mironov–Talwar–Zhang 2019), plus
  `calibrate_noise_for_epsilon` (give it a privacy budget, get the σ to train
  at) and `noise_per_element` (the signal-to-noise diagnostic that explains
  Gate 1). Cross-validated against Opacus; **conservative by construction** in
  the default configuration — see [SECURITY.md](SECURITY.md) for the precise
  scope of this claim.
- **`dp_lora/trainer.py`** — an **FFA-LoRA** DP-SGD trainer (freeze `lora_A`,
  train `lora_B`) with a built-in sanity gate that aborts if training is a
  no-op. See "The no-op" in the report for why this matters.
- **`dp_lora/gate.py`** — held-out perplexity with a `lora_B`-degeneracy check
  and an *improvement-over-base* comparator (absolute thresholds hide no-ops).
- **`dp_lora/calibrate.py`** — Opacus cross-check + σ calibration.

## Install

```bash
pip install git+https://github.com/CiphemonJY/dp-lora-edge.git   # core
pip install "dp-lora-edge[opacus]" @ git+https://github.com/CiphemonJY/dp-lora-edge.git  # + Opacus
```

Or from a local clone:

```bash
git clone https://github.com/CiphemonJY/dp-lora-edge.git
cd dp-lora-edge
pip install -e .
```

## 60 seconds

```python
from dp_lora import calibrate, noise_per_element

# "I want ε=8 after 80 steps at 5% sampling — what σ do I train at?"
print(calibrate(target_epsilon=8.0, steps=80, sampling_rate=0.05))

# "Why did my batch=4 run fail?" — compare noise to a ~0.01 signal:
print(noise_per_element(0.5, 0.5, batch_size=4))   # 0.0625  (drowns it)
print(noise_per_element(0.5, 0.5, batch_size=48))  # 0.0052  (survives)
```

```bash
python examples/calibration_table.py     # reproduce the calibration math
pytest -q                                 # accountant + no-op regression tests
```

## Why trust it

Every claim above is a test. `pytest -q` checks the accountant is monotone in σ
and steps, that subsampling amplifies privacy, that calibration round-trips,
and — the centerpiece — that the no-op bug is mechanically impossible to
reintroduce (`grad(lora_A) ≡ 0`, `grad(lora_B) > 0`, and the trainer's sanity
gate fires on a degenerate setup).

The accountant agrees with Opacus within ~10% in the practical DP regime
(σ ≥ 1, moderate sampling). At extreme low sampling rates and low σ, the
integer-order bound loosens to tens of percent — characterised honestly in
REPORT.md "Accountant fidelity". The accountant is conservative (over-estimates
ε) in the default configuration and common parameter range; see
[SECURITY.md](SECURITY.md) for the precise scope and the per-parameter clipping
caveat.

## Scope & honesty

Research code, not a compliance certificate. The accountant is a *conservative
upper bound* in its tested regime — see [SECURITY.md](SECURITY.md) and the
report's "Accountant fidelity" section before making any external privacy claim.
No data ships here; examples use synthetic text or public `wikitext`.

Apache-2.0.