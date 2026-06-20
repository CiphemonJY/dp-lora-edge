"""
Rényi Differential Privacy (RDP) accountant for the Sampled Gaussian Mechanism.

Self-contained, dependency-free (stdlib + math). Validated against Opacus —
see ``calibrate.py`` and ``REPORT.md``. The accountant is *conservative*: at
low sampling rates it over-estimates ε (claims privacy is weaker than it is),
which is the safe direction for a privacy claim. Known limitation documented
in REPORT.md "Accountant fidelity".

References
---------
- Mironov (2017), "Rényi Differential Privacy".
- Mironov, Talwar, Zhang (2019), "Rényi Differential Privacy of the Sampled
  Gaussian Mechanism", Theorem 4 (the subsampled-Gaussian RDP used here).
- Abadi et al. (2016), "Deep Learning with Differential Privacy".
"""
from __future__ import annotations

import math
from typing import Optional

# Default RDP orders to scan (same spirit as Opacus' default order grid).
_DEFAULT_ALPHAS = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def _sgm_rdp(alpha: int, q: float, sigma: float) -> float:
    """
    Per-step RDP of the Sampled Gaussian Mechanism at INTEGER order ``alpha``,
    for Poisson-subsampling rate ``q`` and noise multiplier ``sigma``
    (Mironov–Talwar–Zhang 2019, Thm. 4):

        RDP_α = (1/(α−1)) · log( Σ_{k=0}^{α} C(α,k)(1−q)^{α−k} q^k · exp(k(k−1)/(2σ²)) )

    Captures privacy amplification by subsampling: for small ``q`` it shrinks
    like q², so sampling a fraction of clients each round buys a much smaller ε
    at the same σ. Reduces to the plain Gaussian RDP (α/(2σ²)) at q=1.
    """
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma * sigma)
    log_terms = []
    for k in range(alpha + 1):
        log_binom = (
            math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        )
        t = (
            log_binom
            + (alpha - k) * math.log(1.0 - q)
            + k * math.log(q)
            + (k * (k - 1)) / (2.0 * sigma * sigma)
        )
        log_terms.append(t)
    m = max(log_terms)
    lse = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return lse / (alpha - 1.0)


def compute_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
    sampling_rate: float = 1.0,
    alpha: Optional[float] = None,
) -> float:
    """
    ε for (ε, δ)-DP after ``steps`` rounds of the Gaussian mechanism, via RDP
    accounting and the RDP→(ε,δ) conversion ε(α) = RDP_α(σ,T) + ln(1/δ)/(α−1),
    minimised over a grid of orders α (exactly how Opacus / TF-Privacy do it).

    ε grows with ``steps`` and shrinks with more noise σ (ε ∝ T/σ²). With
    ``sampling_rate`` < 1 the subsampled-Gaussian RDP is used, accounting for
    privacy amplification by subsampling.

    Returns 0.0 if steps == 0; +inf if σ ≤ 0.
    """
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return float("inf")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if sampling_rate > 1.0:
        sampling_rate = 1.0

    sigma = noise_multiplier
    ln_inv_delta = math.log(1.0 / delta)

    if sampling_rate < 1.0:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        best = float("inf")
        for a in range(2, 64):
            rdp = steps * _sgm_rdp(a, sampling_rate, sigma)
            best = min(best, rdp + ln_inv_delta / (a - 1.0))
        return best

    orders = [alpha] if alpha is not None else _DEFAULT_ALPHAS
    best = float("inf")
    sigma2 = sigma * sigma
    for a in orders:
        if a <= 1.0:
            continue
        rdp = steps * a / (2.0 * sigma2)
        best = min(best, rdp + ln_inv_delta / (a - 1.0))
    return best


def calibrate_noise_for_epsilon(
    target_epsilon: float,
    steps: int,
    delta: float = 1e-5,
    sampling_rate: float = 1.0,
    tol: float = 1e-3,
) -> float:
    """
    Find the smallest noise multiplier σ that achieves ε ≤ ``target_epsilon``.

    ε is monotonically decreasing in σ, so a bisection converges. This is the
    operational inverse of ``compute_epsilon`` — given a privacy budget, return
    the σ to train at — and is what turns "lower sigma" from a knob into a
    decision (see REPORT.md, Arm C).
    """
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    lo, hi = 1e-3, 1.0
    # Grow hi until ε(hi) <= target.
    while compute_epsilon(hi, steps, delta, sampling_rate) > target_epsilon:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("no σ achieves target ε within bounds")
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if compute_epsilon(mid, steps, delta, sampling_rate) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def noise_per_element(noise_multiplier: float, clip_norm: float, batch_size: int) -> float:
    """
    Std-dev of DP noise added per gradient element, for the averaged-gradient
    mechanism: σ·C/n (standard DP-SGD noise scaling). This is the quantity that
    determines whether learning survives the noise — compare it to the
    per-element gradient magnitude (REPORT.md "Signal-to-noise"). Larger
    batches average noise down.

    Note: the original version of this function used σ·C/√batch, which is
    √n larger than standard DP-SGD's σ·C/n. The trainer was updated to match.
    See REPORT.md "Accountant fidelity" for the full discussion.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return noise_multiplier * clip_norm / batch_size
