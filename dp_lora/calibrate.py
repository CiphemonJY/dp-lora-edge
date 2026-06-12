"""
Cross-validate the built-in RDP accountant against Opacus, and calibrate σ for
a target ε.

Opacus is an OPTIONAL dependency (``pip install dp-lora-edge[opacus]``). The
built-in accountant in ``accountant.py`` has no third-party dependency; this
module exists only to *check* it and to report the agreement (see REPORT.md).
"""
from __future__ import annotations

from typing import Dict, Optional

from .accountant import calibrate_noise_for_epsilon, compute_epsilon


def opacus_epsilon(noise_multiplier: float, steps: int, sampling_rate: float,
                   delta: float = 1e-5) -> Optional[float]:
    """ε from Opacus' RDP accountant, or None if Opacus is not installed."""
    try:
        from opacus.accountants import RDPAccountant
    except ImportError:
        return None
    acct = RDPAccountant()
    for _ in range(steps):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sampling_rate)
    return acct.get_epsilon(delta=delta)


def cross_check(noise_multiplier: float, steps: int, sampling_rate: float,
                delta: float = 1e-5) -> Dict:
    """
    Compare the built-in accountant to Opacus at one operating point. Returns
    both ε values and their relative disagreement (None for Opacus if absent).
    """
    ours = compute_epsilon(noise_multiplier, steps, delta, sampling_rate)
    theirs = opacus_epsilon(noise_multiplier, steps, sampling_rate, delta)
    disagreement = None
    if theirs is not None and theirs > 0:
        disagreement = abs(ours - theirs) / theirs
    return {
        "noise_multiplier": noise_multiplier,
        "steps": steps,
        "sampling_rate": sampling_rate,
        "delta": delta,
        "epsilon_ours": round(ours, 4),
        "epsilon_opacus": round(theirs, 4) if theirs is not None else None,
        "relative_disagreement": round(disagreement, 4) if disagreement is not None else None,
    }


def calibrate(target_epsilon: float, steps: int, sampling_rate: float = 1.0,
              delta: float = 1e-5) -> Dict:
    """Smallest σ achieving ε ≤ target, with the ε it actually yields."""
    sigma = calibrate_noise_for_epsilon(target_epsilon, steps, delta, sampling_rate)
    return {
        "target_epsilon": target_epsilon,
        "calibrated_sigma": round(sigma, 4),
        "achieved_epsilon": round(compute_epsilon(sigma, steps, delta, sampling_rate), 4),
        "steps": steps,
        "sampling_rate": sampling_rate,
    }
