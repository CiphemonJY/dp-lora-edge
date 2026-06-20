"""Accountant correctness, monotonicity, calibration round-trip, and Opacus agreement."""
import math

import pytest

from dp_lora.accountant import (
    calibrate_noise_for_epsilon,
    compute_epsilon,
    noise_per_element,
)
from dp_lora.calibrate import opacus_epsilon


def test_epsilon_zero_steps_is_zero():
    assert compute_epsilon(1.0, steps=0) == 0.0


def test_epsilon_decreases_with_more_noise():
    """ε ∝ 1/σ²: more noise must mean less ε (the bug this replaced had it inverted)."""
    eps_low_noise = compute_epsilon(0.5, steps=80)
    eps_high_noise = compute_epsilon(2.0, steps=80)
    assert eps_high_noise < eps_low_noise


def test_epsilon_increases_with_steps():
    assert compute_epsilon(1.0, steps=160) > compute_epsilon(1.0, steps=80)


def test_subsampling_amplifies_privacy():
    """Sampling a fraction of clients must yield smaller ε than full participation."""
    full = compute_epsilon(1.0, steps=80, sampling_rate=1.0)
    sub = compute_epsilon(1.0, steps=80, sampling_rate=0.1)
    assert sub < full


def test_calibration_round_trip():
    """calibrate_noise_for_epsilon must return a σ whose ε is at (or just under) target."""
    for target in (1.0, 4.0, 8.0):
        sigma = calibrate_noise_for_epsilon(target, steps=80)
        eps = compute_epsilon(sigma, steps=80)
        assert eps <= target + 1e-2
        # And it should be the *smallest* such σ — a hair less noise overshoots.
        assert compute_epsilon(sigma * 0.9, steps=80) > target


def test_noise_per_element_averages_down_with_batch():
    """σ·C/n: 12× batch = 12× less per-element noise (the Arm B result).

    Note: the original formula was σ·C/√n (sqrt scaling). This was corrected
    to σ·C/n (standard DP-SGD) after a council review identified the
    discrepancy. The test was updated to match.
    """
    small = noise_per_element(0.5, 0.5, batch_size=4)
    large = noise_per_element(0.5, 0.5, batch_size=48)
    assert math.isclose(small / large, 48 / 4, rel_tol=1e-6)


@pytest.mark.skipif(opacus_epsilon(1.0, 1, 0.1) is None, reason="opacus not installed")
def test_agrees_with_opacus_in_practical_dp_regime():
    """
    In the standard DP-SGD regime (σ ≥ 1, moderate sampling) the integer-order
    bound tracks Opacus' fractional-order accountant within ~10%. It loosens at
    extreme low sampling rates — characterised honestly in REPORT.md.
    """
    from dp_lora.calibrate import cross_check
    r = cross_check(noise_multiplier=1.0, steps=80, sampling_rate=0.48)
    assert r["relative_disagreement"] < 0.10


@pytest.mark.skipif(opacus_epsilon(1.0, 1, 0.1) is None, reason="opacus not installed")
def test_accountant_is_conservative_in_tested_grid():
    """
    The load-bearing safety property: across the tested operating grid our ε is
    always >= Opacus' ε. The accountant may over-state how much privacy you
    spent, but it never under-states it — so a privacy claim made with it is
    never optimistic.

    Note: this is empirically verified on a finite grid (σ ∈ {0.5, 1.0, 2.0},
    q ∈ {0.01, 0.1, 0.48}). It is NOT a mathematical proof for all parameter
    combinations. The per-parameter clipping caveat (see SECURITY.md) can
    override this conservatism when P > n.
    """
    from dp_lora.accountant import compute_epsilon
    for sigma in (0.5, 1.0, 2.0):
        for q in (0.01, 0.1, 0.48):
            ours = compute_epsilon(sigma, 80, 1e-5, q)
            theirs = opacus_epsilon(sigma, 80, q)
            assert ours >= theirs - 1e-6, f"under-estimate at σ={sigma}, q={q}"


def test_import_smoke():
    """Basic import must succeed and expose the public API."""
    import dp_lora
    assert hasattr(dp_lora, '__version__')
    assert hasattr(dp_lora, 'train_dp_lora')
    assert hasattr(dp_lora, 'DPConfig')
    assert hasattr(dp_lora, 'compute_epsilon')
    assert hasattr(dp_lora, 'calibrate_noise_for_epsilon')
