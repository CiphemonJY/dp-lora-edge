"""
dp-lora-edge — differentially private LoRA fine-tuning at calibrated ε,
with a self-contained RDP accountant and honest, anomaly-aware gating.
"""
from .accountant import (
    calibrate_noise_for_epsilon,
    compute_epsilon,
    noise_per_element,
)
from .calibrate import calibrate, cross_check, opacus_epsilon
from .gate import compare_to_base, gate, lora_diagnostics, perplexity
from .trainer import DPConfig, SanityGateError, build_model, train_dp_lora

__version__ = "0.2.2"

__all__ = [
    "compute_epsilon",
    "calibrate_noise_for_epsilon",
    "noise_per_element",
    "cross_check",
    "calibrate",
    "opacus_epsilon",
    "gate",
    "compare_to_base",
    "lora_diagnostics",
    "perplexity",
    "DPConfig",
    "SanityGateError",
    "train_dp_lora",
    "build_model",
]
