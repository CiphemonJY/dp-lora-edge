"""Tests for dp_lora/gate.py — perplexity, diagnostics, and compare_to_base.

These are unit tests that don't require a real model — they test the
logic of the gate functions with mock objects.
"""
import math

import pytest

torch = pytest.importorskip("torch")


class MockModel:
    """Minimal mock that returns a fixed loss from model()."""
    def __init__(self, loss_value=2.5, vocab_size=100):
        self.loss_value = loss_value
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.training = True

    def __call__(self, **kwargs):
        return type("Output", (), {"loss": torch.tensor(self.loss_value)})()

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class MockTokenizer:
    """Minimal mock tokenizer that returns fixed tensors."""
    def __call__(self, texts, max_length=None, padding=None, truncation=None,
                 return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        ids = torch.zeros(n, max_length or 8, dtype=torch.long)
        mask = torch.ones(n, max_length or 8, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": ids, "attention_mask": mask}


class MockParams:
    """Mock parameter with .norm() and .item()."""
    def __init__(self, norm_val):
        self._norm = norm_val

    def norm(self):
        return torch.tensor(self._norm)


def make_mock_model_with_lora(norms):
    """Create a mock model with named parameters having lora_b in their names."""
    model = MockModel()
    params = []
    for i, n in enumerate(norms):
        p = MockParams(n)
        # Attach name attribute via namedtuple pattern
        params.append((f"model.layers.{i}.lora_B.weight", p))
    model.named_parameters = lambda: iter(params)
    return model


def test_lora_diagnostics_all_zero():
    """When all lora_B norms are zero, lora_B_degenerate should be True."""
    from dp_lora.gate import lora_diagnostics
    model = make_mock_model_with_lora([0.0, 0.0, 0.0])
    diag = lora_diagnostics(model)
    assert diag["lora_B_degenerate"] is True
    assert diag["zero_lora_B_count"] == 3
    assert diag["total_lora_layers"] == 3


def test_lora_diagnostics_healthy():
    """When all lora_B norms are non-zero, lora_B_degenerate should be False."""
    from dp_lora.gate import lora_diagnostics
    model = make_mock_model_with_lora([1.5, 2.3, 0.8])
    diag = lora_diagnostics(model)
    assert diag["lora_B_degenerate"] is False
    assert diag["zero_lora_B_count"] == 0
    assert diag["total_lora_layers"] == 3
    assert math.isclose(diag["avg_lora_B_norm"], (1.5 + 2.3 + 0.8) / 3, rel_tol=1e-4)


def test_lora_diagnostics_partial_zero():
    """When some lora_B norms are zero, lora_B_degenerate should be False."""
    from dp_lora.gate import lora_diagnostics
    model = make_mock_model_with_lora([0.0, 1.5, 0.0])
    diag = lora_diagnostics(model)
    assert diag["lora_B_degenerate"] is False
    assert diag["zero_lora_B_count"] == 2
    assert diag["total_lora_layers"] == 3


def test_compare_to_base_pass():
    """compare_to_base should pass when trained beats base by min_improvement."""
    from dp_lora.gate import compare_to_base
    trained = {"perplexity": 9.59, "lora_B_degenerate": False, "avg_lora_B_norm": 9.69}
    result = compare_to_base(base_ppl=13.88, trained=trained, min_improvement=0.05)
    assert result["pass"] is True
    assert result["anomaly"] is False
    # delta = (13.88 - 9.59) / 13.88 = 0.309... = 30.9%
    assert abs(result["delta_pct"] - 30.91) < 0.5


def test_compare_to_base_fail_no_improvement():
    """compare_to_base should fail when trained is worse than base."""
    from dp_lora.gate import compare_to_base
    trained = {"perplexity": 14.49, "lora_B_degenerate": False, "avg_lora_B_norm": 1.15}
    result = compare_to_base(base_ppl=13.88, trained=trained, min_improvement=0.05)
    assert result["pass"] is False
    assert result["delta_pct"] < 0  # negative delta = worse than base


def test_compare_to_base_anomaly_degenerate():
    """compare_to_base should flag anomaly when lora_B is degenerate."""
    from dp_lora.gate import compare_to_base
    trained = {"perplexity": 13.88, "lora_B_degenerate": True, "avg_lora_B_norm": 0.0}
    result = compare_to_base(base_ppl=13.88, trained=trained, min_improvement=0.05)
    assert result["pass"] is False
    assert result["anomaly"] is True


def test_compare_to_base_anomaly_identical_ppl():
    """compare_to_base should flag anomaly when trained == base (no-op signature)."""
    from dp_lora.gate import compare_to_base
    trained = {"perplexity": 13.880, "lora_B_degenerate": False, "avg_lora_B_norm": 1.0}
    result = compare_to_base(base_ppl=13.88, trained=trained, min_improvement=0.05)
    assert result["pass"] is False
    assert result["anomaly"] is True


def test_noise_per_element_correct_formula():
    """Verify the corrected noise formula: sigma * C / n (not sqrt(n))."""
    from dp_lora.accountant import noise_per_element
    # sigma=0.5, C=0.5, batch=4: 0.5 * 0.5 / 4 = 0.0625
    assert math.isclose(noise_per_element(0.5, 0.5, 4), 0.0625, rel_tol=1e-6)
    # sigma=0.5, C=0.5, batch=48: 0.5 * 0.5 / 48 = 0.005208...
    assert math.isclose(noise_per_element(0.5, 0.5, 48), 0.5 * 0.5 / 48, rel_tol=1e-6)


def test_compute_epsilon_rejects_invalid_delta():
    """compute_epsilon should raise for delta outside (0, 1)."""
    from dp_lora.accountant import compute_epsilon
    with pytest.raises(ValueError):
        compute_epsilon(1.0, 80, delta=0.0)
    with pytest.raises(ValueError):
        compute_epsilon(1.0, 80, delta=1.0)
    with pytest.raises(ValueError):
        compute_epsilon(1.0, 80, delta=-0.5)


def test_global_clip_config_exists():
    """DPConfig should support global_clip option."""
    from dp_lora.trainer import DPConfig
    cfg = DPConfig(global_clip=True)
    assert cfg.global_clip is True
    cfg2 = DPConfig()
    assert cfg2.global_clip is False  # default: per-parameter


def test_sampling_rate_config_exists():
    """DPConfig should support sampling_rate for ε accounting."""
    from dp_lora.trainer import DPConfig
    cfg = DPConfig(sampling_rate=0.1)
    assert cfg.sampling_rate == 0.1
    cfg2 = DPConfig()
    assert cfg2.sampling_rate == 1.0  # default: full participation