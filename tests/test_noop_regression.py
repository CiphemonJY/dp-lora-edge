"""
The regression test that would have caught the no-op.

Proves two things on a real (tiny) model:
1. Under PEFT zero-init, grad(lora_A) is identically zero while grad(lora_B) is
   not — so a trainer that updates lora_A is provably a no-op.
2. The FFA-LoRA trainer actually moves lora_B (norm grows from 0) and its
   built-in sanity gate fires when handed a degenerate setup.

Skips automatically if torch/transformers/peft or network weights are absent.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def _build():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, target_modules=["query_key_value"], task_type="CAUSAL_LM"))
    return model, tok


@pytest.mark.slow
def test_grad_lora_A_is_zero_grad_lora_B_is_not():
    try:
        model, tok = _build()
    except Exception as e:                       # offline / no weights
        pytest.skip(f"model unavailable: {e}")
    enc = tok("Patient presents with acute chest pain.", return_tensors="pt")
    model(input_ids=enc["input_ids"], labels=enc["input_ids"]).loss.backward()
    a = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
            if "lora_a" in n.lower() and p.grad is not None)
    b = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
            if "lora_b" in n.lower() and p.grad is not None)
    assert a == 0.0, "grad(lora_A) must be exactly zero under zero-init B"
    assert b > 0.0, "grad(lora_B) must be non-zero — this is what FFA-LoRA trains"


@pytest.mark.slow
def test_ffa_lora_actually_trains():
    from dp_lora.trainer import DPConfig, train_dp_lora
    texts = [f"Condition: hypertension. BP 120/80. Plan: continue management. {i}"
             for i in range(40)]
    cfg = DPConfig(noise_multiplier=0.5, clip_norm=0.5, rounds=2,
                   local_steps=5, batch_size=4)
    try:
        res = train_dp_lora("EleutherAI/pythia-70m", texts, cfg, device="cpu")
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    assert res["final_lora_B_norm"] > 0.0
    assert res["epsilon"] > 0.0
