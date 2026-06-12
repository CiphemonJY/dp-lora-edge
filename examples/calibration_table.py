"""
Reproduce the privacy/utility calibration table from REPORT.md (Arm C).

Uses only the built-in accountant (no GPU, no training). If Opacus is
installed, the cross-check columns are populated too. Run:

    python examples/calibration_table.py
"""
from dp_lora.accountant import compute_epsilon, noise_per_element
from dp_lora.calibrate import calibrate, cross_check

# The experimental operating points from the report. 4 rounds × 20 local steps
# = 80 steps; sampling rate = batch / pool. The pool here is the 57k-prompt
# FHIR corpus; q is reported per config below.
STEPS = 80
CONFIGS = [
    {"name": "Gate 1", "sigma": 0.5, "clip": 0.5, "batch": 4,  "q": 4 / 57000},
    {"name": "Arm B",  "sigma": 0.5, "clip": 0.5, "batch": 48, "q": 48 / 57000},
]

print("This accountant + Opacus at the documented operating points.")
print("Historical *measured* epsilon from the originating runs is in results/gates.json;")
print("this script demonstrates the reusable accountant, not a replay of that run.\n")
print(f"{'config':<8} {'sigma':>6} {'batch':>6} {'noise/elt':>10} "
      f"{'eps(ours)':>10} {'eps(opacus)':>12} {'disagree':>9}")
for c in CONFIGS:
    ne = noise_per_element(c["sigma"], c["clip"], c["batch"])
    cc = cross_check(c["sigma"], STEPS, c["q"])
    op = cc["epsilon_opacus"]
    dis = cc["relative_disagreement"]
    print(f"{c['name']:<8} {c['sigma']:>6} {c['batch']:>6} {ne:>10.4f} "
          f"{cc['epsilon_ours']:>10.2f} {str(op):>12} "
          f"{(f'{dis*100:.1f}%' if dis is not None else '—'):>9}")

print()
cal = calibrate(target_epsilon=8.0, steps=STEPS, sampling_rate=48 / 57000)
print(f"Calibrated for eps=8 at batch=48: sigma={cal['calibrated_sigma']} "
      f"(achieves eps={cal['achieved_epsilon']})")
print(f"  noise/element at calibrated sigma: "
      f"{noise_per_element(cal['calibrated_sigma'], 0.5, 48):.4f}")
