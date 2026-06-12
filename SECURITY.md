# Security & Privacy Notes

This is research code for **differentially private fine-tuning**. Two cautions:

1. **The accountant is conservative but not a compliance certificate.** It
   over-estimates ε at low sampling rates (the safe direction — it claims
   *weaker* privacy than realised). It is cross-checked against Opacus
   (`dp_lora/calibrate.py`); validate against your own accountant before making
   any external privacy claim. See `REPORT.md` → "Accountant fidelity".

2. **No data ships with this repo.** Examples use synthetic text or public
   `wikitext`. Do not commit real or PHI-bearing data. If you train on
   sensitive data, the DP guarantee is only as good as your clip bound, your
   sampling assumptions, and your δ — read `REPORT.md` before relying on it.

Report issues via GitHub issues. Do not include sensitive data in reports.
