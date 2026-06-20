# Security & Privacy Notes

This is research code for **differentially private fine-tuning**. Several
cautions:

1. **The accountant is conservative but not a compliance certificate.** It
   over-estimates ε at low sampling rates (the safe direction — it claims
   *weaker* privacy than realised). It is cross-checked against Opacus
   (`dp_lora/calibrate.py`); validate against your own accountant before making
   any external privacy claim. See `REPORT.md` → "Accountant fidelity".

2. **Per-parameter clipping caveat.** The trainer clips gradients *per LoRA
   parameter tensor* and then averages across the batch. Standard DP-SGD clips
   the *global* per-sample gradient norm across all parameters. When the number
   of LoRA target modules (P) exceeds the batch size (n), per-parameter clipping
   adds more noise than necessary — but because each parameter is clipped
   independently to `C`, the effective sensitivity is `P·C` not `C`, while the
   accountant assumes sensitivity `C`. This means the accountant can
   **under-state ε** when `P > n`. For typical LoRA configs (P=6–12 modules) and
   batch sizes (n=48+), this is not a concern. For extreme configs (P=192, n=4),
   reported ε may be understated by up to 4×. **If your config has P > n, use
   global clipping or increase the batch size.** This is documented as a known
   limitation, not a silent bug.

3. **No data ships with this repo.** Examples use synthetic text or public
   `wikitext`. Do not commit real or PHI-bearing data. If you train on
   sensitive data, the DP guarantee is only as good as your clip bound, your
   sampling assumptions, and your δ — read `REPORT.md` before relying on it.

4. **"Conservative by construction" — scope.** The accountant's integer-order
   RDP bound is conservative vs Opacus' fractional-order bound *in isolation*.
   The three independent sources of conservatism (integer scan, older conversion
   formula, sampling_rate=1.0 default) ensure it over-estimates ε across the
   tested grid (σ ∈ {0.5, 1.0, 2.0}, q ∈ {0.01, 0.1, 0.48}). This has NOT been
   proven for all parameter combinations. The per-parameter clipping issue
   (point 2) can override this conservatism. **Verify against Opacus for your
   specific configuration before making any privacy claim.**

Report issues via GitHub issues. Do not include sensitive data in reports.

5. **Poisson vs. hypergeometric subsampling.** The RDP accountant assumes
   Poisson subsampling (each sample included independently with probability
   q). The trainer uses fixed-size batches (hypergeometric sampling without
   replacement). For small sampling rates (q << 1) the difference is
   negligible. For large q (e.g. q=0.48), Poisson is the standard
   assumption in the DP literature and our ε is valid under it, but the actual
   privacy may be slightly stronger (hypergeometric amplifies privacy more than
   Poisson). This is the safe direction — we may over-state ε, never under-state
   it due to sampling assumptions. See Mironov–Talwar–Zhang 2019 §1.3 for
   discussion.
