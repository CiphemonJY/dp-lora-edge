# Changelog

## [0.2.2] — 2026-06-19

### Added
- **Import smoke test** — `test_import_smoke` verifies basic packaging works.
- **δ validation in `calibrate_noise_for_epsilon`** — raises `ValueError` for
  delta outside (0, 1) and non-positive sampling_rate, matching `compute_epsilon`.
- **`py.typed` marker** — PEP 561 marker so type checkers know the package
  ships type hints.
- **CLI entry point** — `dp-lora-calibrate --target-epsilon 8 --steps 80`
  command-line tool for quick privacy budget calibration.
- **Standalone training script** — `examples/train_dp_lora.py` — quick-start
  FFA-LoRA DP-SGD on wikitext-2. Downloads data, calibrates σ, trains, reports.
- **Poisson/hypergeometric sampling note** — SECURITY.md now documents that the
  accountant assumes Poisson subsampling while the trainer uses fixed-size
  batches. Safe direction (over-states ε for hypergeometric).

### Changed
- **"Conservative by construction" → "conservative in the tested regime"** in
  README. The phrase "by construction" implies mathematical proof; we only
  have empirical verification on a finite grid.
- **Single-seed caveat** added to REPORT.md — gates.json results are from
  single-seed runs, exact perplexity may vary ±1-2 points.

### Fixed
- **calibration_table.py parameter mismatch** — verified resolved (signatures
  match in v0.2.0+). Smoke test confirms the example script runs correctly.

## [0.2.1] — 2026-06-19

### Added
- **Multi-GPU distributed training**: `train_dp_lora` now detects
  `torch.distributed` and automatically splits clients across ranks + averages
  LoRA-B via NCCL all-reduce. Validated on 2-node DGX Spark cluster (1.86x
  speedup). Use `torchrun --nproc_per_node=N` to launch.
- `_all_reduce_lora_b()` — NCCL all-reduce helper, mirrors LISA_FTM's 2-node
  training pattern.
- `DPConfig.dtype` — "float32" (default) or "bfloat16" (~2x faster on modern
  GPUs like GB10, A100, H100).
- `DPConfig.clients_per_round` — number of local clients per round. Set >1
  for multi-GPU parallelism (each rank trains its share independently).
- `tests/test_gate.py` — 4 new tests for dtype, clients_per_round,
  _is_distributed (21 total, all pass).

### Changed
- `build_model` now respects `cfg.dtype` for bfloat16 mixed precision.
- `train_dp_lora` now splits `clients_per_round` across `world_size` ranks.
- ε accounting now counts `steps = rounds * local_steps * total_clients`.
- Trainer docstring updated with multi-GPU usage instructions.

## [0.2.0] — 2026-06-19

### Fixed
- **Noise formula corrected**: `noise_per_element` and trainer noise injection
  now use `σ·C/n` (standard DP-SGD) instead of `σ·C/√n` (√n larger). The
  qualitative findings are unchanged but quantitative noise values are smaller.
- **gates.json delta_pct corrected**: Arm A delta was 93.6% (wrong), now 84.1%
  (matches `compare_to_base` formula). Arm B delta was 72.2% (wrong), now 30.9%.
- **"pip install dp-lora-edge" removed from README**: package is not on PyPI.
  Install instructions now use `pip install git+https://...`.
- **"Never under-states ε" claim scoped**: README and SECURITY.md now document
  that the conservatism guarantee is empirically verified on a finite grid, not
  mathematically proven for all parameters. Per-parameter clipping caveat
  documented (can under-state ε when P > n).
- **"Agrees within 10%" claim scoped**: README now states the regime where this
  holds (σ ≥ 1, moderate sampling) and links to REPORT.md for the full
  characterization.

### Added
- `tests/test_gate.py` — 10 unit tests for gate.py (was previously untested).
- `.github/workflows/ci.yml` — CI runs pytest on Python 3.11+3.12.
- `pyproject.toml` — added classifiers, Issues URL.
- SECURITY.md — per-parameter clipping caveat, scope of conservatism claim.
- `compute_epsilon` — input validation for `delta` and `sampling_rate`.
- `noise_per_element` — input validation for `batch_size`.
- `DPConfig.global_clip` — global per-sample clipping option (standard DP-SGD).
- `DPConfig.sampling_rate` — Poisson subsampling rate for ε accounting.
- README.md — "Validated at scale" section with 2-node DGX training results.

### Changed
- `test_accountant_is_conservative_never_underestimates` →
  `test_accountant_is_conservative_in_tested_grid` (honest name).
- `test_noise_per_element_averages_down_with_batch` updated for `σ·C/n` formula.
- REPORT.md updated: corrected delta values, noise formula, per-parameter
  clipping section, scoped conservatism claims.

## [0.1.0] — 2026-06-12

Initial release. FFA-LoRA trainer, RDP accountant, sanity gates, 3-arm ablation
results, Opacus cross-validation.