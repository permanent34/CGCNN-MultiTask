# Multitask results

This directory contains the saved three-task `mt-large` results and a separate small demo reproduction.

## Main three-task result

- Source predictions: `test_results.csv`
- Source run: `data/mt-run/`
- Rows: 3,635
- Targets: formation energy per atom, band gap, Efermi
- Epochs: 30
- Best validation mean MAE: 0.3259

| Task | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Formation energy | 0.109772 | 0.154312 | 0.980555 |
| Band gap | 0.424296 | 0.697009 | 0.840928 |
| Efermi | 0.447355 | 0.670312 | 0.943981 |

Mean task MAE is 0.327141, but this average combines different properties and should not replace per-task reporting.

The exact source CSV hash is stored in `metrics.json`.

## Reproduction subdirectory

`reproduction-20260914/` verifies the modified training and prediction path on the nine-sample `mt-demo` dataset. It is a smoke test and should not be used as a performance claim.
