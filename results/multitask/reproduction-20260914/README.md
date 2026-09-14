# Multitask CGCNN demo reproduction

## Purpose

This run verifies that the modified multitask path can train and predict three targets simultaneously. It uses `data/mt-demo` with nine samples, so it is a functional smoke test rather than a scientific benchmark.

## Training command

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe <project>\main.py `
  --num-targets 3 `
  --epochs 30 `
  --batch-size 3 `
  --train-size 5 `
  --val-size 2 `
  --test-size 2 `
  --print-freq 1 `
  <project>\data\mt-demo
```

## Dataset split

- Training: 5
- Validation: 2
- Test: 2
- Targets: 3

## Real training results

| Item | Value |
|---|---:|
| Best validation epoch | 8 |
| Best validation MAE | 0.613 |
| Final test MAE | 0.568 |
| Final test loss | 0.3557 |

## Prediction run

The best checkpoint was evaluated on all nine demo structures.

| Task | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Formation energy | 0.394782 | 0.550475 | 0.359460 |
| Band gap | 0.903078 | 1.089489 | 0.186937 |
| Fermi energy | 0.824777 | 1.127878 | 0.099507 |

## What this run proves

- The dataset reads three target columns.
- The model output layer has shape `(3, 128)`.
- The loss ignores missing values through a mask.
- Test output is written as `id + 3 targets + 3 predictions`.
- `predict.py` reconstructs a three-output model from the checkpoint.

## What this run does not prove

- It does not show that multitask learning is better than single-task learning.
- It does not provide reliable performance estimates because the dataset is extremely small.
- It does not test negative transfer.

## Files

- `train_multitask.log`: raw training output
- `demo-multitask_metrics.csv`: parsed epoch-level metrics
- `demo-multitask_summary.json`: best validation and final test summary
- `test_results.csv`: test predictions written by `main.py`
- `predict/`: `predict.py` run, seven-column CSV, and per-task metrics
