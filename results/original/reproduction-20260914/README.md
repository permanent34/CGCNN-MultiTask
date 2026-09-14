# Original CGCNN sample reproduction

## Purpose

This run verifies that the original single-task CGCNN training and prediction pipelines can execute in the current environment. It is a smoke test, not a scientific performance benchmark.

## Training command

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe <project>\main.py `
  --epochs 30 `
  --train-size 6 `
  --val-size 2 `
  --test-size 2 `
  --print-freq 1 `
  <project>\data\sample-regression
```

## Dataset split

- Training: 6
- Validation: 2
- Test: 2
- Targets: 1

## Real results

| Item | Value |
|---|---:|
| Best validation epoch | 28 |
| Best validation MAE | 2.425 |
| Final test MAE | 3.813 |
| Final test loss | 1.9724 |

`sample-regression` has only 10 structures. These values must not be presented as evidence of model quality.

## Prediction run

The best checkpoint was evaluated on all 10 sample structures:

| Item | Value |
|---|---:|
| Prediction rows | 10 |
| MAE | 2.389277 |
| RMSE | 2.916668 |
| R2 | -0.031145 |

## Files

- `environment.txt`: saved environment and GPU information
- `train_sample.log`: raw training output
- `sample-single-task_metrics.csv`: parsed epoch-level metrics
- `sample-single-task_summary.json`: best validation and final test summary
- `test_results.csv`: test predictions written by `main.py`
- `predict/`: `predict.py` run and its metrics
