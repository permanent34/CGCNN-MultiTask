# Single-task shc-max-log rerun

## Purpose

This run repeats single-task CGCNN training for 30 epochs on `data/shc-max-log` and saves the complete console output, parsed epoch metrics, predictions, and test metrics.

## Command

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe <project>\main.py `
  --epochs 30 `
  --batch-size 256 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  --print-freq 10 `
  <project>\data\shc-max-log
```

Run directory was isolated from the project root. Existing checkpoints and result files in the root were not overwritten.

## Dataset

- Total structures: 8,668
- Training split: 80%
- Validation split: 10%
- Test split: 10%
- Test rows: 866
- Targets: 1

## Real results

| Item | Value |
|---|---:|
| Log epoch of best validation | 15 |
| Checkpoint epoch of best model | 16 |
| Best validation MAE | 0.311909 |
| Final test MAE | 0.318826 |
| Final test RMSE | 0.524193 |
| Final test R2 | 0.443920 |

The final test MAE shown in the raw log is rounded to `0.319`. The CSV-derived value is more precise.

## Logging note

`--print-freq 10` prints only some training and validation batches. The parser therefore marks training loss and validation loss as incomplete where the final batch was not printed. Validation MAE and final test MAE are taken from the exact `* MAE` and `** MAE` lines.

## Runtime

- Start: approximately 2026-09-14 16:34:58
- Finish: approximately 2026-09-14 16:41:54
- First epoch included one-time CIF parsing and was much slower than later epochs.

## Files

- `environment.txt`: environment and GPU snapshot
- `train_shc_max_log_30e.log`: raw console output
- `shc-max-log-30e_metrics.csv`: parsed epoch-level metrics
- `shc-max-log-30e_summary.json`: best validation and final test summary
- `test_results.csv`: 866 test predictions
- `metrics.json` / `metrics.md`: test metrics recomputed from the CSV
