# Original single-task results

This directory contains saved single-task predictions and parsed metrics.

## Long run

- Source predictions: `test_results.csv`
- Training run: 120 epochs
- Rows: 866
- Best validation MAE: 0.2946
- Test MAE: 0.311508
- Test RMSE: 0.539655
- Test R2: 0.410632

## Small reproduction

`reproduction-20260914/` reruns the original code on the ten-structure `sample-regression` dataset. It verifies the complete training and prediction workflow but is not a meaningful scientific benchmark.

## Current-code 30-epoch rerun

`rerun-shc-max-log-20260914/` contains a fresh 30-epoch run on the 8,668-structure `shc-max-log` dataset.

- Best validation MAE: 0.311909
- Test MAE: 0.318826
- Test RMSE: 0.524193
- Test R2: 0.443920
- Raw log, parsed epoch data, environment snapshot, and 866 predictions are included.

The training and validation loss curves are marked as reported running averages because the run used `--print-freq 10`. Exact validation and test MAE are taken from the `* MAE` and `** MAE` log lines.
