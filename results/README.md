# Results

本目录只保存由真实运行生成的测试结果与由这些结果重新计算的指标。

- `original/`：原始单任务 CGCNN 的运行结果。
- `multitask/`：三任务 CGCNN 的运行结果。
- `metrics.json` / `metrics.md`：通过 `tools/summarize_results.py` 从测试 CSV 重算的 MAE、RMSE 和 R²。

测试 CSV 不包含表头。多任务文件格式为：

```text
id,target_1,...,target_K,prediction_1,...,prediction_K
```

当前完整逐轮训练日志尚未保存，因此这里暂时不放 loss 曲线。后续重新训练时会保存原始日志，并从日志生成真实曲线。
