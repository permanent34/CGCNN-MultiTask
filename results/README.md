# Results

本目录只保存由真实运行生成的测试结果，以及从这些结果重新计算的指标。

## 最新 MP 汇报结果

- `mp_presentation_20260917/single_task_30e/`：MP 形成能单任务，30 轮；
- `mp_presentation_20260917/multitask_30e/`：MP 形成能、带隙、费米能三任务，30 轮；
- 测试指标由 `test_results.csv` 重新计算；
- `metrics.json` 记录源 CSV 的 SHA-256；
- 训练日志、逐轮指标、预测 CSV 和摘要均保存在对应目录。

## 历史结果

`original/` 和 `multitask/` 中的旧结果仍保留用于追溯，但最新组会 PPT、演讲稿和主 README 只采用 `mp_presentation_20260917/` 下的 MP 结果。
