# CGCNN 实验结果汇总

本文件只汇总项目目录中已经保存的真实结果。所有测试指标都可以由对应 CSV 重新计算。

## 1. 单任务 CGCNN

### 1.1 本次重新运行：30 轮

- 数据：`data/shc-max-log`
- 总结构数：8,668
- 划分：80% / 10% / 10%
- batch size：256
- epoch：30
- 输出任务数：1

| 指标 | 结果 |
|---|---:|
| 最好验证 MAE | 0.311909 |
| 测试集 MAE | 0.318826 |
| 测试集 RMSE | 0.524193 |
| 测试集 R² | 0.443920 |
| 测试集样本数 | 866 |

来源：

```text
results/original/rerun-shc-max-log-20260914/
```

### 1.2 历史 120 轮结果

| 指标 | 结果 |
|---|---:|
| 最好验证 MAE | 0.2946 |
| 测试集 MAE | 0.311508 |
| 测试集 RMSE | 0.539655 |
| 测试集 R² | 0.410632 |
| 测试集样本数 | 866 |

来源：

```text
results/original/test_results.csv
results/original/metrics.json
```

## 2. 多任务 CGCNN

- 数据：`data/mt-large`
- 总结构数：36,358
- 有效测试样本：3,635
- epoch：30
- 任务数：3
- 任务：形成能、带隙、费米能
- 缺失标签：形成能 3 个，带隙 0 个，费米能 6 个

| 任务 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 形成能 | 0.109772 | 0.154312 | 0.980555 |
| 带隙 | 0.424296 | 0.697009 | 0.840928 |
| 费米能 | 0.447355 | 0.670312 | 0.943981 |
| 简单平均 | 0.327141 | - | - |

来源：

```text
results/multitask/test_results.csv
results/multitask/metrics.json
```

注意：简单平均 MAE 混合了三个不同量纲和难度的任务，不能单独作为多任务模型的结论。

## 3. 代码流程测试

### 单任务小样本

- 数据：`data/sample-regression`
- 训练/验证/测试：6 / 2 / 2
- epoch：30
- 最终测试 MAE：3.813
- 对全部 10 个样本预测：MAE 2.389277

该结果只说明代码运行成功，不说明模型性能。

### 多任务小样本

- 数据：`data/mt-demo`
- 训练/验证/测试：5 / 2 / 2
- epoch：30
- 任务数：3
- 最终测试平均 MAE：0.568
- 对全部 9 个样本预测：平均 MAE 0.707545

该结果只说明多任务训练和预测流程运行成功。

## 4. 生成的真实图片

- `figures/training_loss.png`
- `figures/training_mae.png`
- `figures/real_multitask_metrics.png`
- `figures/original_sample_training_loss.png`
- `figures/original_sample_training_mae.png`
- `figures/multitask_demo_training_loss.png`
- `figures/multitask_demo_training_mae.png`

结构示意图：

- `figures/cgcnn_pipeline.png`
- `figures/crystal_graph.png`
- `figures/multitask_model.png`
- `figures/deployment_flow.png`
- `figures/single_vs_multitask.png`

## 5. 结果解释原则

1. 验证集 MAE 和测试集 MAE 分开报告。
2. 多任务必须分任务报告 MAE、RMSE 和 R²。
3. 不同数据集、不同任务数的结果不能直接比较。
4. 小样本结果只能验证代码，不能用于科研结论。
5. 所有结果都应保留原始日志、CSV 和来源哈希。
