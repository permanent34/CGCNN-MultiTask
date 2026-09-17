# CGCNN 实验结果汇总

本文件只汇总当前最新、可追溯的 Materials Project 实验结果。所有指标均可由对应 `test_results.csv` 重新计算。

## 1. MP 单任务

- 目标：`formation_energy_per_atom`，单位 eV/atom；
- 结构数：36,358；
- 训练 / 验证 / 测试：29,086 / 3,635 / 3,635；
- epochs：30；
- batch size：256；
- 输出任务数：1。

| 指标 | 结果 |
|---|---:|
| 最好验证 MAE | 0.0810 |
| 测试 MAE | 0.079801 |
| 测试 RMSE | 0.121411 |
| 测试 R² | 0.987963 |
| 测试样本数 | 3,635 |

来源：

```text
results/mp_presentation_20260917/single_task_30e/
```

## 2. MP 多任务

- 目标：形成能、带隙、费米能；
- 结构数：36,358；
- 训练 / 验证 / 测试：29,086 / 3,635 / 3,635；
- epochs：30；
- batch size：256；
- 输出任务数：3；
- 缺失标签：形成能 3，带隙 0，费米能 6。

| 任务 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 形成能 | 0.104661 | 0.148992 | 0.981872 |
| 带隙 | 0.430826 | 0.688462 | 0.844805 |
| 费米能 | 0.447305 | 0.673865 | 0.943385 |
| 简单平均 MAE | 0.327597 | — | — |

简单平均只用于辅助汇总，不能替代分任务结果。

来源：

```text
results/mp_presentation_20260917/multitask_30e/
```

## 3. 汇报材料

- PPT：`presentation/output/CGCNN_MP_GroupMeeting_20260917_rebuilt_v5_math_eq.pptx`
- 演讲稿：`presentation/output/CGCNN_MP_GroupMeeting_20260917_v5_speech.txt`

## 4. 结果解释原则

1. 单任务与多任务分别设置、分别训练、分别报告。
2. 验证集 MAE 与测试集 MAE 分开。
3. 多任务必须分性质报告 MAE、RMSE 和 R²。
4. 不同性质不能只看简单平均 MAE。
5. 所有指标必须能够追溯到原始日志或测试 CSV。
