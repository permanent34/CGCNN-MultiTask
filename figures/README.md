# Figures

本目录保存由项目代码、架构说明和真实实验日志生成的可追溯图片。

## 最新 MP 汇报图

位于 `figures/mp_report/`：

| Figure | Type | Source |
|---|---|---|
| `cgcnn_conv_formula.png` | 数学排版公式 | `cgcnn/model.py` 的 `ConvLayer` |
| `single_training_curves.png` | 真实训练曲线 | `results/mp_presentation_20260917/single_task_30e/` |
| `single_test_results.png` | 真实测试预测与残差 | 同一运行的 `test_results.csv` |
| `multitask_training_curves.png` | 真实训练曲线 | `results/mp_presentation_20260917/multitask_30e/` |
| `multitask_metric_bars.png` | 分任务指标图 | 多任务 `metrics.json` |
| `multitask_test_results.png` | 三任务预测散点图 | 多任务 `test_results.csv` |

## 通用结构示意

- `cgcnn_pipeline.png`：CGCNN 主干流程；
- `crystal_graph.png`：晶体图表示；
- `multitask_model.png`：共享主干与多输出结构；
- `deployment_flow.png`：部署流程。

小样本流程图只用于验证代码可以运行，不作为性能结论。
