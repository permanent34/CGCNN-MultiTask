# CGCNN 多任务预测复现与改进

本项目复现 Crystal Graph Convolutional Neural Network（CGCNN），并在原始单任务回归代码基础上，增加了共享表示的多任务回归能力。当前版本可以一次预测多个材料性质，并对缺失标签使用掩码损失，不改动 CGCNN 的图卷积主体。

> 重要说明：README 中的实验数值均来自 `results/` 或已保存 checkpoint/test CSV。不同数据集、不同任务数和不同单位的结果不能直接横向比较。

## 1. 项目简介

CGCNN 是一种直接以晶体结构为输入的图神经网络。它把晶体中的原子看作节点，把原子之间的近邻关系看作边，通过图卷积学习晶体表示，再预测形成能、带隙、费米能等材料性质。

本项目完成了以下工作：

- 复现原始 CGCNN 的单任务回归训练与预测流程；
- 在 `cgcnn/model.py`、`cgcnn/data.py`、`main.py`、`predict.py` 中加入多任务支持；
- 使用 Materials Project 数据完成三任务训练；
- 保存测试预测 CSV，并从 CSV 重新计算 MAE、RMSE、R²；
- 增加独立复现实验目录、环境记录、训练日志解析和图表生成脚本。

## 2. CGCNN 论文简介

论文：Tian Xie and Jeffrey C. Grossman, *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, Physical Review Letters, 2018.

### CGCNN 解决什么问题？

传统材料性质预测方法通常需要人工设计特征，或者使用不直接包含晶体局部连接关系的模型。CGCNN 使用晶体图自动学习结构表示，减少人工特征工程，并能利用周期边界条件下的局部原子环境。

### 为什么使用晶体图？

晶体中每个原子与周围原子之间存在周期性近邻关系。图中的节点和边可以自然表示：

- Node（节点）：一个原子及其初始特征；
- Edge（边）：中心原子与其邻域原子的连接；
- Graph（图）：由 nodes、edges 和图结构组成的晶体表示。

本项目使用的原子初始特征是 92 维向量；每条边使用距离的高斯展开特征。默认最多保留 12 个近邻，截断半径为 8 Å，距离步长为 0.2 Å，因此边特征为 41 维。

### Graph Convolution 是什么？

图卷积让每个原子聚合邻居信息，更新自己的特征。连续做多层图卷积后，原子的表示会包含更大范围的局部环境信息。

直观过程可以写成：

```text
中心原子特征 + 邻居原子特征 + 边特征
                    ↓
              拼接与线性变换
                    ↓
              门控与 Softplus
                    ↓
            聚合邻居并更新中心原子
```

### 如何从原子特征得到整个晶体的特征？

每个原子经过若干层图卷积后，对同一晶体中的所有原子特征取平均，得到整个晶体的表示。这一步称为 pooling。

### 最终如何预测材料性质？

晶体表示经过全连接层后输出目标性质。原始 CGCNN 回归输出 1 个值；本项目通过 `num_targets` 参数将输出改为 K 个值。

## 3. 项目结构

```text
CGCNN-MultiTask/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── main.py
├── predict.py
├── download_mp_data.py
├── cgcnn/
│   ├── data.py
│   └── model.py
├── data/
│   ├── sample-regression/
│   ├── sample-classification/
│   ├── mt-demo/
│   └── mt-large/                 # 本地大数据，默认不上传
├── pre-trained/
├── results/
│   ├── original/
│   └── multitask/
├── figures/
├── docs/
└── tools/
```

主要文件：

| 文件 | 作用 |
|---|---|
| `cgcnn/data.py` | CIF 读取、晶体图构建、多列标签读取、批处理 |
| `cgcnn/model.py` | 图卷积层、pooling 和最终回归/分类输出 |
| `main.py` | 训练、验证、测试、保存 checkpoint 和预测 CSV |
| `predict.py` | 加载 checkpoint，对新 CIF 目录进行预测 |
| `download_mp_data.py` | 从 Materials Project 下载 CIF 和性质标签 |
| `tools/summarize_results.py` | 从测试 CSV 计算 MAE、RMSE、R² |
| `tools/parse_training_log.py` | 从真实训练日志提取逐轮 loss/MAE |

## 4. 环境配置

当前已验证环境：

```text
操作系统：Windows
Conda：D:\Miniconda\miniconda
环境名：cgcnn
Python：3.10.20
PyTorch：2.7.0+cu128
CUDA runtime：12.8
cuDNN：9.7.1
GPU：NVIDIA GeForce RTX 5060 Laptop GPU
Compute Capability：12.0
驱动：582.05
```

安装方式：

```powershell
D:\Miniconda\miniconda\Scripts\conda.exe create -n cgcnn python=3.10
D:\Miniconda\miniconda\Scripts\conda.exe activate cgcnn
python -m pip install -r requirements.txt
```

如果直接输入 `conda` 找不到命令，可以使用 Conda 的绝对路径，或者先初始化 PowerShell：

```powershell
D:\Miniconda\miniconda\Scripts\conda.exe init powershell
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

本项目不要求 `torch_geometric`。图结构由 `cgcnn/data.py` 中的 PyTorch Dataset 和自定义 batch 函数构建。

## 5. CGCNN 部署过程

```text
获取源码
  ↓
创建 Python 3.10 Conda 环境
  ↓
安装 PyTorch CUDA 版本
  ↓
安装 pymatgen、numpy、scikit-learn 等依赖
  ↓
准备 CIF、id_prop.csv、atom_init.json
  ↓
运行 main.py 训练
  ↓
保存 checkpoint 和 test_results.csv
  ↓
运行 predict.py 对新材料预测
```

数据集需要三个核心部分：

```text
dataset/
├── id_prop.csv
├── atom_init.json
└── *.cif
```

单任务 CSV：

```text
id,property
```

多任务 CSV：

```text
id,property1,property2,property3
```

多任务中某一行缺少一个性质时，可以留空或写 `nan`。程序使用掩码损失，不把缺失值当作 0。

## 6. MP 单任务复现

单任务实验只使用 Materials Project 数据。结构与标签来自 `data/mt-large` 的同源 MP 记录，单任务视图使用其中的 `formation_energy_per_atom` 列。

训练命令：

```powershell
python main.py --epochs 30 --batch-size 256 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --num-targets 1 data/mp-single-formation
```

实验设置：

- 原始 MP ID 清单：46,744；
- 成功保留的 MP 结构：36,358；
- 训练 / 验证 / 测试：29,086 / 3,635 / 3,635；
- 输出目标：1 个，形成能 per atom，单位 eV/atom；
- batch size：256；epochs：30；SGD 初始学习率：0.01。

真实测试结果：

| 指标 | 数值 | 来源 |
|---|---:|---|
| 最好验证 MAE | 0.0810 | `single_summary.json` |
| 测试 MAE | 0.079801 | 3,635 条测试记录重算 |
| 测试 RMSE | 0.121411 | 3,635 条测试记录重算 |
| 测试 R² | 0.987963 | 3,635 条测试记录重算 |

结果为独立单任务结果页，不和 MP 多任务指标混合。原始日志、逐轮指标、测试 CSV 和 checkpoint 元数据保存在 `results/mp_presentation_20260917/single_task_30e/`。大型 CIF 派生视图和晶体图缓存属于本地可再生成数据，不提交到 GitHub。

## 7. 多任务预测改进

原始 CGCNN 的输出层：

```python
nn.Linear(h_fea_len, 1)
```

多任务输出层：

```python
nn.Linear(h_fea_len, num_targets)
```

结构如下：

```text
Crystal Structure
        ↓
Crystal Graph
        ↓
CGCNN Graph Convolutions
        ↓
Mean Pooling
        ↓
Shared Representation
        ↓
Shared Linear Output Layer
   ↓         ↓         ↓
Task 1   Task 2   Task 3
```

当前实现是在同一共享表示之后，用一行输出神经元对应一个任务。它实现了参数共享，但还没有加入任务专属多层网络、任务权重学习或不确定性建模。

MP 三任务测试集指标：

| 材料性质 | MAE | RMSE | R² | 有效测试数 |
|---|---:|---:|---:|---:|
| 形成能 `formation_energy_per_atom` | 0.104661 | 0.148992 | 0.981872 | 3,635 |
| 带隙 `band_gap` | 0.430826 | 0.688462 | 0.844805 | 3,635 |
| 费米能 `efermi` | 0.447305 | 0.673865 | 0.943385 | 3,635 |

三个任务的简单平均 MAE 约为 0.327597。由于三个性质单位不同、难度不同，平均值只能作为总体参考，不能替代分任务指标。完整结果位于 `results/mp_presentation_20260917/multitask_30e/`。

## 8. 代码修改

### `cgcnn/model.py`

- 新增 `num_targets` 参数；
- 回归输出层由 `Linear(h_fea_len, 1)` 改为 `Linear(h_fea_len, num_targets)`；
- 图卷积、pooling 和共享表示结构保持不变。

### `cgcnn/data.py`

- 将 CSV 中第 2 列之后的所有字段读取为多个标签；
- 新增 `_safe_float`，支持空值和 `nan`；
- 对新版 pymatgen 的 site 原子序数读取做了兼容；
- target shape 从 `[1]` 变为 `[K]`。

### `main.py`

- 新增 `--num-targets`；
- 回归损失由普通 MSE 改为 `masked_mse_loss`；
- `Normalizer` 改为按任务逐列统计，并忽略 NaN；
- MAE 只统计有效标签；
- 测试 CSV 支持 `id + K 个真实值 + K 个预测值`；
- 修复 Windows 下 CSV 多余空行问题。

### `predict.py`

- 从 checkpoint 中读取 `num_targets`；
- 对不包含该字段的旧模型默认使用 1；
- 多任务预测结果按 7 列格式写入 CSV。

## 9. 实验结果与汇报文件

最新 MP 汇报结果目录：

```text
results/mp_presentation_20260917/
├── environment.txt
├── single_task_30e/
│   ├── train_mp_single_30e.log
│   ├── single_metrics.csv
│   ├── single_summary.json
│   ├── test_results.csv
│   ├── metrics.json
│   └── metrics.md
└── multitask_30e/
    ├── train_mp_multitask_30e.log
    ├── multitask_metrics.csv
    ├── multitask_summary.json
    ├── test_results.csv
    ├── metrics.json
    └── metrics.md
```

`metrics.json` 记录源测试 CSV 的 SHA-256，`tools/summarize_results.py` 可以重新计算 MAE、RMSE 和 R²。训练曲线和预测散点图位于 `figures/mp_report/`。

最新组会材料：

- `presentation/output/CGCNN_MP_GroupMeeting_20260917_rebuilt_v5_math_eq.pptx`
- `presentation/output/CGCNN_MP_GroupMeeting_20260917_v5_speech.txt`

单任务和多任务分别训练、分别保存、分别在 PPT 中展示。多任务结果按性质分项报告，不用简单平均 MAE 作为唯一结论。

## 10. 遇到的问题与解决方法

| 问题 | 原因 | 解决方法 |
|---|---|---|
| 新 PyTorch 中 `site.specie` 兼容性变化 | pymatgen API 版本变化 | 增加 `_site_atomic_number` 兼容函数 |
| 多任务标签含缺失值 | 不是所有材料都有全部性质 | 引入 NaN 掩码 MSE 和逐列归一化 |
| `NaN × 0` 仍是 NaN | NaN 先乘掩码会污染损失 | 先将 NaN 替换为 0，再乘掩码 |
| Windows CSV 出现空行 | CSV 换行与文本模式叠加 | 使用 `open(..., newline='')` |
| 多任务 CSV 仍按两列解析 | 原代码写死单标签 | 将剩余列统一读取为 target 向量 |
| 同目录 checkpoint 覆盖 | `main.py` 写到当前工作目录 | 每次实验使用独立结果目录 |
| 训练日志没有逐轮保存 | 仅输出到终端 | 使用 `Tee-Object` 保存原始日志 |
| 小样本 loss 波动大 | 数据量不足 | 只用于流程验证，不用于结论 |

## 11. 学习收获

- Python：用脚本解析日志、生成指标和图片；
- PyTorch：理解 Tensor shape、loss、backward、optimizer 和 checkpoint；
- GPU：确认 PyTorch CUDA 与 RTX 5060 的兼容性；
- Deep Learning：理解训练集、验证集、测试集的职责；
- GNN：理解 node、edge、message aggregation、pooling；
- Git：理解工作区、暂存区、commit、remote 和 push 的关系；
- 科研规范：不伪造指标，保留原始 CSV、日志和来源哈希。

## 12. 思考与启发

多任务学习的优势：

- 多个任务共享晶体结构表示，可能提高数据利用效率；
- 一个模型可以同时输出多个性质，推理更方便；
- 相关任务之间可能互相提供正则化作用。

可能的问题：

- 不同性质尺度差异大，简单平均 loss 可能让某一任务主导训练；
- 任务不相关时可能出现 negative transfer；
- 缺失标签的分布可能引入偏差；
- 当前模型没有任务权重、梯度冲突处理或不确定性估计。

不同材料性质之间的关系：

- 形成能、带隙和费米能都与电子结构有关，但并非简单线性相关；
- 带隙中大量金属材料取值为 0，使分布呈混合特征，预测更困难；
- 应分别报告每个任务的指标，而不是只看总 MAE。

CGCNN 的局限：

- 只使用局部截断范围内的结构信息；
- 不显式建模长程静电相互作用、磁性、自旋、温度和压力；
- 对训练数据分布之外的材料泛化有限；
- 图构建受截断半径和最大邻居数影响；
- 当前多任务版本没有真正验证 negative transfer。

## 13. 后续计划

1. 在相同数据上分别训练单任务和共享多任务模型，做严格对比；
2. 为每个任务分别输出 MAE、RMSE、R²，而不是只看平均 MAE；
3. 加入任务权重或不确定性加权损失；
4. 复现并学习 SchNet、Matformer、ALIGNN 等材料图神经网络；
5. 尝试把晶体结构表示与更丰富的物理先验结合；
6. 建立统一的实验配置、日志和结果记录流程。
