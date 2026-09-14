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

## 6. 原始 CGCNN 复现

原始单任务训练命令：

```powershell
python main.py --epochs 120 --batch-size 256 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 data/shc-max-log
```

已保存的 120 轮单任务结果显示：

| 指标 | 数值 | 来源 |
|---|---:|---|
| 最好验证 MAE | 0.2946 | 顶层 `model_best.pth.tar` 元数据 |
| 测试 MAE | 0.311508 | 866 条真实测试记录重算 |
| 测试 RMSE | 0.539655 | 866 条真实测试记录重算 |
| 测试 R² | 0.410632 | 866 条真实测试记录重算 |

注意：最好验证 MAE 和最终测试 MAE 不是同一个指标。模型选择使用验证集，最终泛化能力要看测试集。

本次又使用当前代码重新运行了 30 轮，并保存了完整终端日志：

```powershell
python main.py --epochs 30 --batch-size 256 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --print-freq 10 data/shc-max-log
```

本次 30 轮真实结果：

| 指标 | 数值 | 来源 |
|---|---:|---|
| 最好验证 MAE | 0.311909 | 日志中的 `* MAE` 与最佳 checkpoint |
| 测试 MAE | 0.318826 | 866 条新测试记录重算 |
| 测试 RMSE | 0.524193 | 866 条新测试记录重算 |
| 测试 R² | 0.443920 | 866 条新测试记录重算 |

该次运行的原始日志位于 `results/original/rerun-shc-max-log-20260914/`。由于命令使用 `--print-freq 10`，日志中的逐轮 training loss 和 validation loss 是最后一个打印 batch 的 running average；validation MAE 和测试 MAE 来自完整的 `* MAE`/`** MAE` 行。

另有 10 个 CIF 的单任务代码冒烟测试，训练 6、验证 2、测试 2。该实验只用于验证流程，测试 MAE 为 3.813，不应作为模型性能结论。

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

三任务已保存结果的测试集指标：

| 材料性质 | MAE | RMSE | R² | 有效测试数 |
|---|---:|---:|---:|---:|
| 形成能 `formation_energy_per_atom` | 0.109772 | 0.154312 | 0.980555 | 3635 |
| 带隙 `band_gap` | 0.424296 | 0.697009 | 0.840928 | 3635 |
| 费米能 `efermi` | 0.447355 | 0.670312 | 0.943981 | 3635 |

这三个任务平均值约为 0.327141。由于三个性质单位不同、难度不同，平均值只能作为总体参考，不能替代分任务指标。

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

## 9. 实验结果

结果目录：

```text
results/
├── original/
│   ├── test_results.csv
│   ├── metrics.json
│   ├── metrics.md
│   └── reproduction-20260914/
└── multitask/
    ├── test_results.csv
    ├── metrics.json
    ├── metrics.md
    └── reproduction-20260914/
```

`metrics.json` 记录了源 CSV 的 SHA-256，用于检查结果来源。`tools/summarize_results.py` 可以从 CSV 重新计算指标。

本次 30 轮单任务重跑的逐轮数据和真实曲线位于：

- `results/original/rerun-shc-max-log-20260914/`
- `figures/training_loss.png`
- `figures/training_mae.png`

多任务 `mt-demo` 是一次 9 个样本的流程测试：

- 训练 5、验证 2、测试 2；
- 最终测试平均 MAE 为 0.568；
- 预测全部 9 个样本时，三任务平均 MAE 为 0.708。

小样本训练只能证明代码可以运行，不能证明多任务模型优于单任务模型。

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
