# CGCNN 部署与复现记录

## 1. 实际环境

```text
项目目录：D:\VScode_project\CGCNN\cgcnn-master\cgcnn-master
Conda：D:\Miniconda\miniconda
环境：cgcnn
Python：3.10.20
PyTorch：2.7.0+cu128
CUDA runtime：12.8
cuDNN：9.7.1
GPU：NVIDIA GeForce RTX 5060 Laptop GPU
Compute Capability：12.0
NVIDIA Driver：582.05
```

环境验证命令：

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

结果：

```text
2.7.0+cu128
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

并完成了真实 CUDA 矩阵乘法验证，输出 device 为 `cuda:0`。

## 2. 为什么没有重新安装环境

检查发现现有 `cgcnn` 环境已经满足：

- Python 3.10.x；
- PyTorch CUDA 版本；
- RTX 5060 `sm_120` 可用；
- pymatgen、numpy、scikit-learn、pandas、matplotlib 等依赖完整；
- `pip check` 没有发现依赖冲突。

因此没有重新安装或升级依赖，避免破坏已验证过的环境。

## 3. 数据集准备

每个数据集目录至少包含：

```text
dataset/
├── id_prop.csv
├── atom_init.json
└── *.cif
```

单任务标签：

```text
material_id,property
```

多任务标签：

```text
material_id,property1,property2,property3
```

缺失值可以留空或写 `nan`。

## 4. 单任务复现

示例命令：

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe main.py `
  --epochs 120 `
  --batch-size 256 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  data/shc-max-log
```

输出文件：

```text
checkpoint.pth.tar
model_best.pth.tar
test_results.csv
```

注意：这些文件默认写到“当前工作目录”。每次实验应使用独立目录，否则会覆盖已有结果。

## 5. 多任务复现

示例命令：

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe main.py `
  --num-targets 3 `
  --epochs 30 `
  --batch-size 256 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  data/mt-large
```

多任务测试 CSV 格式：

```text
id,target_1,target_2,target_3,pred_1,pred_2,pred_3
```

## 6. 预测

```powershell
D:\Miniconda\miniconda\envs\cgcnn\python.exe predict.py `
  results/multitask/reproduction-20260914/model_best.pth.tar `
  data/mt-demo `
  --print-freq 1
```

预测同样把 `test_results.csv` 写到当前工作目录，因此建议在独立预测目录中运行。

## 7. 日志保存方法

PowerShell：

```powershell
python main.py ... 2>&1 | Tee-Object -FilePath results/run/train.log
```

解析日志：

```powershell
python tools/parse_training_log.py results/run/train.log results/run --prefix experiment
```

生成指标：

```powershell
python tools/summarize_results.py results/run/test_results.csv results/run --task-names property1 property2 property3
```

## 8. Git 与大文件策略

默认不上传：

- `data/mt-large/`
- `data/mt-run/`
- `data/shc/`
- `data/shc-log/`
- `data/shc-max-log/`
- 用户生成的 `*.pth.tar`
- 根目录临时 `test_results.csv`
- IDE、缓存和日志临时文件

默认保留：

- `data/sample-regression/`
- `data/sample-classification/`
- `data/mt-demo/`
- 官方 `pre-trained/*.pth.tar`

## 9. 实际遇到的问题

1. 当前 PowerShell 初始 PATH 找不到 `conda`，需要调用绝对路径。
2. `torch_env` 虽然能报告 CUDA 可用，但 RTX 5060 的 `sm_120` 不受旧 PyTorch 支持。
3. `torch_env_clean` 是 CPU 版 PyTorch。
4. 部分 CIF 会被 pymatgen 报告组成或坐标精度 warning，但流程可以继续。
5. 默认输出路径是当前工作目录，容易覆盖不同实验。
6. 原代码没有逐轮日志，后来用 `Tee-Object` 保存真实终端输出。
7. 多 batch 日志需要取每轮括号中的平均值，不能取第一批瞬时值。
