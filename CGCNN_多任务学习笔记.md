# CGCNN 多任务改造 · 完整学习笔记

> 项目：把一个"一次预测 1 个性质"的 CGCNN 改成"一次预测 3 个性质"（多任务回归），
> 并用 Materials Project 真实数据完成"找数据 → 洗数据 → 大规模训练 → 预测评估"全流程。
> 日期：2026-09  环境：Windows 11 / Python 3.10.20 / PyTorch 2.7.0+cu128 / RTX 5060 Laptop / pymatgen 2025.10.7

---

## 目录
1. [CGCNN 是什么（30 秒版）](#1-cgcnn-是什么30-秒版)
2. [数据流与 tensor shape](#2-数据流与-tensor-shape)
3. [为什么原来只能预测 1 个性质](#3-为什么原来只能预测-1-个性质)
4. [多任务改造：4 个文件改了啥](#4-多任务改造4-个文件改了啥)
5. [找数据与洗数据](#5-找数据与洗数据)
6. [下载脚本的三次迭代（含一次重要的调试教训）](#6-下载脚本的三次迭代含一次重要的调试教训)
7. [大规模训练结果](#7-大规模训练结果)
8. [踩坑清单](#8-踩坑清单)
9. [自测题](#9-自测题)
10. [产物位置与下一步](#10-产物位置与下一步)

---

## 1. CGCNN 是什么（30 秒版）

**给一个晶体结构，预测它的性质。**

- 晶体 = 原子按规律排列 → 看成一张**图**：原子 = 点(node)，近邻关系 = 线(edge)。
- 每个原子带一张"身份证"：`atom_init.json` 里的 92 维向量（表示元素种类）。
- 神经网络读这张图，提取"整个晶体"的特征，最后输出性质数值。

**图卷积（通俗版）**：每个原子一开始只认识自己；图卷积 = 每个原子向邻居"打听消息"，
把邻居信息和自己的信息合并、更新自己。做 3 层 = 打听 3 轮，信息越传越远。

**pooling（通俗版）**：一个晶体有几十个原子，最后要的是"整个晶体"的特征 → 把这几十个
原子的向量**取平均**合成一个向量。

---

## 2. 数据流与 tensor shape

```
CIF 文件 + id_prop.csv
   ↓ ① Dataset（CIFData.__getitem__）每次拿一个样本
atom_fea      [原子数, 92]       每个原子的身份证
nbr_fea       [原子数, 12, 41]   最近 12 个邻居的距离信息（41 维高斯展开）
nbr_fea_idx   [原子数, 12]       这 12 个邻居是谁
target        [1] 或 [3]         标准答案（性质个数）
   ↓ ② DataLoader / collate_pool 打包成 batch
target        [batch_size, 1] 或 [batch_size, 3]
   ↓ ③ Model（CrystalGraphConvNet.forward）
embedding(92→64) → 3×图卷积(64→64) → pooling(→[N0,64])
→ conv_to_fc(64→128) → fc_out(128→1 或 3)    输出 [N0, 1] 或 [N0, 3]
   ↓ ④ Loss：output 和 target 逐元素比较 → 一个数
   ↓ ⑤ backward + optimizer.step() 更新参数
```

关键：**`nn.Linear` 的第二个参数 = 输出神经元个数 = 一次吐出几个数。**

---

## 3. 为什么原来只能预测 1 个性质

"输出 1 个"被三个地方同时写死，且必须一致：

| 位置 | 原来 |
|---|---|
| 数据 | `id_prop.csv` 每行 = id + **1 个**数 |
| Dataset | `target.shape = [1]`，batch 后 `[batch,1]` |
| 模型 | `fc_out = nn.Linear(h_fea_len, 1)` ← 根源 |
| Loss | `output[batch,1]` vs `target[batch,1]` |

→ 要预测 3 个，就同时把"数据列数、模型输出维度、命令行参数"从 1 改成 3。

---

## 4. 多任务改造：4 个文件改了啥

原则：**图卷积 / pooling / embedding 一行没动**，只动"入口（数据）"和"出口（输出层）"。

| 文件 | 改了什么 | shape 变化 |
|---|---|---|
| `cgcnn/model.py` | 构造函数加 `num_targets=1`；`fc_out` 的 `1 → num_targets` | 输出 `[N0,1] → [N0,3]` |
| `cgcnn/data.py` | `__getitem__` 读整行第 2~4 列；空值/`nan` → `float('nan')`（新增 `_safe_float`） | target `[1] → [3]` |
| `main.py` | 加 `--num-targets`；回归 loss 换 `masked_mse_loss`；`Normalizer` 改逐列统计（NaN 安全）；`test_results.csv` 写 7 列（顺带修了 Windows 下 CSV 空行问题：`open(..., newline='')`） | loss 输入 `[N0,3]` |
| `predict.py` | 从 checkpoint 读 `num_targets`（`getattr(model_args,'num_targets',1)` 兼容旧模型）；其余与 main.py 同步 | 输出 3 个数 |

### 4.1 掩码 loss（为什么不能直接忽略 NaN）

某个性质缺失（NaN）时，`NaN × 0 = NaN`，直接乘掩码没用。
正确做法：先把 NaN 位置替换成 0，再乘掩码，最后除以有效个数：

```python
def masked_mse_loss(prediction, target):
    mask = ~torch.isnan(target)
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    diff2 = (prediction - target_clean) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)
```

### 4.2 逐列归一化

3 个性质单位/范围不同，先各自减均值除标准差。所以 `Normalizer` 从"一个数"改成"每列一个数"（向量）。
注意：本机 torch 2.7 **没有 `torch.nanstd`**，NaN 安全的 std 要手动算。

---

## 5. 找数据与洗数据

### 5.1 CGCNN 一个数据集 = 3 类文件
- 一堆 `*.cif`（输入）
- `id_prop.csv`（答案：第 1 列 id，后面每列一个性质）
- `atom_init.json`（元素特征，**直接复制 sample 的即可**，不用自己造）

### 5.2 数据来源：Materials Project（MP）
- 免费注册 → 个人设置拿 API key。
- `data/material-data/mp-ids-*.csv` 是论文用过的材料 ID 清单（`mp-ids-46744.csv` 最大，4.6 万个）。
- 用 pymatgen 的 `MPRester` 下载结构 + 性质。

### 5.3 洗数据铁律
1. `id_prop.csv` **不要表头**（原代码 `csv.reader` 每行都当样本）。
2. 缺失**留空 / nan，不要写 0**（0 会被当成"答案是 0"）。
3. 下载失败**跳过不中断**（真实数据一定有坏条目）。
4. **0 不一定是脏数据**：金属带隙就是 0（本数据集 43% 是金属）。
5. 先小批量跑通，再放大。

---

## 6. 下载脚本的三次迭代（含一次重要的调试教训）

### v1：逐 id 查询（9 个样本用）
可靠但慢。

### v2：批量 + 逐个回退（1000 个花了 30+ 分钟）
以为"批量查询成功 749、逐个回退没救回 251 → 批量可靠"，**判断错了**。

### v3：删掉回退、纯批量（1251 个"全部失效"、0 个下载、0.1 分钟）
**真相**：批量查询（多个 id 逗号拼成一个 `material_ids` 参数）其实**一直返回空**，
v2 的 749 个成功全是"逐个查"兜底干的。之前的推断"未找到 251 == 批量缺失 251"是**巧合**。

> 🎓 教训：**不要只看总数对得上就下结论，要看过程。** 0.1 分钟跑完本身就说明"快得不正常"。

### v4：并发逐 id（1251 个 2.8 分钟，全量 49 分钟）
- 确认只有"一个请求一个 id"可靠（这正是 pymatgen 自带方法 `get_structure_by_material_id` 的用法）。
- 每个 id **1 次请求**拿 structure + 3 性质（`_fields=structure,<props>`）。
- 多线程并发（默认 8，实测 16 也不限流），每线程独立连接。
- 断点续传：`id_prop.csv` 即进度；每 500 个存一次盘；实时打印速度 + ETA。

### 实测速度
- 解析：0.06 秒/样本
- 下载：约 780 个 id/分钟（16 并发）
- 全量 46,744 请求 → **49.1 分钟**，0 错误

---

## 7. 大规模训练结果

### 7.1 数据集
- `data/mt-large/`：请求 46,744 → **有效 36,358**（77.8%），失效 10,386，CIF 36,358 个，0 重复。
- 3 个性质：形成能 `formation_energy_per_atom`、带隙 `band_gap`、费米能 `efermi`。
- 质量：36,358×3 ≈ 10.9 万值里仅 9 个缺失（被 NaN 掩码处理）。

### 7.2 训练配置
```
python main.py --num-targets 3 --epochs 30 --batch-size 256 \
               --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 data/mt-large
```
- 训练/验证/测试 ≈ 29,086 / 3,635 / 3,635。
- 第 1 轮约 36~40 分钟（解析全部 CIF + lru_cache 缓存），之后每轮很快。

### 7.3 测试集结果（3,635 个材料）
| 性质 | MAE | RMSE | R² | 解读 |
|---|---|---|---|---|
| 形成能 (eV/atom) | **0.110** | 0.154 | **0.981** | 预测非常好 |
| 带隙 (eV) | 0.424 | 0.697 | 0.841 | 难：43% 是金属（带隙=0），分布断层 |
| 费米能 (eV) | 0.447 | 0.670 | 0.944 | 一般 |

- 30 轮多任务 vs 论文单任务形成能 MAE 0.039（~968 轮）→ 已经相当接近，续训可逼近。
- ⚠️ "总 MAE 0.327"是把 3 个不同量纲的性质混在一起平均，**要看分性质的 MAE**。

---

## 8. 踩坑清单

1. **shape 不匹配**：output 和 target 列数必须一致；报错先数形状。
2. **`id_prop.csv` 加表头** = 多一个假样本，直接报错。
3. **缺失填 0**：把"缺失"当"答案是 0"。
4. **同目录残留别的任务的 `model_best.pth.tar`**：main.py 结尾会加载它测试，不同任务间 shape 不匹配（原项目行为）。
5. **`torch.nanstd` 不存在**（torch 2.7）：NaN 安全 std 手动算。
6. **Windows 下 csv 写入空行**：`open('test_results.csv','w')` 配合 csv 默认 `\r\n` 会重复换行 → 用 `newline=''`。
7. **批量查询返回空**：MP summary 端点多个 id 逗号拼接不可靠 → 用并发逐 id。
8. **训练时电脑睡眠**：长任务记得电源设置改成"从不睡眠"。

---

## 9. 自测题

1. CGCNN 输入输出是什么？晶体为什么能表示成图？node / edge 各是什么？
2. Dataset 和 DataLoader 分别做什么？
3. `nn.Linear(128, 1)` 的 `1` 是什么意思？输出 3 个为什么改成 `3`？
4. 单任务 target `[1]` → 多任务 `[3]`：batch 后、模型输出后各是什么 shape？
5. 为什么 output 和 target 的 shape 必须匹配？
6. 3 个性质怎么合成一个 loss？缺失 target 为什么不能直接算 loss？掩码怎么解决？
7. predict.py 怎么知道模型要输出 3 个？
8. 为什么"总 MAE"要拆成"分性质 MAE"看？
9. 带隙为什么比形成能难预测？

---

## 10. 产物位置与下一步

| 产物 | 位置 |
|---|---|
| 4 个多任务代码文件 | `main.py` `predict.py` `cgcnn/model.py` `cgcnn/data.py` |
| 下载脚本（并发逐 id） | `download_mp_data.py` |
| 小数据集（9 个） | `data/mt-demo/` |
| 大数据集（36,358 个） | `data/mt-large/` |
| 训练产物 | `data/mt-run/`（`model_best.pth.tar`、`checkpoint.pth.tar`、`test_results.csv`） |

下一步可选项：
1. 续训到 100+ 轮：`--resume data/mt-run/checkpoint.pth.tar --epochs 120`
2. 同数据单任务对比（`--num-targets 1` + 单列数据），看多任务是否拖累单个任务
3. 用 `pre-trained/*.pth.tar` 预训练模型做单任务预测，熟悉 predict.py
4. 每个性质单独报 MAE（把 `mae` 拆成按列算，小改动练手）

---

*写完这份笔记，你应该能用自己的话回答第 9 节的自测题——能答上来，这次学习就到位了。*
