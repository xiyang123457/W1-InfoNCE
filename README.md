# InfoNCE on CIFAR-10

从零实现 SimCLR 式对比学习：**双视图随机增强 → ResNet-18 编码 → 128 维投影 → NT-Xent 损失**。

全程无监督 —— 训练时不喂任何标签，输入只有图像。唯一被要求学会的事是：
「同一张图的两个随机视图应该互相靠近，与 batch 内其余 510 个视图互相远离」。

---

## 结果一览

| 阶段 | 指标 | 实测 |
| --- | --- | --- |
| 数据与增强 | 同一张图两遍 transform 是否相同 | `torch.equal(v1, v2) = False`（平均绝对差 0.1353） |
| 随机初始化（B=256、τ=0.5） | NT-Xent loss vs 理论基线 `ln(2B−1)` | **6.1753** vs 6.2364 |
| 训练 15 epoch | loss | **6.1753 → 4.7429** |
| 表征质量 | 正/负样本对余弦距离的 gap | **0.0902 → 0.8192**（约 9 倍） |

产物图（`runs/`，已被 `.gitignore` 忽略）：

| 文件 | 内容 |
| --- | --- |
| `W2_curve_tau0.5_loss.png` | 训练曲线（step 轴，含随机初始化起点与两条基线） |
| `W2_tsne_1_random_init.png` | 随机初始化时的 t-SNE —— 十色均匀散开，无结构 |
| `W2_tsne_2_trained.png` | 训练后的 t-SNE —— truck / automobile / ship+airplane / horse+deer / frog 各自成簇 |
| `W2_tsne_3_pos_neg_dist.png` | 正/负样本对余弦距离的两条直方图，明显分离 |

---

## 环境

```bash
conda activate pytorch     # ⚠️ 本机 PowerShell 里不生效，见下方「已知坑」
python --version
```

实测环境（2026-10，`D:\anaconda\envs\pytorch\python.exe`）：

| 项 | 版本 |
| --- | --- |
| torch / torchvision | 2.9.0+cu130 / 0.24.0+cu130 |
| CUDA | 13.0（RTX 5060 Laptop，arch_list 含 sm_120） |
| numpy / matplotlib / tqdm | 2.2.6 / 3.10.9 / 4.70.1 |
| scikit-learn | 1.7.2（只 t-SNE 用） |

依赖清单见 `requirements.txt`。带 `+cu130` 的本地版本号不能直接 `pip install -r`，需配索引：

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

**数据集**：`torchvision.datasets.CIFAR10` 会自动下载。也可手动下
`cifar-10-python.tar.gz`（md5 `c58f30108f718f92721af3b95e74349a`，170498071 字节）放进 `data/`，
torchvision 校验通过后会跳过下载直接解压。

---

## 快速开始

```bash
# 1) 验证损失实现：随机初始化下 loss 应 ≈ ln(2B-1) = 6.2364
python src/sanity_check.py

# 2) 正式训练（B=256、τ=0.5、15 epoch，约 15 分钟）
python src/train.py --epochs 15 --num-workers 4

# 3) 可视化（需要第 2 步产出的 runs/ckpt_tau0.5.pth）
python src/tsne_vis.py
```

期望输出：

```
# 1)
5 次平均 loss   = 6.13 ~ 6.17
理论 ln(2B-1)   = 6.2364
[PASS] B2：随机初始化下 loss ≈ ln(2B-1)，labels / 归一化 / 2B 拼接全部正确

# 2)
[init] 随机初始化 loss（未训练）= 6.1753
[epoch 15/15] mean loss = 4.7459  (63.2 s)
起始 loss 6.1753 → 末尾 loss 4.7429
[PASS] W2-A(tau0.5)：loss 从 6.1753 降到 4.7429

# 3)
正样本对余弦距离  mean = 0.1561
负样本对余弦距离  mean = 0.9753
分离度 gap = +0.8192
[PASS] W2-B：t-SNE 与正负距离图均已生成，正负样本对明显分离
```

> **注**：真实数据上的 loss 起点是 **6.1753**，不是理论值 6.2364。原因见下方「关键设计」第 4 条。

---

## 项目结构

```
.
├── README.md
├── PROGRESS.md            逐阶段进度、实测数字、踩过的所有坑
├── requirements.txt
└── src/
    ├── augment.py         SimCLR 双视图增强流水线
    ├── data.py            CIFAR-10 下载/解压 + 双视图 Dataset
    ├── model.py           Encoder（ResNet-18 改 stem）+ ProjectionHead
    ├── loss.py            NTXentLoss
    ├── sanity_check.py    随机初始化下的 loss 验收
    ├── train.py           训练循环 + 指标落 JSON + 自动出图
    ├── plot_curve.py      从 JSON 独立画图（不重跑训练）
    └── tsne_vis.py        t-SNE 可视化 + 正负样本对距离
```

每个脚本都能**单独运行**并自带断言，末尾打印 `[PASS] ...` 才算通过；文件末尾附有「怎么验证跑对了」和异常对照表。

---

## 各脚本用法

### `src/augment.py` — 增强流水线

```bash
python src/augment.py
```
六个算子：`RandomResizedCrop(32, scale=(0.2,1.0))` → `RandomHorizontalFlip` → `ColorJitter(0.4, hue=0.1)`
→ `RandomGrayscale(0.2)` → `GaussianBlur(3, σ=0.1~2.0)` → `ToTensor`。
自检：同一张图过两遍必须**不全等**。

### `src/data.py` — 数据集

```bash
python src/data.py
```
`CIFAR10Pair.__getitem__` 返回 `(v1, v2)` —— 同一个 `transform` 对象被**调用两次**（不是 clone）。
`targets` 保留在 Dataset 里，供将来做线性探针。

### `src/model.py` — 网络

```bash
python src/model.py
```
`Encoder`：`resnet18(weights=None)`，`conv1` 换成 `3×3/stride1/pad1`、`maxpool` 换 `Identity`、`fc` 换 `Identity`。
`ProjectionHead`：`Linear(512,512) → BatchNorm1d → ReLU → Linear(512,128)`。
参数量：Encoder 11,168,832 / Head 329,344。

### `src/loss.py` — NT-Xent

```bash
python src/loss.py
```
`NTXentLoss(z: (2B,128)) -> scalar`：L2 归一化 → 余弦相似度矩阵 → 屏蔽对角线 → 交叉熵。
正样本索引是 `i + B`（前 B 行配后 B 行），**不是对角线**。

### `src/sanity_check.py` — 损失实现验收

```bash
python src/sanity_check.py
```

### `src/train.py` — 训练

```bash
python src/train.py [--tau 0.5] [--epochs 15] [--num-workers 4] [--batch-size 256]
```
产出 `runs/ckpt_tau{τ}.pth`、`runs/metrics_tau{τ}.json`、`runs/W2_curve_tau{τ}_loss.png`。
指标**每个 epoch 落一次盘**，训练中途崩了也不丢已完成的曲线。

### `src/plot_curve.py` — 重画曲线（不重跑训练）

```bash
python src/plot_curve.py --metrics runs/metrics_tau0.5.json
python src/plot_curve.py --metrics runs/metrics_tau0.5.json --x-axis epoch
python src/plot_curve.py --metrics runs/metrics_tau0.5.json runs/metrics_tau0.07.json   # 多组对比
```
秒级完成。改标题、换横轴、叠加对比都不必重新训练。

### `src/tsne_vis.py` — 表征可视化

```bash
python src/tsne_vis.py [--ckpt runs/ckpt_tau0.5.pth] [--n-per-class 200] [--perplexity 30]
```
两张 t-SNE（随机初始化 / 训练后）**各出一张 PNG**，外加正负样本距离直方图。

---

## 关键设计

1. **2B 在像素层拼接**：`torch.cat([v1, v2], dim=0)` → `(2B,3,32,32)` 后一次 forward。
   前 B 个是视图 1、后 B 个是视图 2，这决定了正样本索引必须是 `i + B`。
   不分别 forward 再拼 `z`，是因为 BN 会基于 512 个样本更新一次 vs 256 个样本更新两次，`running_stats` 不同。

2. **stem 必须改**：原生 ResNet-18 的 stem 是为 224×224 设计的，用在 32×32 上 layer4 的空间尺寸会塌成 `1×1`，
   **而输出形状照样是 `(B,512)`、照样不报错**。代码里用 forward hook 专门抓 `layer4` 输出验证它是 `4×4` ——
   光看最终 shape 发现不了这个问题。

3. **`drop_last=True` 是硬要求**：否则末尾不足 256 的 batch 会让 `labels = i + B` 越界或错位。
   顺带保证每个 batch 的负样本数恒为 `2B−1 = 511`，`ln(511)` 才好对照。

4. **真实数据的 loss 起点低于 `ln(2B−1)`**：理论值 6.2364 假设「所有候选 logits 近似相等」。
   实际上随机初始化的网络对同一张图的两个视图已经比随机两张图更相似（实测余弦相似度差 +0.09），
   所以起点被压到 6.1753。**`ln(2B−1)` 是「完全无信息」的基线，不是精确期望值。**

5. **训练 / 指标 / 画图三者解耦**：训练只写 JSON，画图只读 JSON。改图零成本，崩了不丢数据。

6. **复现要设四个种子**：`random.seed` + `torch.manual_seed` + `np.random.seed` + `torch.cuda.manual_seed_all`。
   因为 torchvision 的 `ColorJitter` 用的是 **torch 的 RNG**，而 `RandomResizedCrop` / `RandomHorizontalFlip` /
   `RandomGrayscale` 用的是 **Python 的 `random`** —— 这套增强同时踩在两个随机源上，只设一个复现不了。

---

## 已知限制

- **15 epoch 远未收敛**：SimCLR 原论文在 CIFAR-10 上是 1000 epoch。这里的目标是「趋势对」，不是刷指标。
- **没有线性探针**：目前只用 t-SNE 和正负距离说明表征质量，没有量化的分类精度。
- **t-SNE 用的是 `z`（投影头输出）而非 `h`（编码器输出）**：论文指出投影头会丢掉部分对下游任务有用的信息，
  用 `h` 通常能得到更清晰的簇。`model.encode()` 已经留好了接口。
- **无监督的簇不会像监督学习那样干净**：SimCLR 只学「同一张图的两个视图要像」，不知道「类别」，
  视觉相似的类（cat/dog、deer/horse、automobile/truck）天然会靠得近。
- **未评估 batch size 的影响**：负样本全部来自 batch 内，B 越大任务越难、表征越好，这里固定 B=256。

---

## 参考

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML 2020
- Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018

## 文档位置

| 文档 | 内容 |
| --- | --- |
| [`PROGRESS.md`](PROGRESS.md) | **W1** 的逐阶段记录：A1–B2 的实测数字、判据、踩过的坑 |
| 独立仓库 `W2 CLIP demo` 的 `PROGRESS.md` | **W2** 的完整记录：训练日志、t-SNE、CLIP zero-shot demo 及 prompt 消融 |

> 本仓库同时承载 W2-A/B 的代码（`train.py` / `plot_curve.py` / `tsne_vis.py`），
> 因为它们直接复用这里的 `model.py` / `loss.py` / `data.py`；W2 的**日志与文档**则在独立仓库。
