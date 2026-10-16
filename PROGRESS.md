# W1 · InfoNCE 进度记录

- **最后更新**：2026-10-16
- **上游计划书**：`W1-D4+D5计划书（骨架+InfoNCE初版）`
- **本节产物**：能跑通的 InfoNCE 初版代码（= W1 的 P3 交付物）

---

## 一句话状态

**W1（A1–B2）全部完成并实测通过；W2 的 A（正式训练）与 B（t-SNE 可视化）也已完成。**

W1 证明「跑得通」（loss ≈ ln(2B−1)，等于什么都没学到）；W2 证明「真的在学」
（loss 6.1753 → 4.7429，正负样本分离度 0.09 → 0.82）。**W2 详情见文末章节。**

---

## 1. 进度总览

| 阶段 | 步骤 | 内容 | 状态 | 实测验收 |
| --- | --- | --- | --- | --- |
| A | A1 | 环境 | ✅ | `torch 2.9.0+cu130` / `cuda 13.0` / `is_available()=True` / RTX 5060 Laptop GPU |
| A | A2 | 仓库 | ✅ | `git init` 完成，`.gitignore` 已忽略 `data/` |
| A | A3 | 数据集 | ✅ | md5 `c58f30108f718f92721af3b95e74349a`，170498071 字节；50000/10000，10 类 |
| A | A4 | 增强流水线 | ✅ | `torch.equal(v1,v2)=False`，平均绝对差 **0.1353** |
| A | A5 | 骨架 + shape | ✅ | `h (2B,512)` / `z (2B,128)` / `layer4 (2B,512,4,4)` |
| B | B1 | 实现 InfoNCE | ✅ | 纯随机 z 下 loss = **6.2528**，命中计划书的 6.259（τ=0.5） |
| B | B2 | sanity check | ✅ | 真实数据 B=256 τ=0.5：5 次均值 **6.1387 / 6.1449**，偏离 ln511 约 **−0.09 ~ −0.10**（判据 ±0.1，**压线**，原因见 §4.4） |
| B | B5 | 进 GitHub | ⬜ 未开始 | 需配 remote 后 push |

图例：✅ 已实测通过｜⬜ 未开始

---

## 2. 仓库结构

```
W1 InfoNCE/
├── .gitignore            忽略 data/ runs/ *.pth __pycache__ 等
├── requirements.txt      记录环境（非重建用；+cu130 本地版本需配 --index-url）
├── PROGRESS.md           本文件
└── src/
    ├── augment.py        A4 · 双视图增强流水线（含自检）
    ├── data.py           A3 · CIFAR-10 下载/解压 + 双视图 Dataset（含自检）
    ├── model.py          A5 · Encoder + ProjectionHead + SimCLR（含自检）
    ├── loss.py           B1 · NTXentLoss 损失（含自检）
    └── sanity_check.py   B2 · 真实数据 sanity check（含自检）
```

**尚未创建**：`README.md`（B5）

---

## 3. 各模块职责与契约

| 模块 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| `augment.build_augmentation()` | — | `transforms.Compose`（调用两次） | ✅ |
| `data.CIFAR10Pair` | index | `(v1, v2)`，各 `(3,32,32)` | ✅ |
| `data.build_loader()` | — | `DataLoader`，产出 `(v1_batch, v2_batch)` 各 `(B,3,32,32)` | ✅ |
| `model.Encoder` | `(N,3,32,32)` | `h (N,512)` | ✅ |
| `model.ProjectionHead` | `(N,512)` | `z (N,128)` | ✅ |
| `model.SimCLR` | `(N,3,32,32)` | `z (N,128)`；另有 `encode()` 只出 `h`（W2 线性探针用） | ✅ |
| `loss.NTXentLoss` | `z (2B,128)` | `scalar` | ✅ |

### 关键设计点（写代码时已落地的决策）

1. **双视图的来源**：同一个 `transform` 对象被**调用两次**，不是算一次后 `clone()`。
2. **2B 的拼接位置**：在**像素层**拼，`torch.cat([v1, v2], dim=0)` → `(2B,3,32,32)` 后一次 forward。
   - 前 B 个是视图 1，后 B 个是视图 2 → **B1 的正样本索引必须是 `i + B`，不是对角线**。
   - 之所以不分别 forward 再拼 `z`：像素层拼时 BN 基于 512 个样本更新一次；分别 forward 会基于 256 个样本更新两次，`running_stats` 不同，会影响 W2 线性探针的 `eval()` 表现。
3. **stem 改造**：`conv1` → `3×3/stride1/pad1/bias=False`，`maxpool` → `Identity()`，`fc` → `Identity()`。
4. **`layer4 = 4×4` 靠 forward hook 验证**：原生 stem 下输出形状**照样是 `(B,512)` 且不报错**，只是空间分辨率塌成 `1×1` —— 光看 `h.shape` **发现不了**。
5. **标签保留**：`CIFAR10Pair.targets` 存了 50000 条标签，`__getitem__` 不返回。W2 的线性探针直接取用，不必重读数据。
6. **`drop_last=True`**：保证每个 batch 都是完整 256，使 `ln(2B−1)` 恒为 `ln(511)=6.2364`；否则 50000 % 256 = 72 的尾巴 batch 会打出 `ln(143)≈4.96`，容易被误判成 labels 写错。

---

## 4. 实测数字（用于复现核对）

### A3 · 数据集
| 项 | 值 |
| --- | --- |
| 压缩包大小 | 170498071 字节（≈162.6 MB） |
| md5 | `c58f30108f718f92721af3b95e74349a` |
| 解压落点 | `data/cifar-10-batches-py/` |
| 训练 / 测试 | 50000 / 10000 |
| 类别 | airplane automobile bird cat deer dog frog horse ship truck 共 10 类 |

### A4 · 增强
```
v1.shape = (3, 32, 32)
v2.shape = (3, 32, 32)
torch.equal(v1, v2) = False
(v1 - v2).abs().mean() = 0.1353
```

### A5 · 骨架
```
x1.shape     = (4, 3, 32, 32)      # 自检用 B=4
x_cat.shape  = (8, 3, 32, 32)      # 2B
h.shape      = (8, 512)
z.shape      = (8, 128)
layer4 输出  = (8, 512, 4, 4)      # 关键：证明 stem 改对了
Encoder 参数量       = 11,168,832
ProjectionHead 参数量 = 329,344
```

> 参数量已按本机实测校订（原预估 11,676,702 / 328,320 有误，后者漏算了 ProjectionHead 两层 BN 各 512 个 `mean/var`）。

### B1 · NTXentLoss 自检（随机 z，B=8 → 2B=16）
```
z.shape          = (16, 128)
S.shape          = (16, 16)
对角线（未屏蔽）= 1.0000
非对角 std       = 0.0888
loss             = 2.6679
ln(2B-1)         = 2.7081
偏差             = -0.0401
```

### B2 · sanity check（真实数据，B=256，τ=0.5，每次取 5 个 batch）

| 运行 | batch1 | batch2 | batch3 | batch4 | batch5 | 均值 | std | 偏离 ln511 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 次 | 6.1655 | 6.1213 | 6.1373 | 6.1570 | 6.1434 | **6.1449** | 0.0154 | −0.0915 |
| 第 2 次 | 6.1794 | 6.0847 | 6.1429 | 6.1551 | 6.1317 | **6.1387** | 0.0313 | −0.0976 |

### 4.4 为什么真实数据上 loss 低于 ln(2B−1)（−0.09 不是 bug）

关键对照：**把 z 换成纯随机独立向量后，loss = 6.2528（偏差 +0.016），精确命中计划书的 6.259。**

| 场景 | loss | 偏差 |
| --- | --- | --- |
| 纯随机独立 z（mask 对角线） | 6.2607 / 6.2514 / 6.2504 / 6.2439 / 6.2575，均值 **6.2528** | **+0.016** ✅ 对上计划书的 6.259 |
| 纯随机 z（不 mask 对角线） | 6.2847 / 6.2808 / 6.2654，均值 6.2770 | +0.041 |
| 真实数据 + 随机初始化模型 | 6.1313 | −0.105 |

原因（诊断脚本实测）：随机初始化的网络下，**同一张图的两个视图已经比"随便两张图"更相似**：

```
正样本相似度均值 = +0.4432
负样本相似度均值 = +0.3530
正 - 负          = +0.0902
```

两个视图像素相关 → 过同一个（平滑的）随机卷积网络 → 输出自然相关 → 正样本 logit 被系统性抬高 → loss 低于基线。

所以 `ln(2B−1)` 是「完全没有信息」的**基线**，不是「随机初始化下 loss 的精确值」。那 0.09 的差距就是训练信号的雏形。

**结论：实现正确；计划书那张 τ 敏感性表是在独立随机向量上标定的。**

### 4.5 loss 的波动来自哪里（实测拆解）

| 只改变这一项 | 实测 loss | 变化量 |
| --- | --- | --- |
| 增强随机种子 | 6.1678 → 6.1011 | **0.067**（最大） |
| 换一批图（batch 组成） | 6.1678 → 6.1437 | 0.024 |
| 模型随机初始化 | 6.1678 → 6.1635 | 0.004（最小） |
| **全部固定后重跑** | 6.1678 → **6.1678** | **0（逐位相同）** |

- → **代码是确定性的**；波动 100% 来自随机性
- → 主要波动源是**增强**，因为它直接决定「同一张图的两个视图有多像」
- → B2 只跑 5 个 batch 时，均值本身仍有 ±0.03 量级抖动

⚠️ **已知风险（已决定维持不改）**：真实数据偏差实测范围 **−0.06 ~ −0.14**，**越过了 ±0.1 的判据边界**。
若某次 `sanity_check.py` 断言失败，先看偏差是不是 −0.10 ~ −0.14 这种量级 —— 那是正常波动，**不是代码 bug**。
真正的错误信号是 4.x / `inf` / `nan` / 偏离幅度 > 0.5。

---

## 5. 环境备忘（踩过的坑）

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `conda activate pytorch` 后 `torch.cuda.is_available()` 仍是 `False` | 本机 PowerShell 未做 conda init，命令静默落到了 base 环境的 `torch 2.14.0+cpu` | **一律用绝对路径**：`& "D:\anaconda\envs\pytorch\python.exe"`，不要依赖 `conda activate` |
| `curl` 报 `Failed to connect to www.cs.toronto.edu port 443 via 127.0.0.1` | 代理环境变量指向了 `7890`，本机代理实际端口是 **`7897`** | 用 `$env:HTTPS_PROXY="http://127.0.0.1:7897"` |
| CIFAR-10 下载慢（代理实测 ≈71 KB/s） | torchvision 单线程、无断点续传 | 用迅雷/aria2c 多连接下载后丢进 `data/`，torchvision 校验 md5 通过会跳过下载 |
| 设了 `random.seed(42)` 重取同一批数据，结果**仍然不一致** | torchvision 的 `ColorJitter` 用的是 **torch 的 RNG**（`torch.empty(1).uniform_()`），而 `RandomResizedCrop` / `RandomHorizontalFlip` / `RandomGrayscale` 用的是 Python `random`。这套增强**同时踩在两个 RNG 上** | 要复现实验必须四个种子一起设：`random.seed` + `torch.manual_seed` + `np.random.seed` + `torch.cuda.manual_seed_all` |
| `print` 中文/特殊符号报 `UnicodeEncodeError: 'gbk' codec ...` | Windows 控制台默认 GBK，编不出 `−`(U+2212) 等字符 | 换成 ASCII `-`，或设 `$env:PYTHONIOENCODING="utf-8"` |

### 标准自检命令

```powershell
cd "d:\learning project\W1 InfoNCE"
& "D:\anaconda\envs\pytorch\python.exe" src\augment.py   # A4
& "D:\anaconda\envs\pytorch\python.exe" src\data.py      # A3
& "D:\anaconda\envs\pytorch\python.exe" src\model.py     # A5
& "D:\anaconda\envs\pytorch\python.exe" src\loss.py      # B1
& "D:\anaconda\envs\pytorch\python.exe" src\sanity_check.py   # B2（要 GPU，约 10 秒）
```

五个脚本都是**自断言**的：末尾打印 `[PASS] ...` 才算过，断言失败会非零退出。

---

## 6. 下一步（仅剩 B5）

### B1 / B2 已完成 —— 实现要点留档

`NTXentLoss.forward(z: (2B,128)) -> scalar` 的 5 步：

1. `F.normalize(z, dim=1)` —— **L2 归一化**（漏掉 → 相似度尺度不稳、τ 失效、易出 `nan`）
2. `sim = z @ z.t() / τ` → `(2B, 2B)`，τ = 0.5
3. `sim.masked_fill(eye, -inf)` —— 屏蔽对角线（对角 logit 恒为 `1/τ = 2.0`，是会挤占 softmax 分母的大值）
4. `labels = [B..2B-1] + [0..B-1]` —— 正样本是 `i+B`，**不是对角线**（最容易错的一步）
5. `F.cross_entropy(sim, labels)` —— `S` 对称 + labels 覆盖双向，**一次 CE 即可**，无需手动对称化

排错对照：

| 现象 | 判断 |
| --- | --- |
| 实测 ≈ 4.x 且不下降 | labels 写成了对角线 `i`（且未屏蔽对角线） |
| `loss = inf` | 对角线已屏蔽成 `-inf`，labels 却指到对角线上 |
| `loss = nan` | 忘了 L2 归一化，或 τ = 0 |
| 实测 6.10 ~ 6.18 | **正常**，见 §4.4 |
| 偏高到 7.07 | **正常**，那是 τ=0.07 的偏移，不是 labels 错 |

### B5 · 进 GitHub

- `git remote add origin <仓库地址>` → `git push -u origin master`
- `README.md` 至少写三样：① 环境（conda 环境名 + requirements.txt）② 跑的命令（一行）③ 期望输出 `loss ≈ 6.236`

---

## 7. 纪律备注

计划书第 6 章明确：**代码本人写**，计划书只给契约、坑、验收数字。
本仓库的注释按「五层注释」规范写（文件头含自检数字 → 分节横幅 → 设计决策 → 变量三行 → 调用四行 → 末尾附「怎么验证跑对了」+ 异常对照表），目的是**不查文档也能看懂、出错能自查**。

---
---

# W2 · 对比学习收口（计划 2026-10-20 ~ 10-26）

> 上游计划书：`W2详细计划（10.20-10.26）`
> **核心转变**：W1 证明「**跑得通**」（loss ≈ ln(2B−1)，模型什么都没学到），
> W2 证明「**真的在学**」（loss 掉下去 + t-SNE 看得出来）。这两个数字的差就是本周的价值。

## W2.1 进度

| 节 | 日 | 产物 | 状态 | 实测 |
| --- | --- | --- | --- | --- |
| A | D1 | InfoNCE 训练曲线 + checkpoint | ✅ 完成 | 起始 **6.1753** → 末尾 **4.7429**（τ=0.5、15 epoch） |
| B | D2 | t-SNE 随机 vs 训练后 + 正负距离 | ✅ 完成 | gap = **+0.8192** |
| P4 | — | τ 敏感性（0.07 / 0.5 两组） | ❌ **已砍** | 按计划书 F 节「时间不够就砍 τ」规则执行 |

## W2.2 新增文件

| 文件 | 作用 |
| --- | --- |
| `src/train.py` | A · 正式训练循环（AdamW） + 指标每 epoch 落 JSON + 自动出图 |
| `src/plot_curve.py` | A · 从 JSON 独立画图（秒级，完全不碰训练） |
| `src/tsne_vis.py` | B · t-SNE（随机 vs 训练后）+ 正负样本对距离 |

## W2.3 实测数字

### A · 训练（τ=0.5、B=256、AdamW lr=3e-4 wd=1e-4、15 epoch、seed=0）

| 项 | 值 |
| --- | --- |
| 起始 loss（随机初始化，训练前单独测） | **6.1753** |
| 第 20 步 | 5.6826 |
| 末尾 loss（最后 5 个记录点均值） | **4.7429** |
| 下降幅度 | +1.4323 |
| epoch 1 → 15 均值 | 5.2987 → 4.7459（单调下降，无异常抖动） |
| 耗时 | 925.9 s（61.7 s/epoch） |

- 计划书 A3 目标区间「4.5 ~ 5.5」→ 实测 **4.74**，落在区间内 ✅
- **`num_workers` 的影响（实测）**：2 → 77 s/epoch；**4 → 61.7 s/epoch**（提速 20%）
  → 坐实了「瓶颈是 CPU 端随机增强，不是 GPU」

### B · t-SNE 与正负样本对距离（2000 张测试图，每类 200，**不带增强**）

| 项 | W2 训练后 | W1 训练前（换算） | 变化 |
| --- | --- | --- | --- |
| 正样本对余弦距离 | **0.1561**（std 0.1469） | 0.5568 | ↓ 拉近 |
| 负样本对余弦距离 | **0.9753**（std 0.2625） | 0.6470 | ↑ 推远 |
| **分离度 gap** | **+0.8192** | **+0.0902** | **约 9 倍** |
| 特征各维标准差均值 | 1.1720（远 > 0.05，**无退化解**） | — | — |

> ★ **这一行对比就是 W2 的验收**：W1 的 gap 只有 0.09（几乎等于没有信号），
> 训练 15 epoch 后到 0.82。图 1 的「左糊右分」和图 2 的「两条直方图分开」是同一件事的两种呈现。

## W2.4 产物清单（`runs/`，已被 .gitignore 忽略）

```
runs/
├── W2_curve_tau0.5_loss.png          P1 · 训练曲线（step 轴，含起点与两条基线）
├── W2_tsne_1_random_init.png         P2 · 随机初始化 t-SNE（10 色，应为一团糊）
├── W2_tsne_2_trained.png             P2 · 训练后 t-SNE（10 色，应按类别成簇）
├── W2_tsne_3_pos_neg_dist.png        P2 · 正负样本对余弦距离直方图
├── ckpt_tau0.5.pth                   权重（D2 的输入）
├── metrics_tau0.5.json               指标 JSON（人类可读）
└── train_tau0.5.log                  训练日志
```

## W2.5 关键设计（踩过的坑与决策）

1. **训练 / 指标 / 画图 三者解耦**：`train.py` 只训练并把指标写 `metrics_tau*.json`
   （**每 epoch 落一次盘**），`plot_curve.py` 只读 JSON 画图。
   → 改标题/换横轴/两组对比都是秒级；训练中途崩了也不丢已完成的曲线。
   动机：原实现把画图焊在训练末尾，且只在全部结束时 `torch.save` 一次 —— 跑 18 个 epoch 崩掉就全丢。
2. **起点 loss 必须单独测**：在训练循环**之前**跑一次 `no_grad` 前向得到的才是真起点（6.1753）。
   若拿循环里第 1 个记录点（第 20 步 = 5.6826）当起点，τ 对比表会填错。
3. **每 100 步打一行纯文本进度**：tqdm 在输出被重定向/管道时**会自动静默**，
   这行 `print` 是那时唯一能看出「训到哪一步」的东西。
4. **t-SNE 用测试集 + 干净图**：用训练集有记忆效应；用增强图会让特征被拉向不同方向，图变糊。
5. **t-SNE 拆成三张独立 PNG，不拼盘**：`_1_random_init` / `_2_trained` / `_3_pos_neg_dist`。
   最初做成「随机 vs 训练后」左右并排，实测发现**每张子图只占一半宽度，2000 个点被压扁、簇边界看不清**，
   遂改为各出一张完整画布，「对比」交给文件名编号和阅读顺序。
   配色也一并换成高饱和的 ColorBrewer Set1 系（tab10 偏柔和）+ 黑色描边（密集区点仍可分辨）。
6. **`step` 而非 `epoch` 作横轴**：20 epoch 只有 20 个点，画不出「前 100 步陡降」这种细节；
   `plot_curve.py --x-axis epoch` 可按需切换。

## W2.6 待办

- [ ] **W1 欠账**：D3 自检 5 题（30min）、P2 论文方法框图（TII 2025，G1 验收物）
- [ ] **W1 的 B5**：推 GitHub —— 卡在账号（GCM 当前登录 `up70xiyang`，目标仓库属 `xiyang123457`）
- [ ] 周日 20:00 前发周报（草稿见 W2 计划书 G 节）

> **W2 代码线到此闭合**：A（训练）+ B（可视化）两个 P 级产物已交付，
> 剩 C（读论文）和 W1 欠账都是非代码工作。
