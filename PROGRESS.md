# W1 · InfoNCE 进度记录

- **最后更新**：2026-10-16
- **上游计划书**：`W1-D4+D5计划书（骨架+InfoNCE初版）`
- **本节产物**：能跑通的 InfoNCE 初版代码（= W1 的 P3 交付物）

---

## 一句话状态

**A 阶段（搭壳）全部完成且逐项实测通过；B 阶段（填 InfoNCE）尚未开始。**

A5 的 shape 打印已过 —— 这是计划书定义的 A→B 交接点，可以往下走。

---

## 1. 进度总览

| 阶段 | 步骤 | 内容 | 状态 | 实测验收 |
| --- | --- | --- | --- | --- |
| A | A1 | 环境 | ✅ | `torch 2.9.0+cu130` / `cuda 13.0` / `is_available()=True` / RTX 5060 Laptop GPU |
| A | A2 | 仓库 | ✅ | `git init` 完成，`.gitignore` 已忽略 `data/` |
| A | A3 | 数据集 | ✅ | md5 `c58f30108f718f92721af3b95e74349a`，170498071 字节；50000/10000，10 类 |
| A | A4 | 增强流水线 | ✅ | `torch.equal(v1,v2)=False`，平均绝对差 **0.1353** |
| A | A5 | 骨架 + shape | ✅ | `h (2B,512)` / `z (2B,128)` / `layer4 (2B,512,4,4)` |
| B | B1 | 实现 InfoNCE | ⬜ 未开始 | `NTXentLoss` 尚未编写 |
| B | B2 | sanity check | ⬜ 未开始 | 目标：τ=0.5、B=256 时 `loss ≈ 6.26 ± 0.02` |
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
    └── model.py          A5 · Encoder + ProjectionHead + SimCLR（含自检）
```

**尚未创建**：`src/loss.py`（B1）、`src/sanity_check.py`（B2）、`README.md`（B5）

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
| `loss.NTXentLoss` | `z (2B,128)` | `scalar` | ⬜ B1 待写 |

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

---

## 5. 环境备忘（踩过的坑）

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `conda activate pytorch` 后 `torch.cuda.is_available()` 仍是 `False` | 本机 PowerShell 未做 conda init，命令静默落到了 base 环境的 `torch 2.14.0+cpu` | **一律用绝对路径**：`& "D:\anaconda\envs\pytorch\python.exe"`，不要依赖 `conda activate` |
| `curl` 报 `Failed to connect to www.cs.toronto.edu port 443 via 127.0.0.1` | 代理环境变量指向了 `7890`，本机代理实际端口是 **`7897`** | 用 `$env:HTTPS_PROXY="http://127.0.0.1:7897"` |
| CIFAR-10 下载慢（代理实测 ≈71 KB/s） | torchvision 单线程、无断点续传 | 用迅雷/aria2c 多连接下载后丢进 `data/`，torchvision 校验 md5 通过会跳过下载 |

### 标准自检命令

```powershell
cd "d:\learning project\W1 InfoNCE"
& "D:\anaconda\envs\pytorch\python.exe" src\augment.py   # A4
& "D:\anaconda\envs\pytorch\python.exe" src\data.py      # A3
& "D:\anaconda\envs\pytorch\python.exe" src\model.py     # A5
```

三个脚本都是**自断言**的：末尾打印 `[PASS] ...` 才算过，断言失败会非零退出。

---

## 6. 下一步（B 阶段）

### B1 · 实现 InfoNCE（`src/loss.py`）

要写的只有 `NTXentLoss.forward(z: (2B,128)) -> scalar`，步骤：

1. 对 `z` 做 **L2 归一化**（忘了这步 → 相似度尺度不稳、τ 失效）
2. `S = z @ z.T / τ` → `(2B, 2B)`，τ 先固定 **0.5**
3. `labels = [(i+B) for i in range(B)] + [i for i in range(B)]` ← **最容易错的地方**
4. `loss = CrossEntropyLoss()(S, labels)`

### B2 · sanity check（`src/sanity_check.py`）

- 真实数据、`B=256`、τ=0.5
- **期望 `loss ≈ 6.26 ± 0.02`**（`ln(511) = 6.2364`，τ=0.5 时理论偏移约 +0.02）
- 排错对照：
  - 实测 ≈ 4.x 且不下降 → labels 写成了对角线 `i`
  - loss = `inf` → 对角线没 mask 成 `-inf` 且 labels 指到了对角线
  - loss = `nan` → 忘了 L2 归一化，或 τ=0
  - 偏高到 7.07 → 正常，那是 τ=0.07 的偏移，不是 labels 错

### B5 · 进 GitHub

- `git remote add origin <仓库地址>` → `git push -u origin master`
- `README.md` 至少写三样：① 环境（conda 环境名 + requirements.txt）② 跑的命令（一行）③ 期望输出 `loss ≈ 6.236`

---

## 7. 纪律备注

计划书第 6 章明确：**代码本人写**，计划书只给契约、坑、验收数字。
本仓库的注释按「五层注释」规范写（文件头含自检数字 → 分节横幅 → 设计决策 → 变量三行 → 调用四行 → 末尾附「怎么验证跑对了」+ 异常对照表），目的是**不查文档也能看懂、出错能自查**。
