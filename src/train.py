"""W2-A · InfoNCE 正式训练（D1）

【用途】
    把 W1「只做了 forward」的 InfoNCE 补成完整训练：加 optimizer、训练循环、
    loss 记录、曲线出图、checkpoint 保存。
    ★ 目标不是「跑通」，是让 loss 从随机初始化的基线**掉下去** ——
      这是「模型真的在学」的唯一证据。W1 证明了跑得通，这一步证明它真的在学。

【输入】无（CIFAR-10 已在 data/ 就位）
【输出】
    runs/W2_curve_tau{τ}_loss.png   训练曲线（单张 PNG，一条指标）
    runs/ckpt_tau{τ}.pth            checkpoint —— ★ D2 的 t-SNE 必须用它，
                                    不存就得重跑 20 epoch

【跑完应看到的自检数字】B=256、τ=0.5、20 epoch：
    起始 loss ≈ 6.14     ← ★ W1 实测的随机初始化基线（不是 6.2364，原因见 PROGRESS.md §4.4）
    末尾 loss ≈ 4.5 ~ 5.5
    趋势    单调下降（step 级会有抖动，看移动平均线）
    τ=0.07 那组：起始 loss ≈ 7.07（τ 的常数偏移，正常，见 §B3 表）

【为什么起点是 6.14 不是 6.26】
    ln(2B−1) = 6.2364 是「所有候选 logits 近似相等」的纯理论基线；
    真实数据上，同一张图的两个视图天然比随机两张图更像（实测相似度差 +0.0902），
    所以起点会被压低到 ≈ 6.14。这是 W1 已经查清并记录在案的，不是 bug。
"""

# --- 标准库 ---
# argparse：命令行参数。让 τ / epoch 数可调，同一份脚本跑两组实验
import argparse
# math：算 ln(2B−1)。用作曲线上的参考线
import math
# random：★ 必须单独设 —— torchvision 的 RandomResizedCrop / RandomHorizontalFlip /
#   RandomGrayscale 用的是 Python random，而 ColorJitter 用的是 torch RNG。
#   只设 torch 的种子复现不了数据（W1 实测踩过，见 PROGRESS.md §5）
import random
# json：把训练指标写成人类可读的 JSON —— ★ 这是「训练」与「画图」解耦的关键：
#   权重留在 .pth（二进制，torch 专用），指标单独成 JSON（记事本/Excel/画图脚本都能读）
import json
# sys：判断 stderr 是否是终端 —— 决定 tqdm 进度条显示还是静默
import sys
# time：记录总耗时，用于估算「跑满 20 epoch 要多久」
import time
# pathlib.Path：跨平台路径拼接
from pathlib import Path

# --- 第三方 ---
# numpy：设 numpy 的随机种子（完整复现的四个种子之一）
import numpy as np
# torch：张量 + 自动求导
import torch
# torch.nn：这里只用来做模块基类的类型标注（actual 使用见 model.py）
import torch.nn as nn
# torch.utils.data.DataLoader：批量取数据。★ 这里显式构造而不是复用 data.build_loader()，
#   因为训练配置（num_workers / drop_last）应集中在这一个文件里，方便对照计划书 A1/A2 表
from torch.utils.data import DataLoader
# tqdm：进度条。让「step / loss」实时可见（计划书 A4 验收要求终端能看到两个数字在滚）
from tqdm import tqdm

# --- 本项目 ---
# build_augmentation：A4 的增强流水线。★ 每个视图都由它现场生成，绝不复用
from augment import build_augmentation
# CIFAR10Pair：A3 的双视图 Dataset，返回 (v1, v2)
# PROJECT_ROOT：项目根目录，用于定位 runs/
from data import CIFAR10Pair, PROJECT_ROOT, N_TRAIN
# SimCLR：A5 的 Encoder + ProjectionHead
from model import SimCLR
# NTXentLoss：B1 的损失
from loss import NTXentLoss
# plot：画图函数。★ 从 plot_curve.py 导入而不是在本文件里自己实现 ——
#   单一真相源：训练结束自动出一张图，之后想改标题/换横轴就用同一个函数单独重画，
#   不必重跑 26 分钟训练
from plot_curve import plot as plot_from_metrics


# ==== 1. 训练超参（全部集中在这里，做实验只改这一块） ====

# 常量 SEED：int，全局随机种子
#   示例值：0
#   为什么固定：计划书 A2 要求「便于复现；D2 扫 τ 时种子必须相同」——
#   两组 τ 实验只有 τ 不同，其余全同，结论才干净
SEED = 0

# 常量 BATCH_SIZE：int，原图数量 B
#   示例值：256 → 2B = 512
#   为什么是 256：与 W1 的 sanity check 一致（ln511 可对照），也正好是「负样本数 = 2B−1 = 511」
#   为什么不改大：负样本全部来自 batch 内，越大越好，但 8GB 显存扛不住更大
BATCH_SIZE = 256

# 常量 LR：float，AdamW 学习率
#   示例值：3e-4
#   为什么是这个量级：SimCLR 对小模型（ResNet-18 级）的常用组合。★ lr 对对比学习很敏感，
#   太大 → loss 直接 nan；太小 → loss 卡在基线不动（两种情况在计划书 F 节都是卡点）
LR = 3e-4

# 常量 WEIGHT_DECAY：float，权重衰减
#   示例值：1e-4
#   为什么加：抑制权重无限增大；SimCLR 原文用 1e-6，小模型上 1e-4 更稳
WEIGHT_DECAY = 1e-4

# 常量 EPOCHS：int，训练轮数
#   示例值：20
#   为什么是 20~30：CIFAR-10 单 epoch 秒级，跑得起；目标是「看趋势」而不是刷最优指标。
#   ★ 20 epoch × 195 步 ≈ 3900 步，是本机几分钟到十几分钟量级
EPOCHS = 20

# 常量 TAU：float，温度系数
#   示例值：0.5（另一组实验用 0.07）
#   为什么默认 0.5：起点干净（起始 loss ≈ 6.14，容易判断「有没有掉下去」）
TAU = 0.5

# 常量 NUM_WORKERS：int，DataLoader 子进程数
#   示例值：2（计划书 A1 的指定值）
#   为什么是 2：增强（RandomResizedCrop + ColorJitter + GaussianBlur）是 CPU 瓶颈，
#   单进程喂不满 GPU。⚠️ Windows 用 spawn 起子进程，会各自复制一份 Dataset（约 150MB），
#   内存翻倍；若出现卡死/报 spawn 错，把它改成 0 即可（慢但稳）
NUM_WORKERS = 2

# 常量 LOG_EVERY：int，每多少步把 loss 记一次
#   示例值：20
#   为什么不必每步都记：单步 loss 抖动大（batch 内负样本噪声），密采样只是增加噪声点；
#   20 步一次已足够画出平滑趋势
LOG_EVERY = 20

# 常量 RUNS_DIR：Path，产物目录
#   示例值：D:\learning project\W1 InfoNCE\runs
#   为什么放这里：.gitignore 已忽略 runs/，权重和图片不进仓库
RUNS_DIR = PROJECT_ROOT / "runs"


# ==== 2. 随机种子 ====


def set_seed(seed: int) -> None:
    """固定四个随机源，让实验可复现。

    为什么是四个：Python random（几何增强）、numpy（数据管线备用）、torch CPU（ColorJitter）、
    torch CUDA（卷积/初始化）。少设任何一个，两次运行都会有差异。
    """
    # 调用 random.seed(seed)：Python 标准库随机源
    #   作用：约束 RandomResizedCrop / RandomHorizontalFlip / RandomGrayscale
    #   坑：它【管不到】ColorJitter —— 后者用 torch RNG，这就是 W1 那次「设了种子还不一致」的原因
    random.seed(seed)
    # 调用 np.random.seed(seed)：numpy 随机源
    np.random.seed(seed)
    # 调用 torch.manual_seed(seed)：torch 的 CPU 随机源
    #   作用：约束权重初始化 + ColorJitter 的 factor 采样
    torch.manual_seed(seed)
    # 调用 torch.cuda.manual_seed_all(seed)：所有 GPU 的随机源
    #   作用：影响 dropout / 随机算子；本项目没有 dropout，但设上是好习惯
    torch.cuda.manual_seed_all(seed)


# ==== 3. DataLoader ====


def build_train_loader(batch_size: int, num_workers: int) -> DataLoader:
    """构造训练用 DataLoader，每个 batch 给出 (v1, v2) 各 (B,3,32,32)。

    返回
        DataLoader
    关键
        drop_last=True 是硬要求 —— 见下方注释
    """
    # 变量 dataset：CIFAR10Pair，50000 张训练图的双视图数据集
    #   为什么显式传 transform：训练必须用带随机性的增强流水线；默认值也是它，写明是为了可读
    dataset = CIFAR10Pair(train=True, transform=build_augmentation(), download=False)

    # 调用 DataLoader(dataset, ...)：签名
    #   `DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
    #               drop_last=False, pin_memory=False, persistent_workers=False) -> DataLoader`
    #   作用：批量取 (v1, v2)。默认 collate 遇到 tuple 会按位置转置 →
    #        输出 (v1_batch, v2_batch)，各 (B,3,32,32)
    #   关键参数：
    #     shuffle=True   —— 训练集必须打乱，否则 batch 内样本同质，负样本区分度低
    #     drop_last=True —— ★★ 计划书 A1 第 2 条的重点：不丢掉尾巴 batch 的话，
    #       最后一批只有 B'=72 个样本，而 loss 里的正样本索引是 `i+B`（常量 256）
    #       → 索引越界 / 错位，报 IndexError 或算出无意义的 loss
    #     pin_memory=True —— 锁页内存，H2D 拷贝更快（有 GPU 时才有意义）
    #     persistent_workers=True —— 复用 worker 进程，避免每个 epoch 重启（省几秒/epoch）
    #   坑：num_workers>0 时整个 main 必须放在 `if __name__ == "__main__"` 里 ——
    #       Windows 用 spawn 起子进程，不加保护会递归启动直到崩
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ==== 4. 训练一个 epoch ====


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int,
    step_offset: int,
) -> tuple[list[float], list[int], int]:
    """跑完一个 epoch，返回 (loss 列表, 对应的全局 step 列表, 本 epoch 步数)。"""
    # 调用 model.train()：切训练模式
    #   作用：BatchNorm 用当前 batch 统计量并更新 running stats；Dropout 打开（本项目没有）
    #   坑：★ 忘了它 → BN 用 running stats，训练行为与预期不符，且 running stats 永不更新，
    #       D2 用 eval() 提特征时会得到错的特征
    model.train()

    # 变量 losses：list[float]，本 epoch 记录下来的 loss
    # 变量 steps：list[int]，与 losses 一一对应的全局步数（横轴用）
    losses, steps = [], []

    # 变量 running_loss：float，两个记录点之间的平均 loss
    #   为什么用区间平均而不是单点：单步 loss 抖动大（一个 batch 里负样本偶然好区分）
    running_loss = 0.0

    # 变量 t_epoch：float，本 epoch 起始时刻（用来算 it/s，估算还剩多久）
    t_epoch = time.time()

    # 调用 tqdm(loader, ...)：包一层进度条
    #   作用：每一步实时显示 loss，满足计划书 A4「终端能看到 step / epoch / loss 滚动」
    #   关键参数：
    #     desc —— 左侧说明
    #     leave=False —— 做完的 epoch 不占屏
    #     disable=not sys.stderr.isatty() —— ★ 输出被重定向/被工具捕获时自动静默。
    #       否则 tqdm 每步写一行，3900 步会把真正重要的日志冲掉；在真实终端里（tty）照常显示
    pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not sys.stderr.isatty())

    for i, (v1, v2) in enumerate(pbar):
        # 调用 tensor.to(device)：把数据搬到 GPU
        #   作用：模型和输入必须同设备，否则报 "Expected all tensors to be on the same device"
        #   关键参数：non_blocking=True 配合 pin_memory，让拷贝与计算重叠（微小提速）
        #   坑：DataLoader 产出永远在 CPU 上，这一步不能省
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        # 调用 torch.cat([v1, v2], dim=0)：签名 `-> Tensor(2B,3,32,32)`
        #   作用：拼成 2B。★ 前 B 个是视图 1、后 B 个是视图 2 —— 这个顺序正是
        #       loss.py 里 labels = [i+B] + [i] 依赖的前提，也正好使 drop_last 成为必须
        #   关键参数：dim=0（沿样本维）
        x = torch.cat([v1, v2], dim=0)

        # --- 标准的五步训练循环（顺序错一个就可能 nan）---

        # 第 1 步：前向。model(x) → z (2B,128)
        #   坑：这里【不能】包 torch.no_grad() —— 那会切断计算图，backward 直接报错
        z = model(x)

        # 第 2 步：算损失。criterion(z) → 0 维标量
        loss = criterion(z)

        # 第 3 步：清空上一步的梯度
        #   作用：PyTorch 的梯度是【累加】的，不清零就会把上一步的梯度叠进来 → loss 发散
        #   坑：★ 漏掉这行是最隐蔽的 bug 之一 —— 不报错，只是 loss 一路涨或变 nan
        optimizer.zero_grad(set_to_none=True)

        # 第 4 步：反向传播，算梯度
        #   坑：若前面任何一步用了 no_grad / detach，这里会报 "element 0 of tensors does not require grad"
        loss.backward()

        # 第 5 步：更新参数
        optimizer.step()

        running_loss += loss.item()

        # 每 log_every 步记录一次（区间平均）
        if (i + 1) % log_every == 0:
            avg = running_loss / log_every
            losses.append(avg)
            # ★ 记「已完成的步数」= i+1，不是 0-based 的 i ——
            #   否则横轴最后一个点会标成 194，而实际上已经跑完 195 步
            steps.append(step_offset + i + 1)
            running_loss = 0.0
            pbar.set_postfix(loss=f"{avg:.4f}")

            # ★ 每 5 个记录点（= log_every×5 = 100 步）额外打一行【纯文本】进度。
            #   为什么必须补这一行：tqdm 只在真实终端里显示；一旦你把输出重定向到文件
            #   （python train.py --epochs 15 > log.txt）或通过管道 / tee，
            #   进度条会被自动关闭 —— 那时这行 print 是唯一能看出「训到哪一步了」的东西
            if (i + 1) % (log_every * 5) == 0:
                # 变量 ips：float，本 epoch 的实际速度（步/秒）
                #   用途：估算剩余时间 —— 195 步 ÷ 2.5 it/s ≈ 78 s/epoch，乘剩余 epoch 数即可
                ips = (i + 1) / max(1e-9, time.time() - t_epoch)
                print(f"    [epoch {epoch:>2} | step {step_offset + i + 1:>5}] "
                      f"loss = {avg:.4f}  ({ips:.2f} it/s)")

    # 收尾：把最后不足 log_every 步的残余也记下来（否则丢掉一段）
    if running_loss > 0.0:
        losses.append(running_loss / ((i + 1) % log_every or log_every))
        steps.append(step_offset + i + 1)

    # 第三个返回值是本 epoch 的真实步数：main 里靠它累加 step_offset，
    #   不能写成「最后一个记录点 + 20」—— 尾巴不足 20 步时会算偏
    return losses, steps, len(loader)


# ==== 5. 指标落盘（JSON） ====


def save_metrics(
    path: Path,
    config: dict,
    init_loss: float,
    losses: list[float],
    steps: list[int],
    epoch_index: list[int],
    epoch_mean: list[float],
    elapsed_s: list[float],
    end_loss: float | None = None,
) -> dict:
    """把训练指标写成 JSON（覆盖写），返回写出的 dict（调用方可直接拿去画图）。

    为什么不用 checkpoint 代替这个文件：
        ckpt 里必须装张量（权重），JSON 装不了；把指标塞进 ckpt 的后果是
        「只有 torch 能读」。分开存之后，指标能用记事本/Excel 打开，
        也能被 plot_curve.py 独立读取 —— 改图不必重跑训练。
    为什么每个 epoch 都调一次：
        torch.save 只在训练【完全结束】时执行。若 20 epoch 跑到第 18 个崩了，
        权重和曲线会一起丢。每 epoch 落一次盘，崩了也保住已完成的部分。

    参数
        end_loss：训练结束时才有的收敛值；训练中途落盘时传 None
    返回
        dict：与写入文件内容一致
    """
    # 变量 payload：dict，要序列化的全部内容
    #   结构说明见 plot_curve.py 末尾「JSON 结构」一节
    payload = {
        "config": config,
        "log_every": LOG_EVERY,
        "init_loss": init_loss,
        "end_loss": end_loss,
        "steps": list(steps),
        "losses": list(losses),
        "epoch_index": list(epoch_index),
        "epoch_mean": list(epoch_mean),
        "elapsed_s": list(elapsed_s),
    }

    # 调用 Path.open("w", encoding="utf-8")：以文本写模式打开（存在则截断重写）
    #   关键参数：encoding="utf-8"
    #   坑：★ 必须显式写 encoding。Windows 默认 GBK，一旦 payload 里出现非 ASCII 字符
    #       （现在没有，以后加了中文备注就会踩）会直接抛 UnicodeEncodeError
    with open(path, "w", encoding="utf-8") as f:
        # 调用 json.dump(payload, f, ...)：签名 `-> None`
        #   作用：序列化写入
        #   关键参数：
        #     ensure_ascii=False —— 中文按原样写（而不是 \uXXXX），文件可读
        #     indent=2 —— 人工看得懂；这里只有几百个数字，体积不是问题
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


# ==== 6. 主流程 ====


def main():
    # 变量 parser：argparse.ArgumentParser，命令行解析器
    #   为什么需要它：同一份脚本要跑 τ=0.5 和 τ=0.07 两组，还要能先跑 2 epoch 试通
    parser = argparse.ArgumentParser(description="W2-A: InfoNCE training")
    parser.add_argument("--tau", type=float, default=TAU, help="温度系数（0.5 或 0.07）")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader 子进程数")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size B")
    args = parser.parse_args()

    # 变量 device：torch.device
    #   为什么显式判断：若落到 CPU，必须立刻发现 —— 训练会慢 50 倍以上，
    #   而且 99% 是「误用了 base 环境的 CPU 版 torch」（见 PROGRESS.md 环境备忘）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError(
            "检测到没有 CUDA。必须用 & 'D:\\anaconda\\envs\\pytorch\\python.exe' 运行，"
            "不能依赖 conda activate（见 PROGRESS.md 环境备忘）"
        )

    # 调用 set_seed(SEED)：固定四个随机源
    #   关键参数：必须在【建模型和 DataLoader 之前】调用，否则初始化已经消耗过随机数了
    set_seed(SEED)

    # 变量 tau_tag：str，文件名里的 τ 标记
    #   示例值：tau0.5 / tau0.07
    tau_tag = f"tau{args.tau}"

    # 变量 config：dict，这次训练的全部超参
    #   为什么抽成一个 dict：它要同时写进 ckpt 和 metrics JSON ——
    #   两处共用同一份，才不会出现「JSON 里记的 tau 和 ckpt 里的不一致」这种事故
    config = {
        "tau": args.tau,
        "batch_size": args.batch_size,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": args.epochs,
        "seed": SEED,
    }

    # 变量 metrics_path：Path，指标 JSON 落点
    #   为什么提前定义：训练循环里每个 epoch 都要往它写，不能拖到最后才算路径
    metrics_path = RUNS_DIR / f"metrics_{tau_tag}.json"
    # 调用 RUNS_DIR.mkdir(parents=True, exist_ok=True)：建产物目录
    #   关键参数：parents=True 连父目录一起建；exist_ok=True 已存在不报错
    #   坑：★ 必须提前建 —— 否则第一个 epoch 结束时写 JSON 会 FileNotFoundError
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[config] device={device}  B={args.batch_size}  tau={args.tau}  "
          f"lr={LR}  wd={WEIGHT_DECAY}  epochs={args.epochs}  seed={SEED}")
    print(f"[config] 每 epoch 步数 = {N_TRAIN // args.batch_size}（drop_last=True，尾巴丢弃）")

    # 变量 model：SimCLR —— ★ 权重共享的那一座塔，两个视图都喂给它
    model = SimCLR().to(device)

    # 变量 criterion：NTXentLoss，τ 由命令行决定
    criterion = NTXentLoss(temperature=args.tau)

    # 调用 torch.optim.AdamW(params, lr, weight_decay)：签名
    #   `AdamW(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2) -> Optimizer`
    #   作用：把梯度转成参数更新。AdamW = Adam + 解耦的权重衰减（比 Adam 的 L2 更规范）
    #   关键参数：model.parameters() —— ★ 必须传【模型】的参数；写错成空列表也不报错，
    #   只是训练「跑得通但 loss 一动不动」，是极隐蔽的坑
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 变量 loader：DataLoader，B=256，drop_last=True
    loader = build_train_loader(args.batch_size, args.num_workers)

    # --- ★ 训练前先测一次「随机初始化」的 loss = 真正的起点 ---
    # 为什么必须单独测一遍：
    #   计划书 B3 的 τ 对比表要填「起始 loss」，语义是「没训练过时的 loss」
    #   （τ=0.5 应 ≈ 6.14、τ=0.07 应 ≈ 7.07）。
    #   如果偷懒拿训练循环里第 1 个记录点当起点，读到的是「已经训练过 20 步」的值 ——
    #   本机实测会低到 5.49（20 步掉了 0.65），直接把这栏填错。
    # 为什么不开 no_grad 之外的开关：保持与 W1 的 sanity_check.py 完全同一种测法
    #   （train 模式下 BN 用 batch 统计量），这样两处的数字才能互相对得上。
    with torch.no_grad():
        # 调用 next(iter(loader))：取一个 batch
        #   副作用：消耗掉一个 batch 的数据与随机数。两组 τ 实验都会消耗同样多，所以不影响横向对比
        v1, v2 = next(iter(loader))
        # 前向 → 算 loss。这就是「起点」
        init_loss = criterion(model(torch.cat([v1, v2], dim=0).to(device))).item()
    print(f"[init] 随机初始化 loss（未训练）= {init_loss:.4f}   "
          f"← 填进 τ 对比表的「起始 loss」；理论 ln(2B-1) = {math.log(2 * args.batch_size - 1):.4f}")

    # 变量 all_losses / all_steps：累积全训练的曲线（step 轴用）
    all_losses: list[float] = []
    all_steps: list[int] = []
    # 变量 epoch_index / epoch_mean：epoch 轴数据
    #   为什么两份都留：step 轴看细节（前 20 步陡降），epoch 轴看趋势（给导师/周报），
    #   它们是同一批实验的两种呈现，不存在重复计算的浪费
    epoch_index: list[int] = []
    epoch_mean: list[float] = []
    # 变量 elapsed_s：list[float]，每个 epoch 的耗时
    #   为什么记它：77 s/epoch 与 148 s/epoch 差一倍，这是判断「要不要调 num_workers」的依据
    elapsed_s: list[float] = []
    step_offset = 0

    # 变量 t0：float，起始时刻（算总耗时）
    t0 = time.time()

    # --- 训练主循环 ---
    for epoch in range(1, args.epochs + 1):
        # 变量 epoch_t0：float，本 epoch 起跑时刻（用于记录 s/epoch）
        epoch_t0 = time.time()

        # 调用 train_one_epoch(...)：跑一轮，返回本轮记录点
        ep_losses, ep_steps, n_steps = train_one_epoch(
            model, loader, criterion, optimizer, device,
            epoch, args.epochs, LOG_EVERY, step_offset,
        )
        all_losses.extend(ep_losses)
        all_steps.extend(ep_steps)

        # 变量 ep_mean：float，本 epoch 记录点的均值
        #   为什么逐 epoch 打印：epoch 级均值比 step 级稳定，是判断「真的在降」的主依据
        ep_mean = sum(ep_losses) / len(ep_losses) if ep_losses else float("nan")
        epoch_index.append(epoch)
        epoch_mean.append(ep_mean)
        elapsed_s.append(round(time.time() - epoch_t0, 1))
        print(f"[epoch {epoch:>2}/{args.epochs}] mean loss = {ep_mean:.4f}  "
              f"({elapsed_s[-1]:.1f} s)")

        step_offset += n_steps

        # ★ 每个 epoch 落一次盘 —— 训练中途崩了也不白跑
        #   调用 save_metrics(...)：把当前累积进度写成 JSON
        #   关键参数：end_loss 传 None（还没训练完，没有收敛值）
        #   坑：★ 它是【覆盖写】，所以每次都传完整累积列表 —— 只传本 epoch 的增量的话，
        #       文件里最后只会剩最后一个 epoch 的数据
        save_metrics(
            metrics_path, config, init_loss, all_losses, all_steps,
            epoch_index, epoch_mean, elapsed_s, end_loss=None,
        )

    # 变量 elapsed：float，总耗时（秒）
    elapsed = time.time() - t0

    # --- 结果统计 ---
    # 变量 start_loss：float，★ 真正的起点 = 训练前测的随机初始化 loss
    #   千万不要用 all_losses[0]：那是训练 20 步之后的值，本机实测会低 0.65（5.49 vs 6.14），
    #   填进 τ 对比表就是错的
    start_loss = init_loss
    # 变量 end_loss：float，收敛 loss
    #   为什么取最后 5 个记录点：末尾同样要抹掉单批抖动
    end_loss = sum(all_losses[-5:]) / min(5, len(all_losses))
    # 变量 first_recorded：float，训练循环里第 1 个记录点（第 LOG_EVERY 步）
    #   它的作用只有一个：展示「前 20 步已经掉了多少」，说明 lr 没设错
    first_recorded = all_losses[0]

    print("-" * 62)
    print(f"起始 loss（随机初始化，未训练）  = {start_loss:.4f}")
    print(f"第 {LOG_EVERY} 步时                  = {first_recorded:.4f}")
    print(f"末尾 loss（最后 5 个记录点均值） = {end_loss:.4f}")
    print(f"下降幅度 = {start_loss - end_loss:+.4f}")
    print(f"总耗时 = {elapsed:.1f} s（{elapsed / max(1, args.epochs):.1f} s/epoch）")

    # --- 存产物 ---
    # 变量 ckpt_path：Path，checkpoint 落点
    #   ★ D2 的 t-SNE「训练后」那条必须用它；不存就得重跑
    ckpt_path = RUNS_DIR / f"ckpt_{tau_tag}.pth"
    # 调用 torch.save(dict, path)：把权重和配置一起存
    #   关键参数：存 state_dict 而不是整个 model 对象 —— 后者依赖类定义路径，换目录就加载不了
    #   坑：★ 一定要把 config 一起存，否则 D2 加载时不知道这个权重是 τ=0.5 还是 0.07 训出来的
    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": all_losses,
            "steps": all_steps,
            "config": config,
            # 结果放【顶层】，与 config（超参）分开 —— 两者语义不同，混在一起会让
            # 「读 config 复现实验」时读到一个不属于超参的字段
            "init_loss": init_loss,
            "end_loss": end_loss,
        },
        ckpt_path,
    )
    print(f"[saved] {ckpt_path}")

    # ★ 最终落盘一次：这次带上 end_loss（训练中途的每次落盘都是 None）
    #   调用 save_metrics(...)：返回写出的 payload，直接拿去画图，省一次读盘
    metrics = save_metrics(
        metrics_path, config, init_loss, all_losses, all_steps,
        epoch_index, epoch_mean, elapsed_s, end_loss=end_loss,
    )
    print(f"[saved] {metrics_path}")

    # 变量 plot_path：Path，曲线落点
    #   命名带 tau_tag —— 两个 τ 是两次独立实验，各自一张 PNG（分开看各自都有意义）
    plot_path = RUNS_DIR / f"W2_curve_{tau_tag}_loss.png"
    # 调用 plot_from_metrics(metrics_list, out_path, x_axis, window)：来自 plot_curve.py
    #   作用：训练结束时自动出一张图，省得用户还要手动跑一次
    #   关键参数：★ 第一个参数是【列表】，单组就写成 [metrics]
    #   坑：传单个 dict 会报 "string indices must be integers" 之类的怪错 —— 因为 dict 不可按位取
    plot_from_metrics([metrics], plot_path, x_axis="step", window=50)
    print(f"[saved] {plot_path}")
    print("[hint] 想改标题 / 换横轴 / 两组对比，都不必重跑训练，直接：")
    print(f'       & "D:\\anaconda\\envs\\pytorch\\python.exe" src\\plot_curve.py --metrics {metrics_path}')

    # --- 自检 ---
    assert end_loss < start_loss - 0.3, (
        f"loss 只从 {start_loss:.4f} 降到 {end_loss:.4f}，没有明显下降。"
        "排查顺序：① labels 是否为 i+B ② 有没有 loss.backward() ③ lr 是否太小 "
        "④ z 是否做了 L2 归一化（见 loss.py 末尾对照表）"
    )
    print(f"[PASS] W2-A({tau_tag})：loss 从 {start_loss:.4f} 降到 {end_loss:.4f}")


if __name__ == "__main__":
    # ★ num_workers>0 时必须有这层保护：Windows 用 spawn 起子进程，
    #   不加保护子进程会重新 import 本模块并再次执行训练，递归爆炸
    main()


# ==== 怎么验证跑对了 ====
#
# 第一步：先用 2 个 epoch 快速试通
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\train.py --epochs 2
#
# 期望（数字是本机实测值）：
#   [config] device=cuda  B=256  tau=0.5  lr=0.0003  wd=0.0001  epochs=2  seed=0
#   [config] 每 epoch 步数 = 195（drop_last=True，尾巴丢弃）
#   [init] 随机初始化 loss（未训练）= 6.1x     ← ★ 这才是要填表的「起始 loss」
#   [epoch  1/2] mean loss = 5.3x  (77.0 s)
#   [epoch  2/2] mean loss = 5.1x  (77.0 s)
#   起始 loss（随机初始化，未训练）  = 6.1x
#   第 20 步时                  = 5.5x
#   末尾 loss（最后 5 个记录点均值） = 5.0x
#   下降幅度 = +1.0x
#   [saved] ...\runs\ckpt_tau0.5.pth
#   [saved] ...\runs\metrics_tau0.5.json      ← ★ 每个 epoch 都会刷一次，崩了不丢
#   [saved] ...\runs\W2_curve_tau0.5_loss.png
#   [hint] 想改标题 / 换横轴 / 两组对比，都不必重跑训练，直接跑 plot_curve.py
#   [PASS] W2-A(tau0.5)：loss 从 6.1x 降到 5.0x
#
#   ⚠️ 为什么「第 20 步时」比起点低 0.6：lr=3e-4 下前 20 步下降最快。所以报表面向的两个数
#      是「起始（未训练）」和「末尾」，中间那个只是给你确认 lr 没设错的参照。
#
# 第二步：正式跑 20 epoch
#   & "D:\anaconda\envs\pytorch\python.exe" src\train.py --epochs 20
#   期望末尾落在 4.5 ~ 5.5（计划书 A3 给的参考量级）
#   ⚠️ 本机实测约 **77 s/epoch**（首次跑会慢一倍，那是冷启动），20 epoch ≈ 26 分钟。
#      瓶颈是 CPU 端的随机增强，不是 GPU。想提速：--num-workers 4 或 6
#
# 第三步：τ=0.07 那组（P4，挂在后面跑）
#   & "D:\anaconda\envs\pytorch\python.exe" src\train.py --tau 0.07 --epochs 20
#   期望：[init] 那行 ≈ **7.07**（τ 的常数偏移，正常，不是 labels 错）
#         收敛 loss 通常比 τ=0.5 那组**更高、更抖** —— τ 越小分布越锐，只盯最难的负样本
#
# 改图（秒级，完全不用碰训练）：
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json --x-axis epoch
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json runs\metrics_tau0.07.json
#
# 训练中途崩了会怎样：
#   ✅ runs\metrics_tau0.5.json 保住了前 N 个 epoch 的数据 —— plot_curve.py 照样能出部分曲线
#   ❌ runs\ckpt_tau0.5.pth 只在训练【完全结束】时写一次 —— 半路崩掉就没有权重可给 D2 用
#
# 异常对照表（对应计划书 F 节）：
#   loss 卡在 6.14 一动不动
#       → ① labels 写错 ② 漏了 loss.backward() ③ optimizer 收的是空参数列表 ④ lr 太小
#   loss = nan
#       → lr 过大（先降到 1e-4 试）/ 忘了 L2 归一化 / τ=0
#   最后一批 IndexError
#       → drop_last 被改成 False 了。回来看 build_train_loader 的注释
#   loss 掉到 1.x 以下
#       → 退化解嫌疑：打印 z 的方差，看是不是所有样本都塌到同一个方向
#   Windows 上报 spawn / 卡在第一个 batch
#       → 把 --num-workers 0 加上（慢一点但稳）
#   RuntimeError: 检测到没有 CUDA
#       → 用错 python 了。必须绝对路径调 pytorch 环境的解释器
#   起始 loss ≈ 7.07 而不是 6.14
#       → 正常：你跑的是 τ=0.07 那组，偏移是 τ 的常数效应（见 W2 计划书 B3 表）
