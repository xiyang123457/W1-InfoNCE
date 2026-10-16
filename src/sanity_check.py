"""B2 · InfoNCE sanity check —— 本周最重要的一行数字

【用途】
    在【真实数据 + 随机初始化的模型】上跑一次完整前向，确认 loss 落在 ln(2B−1) 附近。
    这个数字对了，说明 labels 构造、L2 归一化、2B 拼接三件事全对；
    数字不对，说明 loss 逻辑有 bug（99% 是 labels）。

    ★ 注意本脚本【不训练】：不建 optimizer、不 backward。因为一旦更新了参数，
      loss 就不再是「随机初始化」的值，这个验收也就失效了。

【输入】无（用 data.py 的真 CIFAR-10，B=256）
【输出】打印每个 batch 的 loss、均值、与 ln(2B−1) 的偏差

【跑完应看到的自检数字】`python src/sanity_check.py`：
    z.shape = (512, 128)          ← 2B × 128
    S.shape = (512, 512)
    τ = 0.5、B = 256 → 2B − 1 = 511
    ln(511)            = 6.2364   ← 理论期望
    5 次平均 loss      ≈ 6.24 ~ 6.28
    偏差               ≈ +0.01 ~ +0.03
    [PASS] B2 ...

【为什么这个数字重要】
    I(z;z⁺) ≥ log(2B−1) − L。L = ln(2B−1) 恰好对应互信息下界 = 0，
    即「模型什么都没学到」—— 这是训练的【起点】，不是终点。
    训练的唯一目标就是把这个 loss 压到 6.236 以下（W2 的曲线任务）。
"""

# --- 标准库 ---
# math：算 ln(2B−1)。用 math.log 比 torch.log 少一层张量包装
import math

# --- 第三方 ---
import torch

# --- 本项目 ---
# build_loader：A3 的双视图 DataLoader，产 (v1_batch, v2_batch) 各 (B,3,32,32)
from data import build_loader, BATCH_SIZE
# SimCLR：A5 的 Encoder + ProjectionHead，(2B,3,32,32) → z (2B,128)
from model import SimCLR
# NTXentLoss：B1 的损失，z (2B,128) → scalar
from loss import NTXentLoss, TEMPERATURE


# ==== 1. 验收常量 ====

# 常量 REPEATS：int，重复几次取平均
#   示例值：5
#   为什么重复：单个 batch 的 loss 有随机波动（batch 内样本不同）；平均后波动被抹平，
#   才有资格谈 ±0.02 这种精度
REPEATS = 5

# 常量 TOLERANCE：float，允许的最大偏差
#   示例值：0.1
#   为什么是 0.1：计划书 B4 定的判据（±0.1 通过）。τ=0.5 时理论偏移只有 +0.02，
#   留 0.1 的余量给「不同 batch 的抖动」和「不同机器/版本下的浮点差异」
TOLERANCE = 0.1


def main():
    # 变量 device：torch.device，"cuda" 或 "cpu"
    #   示例值：cuda
    #   为什么显式指定并 to(device)：模型和数据必须在同一设备上，否则报
    #   "Expected all tensors to be on the same device"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 变量 model：SimCLR，随机初始化的 Encoder + ProjectionHead
    #   为什么默认就是随机初始化：model.py 里 resnet18(weights=None)，
    #   不能加载 ImageNet 权重，否则这个 sanity check 的语义就不成立了
    model = SimCLR().to(device)

    # 调用 model.train()：切到训练模式
    #   作用：让 BatchNorm 用「当前 batch 的统计量」，并更新 running_mean/running_var
    #   关键参数：无
    #   坑：★ 这一步不能省。若停在 eval() 模式，BN 会用随机初始化的 running stats
    #   （mean=0, var=1），输出分布与真实训练时不同 —— 虽然 loss 量级差不多，
    #   但那就不是在测「训练时会发生什么」了
    model.train()

    # 变量 criterion：NTXentLoss，τ = 0.5
    #   为什么 τ 必须是 0.5：验收期望值 6.26 是按 τ=0.5 标定的（见 loss.py 末尾对照表）
    criterion = NTXentLoss(temperature=TEMPERATURE)

    # 变量 loader：DataLoader，B=256，drop_last=True
    #   为什么 drop_last 必须为 True：保证每个 batch 都是完整 256 → 2B−1 恒为 511。
    #   否则 50000 % 256 = 72 的尾巴 batch 会打出 ln(143) ≈ 4.96，看起来像 labels 错
    loader = build_loader(train=True, batch_size=BATCH_SIZE, download=False)

    # 变量 expect：float，理论期望值 ln(2B−1)
    #   示例值：6.2364 = ln(511)
    expect = math.log(2 * BATCH_SIZE - 1)

    # 变量 losses：list[float]，每次迭代的 loss
    #   为什么先收集再统计：要同时给出「每一次的值」和「均值 ± 波动」，
    #   只看均值会掩盖「某一次明显异常」这种情况
    losses = []

    print(f"设备 = {device}   τ = {TEMPERATURE}   B = {BATCH_SIZE}   2B-1 = {2 * BATCH_SIZE - 1}")
    if device.type == "cpu":
        print("[!] 警告：用的是 CPU。先确认是不是误用了 base 环境的 python（见 PROGRESS.md 环境备忘）")
    print("-" * 60)

    # 调用 torch.no_grad()：上下文管理器，关闭自动求导
    #   作用：不构建计算图，省显存、更快
    #   关键参数：无
    #   坑：★ 只适合「验收」这种只看数值的场景。真实训练必须去掉它，
    #       否则 loss 没有梯度、backward() 会直接报错
    with torch.no_grad():
        # 调用 loader：迭代 DataLoader，每次产出 (v1, v2) 各 (B,3,32,32)
        #   作用：拿真实的两视图 batch
        #   关键参数：由 build_loader 决定（shuffle=True, drop_last=True）
        #   坑：v1/v2 在 CPU 上（DataLoader 默认），必须 .to(device) 再喂给 GPU 上的模型
        for i, (v1, v2) in enumerate(loader):
            if i >= REPEATS:
                break

            v1 = v1.to(device)
            v2 = v2.to(device)

            # 调用 torch.cat([v1, v2], dim=0)：签名 `-> Tensor(2B,3,32,32)`
            #   作用：拼成 2B —— ★ 前 B 个是视图 1、后 B 个是视图 2，
            #   这个顺序正是 loss.py 里 labels = [i+B] + [i] 所依赖的前提
            #   关键参数：dim=0（拼样本维）
            #   坑：必须与 CIFAR10Pair 返回的 (v1, v2) 顺序一致；调换 v1/v2 其实也成立
            #       （labels 会跟着变成 [i−B]+[i+B] 的镜像），但别只换一半
            x = torch.cat([v1, v2], dim=0)

            # 调用 model(x)：签名 `SimCLR.forward((2B,3,32,32)) -> z (2B,128)`
            #   作用：编码 + 投影
            #   关键参数：x 在 device 上
            #   坑：model.train() 下 BN 要求每个 channel 有 >1 个空间位置，
            #       2B=512 完全够；但别把 B 调到 1 以下
            z = model(x)

            # 调用 criterion(z)：签名 `NTXentLoss.forward((2B,128)) -> scalar`
            #   作用：算 InfoNCE
            #   关键参数：z 的行序 = [视图1..., 视图2...]
            #   坑：loss 是 0 维张量，.item() 才能拿出来当 float 用
            loss = criterion(z)

            losses.append(loss.item())
            print(f"batch {i + 1}/{REPEATS}:  z.shape = {tuple(z.shape)}   loss = {loss.item():.4f}")

    # 变量 mean_loss：float，5 次的平均
    #   示例值：6.2590
    #   为什么用平均而不是最后一次：单次有 batch 抖动，平均才是可比的量
    mean_loss = sum(losses) / len(losses)

    # 变量 std_loss：float，5 次的标准差（总体，除以 n）
    #   示例值：0.0021
    #   为什么算它：它反映「batch 抖动有多大」。若 std 突然很大（如 0.5），
    #   说明某个 batch 数据异常（比如重复采样），值得回头看
    var = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
    std_loss = var ** 0.5

    # 变量 dev：float，与理论值的偏差
    #   示例值：+0.0226（τ=0.5 时的常数偏移）
    dev = mean_loss - expect

    print("-" * 60)
    print(f"{REPEATS} 次平均 loss   = {mean_loss:.4f}   (std = {std_loss:.4f})")
    print(f"理论 ln(2B-1)   = {expect:.4f}")
    print(f"偏差            = {dev:+.4f}")

    assert not any(math.isnan(x) for x in losses), "出现 nan —— 查 L2 归一化 / τ 是否为 0（见 loss.py 对照表）"
    assert not any(math.isinf(x) for x in losses), "出现 inf —— 查 labels 是否指向了对角线"
    assert abs(dev) < TOLERANCE, (
        f"偏差 {dev:+.4f} 超出 ±{TOLERANCE}。若实测 ≈ 4.x → labels 写成了对角线；"
        f"若 ≈ 7.07 → 是 τ=0.07 的偏移（先确认你的 τ）；详见 loss.py 末尾对照表"
    )
    print("[PASS] B2：随机初始化下 loss ≈ ln(2B-1)，labels / 归一化 / 2B 拼接全部正确")


if __name__ == "__main__":
    main()


# ==== 怎么验证跑对了 ====
#
# 命令：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\sanity_check.py
#
# 期望输出：
#   设备 = cuda   τ = 0.5   B = 256   2B-1 = 511
#   ------------------------------------------------------------
#   batch 1/5:  z.shape = (512, 128)   loss = 6.26xx
#   ...（共 5 行，都在 6.24 ~ 6.28 之间）
#   ------------------------------------------------------------
#   5 次平均 loss   = 6.24 ~ 6.28   (std = 0.00xx)
#   理论 ln(2B-1)   = 6.2364
#   偏差            = +0.01 ~ +0.03
#   [PASS] B2：随机初始化下 loss ≈ ln(2B−1)，labels / 归一化 / 2B 拼接全部正确
#
# 异常对照表（就是计划书 B4 那几条 + 5 章卡点预案）：
#   RuntimeError: Torch not compiled with CUDA enabled
#       → 用错环境了。必须 & "D:\anaconda\envs\pytorch\python.exe"，不能 conda activate
#   CUDA out of memory
#       → 把 BATCH_SIZE 从 256 降到 128。★ 注意降了之后期望值要改成 ln(255)=5.5413
#   loss ≈ 4.x 且几乎不变（不下降）
#       → 先查 labels：正样本索引可能写成了 i（对角线）
#   loss = inf
#       → 对角线被 mask 成 −inf 的同时 labels 指到了对角线
#   loss = nan
#       → L2 归一化忘了，或 τ = 0
#   5 次 loss 里有某一次明显不同（如 4.96）
#       → 那个 batch 不是完整 256 —— 检查 drop_last 是否被改成 False
#   偏差 +0.8 左右
#       → 大概率是 τ 被改成了 0.07。ln 期望要跟着换，或把 τ 调回 0.5
