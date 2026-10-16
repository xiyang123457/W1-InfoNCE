"""B1 · InfoNCE（NT-Xent）损失

【用途】
    拿一批 (2B,128) 的表示 z，算出「同一张图的两个视图应该互相靠近、与其余 510 个
    视图应该互相远离」的损失。
    ★ 这是整条流水线里【唯一】告诉模型「该怎么学」的地方 —— Encoder 本身不知道
      谁和谁是一对，配对关系全部由这里的 labels 提供。

【输入】
    z：torch.Tensor，float32，shape (2B, 128)
        ★ 行序约定：第 0 ~ B-1 行是视图 1，第 B ~ 2B-1 行是视图 2。
          这个顺序由 data.py → torch.cat([v1, v2], dim=0) 决定，不能乱。
【输出】
    loss：0 维标量 Tensor（可直接 .backward()）

【跑完应看到的自检数字】`python src/loss.py`（随机 z，B=8 → 2B=16）：
    S.shape = (16, 16)
    loss ≈ 2.72 ± 0.05          ← ln(2B−1) = ln(15) = 2.7081，τ=0.5 时略高一点
    真实数据上（B=256、τ=0.5）应得 ≈ 6.26 —— 见 sanity_check.py（B2 的核心验收）

【为什么随机初始化下 loss 恰好是 ln(2B−1) 附近】
    SimCLR 的 InfoNCE 本质是「在 2B−1 个候选里做 2B−1 分类」。随机初始化时
    所有 logits 近似相等（128 维单位向量两两点积的 std ≈ 0.088，很小），
    softmax 近似均匀 → 每个候选概率 ≈ 1/(2B−1) → loss ≈ −ln(1/(2B−1)) = ln(2B−1)。
    B=256 → ln(511) = 6.2364。
    互信息视角：I(z;z⁺) ≥ log(2B−1) − L。L = ln(2B−1) 恰好对应互信息下界 = 0，
    即「模型什么都没学到」。训练的唯一目标就是把这个 loss 压到 6.236 以下。
"""

# --- 第三方 ---
# torch：张量运算。这里用 arange/eye/masked_fill 构造 labels 与对角线掩码
import torch
# torch.nn：Module 基类。写成 nn.Module 才能进 nn.Sequential、能被统一管理
#   （本损失没有可学参数，继承 Module 主要是为了接口一致、方便放进训练脚本）
import torch.nn as nn
# torch.nn.functional：函数式接口。F.normalize 做 L2 归一化、F.cross_entropy 做 softmax+CE
#   为什么用 F 而不是 nn.CrossEntropyLoss()：
#     nn 版会记住 weight/ignore_index 等状态，这里不需要，一次性的 F 调用更直观
import torch.nn.functional as F


# ==== 1. 常量 ====

# 常量 TEMPERATURE：float，温度系数 τ
#   示例值：0.5
#   为什么是 0.5：SimCLR 在小图/小模型上的常用取值。τ 越小，softmax 越尖锐、
#   对困难负样本惩罚越重（MoCo 用 0.07，训练更激进但更不稳）。
#   ★ τ 会改变 sanity check 的实测值（常数偏移），详见本文件末尾对照表：
#       τ=0.5  → 实测 ≈ 6.26
#       τ=0.07 → 实测 ≈ 7.07（属于正常，不是 labels 错）
TEMPERATURE = 0.5


# ==== 2. NTXentLoss ====


class NTXentLoss(nn.Module):
    """NT-Xent / InfoNCE 损失（SimCLR 版）。

    参数
        temperature：float —— 温度系数 τ，默认 0.5
    返回
        forward(z: (2B,128)) → scalar Tensor
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()

        # 变量 self.temperature：float，温度系数
        #   示例值：0.5
        #   为什么存成属性：前向里要用它除相似度；也方便 W2 扫 τ ∈ {0.07,...,2.0}
        #   坑：★ 绝不要传 0 —— 除零会得到 inf/nan，且是静默的（不报错，只是 loss 是 nan）
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (2B,128) → loss: 0 维标量 Tensor"""
        # 变量 n_views：int，总视图数 = 2B
        #   示例值：512（B=256 时）
        #   为什么先取它：下面所有 shape 都从它派生，写死 B 会在换 batch size 时出错
        n_views = z.shape[0]

        # 变量 n_samples：int，原图数量 = B
        #   示例值：256
        #   为什么是 n_views // 2：前 B 行是视图 1、后 B 行是视图 2（见文件头「行序约定」）
        #   坑：若 z 的行数是奇数（忘了 cat 或只喂了一个视图），这里会静默取整出错 ——
        #       所以下面加了断言
        n_samples = n_views // 2

        assert n_views % 2 == 0, (
            f"z 的样本数必须是偶数（2B），当前 {n_views}。"
            "检查是不是忘了 torch.cat([v1, v2], dim=0)"
        )
        assert z.dim() == 2, f"z 必须是 (2B,128) 的二维张量，当前 {tuple(z.shape)}"

        # --- 第 1 步：L2 归一化 ---
        # 调用 F.normalize(z, dim=1, eps=1e-12)：签名 `-> Tensor`，逐行做 v/||v||₂
        #   作用：把每个 128 维向量压到单位球面上，使点积 = 余弦相似度 ∈ [−1,1]
        #   关键参数：dim=1 —— 按「每个样本的特征维」归一化（dim=0 是跨样本归一，完全错）
        #   坑：★ 忘了这一步的后果不是报错，而是「相似度尺度随初始化漂移」，τ 失去意义、
        #       loss 数值也对不上 —— 计划书 B3 坑 2，也是 nan 的常见来源之一
        z = F.normalize(z, dim=1, eps=1e-12)

        # --- 第 2 步：相似度矩阵 ---
        # 调用 z @ z.t()：矩阵乘，签名 `-> (2B, 2B)`
        #   作用：第 (i,j) 项 = z_i · z_j = 余弦相似度；对角线恒为 1（自己和自己）
        #   关键参数：z.t() 是转置。★ 不要写成 z @ z —— shape 对不上会直接报错
        #   坑：结果是对称矩阵，这也是「一次 CrossEntropy 就够、不需要对称化两次」的原因
        sim = z @ z.t()

        # 调用 sim / self.temperature：标量除法
        #   作用：得到 logits。τ 越小，logits 被放大得越厉害，softmax 越尖锐
        #   坑：τ 是 float 时不会改 dtype；若 τ 被写成 Tensor 且要求梯度，会把 loss 变成非标量
        sim = sim / self.temperature

        # --- 第 3 步：屏蔽对角线 ---
        # 变量 diag_mask：torch.Tensor(bool)，shape (2B,2B)
        #   示例值：True 只在对角线上
        #   为什么屏蔽：
        #     对角线的 logit 恒为 1/τ = 2.0（τ=0.5），是整行里最大的值。
        #     它虽然【永远不会成为正样本】（labels 指向 i+B 而不是 i），但作为候选它会
        #     占掉 softmax 分母里很大一份，把 loss 抬高、把梯度的有效空间挤掉。
        #     标准做法就是置 −inf 让它彻底不参与。
        #   坑：置 −inf 后，如果 labels 误指向对角线，log_softmax 会得到 −inf → loss = inf。
        #       这恰好是一个「一眼可辨」的错误信号（计划书 B4 的排除线索）
        diag_mask = torch.eye(n_views, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # --- 第 4 步：构造 labels（★ 最容易错的一步）---
        # 变量 labels：torch.Tensor(long)，shape (2B,)
        #   示例值（B=4）：[4, 5, 6, 7, 0, 1, 2, 3]
        #   含义：第 i 个视图的正样本是第 i+B 个视图；第 i+B 个视图的正样本是第 i 个。
        #     前半段 arange(B, 2B) = 前 B 个视图（视图 1）→ 去后 B 段找搭档
        #     后半段 arange(0, B)  = 后 B 个视图（视图 2）→ 去前 B 段找搭档
        #   为什么不能是对角线 i：正样本不是「自己」，而是「同一张图的另一个视图」。
        #     写成 i 的话：
        #       - 配合上面的 −inf 屏蔽 → loss = inf（正样本位置被屏蔽了）
        #       - 不屏蔽 → loss ≈ 0（自己和自己相似度最大，问题太容易）
        #   为什么一次 CE 就覆盖双向：S 是对称的，labels 又同时写了两个方向 ——
        #     再手动对称化（loss(S,labels)+loss(S.T,labels))/2 是重复劳动，结果一样
        #   坑：★ 计划书说「99% 的错都在这里」。改完先看 loss 是不是约等于 ln(2B−1)
        labels = torch.cat(
            [
                torch.arange(n_samples, n_views, device=z.device),
                torch.arange(0, n_samples, device=z.device),
            ]
        )

        # --- 第 5 步：交叉熵 ---
        # 调用 F.cross_entropy(sim, labels)：签名 `-> scalar Tensor`
        #   作用：对每一行做 log_softmax 后取 labels 位置的对数概率，再对 2B 行求平均
        #   关键参数：sim 是 (2B,2B) 的 logits，labels 是 (2B,) 的 long 索引
        #   坑：cross_entropy 内部已经含 softmax —— 别在外面再加一次 nn.Softmax，
        #       那会把 logits 当概率再取 log，loss 会变成无意义的小数
        loss = F.cross_entropy(sim, labels)

        return loss


# ==== 3. 自检（随机 z 的形状 + 数值双向验证） ====

if __name__ == "__main__":
    # 变量 B：int，原图数量
    #   示例值：8 → 2B = 16 → 期望 loss ≈ ln(15) = 2.7081
    #   为什么用小 B：这个自检只验证「公式对不对」，与 batch size 无关，小一点跑得快；
    #   真实的 B=256 验收在 sanity_check.py
    B = 8

    # 变量 z：Tensor，(2B,128)
    #   示例值：标准正态随机数（随机初始化的网络输出就近似这个分布）
    #   为什么先 detach：自检不需要梯度，避免多余的计算图
    z = torch.randn(2 * B, 128).detach()

    # 变量 criterion：NTXentLoss，τ=0.5
    criterion = NTXentLoss(temperature=0.5)

    # 调用 criterion(z)：签名 `NTXentLoss.forward(z: (2B,128)) -> Tensor`
    #   作用：算出一个标量损失
    #   关键参数：z 的行序必须是「前 B 个视图 1、后 B 个视图 2」
    #   坑：自检里 z 是独立随机向量 —— 也就是说它其实【没有】正样本关系，
    #       但 loss 依然会 ≈ ln(2B−1)。这正是「随机初始化时模型什么都没学到」的体现
    loss = criterion(z)

    # 变量 expect：float，理论期望值
    #   示例值：2.7081 = ln(15)
    #   为什么可以算出来：见文件头「为什么随机初始化下 loss 恰好是 ln(2B−1) 附近」
    import math
    expect = math.log(2 * B - 1)

    # 变量 sim_check：Tensor，(2B,2B)，单独重算一遍相似度用于打印
    #   为什么重算：只想看形状，不修改 criterion 内部状态
    sim_check = F.normalize(z, dim=1) @ F.normalize(z, dim=1).t()

    print(f"z.shape          = {tuple(z.shape)}")
    print(f"S.shape          = {tuple(sim_check.shape)}")
    print(f"对角线（未屏蔽）= {sim_check.diag()[0].item():.4f}   <- 恒为 1.0")
    print(f"非对角 std       = {sim_check[~torch.eye(2 * B, dtype=torch.bool)].std().item():.4f}   <- 128 维单位向量点积，约 0.088")
    print(f"loss             = {loss.item():.4f}")
    print(f"ln(2B-1)         = {expect:.4f}   (B={B} → 2B-1={2 * B - 1})")
    print(f"偏差             = {loss.item() - expect:+.4f}")

    assert not torch.isnan(loss), "loss 是 nan —— 查是否漏了 L2 归一化，或 temperature 设成了 0"
    assert not torch.isinf(loss), "loss 是 inf —— 检查 labels 是否误指向了对角线（被屏蔽成 -inf）"
    assert abs(loss.item() - expect) < 0.15, (
        f"loss {loss.item():.4f} 偏离 ln(2B-1)={expect:.4f} 太多 —— 先查 labels 构造"
    )
    print("[PASS] B1：NTXentLoss 形状与数值均正确")


# ==== 怎么验证跑对了 ====
#
# 命令：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\loss.py
#
# 期望输出：
#   z.shape          = (16, 128)
#   S.shape          = (16, 16)
#   对角线（未屏蔽）= 1.0000   <- 恒为 1.0
#   非对角 std       = 0.088 前后
#   loss             = 2.72 前后     ← ★ 核心：ln(15) = 2.7081，τ=0.5 时约 +0.01
#   ln(2B-1)         = 2.7081   (B=8 → 2B-1=15)
#   偏差             = +0.01 前后
#   [PASS] B1：NTXentLoss 形状与数值均正确
#
# 异常对照表（对应计划书 B4 的四条排错线索）：
#   loss ≈ 0 或极小，且几乎不变
#       → labels 写成了对角线 i，且【没有】屏蔽对角线。自己和自己相似度最大，问题太容易
#   loss = inf
#       → labels 误指向了对角线，而对角线已被 masked_fill 成 −inf
#   loss = nan
#       → 忘记 F.normalize（数值溢出），或 temperature = 0（除零）
#   实测 ≈ 7.07 而不是 6.26
#       → 正常！那是 τ=0.07（MoCo 取值）的常数偏移，不是 labels 错。
#         τ 越小 logits 被放得越大、softmax 越不均匀，loss 就越高
#   形状报错 "size mismatch" / "dimension out of range"
#       → z 不是 (2B,128)。检查 torch.cat 的 dim，以及是否误用了 stack（会多一维）
