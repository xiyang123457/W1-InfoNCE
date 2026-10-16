"""A5 · SimCLR 骨架：Encoder（ResNet-18 改 stem）+ ProjectionHead

【用途】
    提供「图像 → 512 维特征 h」的编码器，和「h → 128 维表示 z」的投影头。
    本文件只搭壳、只验形状 —— loss 在 loss.py（B1 才填），训练/验收在 sanity_check.py。

【输入】x：torch.Tensor，float32，形状 (N, 3, 32, 32)，值域 [0, 1]
【输出】
    Encoder(x)        → h，形状 (N, 512)
    ProjectionHead(h) → z，形状 (N, 128)
    SimCLR(x)         → z，形状 (N, 128)（上面两个串起来，训练脚本一行调用）

【跑完应看到的自检数字】`python src/model.py`（B=4 的随机张量，不依赖数据下载）：
    x1.shape        = (4, 3, 32, 32)     ← 单向输入
    x_cat.shape     = (8, 3, 32, 32)     ← 2B，两视图拼在一起
    h.shape         = (8, 512)           ← ★ 特征
    z.shape         = (8, 128)           ← ★ 投影表示
    layer4 输出空间尺寸 = 4×4             ← ★ 证明 stem 改对了（原生 stem 会塌成 1×1）
    Encoder 参数量 = 11,168,832          ← 实测值（改 stem、去 fc 之后）
    ProjectionHead 参数量 = 329,344      ← 实测值（含两层 BN 各 512 个 mean/var 参数）

⚠️ 先澄清一个高频误解
    SimCLR 的「双塔」是【同一个编码器跑两遍】（siamese / 权重共享），
    不是两个不同的网络。看到「双塔」别去建两个 Encoder 实例 ——
    正确做法：建一个 Encoder，把两个视图分别喂进去。
"""

# --- 第三方 ---
# torch：张量基础库。自检里用来造随机输入、读 tensor.shape
import torch
# torch.nn：网络层与 Module 基类。Encoder / ProjectionHead 都要继承 nn.Module，
#   才能自动登记参数（optimizer 才能收走）、才有 train/eval 切换
import torch.nn as nn
# torchvision.models：只借 ResNet-18 的结构定义，权重一律传 None（见 Encoder 说明）
from torchvision import models


# ==== 1. 结构常量 ====

# 常量 FEATURE_DIM：int，Encoder 输出的特征维度 h
#   示例值：512 —— ResNet-18 的 layer4 输出通道数
#   为什么是 512：跟着 backbone 走，ProjectionHead 第一层输入必须与之对齐，改一处就要改两处
FEATURE_DIM = 512

# 常量 PROJ_DIM：int，投影头输出的表示维度 z
#   示例值：128
#   为什么是 128（而不是直接拿 512 维的 h 算损失）：SimCLR 论文实测「在 z 上算对比损失」
#   优于「在 h 上算」。投影头可以牺牲掉一部分对实例判别无用、但对下游分类有用的信息
#   （典型如颜色），把更「干净」的不变性留在 h 里。W2 的线性探针冻结 backbone、拿 h
#   做分类，正好检验这件事
PROJ_DIM = 128

# 常量 PROJ_HIDDEN：int，投影头中间层宽度
#   示例值：512 —— 与 FEATURE_DIM 相同，构成 512 → 512 → 128 的瓶颈
#   为什么先撑住再压缩：SimCLR 原文的 MLP 是「输入维度 → 输入维度 → 输出维度」，
#   一步降维到 128 会丢太多信息，加一层等宽隐层让变换有足够容量
PROJ_HIDDEN = 512


# ==== 2. Encoder（ResNet-18 改 stem、去 fc） ====


class Encoder(nn.Module):
    """ResNet-18 改造版：适配 32×32 输入，输出 512 维特征。

    参数
        无
    返回
        forward(x: (N,3,32,32)) → h: (N,512)
    """

    def __init__(self):
        # 调用 super().__init__()：建立 nn.Module 的参数自动登记机制
        #   作用：之后 self.backbone = ... 里的所有子模块/参数才会被 model.parameters()
        #   收集到 —— 漏掉这一句，optimizer 拿到的参数列表会是空的，训练「跑得通但不学」
        super().__init__()

        # 调用 models.resnet18(weights=None)：签名
        #   `resnet18(weights: Optional[WeightsEnum] = None, progress=True, **kwargs)
        #    -> ResNet`
        #   作用：返回一个 ResNet-18 实例，结构定义齐全、参数随机初始化
        #   关键参数：weights=None —— ★ 不加载 ImageNet 预训练权重。自监督实验要的是
        #       「从零学」，预训练权重会把结论彻底污染（W2 想对比「有/无预训练」再单独开实验）
        #   坑：老写法 `pretrained=False` 在 torchvision ≥0.13 已废弃，会报警告；
        #       0.24 版本请一律用 weights=None
        backbone = models.resnet18(weights=None)

        # --- 改造 1：换 stem（对抗 32×32 小图）---
        # 原生 stem 是 `7×7, stride=2` + `MaxPool(3, stride=2)`，为 224×224 设计。
        # 用在 32×32 上，逐层尺寸（训练模式实测）：
        #     input 32×32 → conv1 16×16 → maxpool 8×8 → layer1 8×8
        #     → layer2 4×4 → layer3 2×2 → layer4 【1×1】
        # ★ 真实的失败模式是「会报错」，不是「悄悄变差」：
        #   - batch_size=1 时直接抛 ValueError —— BatchNorm2d 在 train 模式下要求每个
        #     channel 有 >1 个空间位置，而 layer4 已塌成 1×1：
        #     `ValueError: Expected more than 1 value per channel when training,
        #      got input size torch.Size([1, 512, 1, 1])`
        #   - batch_size>1 时能跑通、最终也是 (B,512)，但整个 32×32 被压成一个点，
        #     空间信息几乎全丢 —— 所以「不报错」不等于「对」
        # 改法是 CIFAR 版 ResNet 的标准做法：3×3 + stride=1 + padding=1，并去掉 maxpool。
        # 改后逐层：32×32 → 32×32 → 32×32 → 32×32 → 16×16 → 8×8 → 【4×4】
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            # 变量 bias：bool，是否带偏置
            #   值：False
            #   为什么是 False：紧跟其后的 bn1 会减掉均值，conv 的 bias 会被完全抵消，
            #   留着只是白白多 64 个无效参数（torchvision 原生 conv1 也是 bias=False）
            bias=False,
        )
        # 调用 nn.Identity()：占位模块，forward 时原样返回输入
        #   作用：把 MaxPool 换成「什么都不做」，保留 32×32 的空间分辨率
        #   关键参数：无
        #   坑：不要直接 del backbone.maxpool —— ResNet.forward 里会显式调用
        #       self.maxpool(x)，删了会 AttributeError
        backbone.maxpool = nn.Identity()

        # --- 改造 2：去掉 fc ---
        # 调用 nn.Identity()：同样做占位
        #   作用：让 backbone 输出 (N, 512) 的特征图池化结果，而不是 (N, 10) 的类别 logits
        #   关键参数：无
        #   坑：必须替换而不是 del —— ResNet.forward 结尾会调用 self.fc(x)。
        #       ResNet-18 的 fc 是 Linear(512,10)，换了之后参数量少 5130
        backbone.fc = nn.Identity()

        # 变量 self.backbone：models.ResNet，改造后的主干
        #   示例值：ResNet(conv1=3×3/s1, maxpool=Identity, layer1~4 原样, fc=Identity)
        #   为什么存成属性而不是局部变量：nn.Module 通过 __setattr__ 自动登记子模块，
        #   写成属性后参数才会进 model.parameters()，也方便自检里挂 forward hook 看中间形状
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,32,32) → h: (N,512)"""
        # 调用 self.backbone(x)：签名 `-> Tensor(N, 512)`
        #   作用：一轮 conv + 4 个 stage + 全局平均池化
        #   关键参数：x 必须是 4 维 (N,C,H,W)；本版只吃 32×32（stem 是为此定制的）
        #   坑：训练模式下 BN 需要 batch 内样本数 > 1（见 __init__ 里的 ValueError 说明），
        #       所以别用 B=1 做形状检查
        return self.backbone(x)


# ==== 3. ProjectionHead（512 → 512 → 128） ====


class ProjectionHead(nn.Module):
    """把 512 维特征 h 投到 128 维表示 z。

    参数
        无
    返回
        forward(h: (N,512)) → z: (N,128)
    """

    def __init__(self):
        super().__init__()

        # 变量 self.net：nn.Sequential，四层堆叠
        #   示例值：Linear(512,512) → BatchNorm1d(512) → ReLU → Linear(512,128)
        #   为什么用 Sequential：结构简单且无分支，按顺序调用最直观
        #
        # ★ 关键点：输入是 (N, 512) —— 二维，不是 (N,C,H,W)。
        #   所以中间必须用 nn.BatchNorm1d，用 BatchNorm2d 会直接报维度错误
        #   （BatchNorm2d 要求 4 维输入）。这是 A5 最容易踩的一个坑。
        self.net = nn.Sequential(
            # 第 1 层：Linear(512, 512)
            #   作用：先在同维度里做一次非线性变换，让投影头有足够容量
            nn.Linear(FEATURE_DIM, PROJ_HIDDEN),
            # 第 2 层：BatchNorm1d(512)
            #   作用：稳定中间层分布，加速收敛；SimCLR 原文的投影头就带这一层
            #   坑：★ 训练时 batch 内样本数必须 > 1（这里实际喂进来的是 2B=512），
            #       若用 B=1 会抛 ValueError
            nn.BatchNorm1d(PROJ_HIDDEN),
            # 第 3 层：ReLU
            #   作用：引入非线性。没有它，两层 Linear 等价于一层 —— 投影头就退化了
            nn.ReLU(inplace=True),
            # 第 4 层：Linear(512, 128)
            #   作用：压到 128 维。★ 注意这里【不加】BN，也不加 ReLU ——
            #       后端输出的 z 要保持线性，后续做 L2 归一化 + 点积才对
            nn.Linear(PROJ_HIDDEN, PROJ_DIM),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N,512) → z: (N,128)"""
        return self.net(h)


# ==== 4. SimCLR 组合（Encoder + ProjectionHead） ====


class SimCLR(nn.Module):
    """把 Encoder 与 ProjectionHead 串起来，一步出 z。

    参数
        无
    返回
        forward(x: (N,3,32,32)) → z: (N,128)
    关键
        这是「双塔」里的【那一座塔】。两个视图共用同一个实例 —— 这正是权重共享的含义
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = ProjectionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,32,32) → z: (N,128)"""
        h = self.encoder(x)
        return self.head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """只走 Encoder，返回 h: (N,512)。W2 的线性探针会用它（冻结 backbone 取特征）。"""
        return self.encoder(x)


# ==== 5. 自检（A5 验收：形状 + stem 是否改对） ====


def _count_params(module: nn.Module) -> int:
    """统计 trainable 参数量。

    返回
        int：所有 requires_grad=True 的参数元素总数
    关键
        用它核对「有没有意外加载预训练权重 / 有没有漏掉某层」—— 数字对不上就是结构被改过
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 变量 B：int，真实训练时的 batch size
    #   示例值：4（这里只为验形状，不用 256 —— 形状对不对与 B 无关，但跑得快）
    #   为什么不用 1：★ B=1 时 BN 在 train 模式下必然报 ValueError；
    #       而且 B=1 也验证不出「两个视图是否真的不同」
    B = 4

    # 变量 x1 / x2：Tensor，(B,3,32,32)
    #   示例值：标准正态随机数（真实场景是经增强后的两视图，这里只关心形状）
    #   为什么可以用随机张量：A5 的验收目标是「forward 不报错、形状对」，
    #   随机张量即可暴露 stem/ BN 维度问题，且不依赖数据集是否下载完
    x1 = torch.randn(B, 3, 32, 32)
    x2 = torch.randn(B, 3, 32, 32)

    # 调用 torch.cat([x1, x2], dim=0)：签名 `torch.cat(tensors, dim=0) -> Tensor`
    #   作用：沿 batch 维拼成 (2B,3,32,32)。★ 前 B 个是视图 1、后 B 个是视图 2 ——
    #       这个「前 B 后 B」的排布决定了 B1 里正样本索引必须是 i+B（不是对角线）
    #   关键参数：dim=0 沿样本维拼，不是 dim=1（那是拼通道，会变成 6 通道报错）
    #   坑：cat 不是 stack —— cat 的每个输入都是 4 维，拼完还是 4 维；
    #       stack([x1,x2]) 会变成 (2,B,3,32,32)，后面所有 shape 都会对不上
    x_cat = torch.cat([x1, x2], dim=0)

    # 变量 model：SimCLR，含 Encoder + ProjectionHead，权重共享的那一座塔
    #   为什么只建一个实例：双视图都喂给它 —— 这就是「权重共享」的实现方式
    model = SimCLR()

    # 变量 mid：dict，记录中间层输出形状（用 forward hook 抓）
    #   为什么抓 layer4：stem 改对了 layer4 应是 4×4；若还是 1×1，说明 conv1/maxpool 没改成功，
    #   但最终 shape 依然会是 (2B,512) —— 光看输出形状【发现不了】这个问题
    mid = {}

    def _hook(_module, _inputs, output):
        mid["layer4"] = tuple(output.shape)

    # 调用 register_forward_hook(fn)：签名 `register_forward_hook(hook) -> RemovableHandle`
    #   作用：在 layer4 前向结束后触发 fn，拿到它的输出张量
    #   关键参数：fn(module, inputs, output)，必须接收这三个位置参数
    #   坑：hook 会一直挂着，长训练里别忘了 remove()，否则重复注册会累积内存
    handle = model.encoder.backbone.layer4.register_forward_hook(_hook)

    # 调用 model(x_cat)：签名 `SimCLR.forward(x: (2B,3,32,32)) -> z: (2B,128)`
    #   作用：一次前向走完 Encoder + ProjectionHead
    #   关键参数：输入必须是 4 维、32×32（本版 stem 是为 32×32 定制的）
    #   坑：这里没有 optimizer / loss —— A5 只验 forward，别急着加训练逻辑
    z = model(x_cat)
    h = model.encode(x_cat)

    handle.remove()

    print(f"x1.shape     = {tuple(x1.shape)}")
    print(f"x_cat.shape  = {tuple(x_cat.shape)}   <- 2B")
    print(f"h.shape      = {tuple(h.shape)}")
    print(f"z.shape      = {tuple(z.shape)}")
    print(f"layer4 输出  = {mid.get('layer4')}   <- 期望 (2B,512,4,4)")
    print(f"Encoder 参数量      = {_count_params(model.encoder):,}")
    print(f"ProjectionHead 参数量 = {_count_params(model.head):,}")

    assert tuple(x_cat.shape) == (2 * B, 3, 32, 32), f"输入拼错了：{x_cat.shape}"
    assert tuple(h.shape) == (2 * B, FEATURE_DIM), f"h 形状不对：{h.shape}"
    assert tuple(z.shape) == (2 * B, PROJ_DIM), f"z 形状不对：{z.shape}"
    assert mid["layer4"][-2:] == (4, 4), (
        f"layer4 空间尺寸是 {mid['layer4'][-2:]}，期望 (4,4) —— "
        "stem 没改对（conv1 要换成 3×3/s1/p1，maxpool 要换成 Identity）"
    )
    print("[PASS] A5：形状契约通过，layer4 保持 4×4 空间分辨率")


# ==== 怎么验证跑对了 ====
#
# 命令：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\model.py
#
# 期望输出：
#   x1.shape     = (4, 3, 32, 32)
#   x_cat.shape  = (8, 3, 32, 32)   <- 2B
#   h.shape      = (8, 512)              ← ★ 计划书 A5.5 的第二个数字
#   z.shape      = (8, 128)              ← ★ 计划书 A5.5 的第三个数字
#   layer4 输出  = (8, 512, 4, 4)   <- 期望 (2B,512,4,4)
#   Encoder 参数量      = 11,168,832
#   ProjectionHead 参数量 = 329,344
#   [PASS] A5：形状契约通过，layer4 保持 4×4 空间分辨率
#
# 异常对照表：
#   ValueError: Expected more than 1 value per channel when training
#       → B 太小（当前 B=4 不会触发）。若把 B 改成 1 必然触发；改成 ≥2 即可
#   h.shape = (8, 512) 但 layer4 输出 = (8,512,1,1)
#       → maxpool 没换成 Identity，或者 conv1 还是 7×7/stride=2。回去检查 __init__ 的两处改造
#   RuntimeError: Given groups=1, weight of size [64,3,3,3], expected input[...]
#       → x 的通道数不是 3，检查是不是把 (N,H,W,C) 当成了 (N,C,H,W)
#   z.shape = (8, 10)
#       → fc 没换成 Identity，输出还是 10 类 logits
#   RuntimeError: running_mean should contain 512 elements not ...
#       → 投影头里误用了 BatchNorm2d。输入是二维 (N,512)，必须用 BatchNorm1d
#   Encoder 参数量 = 11,173,962（比正常多 5,130）
#       → fc 没换掉。多的正好是 512×10+10 = 5,130（ResNet-18 原始的 10 类全连接层）
