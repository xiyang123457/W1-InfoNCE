"""A4 · SimCLR 双视图增强流水线（CIFAR-10，32×32）

【用途】
    给一张 32×32 RGB 图，返回两个「互相不同」的随机视图 (v1, v2)。
    这两个视图构成对比学习里唯一的正样本对 —— 模型要学的就是「同一张图的两个
    外观之间应该互相像」。增强不够狠，模型记住原图就够了，学不到不变性。

【输入】
    PIL.Image，mode="RGB"，size=(32, 32)
【输出】
    build_augmentation() → torchvision.transforms.Compose
    对它调用两次（tf(img), tf(img)）得到两个 torch.Tensor，各为 (3, 32, 32)，值域 [0, 1]

【跑完应看到的自检数字】直接 `python src/augment.py`：
    v1.shape              = torch.Size([3, 32, 32])      ← 形状被 RandomResizedCrop 固定回 32×32
    v2.shape              = torch.Size([3, 32, 32])
    torch.equal(v1, v2)   = False                        ← ★ 本文件唯一的硬性验收
    (v1 - v2).abs().mean()≈ 0.05 ~ 0.40                  ← 实测参考区间（见文末「怎么验证跑对了」）

⚠️ 两个必踩的坑（写代码前先读，成品见下方 build_augmentation）
    坑 1：必须「调用两次」，不是「算一遍再复制」
         Compose 对象每次 __call__ 才重新随机采样。正确做法是同一个 transform
         对象被调用两次；若先算好 v1 再 v1.clone()，两视图完全相同，正样本对退化
         成「自己和自己」，loss 直接失去意义。
    坑 2：GaussianBlur 必须放在最后
         若放在 ColorJitter 之前，模糊会削弱后面的颜色扰动强度，等于白做。
"""

# --- 标准库 ---
# 无（本模块只做变换组合，不需要文件/系统能力）

# --- 第三方 ---
# torchvision.transforms：本模块唯一依赖，提供 RandomResizedCrop / ColorJitter 等
#   增强算子。之所以不用 torchvision.transforms.v2，是因为 v1 接口在 PIL 输入下
#   最稳定，且与后续 Dataset 的 PIL 输出直接对接。
from torchvision import transforms


# ==== 1. 增强算子参数（集中成常量，方便 W2 做敏感性实验） ====

# 常量 CROP_SCALE_MIN：float，RandomResizedCrop 的面积下界
#   示例值：0.2 —— 允许裁到只剩原图 20% 面积
#   为什么是 0.2：这是 SimCLR 对 32×32 小图调过的值。裁得越狠，模型越无法靠
#   「记住原图局部」作弊，只能去学语义不变性。这是整套增强里最关键的一项。
CROP_SCALE_MIN = 0.2

# 常量 JITTER_STRENGTH：float，ColorJitter 的亮度/对比度/饱和度扰动幅度
#   示例值：0.4 —— 每个通道独立在 ±40% 范围内抖动
#   为什么是 0.4：SimCLR 消融实验里颜色扰动是「单点贡献最大」的增强项，给足强度
JITTER_STRENGTH = 0.4

# 常量 JITTER_HUE：float，色相扰动幅度（上限 0.5，即整个色环）
#   示例值：0.1 —— 只在色环上偏 ±10%
#   为什么是 0.1：色相抖动过大会让「红车/蓝车」这类类别语义也被抹掉，反而伤表征
JITTER_HUE = 0.1

# 常量 GRAYSCALE_P：float，随机转灰度的概率
#   示例值：0.2 —— 每张图有 20% 概率被完全去色
#   为什么是 0.2：颜色信息被去掉后，模型仍要求两视图相似 → 强制学到「形状/纹理」
#   这类与颜色无关的线索
GRAYSCALE_P = 0.2

# 常量 BLUR_SIGMA：tuple[float, float]，高斯模糊的标准差采样区间
#   示例值：(0.1, 2.0) —— 每次从 0.1 到 2.0 之间均匀取一个 sigma
#   为什么是这个区间：对应「几乎不糊」到「明显糊」的跨度，且下限 0.1 保证 blur 生效
BLUR_SIGMA = (0.1, 2.0)

# 常量 IMG_SIZE：int，网络输入边长（像素）
#   示例值：32 —— 与 CIFAR-10 原图同尺寸
#   为什么是 32：RandomResizedCrop 每次裁出的子图尺寸不同，必须缩放回统一尺寸才能
#   组成 batch 喂进网络
IMG_SIZE = 32


# ==== 2. 增强流水线构造 ====


def build_augmentation() -> transforms.Compose:
    """构造 SimCLR 用的单条增强流水线（调用两次即得一对视图）。

    返回
        transforms.Compose：一个可调用对象，输入 PIL.Image，输出 (3, 32, 32) 的 Tensor
    关键
        返回值必须被【调用两次】，而不是算一次后复制 —— 见文末「怎么验证跑对了」
    """
    # 设计决策：把 5 个算子固定成「裁剪 → 翻转 → 颜色 → 灰度 → 模糊」这一顺序。
    #   裁剪放最前：后续所有扰动都在「裁完的那块区域」上做，等价于更强的尺度扰动。
    #   模糊放最后：先做颜色再做模糊，否则模糊会把颜色扰动的效果平均掉（坑 2）。
    #   不加入旋转/仿射：32×32 上旋转会引入黑边，且 SimCLR 原始组合里没有它。
    return transforms.Compose(
        [
            # 算子 1：RandomResizedCrop —— 主力增强，随机裁一块再缩放回 32×32
            #   作用：让同一张图的「近景」和「远景」互为正样本，逼模型学尺度不变性
            #   关键参数：scale=(0.2, 1.0) 表示裁剪面积占原图的 20%~100%
            #   坑：cut 出来的子图长宽比也会被随机（默认 ratio=(3/4, 4/3)），
            #       裁到极限时可能只剩物体的一部分，这是预期的，不是 bug
            transforms.RandomResizedCrop(IMG_SIZE, scale=(CROP_SCALE_MIN, 1.0)),
            # 算子 2：RandomHorizontalFlip —— 水平翻转
            #   作用：最便宜的几何增强，几乎不会破坏语义（猫翻过来还是猫）
            #   关键参数：p=0.5 —— 一半概率翻转，保证两视图有约一半机会不一致
            transforms.RandomHorizontalFlip(p=0.5),
            # 算子 3：ColorJitter —— 颜色扰动，单点贡献最大的增强项
            #   作用：破坏「颜色」这条捷径，逼模型依赖形状/纹理
            #   关键参数：brightness/contrast/saturation=0.4，hue=0.1（上限 0.5）
            #   坑：hue 一旦调大（如 0.5），10 类里的「红车 vs 蓝车」语义会被抹掉，
            #       线性探针精度会掉 —— W2 做敏感性实验时重点看这一项
            transforms.ColorJitter(
                brightness=JITTER_STRENGTH,
                contrast=JITTER_STRENGTH,
                saturation=JITTER_STRENGTH,
                hue=JITTER_HUE,
            ),
            # 算子 4：RandomGrayscale —— 以一定概率彻底去色
            #   作用：比 ColorJitter 更极端，直接删掉颜色通道信息
            #   关键参数：p=0.2
            #   坑：若把 p 提到 1.0，等价于训练一个「永远灰度」的模型，
            #       颜色信息在表征里彻底丢失，W2 的线性探针会明显掉点
            transforms.RandomGrayscale(p=GRAYSCALE_P),
            # 算子 5：GaussianBlur —— 轻微模糊
            #   作用：抹掉高频细节（噪点、JPEG 块效应），防止模型背纹理
            #   关键参数：kernel_size=3（必须奇数），sigma 从 (0.1, 2.0) 均匀采样
            #   坑：必须放在 ColorJitter 之后（坑 2），否则会削弱颜色扰动强度
            transforms.GaussianBlur(kernel_size=3, sigma=BLUR_SIGMA),
            # 算子 6：ToTensor —— 最后一步格式转换（不是增强，是必需）
            #   作用：PIL.Image(H,W,C) uint8 [0,255] → Tensor(C,H,W) float32 [0,1]
            #   为什么放最后：前面 5 个算子都在 PIL 域上做（v1 接口颜色变换只认 PIL）
            transforms.ToTensor(),
        ]
    )


# ==== 3. 自检（A4 验收：两视图必须不同） ====

if __name__ == "__main__":
    # 变量 torch：模块，PyTorch 主库
    #   为什么在 __main__ 里才 import：本模块被当作库 import 时不需要 torch，
    #   只在自检时用（torch.equal / 张量运算）
    import torch

    # 变量 np：模块，NumPy —— 用于造一张可复现的假图，让 A4 验收不依赖 CIFAR 是否下载完
    import numpy as np

    # 变量 Image：PIL.Image.Image 类，用于把 numpy 数组包成 PIL 图（增强流的输入格式）
    from PIL import Image

    # 变量 rng：numpy.random.Generator，固定 seed=0 的随机源
    #   示例值：<Generator(PCG64)>
    #   为什么固定 seed：验收要可复现 —— 每次都该得到「不全等」这个同样的结论
    rng = np.random.default_rng(0)

    # 变量 arr：np.ndarray，shape (32, 32, 3)，dtype uint8
    #   示例值：彩色噪声图，值域 [0, 255]
    #   为什么用噪声图：噪声图高频成分最丰富，任何一种增强都能立刻改变它 ——
    #   如果连噪声图的两视图都相等，说明增强根本没生效，问题一定是「调用次数」
    arr = rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # 变量 img：PIL.Image.Image，mode="RGB"，size=(32, 32)
    #   为什么显式写 mode="RGB"：Dataset 读 CIFAR-10 出来的就是 RGB，
    #   保持格式一致才能保证这里测的就是真实数据路径
    img = Image.fromarray(arr, mode="RGB")

    # 变量 tf：transforms.Compose，一条增强流水线
    #   为什么只建一条：SimCLR 的双视图来自【同一个】transform 被调用两次，
    #   不是建两条流水线 —— 建两条也对，但没必要，且容易让人误以为要建两个 Encoder
    tf = build_augmentation()

    # 调用 tf(img)：Compose.__call__，签名 `tf(pil_img) -> Tensor(3,32,32)`
    #   作用：采样一套随机参数并施加全部 6 个算子
    #   关键参数：img 必须是 PIL.Image（前面 5 个算子不吃 Tensor）
    #   坑：★ 必须【再调一次】得到 v2，不能写 v2 = v1.clone()
    v1 = tf(img)
    v2 = tf(img)

    # 变量 diff：torch.Tensor，标量（0 维）
    #   示例值：0.18（不同图片/不同随机种子会在 0.05~0.40 间波动）
    #   为什么算这个：torch.equal 只能告诉你「相不相等」，diff 能告诉你「差多少」——
    #   差 0.001 和差 0.3 都算「不相等」，但前者其实说明增强基本没生效
    diff = (v1 - v2).abs().mean()

    # 调用 torch.equal(v1, v2)：签名 `torch.equal(t1, t2) -> bool`
    #   作用：逐元素严格比较两个张量是否完全相同
    #   关键参数：要求 shape 和 dtype 都一致，否则直接返回 False
    #   坑：这是「严格相等」，只要有一个元素不同就 False —— 正是我们要的判据
    same = torch.equal(v1, v2)

    print(f"v1.shape = {tuple(v1.shape)}")
    print(f"v2.shape = {tuple(v2.shape)}")
    print(f"torch.equal(v1, v2) = {same}")
    print(f"(v1 - v2).abs().mean() = {diff:.4f}")

    # 断言而不是打印：让 CI / 你顺手跑一遍时，错了能直接非零退出
    assert v1.shape == (3, IMG_SIZE, IMG_SIZE), f"形状不对：{v1.shape}"
    assert not same, "两个视图完全相同 —— 检查是不是把 Compose 只调了一次然后 clone"
    assert diff > 0.01, f"两视图差异过小（{diff:.4f}），增强基本没生效"
    print("[PASS] A4：两视图形状正确且确实不同")


# ==== 怎么验证跑对了 ====
#
# 命令（Windows PowerShell，注意用绝对 python 路径绕开 conda activate 不生效的问题）：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\augment.py
#
# 期望输出：
#   v1.shape = (3, 32, 32)
#   v2.shape = (3, 32, 32)
#   torch.equal(v1, v2) = False          ← 核心
#   (v1 - v2).abs().mean() = 0.05 ~ 0.40 ← 参考区间
#   [PASS] A4：两视图形状正确且确实不同
#
# 异常对照表：
#   torch.equal(v1, v2) = True
#       → 99% 是写成了 v2 = v1.clone()。回去改成「同一个 tf 调用两次」
#   (v1 - v2).abs().mean() < 0.01 但 equal=False
#       → 只差了一两个像素，通常是 ColorJitter 的 hue=0 或忘了把 RandomResizedCrop
#         的 scale 放到 (0.2, 1.0)；检查常量区有没有被改动
#   RuntimeError: Expected 3D (unbatched) or 4D (batched) input ...
#       → 把 Tensor 喂给了 RandomResizedCrop。前面必须留 PIL 输入，别提前 ToTensor
#   ValueError: sigma should be a sequence ...
#       → BLUR_SIGMA 写成了单个 float。GaussianBlur 的 sigma 传区间要写成 tuple
