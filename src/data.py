"""A3 · CIFAR-10 数据集 + SimCLR 双视图 Dataset

【用途】
    1) 把 CIFAR-10 下载/解压到项目下的 data/（A3 验收）
    2) 提供 CIFAR10Pair：每次取一张图，返回「同一张图的两次独立增强」——
       即 SimCLR 的 (v1, v2) 正样本对

【输入】无（数据由 torchvision 自动下载）
【输出】
    CIFAR10Pair[i] → (v1, v2)，各为 torch.Tensor (3, 32, 32)，float32，值域 [0, 1]

【跑完应看到的自检数字】`python src/data.py`：
    train 样本数 = 50000 / test 样本数 = 10000      ← 与官方规格一致
    类别数 = 10：airplane automobile bird cat deer dog frog horse ship truck
    data/cifar-10-batches-py/ 存在（解压落点，含 5 个 data_batch + 1 个 test_batch）
    v1.shape = (3, 32, 32) / v2.shape = (3, 32, 32) / equal(v1,v2) = False
    label 已保留：targets 长度 = 50000              ← W2 线性探针要用

【关于标签】
    ★ CIFAR-10 的标签在本任务里【不用】。SimCLR 是自监督 —— 训练时只喂图像。
      所以 __getitem__ 只返回 (v1, v2)，不返回 label。
    ★ 但标签【不能扔】：W2 的 linear probe（冻结 backbone、训线性分类器）要靠它
      量表征质量。因此 CIFAR10Pair 把原始 CIFAR10 实例留成 self.base，
      并把标签复制到 self.targets，W2 直接取用，不必重新下数据。
"""

# --- 标准库 ---
# pathlib.Path：跨平台路径拼接。比手写 r"D:\...\data" 更稳 —— W2/W3 若要换机器，
#   这套路径不用改
from pathlib import Path

# --- 第三方 ---
# torch：张量库。本模块只用它的 Tensor 类型和 stack 拼接
import torch
# torch.utils.data.Dataset：Dataset 基类。继承它才能被 DataLoader 识别（需要
#   __len__ 和 __getitem__ 两个方法）
from torch.utils.data import Dataset, DataLoader
# torchvision.datasets.CIFAR10：官方数据集封装，负责下载 + md5 校验 + 解压 + 读 pickle
from torchvision.datasets import CIFAR10

# --- 本项目 ---
# build_augmentation：A4 的增强流水线。单视图（test）用同一套增强保持接口一致，
#   真正「不带随机性」的评估版留到 W2 的线性探针再写
from augment import build_augmentation


# ==== 1. 路径与规格常量 ====

# 常量 PROJECT_ROOT：pathlib.Path，本项目根目录
#   示例值：D:\learning project\W1 InfoNCE
#   为什么这么取：__file__ 是 src\data.py，parents[0]=src，parents[1]=项目根。
#   这样无论从哪个工作目录运行，数据都落在同一个地方
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 常量 DATA_ROOT：pathlib.Path，数据根目录
#   示例值：D:\learning project\W1 InfoNCE\data
#   为什么单独一层：.gitignore 里忽略的是 data/，下载物全部收在这一层，方便整个删掉重下
DATA_ROOT = PROJECT_ROOT / "data"

# 常量 CIFAR_DIR：pathlib.Path，CIFAR-10 解压后的落点
#   示例值：D:\learning project\W1 InfoNCE\data\cifar-10-batches-py
#   为什么这个文件名不能改：torchvision 内部按这个固定名字找已解压的数据
CIFAR_DIR = DATA_ROOT / "cifar-10-batches-py"

# 常量 CIFAR_URL：str，官方下载地址（A3 验收记录用）
#   实测（2026-10-15）：301 重定向到 https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz
#   文件大小 170498071 字节（≈162.6 MB），md5 = c58f30108f718f92721af3b95e74349a
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# 常量 CIFAR_MD5：str，官方压缩包校验和
#   为什么要记下来：下载慢/断时手动下压缩包丢进 data/，torchvision 会先校验 md5
#   （tar.gz 文件名不能改），通过就跳过下载直接解压
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"

# 常量 N_TRAIN / N_TEST：int，官方样本数
#   示例值：50000 / 10000 —— 自检时对不上说明数据损坏，要重下
N_TRAIN = 50000
N_TEST = 10000

# 常量 BATCH_SIZE：int，训练 batch size（B）
#   示例值：256
#   为什么是 256：
#     ① B2 的 sanity check 期望值是 ln(2B−1) = ln(511) = 6.2364，B=256 正好对上；
#     ② InfoNCE 的负样本数就是 2B−1=511 个，B 太小负样本不足，任务太容易；
#     ③ 32×32 小图 + ResNet-18，256 在 8GB 显存上跑得动（W1 只需跑通，不做长训练）。
#   ⚠️ 别改成 1：原生 stem 会因 BatchNorm 报 ValueError（见 A5.3）；也验证不出两视图差异
BATCH_SIZE = 256

# 常量 NUM_WORKERS：int，DataLoader 的子进程数
#   示例值：0
#   为什么是 0：Windows 上多进程 DataLoader 会反复复制 Dataset（spawn，不是 fork），
#   CIFAR 数据 170MB 会被复制 N 份，W1 跑通阶段不值当。W2 做长训练再调到 4 并配合
#   persistent_workers=True
NUM_WORKERS = 0


# ==== 2. 双视图 Dataset ====


class CIFAR10Pair(Dataset):
    """把 CIFAR-10 包成 SimCLR 的「一对视图」数据集。

    参数
        train：bool —— True 取 50000 张训练集（A5/B2 用），False 取 10000 张测试集
        transform：callable —— 增强流水线，默认 build_augmentation()，会被【调用两次】
        download：bool —— 本地没有时是否自动下载

    返回（__getitem__）
        (v1, v2)：tuple[Tensor, Tensor]，各为 (3, 32, 32)，float32，值域 [0, 1]
    """

    def __init__(self, train: bool = True, transform=None, download: bool = False):
        self.train = train
        # 变量 self.transform：callable，增强流水线
        #   示例值：transforms.Compose([...6 个算子...])
        #   为什么允许外部传入：W2 做增强消融实验时要换不同流水线，写死就没法换
        self.transform = transform if transform is not None else build_augmentation()

        # 调用 CIFAR10(root, train, download, transform)：签名
        #   `CIFAR10(root: str, train: bool, transform=None, target_transform=None,
        #            download=False) -> Dataset`
        #   作用：下载 + md5 校验 + 解压 tar.gz 到 root/cifar-10-batches-py/，然后读 pickle
        #   关键参数：transform=None —— ★ 刻意不加变换，因为我们要在 __getitem__ 里
        #       手动调用两次 transform 拿两个视图；torchvision 只会调一次
        #   坑：root 传 str 而不是 Path —— 老版本 torchvision 对 Path 支持不稳，
        #       这里统一 str(PROJECT_ROOT / "data") 最保险
        self.base = CIFAR10(
            root=str(DATA_ROOT),
            train=train,
            transform=None,
            download=download,
        )

        # 变量 self.targets：list[int]，长度 50000（train）
        #   示例值：[6, 9, 9, 4, 1, 1, 2, 7, 8, 3, ...]（0-9 对应 self.classes 的下标）
        #   为什么现在就用不上还留着：W2 的 linear probe 要拿它算分类精度，
        #   现在留一份，W2 就不用重新读盘、也不会不小心把顺序打乱
        self.targets = self.base.targets

        # 变量 self.classes：list[str]，10 个类名
        #   示例值：['airplane', 'automobile', 'bird', 'cat', 'deer',
        #            'dog', 'frog', 'horse', 'ship', 'truck']
        #   为什么保留：写 README / 周报时要列规格
        self.classes = self.base.classes

    def __len__(self) -> int:
        """返回样本数：train=50000，test=10000。"""
        return len(self.base)

    def __getitem__(self, index: int):
        """取第 index 张图，返回两次独立增强得到的 (v1, v2)。"""
        # 调用 self.base[index]：CIFAR10.__getitem__，签名 `-> (PIL.Image, int)`
        #   作用：从内存里已加载的 pickle 数组取一张原图
        #   关键参数：index ∈ [0, 49999]
        #   坑：因为上面 transform=None，出来的是 PIL.Image(32,32,3) 而不是 Tensor
        img, label = self.base[index]

        # 调用 self.transform(img) —— 第一次
        #   作用：采样一套随机参数（裁剪位置、颜色抖动幅度、是否翻转/去色），施加并返回张量
        #   关键参数：img 必须是 PIL 格式
        #   坑：这一次调用【不改变】transform 对象本身，所以下面还能再调一次
        v1 = self.transform(img)

        # 调用 self.transform(img) —— 第二次（同一个对象！）
        #   作用：重新采样一套【不同】的随机参数，得到同一张图的第二个视图
        #   坑：★ 这里绝不能写 v1.clone() 或 transform(img) 只调一次再复制 ——
        #       两视图相同 = 正样本对退化成「自己和自己」，loss 直接失去意义
        v2 = self.transform(img)

        # 只返回视图，不返回 label —— 见文件头「关于标签」。
        # label 变量在此处被刻意丢弃，但已在 self.targets 里留好了副本
        return v1, v2


# ==== 3. DataLoader 构造 ====


def build_loader(train: bool = True, batch_size: int = BATCH_SIZE, download: bool = False) -> DataLoader:
    """构造一个 DataLoader，每个 batch 直接给出 (v1, v2)。

    返回
        DataLoader：迭代一次产出 (v1, v2)，v1/v2 各为 (B, 3, 32, 32)
    关键
        训练时 shuffle=True —— 正样本对靠「同 index 的两次增强」建立，
        但 batch 内样本必须随机，否则同一类的图挨着，负样本区分度下降
    """
    # 变量 dataset：CIFAR10Pair，双视图数据集
    #   为什么放函数内而不是全局：避免 import 本模块时就触发下载/占内存
    dataset = CIFAR10Pair(train=train, download=download)

    # 调用 DataLoader(dataset, batch_size, shuffle, num_workers, drop_last)
    #   签名：`DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
    #                     drop_last=False, pin_memory=False, ...) -> DataLoader`
    #   作用：批量取数据。它内部会对每个 index 调用 dataset.__getitem__，
    #       拿到 (v1, v2) 元组后按位置分别 stack → 输出 (v1_batch, v2_batch)
    #   关键参数：
    #     shuffle=True  —— 训练集打乱（见上方「关键」说明）
    #     drop_last=True —— ★ 掉最后不满一个 batch 的尾巴。因为 B2 的期望值
    #       ln(2B−1) 依赖「每个 batch 都是完整 256」，尾巴 batch 只有几十个样本时
    #       期望值会变，sanity check 数字对不上不好排查
    #     pin_memory=True —— 锁页内存，H2D 拷贝更快（有 GPU 时才有效）
    #     num_workers=0 —— 见 NUM_WORKERS 常量说明（Windows 上避免 spawn 复制数据）
    #   坑：drop_last=False 时最后一个 batch 的 B 只有 50000 % 256 = 72，
    #       若这个 batch 恰好被打印出来，loss 会是 ln(143)≈4.96，别误判成 labels 错
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
    )


# ==== 4. A3 自检：下载 + 规格核对 + 双视图连通性 ====

if __name__ == "__main__":
    print("[A3] 检查 CIFAR-10 是否就位（首次运行会下载 ≈162.6 MB，可能较慢）")

    # 调用 build_loader(download=True)：首次运行触发下载 → 解压 → 落在 CIFAR_DIR
    #   作用：这一步同时完成 A3 的下载与 A4 的数据连通性验证
    #   关键参数：batch_size=8 —— ★ 自检阶段刻意用小的 B（不要用 1，见下方坑）
    #   坑：download=True 只在本目录找不到数据时生效；已存在会直接跳过（不重复下载）
    loader = build_loader(train=True, batch_size=8, download=True)
    dataset = loader.dataset

    print(f"train 样本数 = {len(dataset)}（期望 {N_TRAIN}）")
    print(f"类别数 = {len(dataset.classes)}：{' '.join(dataset.classes)}")
    print(f"解压落点存在 = {CIFAR_DIR.exists()}  ->  {CIFAR_DIR}")

    # 调用 next(iter(loader))：取一个 batch
    #   作用：验证「两次增强 → stack 成 batch」这条链真的通
    #   坑：batch_size=8 时 v1 是 (8,3,32,32)，不是 (B,3,32,32) 的 256 —— 自检只要形状对
    v1, v2 = next(iter(loader))
    print(f"v1.shape = {tuple(v1.shape)}   dtype = {v1.dtype}   值域 = [{v1.min():.2f}, {v1.max():.2f}]")
    print(f"v2.shape = {tuple(v2.shape)}   dtype = {v2.dtype}")

    # 调用 torch.equal(v1, v2)：见 augment.py 的说明 —— 两个 batch 必须逐元素不全等
    print(f"torch.equal(v1, v2) = {torch.equal(v1, v2)}（期望 False）")
    print(f"targets 已保留：len = {len(dataset.targets)}（W2 线性探针要用，期望 {N_TRAIN}）")

    assert len(dataset) == N_TRAIN, "训练样本数不是 50000，数据可能损坏，删掉 data/ 重下"
    assert tuple(v1.shape) == (8, 3, 32, 32), f"batch 形状不对：{v1.shape}"
    assert not torch.equal(v1, v2), "两个视图批次完全相同 —— 检查 transform 是不是只调了一次"
    assert len(dataset.targets) == N_TRAIN, "标签没保留住，W2 的线性探针会缺数据"
    print("[PASS] A3：CIFAR-10 就位，双视图 Dataset 连通")


# ==== 怎么验证跑对了 ====
#
# 命令：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\data.py
#
# 期望输出（首次会先打印下载进度条）：
#   train 样本数 = 50000（期望 50000）
#   类别数 = 10：airplane automobile bird cat deer dog frog horse ship truck
#   解压落点存在 = True  ->  D:\learning project\W1 InfoNCE\data\cifar-10-batches-py
#   v1.shape = (8, 3, 32, 32)   dtype = torch.float32   值域 = [0.00, 1.00]
#   v2.shape = (8, 3, 32, 32)   dtype = torch.float32
#   torch.equal(v1, v2) = False（期望 False）
#   targets 已保留：len = 50000（W2 线性探针要用，期望 50000）
#   [PASS] A3：CIFAR-10 就位，双视图 Dataset 连通
#
# 异常对照表：
#   下载卡住 / 断线
#       → 浏览器或 IDM 直接下 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#         （重定向后的 cave.cs.toronto.edu 地址也可以），文件名必须保持
#         cifar-10-python.tar.gz，丢进 data/ 后重跑本脚本；torchvision 会先校验 md5
#         = c58f30108f718f92721af3b95e74349a，通过就跳过下载直接解压
#   OSError: [Errno 22] / md5 不匹配
#       → 压缩包下坏了。删掉 data/cifar-10-python.tar.gz 重下，别想着绕过校验
#   targets 长度对但 __getitem__ 报 IndexError
#       → 检查是不是自己改过 __len__（必须与 self.base 一致，50000 / 10000）
#   num_workers > 0 时在 Windows 上卡死
#       → 回到 NUM_WORKERS = 0；要开多进程必须把 build_loader 的调用放进
#         if __name__ == "__main__" 保护块（本文件已经这样做了）
