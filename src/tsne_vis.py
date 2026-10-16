"""W2-B · t-SNE 表征可视化（D2）

【用途】
    把「模型学到了什么」画出来。这是本周唯一一张**能给导师看的图** ——
    loss 曲线是给自己看的（数字掉了），t-SNE 图是给别人看的（特征空间真的按语义分开了）。
    两者是「结果」和「原因」的关系：只有 loss 没有 t-SNE，没法排除
    「loss 掉了但特征全是废的」这种情况。

【输入】
    runs/ckpt_tau0.5.pth —— D1 训练好的 checkpoint（★ 必修；没有它本脚本跑不了）

【输出】三张 PNG（★ 每张一个独立 claim，互不拼盘）
    runs/W2_tsne_1_random_init.png   随机初始化时的 t-SNE（10 色，应是一团糊）
    runs/W2_tsne_2_trained.png       训练后的 t-SNE（10 色，应按类别成簇）
    runs/W2_tsne_3_pos_neg_dist.png  正/负样本对余弦距离的两条直方图

    ★ 为什么 t-SNE 拆成两张而不是左右并排：并排会让每张只占一半宽度，2000 个点被压扁、
      簇边界看不清。拆开后各自有完整画布，「对比」交给文件名编号和阅读顺序。

【跑完应看到的自检数字】2000 张测试图（每类 200）、perplexity=30：
    图 1（随机）：10 个颜色完全混成一团
    图 2（训练后）：按类别聚成簇
          ★ 簇不会像监督学习那么干净，这是正常的（SimCLR 没有「类别」这个目标），
            「比随机那张有序很多」即达标
    图 3：positive mean ≈ 0.1x ~ 0.3x，negative mean ≈ 0.7x ~ 1.0x
          正样本直方图明显左移、负样本留在右边 —— 差 > 0.2 才算拉开
    t-SNE 本身耗时：2000×128 约 3 s（比训练快三个数量级）

【三个前置事实（W2 计划书 B0 已实测）】
    ① sklearn 1.7.2 已装，`from sklearn.manifold import TSNE` 直接可用
    ② TSNE 的旧参数名 `n_iter` 在新版已改名 **`max_iter`**，用旧名会报错
    ③ t-SNE 跑在 CPU 上，和训练互不抢显存
"""

# --- 标准库 ---
# argparse：命令行参数（ckpt 路径、样本数、perplexity）
import argparse
# random：★ 三件随机源之一。t-SNE 的 random_state 单独控制，但增强/初始化也要固定
import random
# pathlib.Path：跨平台路径
from pathlib import Path

# --- 第三方 ---
# numpy：特征矩阵运算、标签筛选、直方图分箱
import numpy as np
# torch：加载权重、前向提特征
import torch
# torch.nn.functional：F.normalize —— 图 2 算余弦距离前必须归一化
import torch.nn.functional as F
# torch.utils.data：Subset（按索引截取子集）、DataLoader（批量提特征）
from torch.utils.data import Subset, DataLoader
# torchvision.datasets.CIFAR10：★ 这里用的是【测试集】和【不带增强】的 transform，
#   与训练用的 CIFAR10Pair（训练集 + 强增强）刻意区分 —— 见文件头「为什么不用增强」
from torchvision.datasets import CIFAR10
# torchvision.transforms：可视化只要 ToTensor（保持与训练同一值域 [0,1]）
from torchvision import transforms
# sklearn.manifold.TSNE：降维。★ 参数名用 max_iter（新版）
from sklearn.manifold import TSNE
# matplotlib：画图。★ 先在 pyplot 之前切 Agg 后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 本项目 ---
# CIFAR10Pair：★ 只有图 2 用它 —— 图 2 需要「同一张图的两个增强视图」才有正样本对
from data import CIFAR10Pair, DATA_ROOT, PROJECT_ROOT
# SimCLR：要两份 —— 随机初始化的 + 加载 checkpoint 的
from model import SimCLR


# ==== 1. 常量 ====

# 常量 N_PER_CLASS：int，每类取多少张测试图
#   示例值：200 → 10 类共 2000 张
#   为什么类别均衡：不均衡的话某些类只有几个点，图上「簇」是假的，纯属采样噪声
#   为什么是 2000：t-SNE 复杂度约 O(N log N)，2000 张约 3 秒；CIFAR-10 测试集每类正好 1000 张，够取
N_PER_CLASS = 200

# 常量 PERPLEXITY：float，t-SNE 的困惑度
#   示例值：30
#   为什么是 30：经验区间是 (5, 50)，样本越多取越大，2000 个样本取 30 是标准起点。
#   ★ 图上如果是「一大块」或「一堆碎点」，先调它（20→5 或 30→50），别急着改代码
PERPLEXITY = 30.0

# 常量 TSNE_ITER：int，t-SNE 迭代次数
#   示例值：1000
#   为什么 1000：sklearn 默认也是 1000；再多收益极小
TSNE_ITER = 1000

# 常量 POS_BATCH：int，正负距离图用的 batch 大小
#   示例值：512
#   为什么是 512：正/负样本对各 512 对，直方图才够平滑；太小（如 32）分布会很毛躁
POS_BATCH = 512

# 常量 COLORS：list[str]，10 个类的颜色（顺序与 CIFAR-10 的 classes 一致）
#   为什么不用 matplotlib 的 tab10：tab10 偏柔和，且 blue/cyan、orange/brown 相邻时
#   在密集散点里不好区分。这里改用 ColorBrewer Set1 那套高饱和配色，
#   并把其中两个在白底上偏浅的（亮黄 #ffff33、浅灰 #999999）换掉
COLORS: list[str] = [
    "#e41a1c",  # 红
    "#377eb8",  # 蓝
    "#4daf4a",  # 绿
    "#984ea3",  # 紫
    "#ff7f00",  # 橙
    "#a65628",  # 棕
    "#f781bf",  # 粉
    "#17becf",  # 青
    "#bcbd22",  # 橄榄
    "#666666",  # 深灰
]


# ==== 2. 工具 ====


def set_seed(seed: int) -> None:
    """固定随机源。★ 四个一起设 —— 只设 torch 的话，DataLoader 的 shuffle 与几何增强仍会漂。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_subset(n_per_class: int) -> tuple[Subset, np.ndarray, list[str]]:
    """构建「类别均衡 + 不带增强」的评估子集。

    返回
        (subset, labels, classes)
        subset：torch.utils.data.Subset，取出来是 (Tensor(3,32,32), int)
        labels：np.ndarray，shape (N,)，与 subset 顺序一一对应 —— ★ t-SNE 上色要它
        classes：list[str]，10 个类名
    """
    # 调用 CIFAR10(train=False, transform=ToTensor())：签名
    #   `CIFAR10(root, train, transform, target_transform, download) -> Dataset`
    #   作用：拿 10000 张测试图
    #   关键参数：
    #     train=False —— ★ 用测试集：可视化要的是「模型没见过的图」，用训练集会有记忆效应
    #     transform=ToTensor() —— ★ 只做格式转换，**不加任何增强**（原因见下方注释）
    #   坑：这里的 transform 必须与训练时的值域一致（训练用 ToTensor 得到 [0,1] 且没 Normalize），
    #       所以这里也【不能】加 Normalize，否则提出来的特征和训练时的分布对不上
    base = CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    # 变量 targets：np.ndarray，shape (10000,)，每张图的类别下标
    #   为什么转 np：后面用 np.where 按类筛选，比 list 推导快得多
    targets = np.array(base.targets)

    # 变量 idx：list[int]，选中的样本下标
    #   按类各取前 n_per_class 张 → 类别均衡
    idx: list[int] = []
    for c in range(len(base.classes)):
        # 调用 np.where(targets == c)[0]：签名 `-> ndarray`
        #   作用：取出所有属于第 c 类的下标
        #   关键参数：切片 [:n_per_class] 只取前 N 个
        idx.extend(np.where(targets == c)[0][:n_per_class].tolist())

    # ★ 排序：保证子集顺序稳定（否则每次运行顺序不同，图没法复现、也没法和基线对齐）
    idx = sorted(idx)

    return Subset(base, idx), targets[idx], list(base.classes)


def load_trained_model(ckpt_path: Path, device: torch.device) -> SimCLR:
    """加载 D1 训练好的 SimCLR。

    关键
        必须先建 SimCLR() 再 load_state_dict —— load_state_dict 只灌参数值，
        它不会按名字造网络结构
    """
    # 变量 ckpt：dict，checkpoint 内容
    #   为什么 weights_only=False：torch ≥ 2.6 起 torch.load 默认只允许加载纯张量，
    #   而我们的 ckpt 里还装了 list / dict（losses、config）—— 不加这个参数会直接报错
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = SimCLR()
    # 调用 model.load_state_dict(state_dict)：签名 `-> _IncompatibleKeys`
    #   作用：把训练好的参数灌进结构
    #   关键参数：strict 默认 True —— 键必须完全对齐；对不上会明确报缺哪些键，便于排查
    #   坑：★ 别用 torch.load 直接加载整个 model 对象，那样依赖类的定义路径，换目录就废
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # 打印这次加载的是哪一组（从 ckpt 自带的 config 读，不靠文件名猜）
    cfg = ckpt.get("config", {})
    print(f"[load] {ckpt_path}")
    print(f"       tau={cfg.get('tau')}  epochs={cfg.get('epochs')}  "
          f"init_loss={ckpt.get('init_loss')}  end_loss={ckpt.get('end_loss')}")
    return model


def make_random_model(device: torch.device) -> SimCLR:
    """新建一个随机初始化的 SimCLR（= 训练前的状态），作为对照组。

    关键
        必须与训练时相同的初始化方式（model.py 里是 resnet18(weights=None) 的默认 He 初始化），
        这样「随机 vs 训练后」的对比才只归因于训练本身
    """
    model = SimCLR().to(device)
    return model


@torch.no_grad()
def extract_features(model: SimCLR, loader: DataLoader, device: torch.device) -> np.ndarray:
    """批量提取投影特征 z。

    返回
        np.ndarray，shape (N, 128)
    关键
        用 z（ProjectionHead 输出）而不是 h（Encoder 输出）—— 对齐 W2 计划书 B1 的说法。
        （SimCLR 原文的 t-SNE 也常用 z；换成 h 只需把 model(x) 改成 model.encode(x)）
    """
    # 调用 model.eval()：切评估模式
    #   作用：BatchNorm 用训练时累积的 running_mean / running_var，而不是当前 batch 的统计量
    #   坑：★ 不切 eval 的话，2000 张图的统计量会把 BN 输出「洗」一遍，
    #       而且同一张图在不同 batch 里特征会不同 —— 那样随机初始化和训练后两张图的
    #       差异就不只是「训练」造成的了
    model.eval()

    feats = []
    for x, _ in loader:
        # 调用 model(x)：签名 `SimCLR.forward((N,3,32,32)) -> z (N,128)`
        #   作用：编码 + 投影
        #   坑：装饰器 @torch.no_grad() 已经关掉求导 —— 可视化不需要梯度，省显存也更快
        z = model(x.to(device))
        feats.append(z.detach().cpu().numpy())

    return np.concatenate(feats, axis=0)


def run_tsne(feats: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    """对特征跑 t-SNE，返回 (N,2) 坐标。

    关键
        init="pca" + learning_rate="auto" —— sklearn ≥1.5 的推荐组合：
        PCA 初始化让结果稳定（随机初始化在不同运行间差异很大），auto 让学习率随样本数自适应
    """
    # 调用 TSNE(n_components, perplexity, max_iter, init, learning_rate, random_state)
    #   签名：`TSNE(n_components=2, perplexity=30.0, max_iter=1000, init='pca',
    #              learning_rate='auto', random_state=None)`
    #   作用：把 128 维降成 2 维，保留「邻域关系」
    #   关键参数：
    #     max_iter —— ★ 新版参数名。旧名 n_iter 在 sklearn ≥1.5 已移除，用旧名直接报错
    #     random_state —— 固定后同一份特征每次跑出的图一致，便于对比
    #   坑：t-SNE 是【非线性、随机、perplexity 敏感】的降维 —— 不同参数给出不同形状，
    #       簇间距和簇大小都不可比。所以它只能「说明有没有结构」，不能「证明结构有多好」
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=TSNE_ITER,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(feats)


# ==== 3. t-SNE 散点图（随机 / 训练后，各出一张）====


def plot_single_tsne(
    xy: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    title: str,
    out_path: Path,
) -> None:
    """画【一张】t-SNE 散点图：N 个点、10 类、10 色。

    ★ 为什么「随机初始化」和「训练后」各出一张 PNG，而不是并排：
       并排会让每张子图只有一半宽度，2000 个点的簇被压扁、边界和内部结构都看不清。
       拆开后每张图有完整画布，簇的形状一目了然；
       「对比」交给文件名编号（_1_random_init / _2_trained）和阅读顺序来做，
       不靠挤在同一张图里。
    """
    fig, ax = plt.subplots(figsize=(8, 7.5))

    handles = []
    for c, cname in enumerate(classes):
        mask = labels == c
        # 调用 ax.scatter(x, y, ...)：散点
        #   关键参数：
        #     s=12 —— 点比并排版（7）略大：现在有完整画布，不用担心糊成一片
        #     alpha=0.85 —— 提高不透明度让颜色更实（原 0.75 偏淡）
        #     edgecolors="black", linewidths=0.25 —— ★ 给每个点加极细黑边：
        #       密集区域点互相覆盖时，黑边能让每个点仍可分辨，簇的边界也更清楚
        #     color=COLORS[c] —— 高饱和 10 色，相邻类别一眼能分开
        h = ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=12,
            alpha=0.85,
            color=COLORS[c % len(COLORS)],
            edgecolors="black",
            linewidths=0.25,
            label=cname,
        )
        handles.append(h)

    ax.set_title(title, fontsize=12)
    # 隐藏坐标刻度：t-SNE 的两个轴【没有物理单位】，标数字只会误导
    ax.set_xticks([])
    ax.set_yticks([])

    # 图例放到画布右侧外部：拆开后有横向空间，不再需要挤在散点上方
    #   markerscale=2 把图例里的点放大，更容易和散点对上
    ax.legend(handles, classes, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, frameon=False, markerscale=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==== 4. 图 2：正/负样本对距离分布 ====


def compute_pos_neg_distance(model: SimCLR, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """用训练后的 encoder 算正/负样本对的余弦距离。

    返回
        (pos_dist, neg_dist)：各 shape (POS_BATCH,)
        余弦距离 = 1 − 余弦相似度，范围 [0,2]；越小越像
    """
    # 变量 dataset：CIFAR10Pair —— ★ 图 2 必须用【带增强】的数据集，
    #   因为「正样本对」的定义就是「同一张图的两个增强视图」；用干净图就没有两视图了
    dataset = CIFAR10Pair(train=True, download=False)
    # 调用 DataLoader(dataset, batch_size, shuffle, drop_last, num_workers)
    #   关键参数：num_workers=0 —— 这里只要一个 batch，起子进程的开销比收益大
    loader = DataLoader(dataset, batch_size=POS_BATCH, shuffle=True,
                        drop_last=True, num_workers=0)

    model.eval()
    with torch.no_grad():
        # 调用 next(iter(loader))：取一个 batch
        #   作用：得到 (v1, v2) 各 (512,3,32,32) —— 同一批图的两种随机增强
        v1, v2 = next(iter(loader))

        # 调用 F.normalize(z, dim=1)：逐样本 L2 归一化
        #   作用：归一化后点积 = 余弦相似度 ∈ [−1,1]
        #   坑：★ 不归一化的话，向量的模长会把「距离」污染，正负样本对没法在同一尺度比较
        z1 = F.normalize(model(v1.to(device)), dim=1)
        z2 = F.normalize(model(v2.to(device)), dim=1)

        # 变量 pos_sim：Tensor (512,)，z1 与【对应位置】的 z2 的余弦相似度
        #   逐元素相乘后按行求和 = 一对一的点积
        pos_sim = (z1 * z2).sum(dim=1)

        # 变量 perm：Tensor (512,)，随机打乱的下标
        #   用途：把 z2 打乱后与 z1 配对 → 这些对几乎必然不是同一张图 → 构成负样本对
        perm = torch.randperm(z1.size(0), device=device)
        # 变量 neg_sim：Tensor (512,)，负样本对的余弦相似度
        neg_sim = (z1 * z2[perm]).sum(dim=1)

    # 余弦距离 = 1 − 余弦相似度
    return (1 - pos_sim).cpu().numpy(), (1 - neg_sim).cpu().numpy()


def plot_pos_neg_distance(pos_dist: np.ndarray, neg_dist: np.ndarray, out_path: Path) -> None:
    """把正/负样本对的余弦距离画成两条直方图。

    ★ 为什么两张直方图放同一个 PNG：同样是「必须一起看才有意义」——
       单看正样本分布，你不知道它算不算「近」；只有和负样本分布并排，
       「左移分离」这件事才成立。这一张图比 t-SNE 更「硬」：
       它直接量化了「拉近正样本、推远负样本」到底做到没有。
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # 变量 bins：np.ndarray，直方图分箱边界
    #   余弦距离范围是 [0,2]（相似度 [−1,1]），分 60 箱 → 每箱宽约 0.033
    bins = np.linspace(0.0, 2.0, 61)

    # 调用 ax.hist(data, bins, alpha, density, label)
    #   关键参数：density=True —— 归一化成概率密度。
    #     这里正负各 512 个样本、数量相同，用 count 也行；但用 density 更通用
    #     （将来若改成正样本少、负样本多，图仍然可比）
    ax.hist(pos_dist, bins=bins, alpha=0.65, density=True,
            label=f"positive pairs  (mean = {pos_dist.mean():.3f})")
    ax.hist(neg_dist, bins=bins, alpha=0.65, density=True,
            label=f"negative pairs  (mean = {neg_dist.mean():.3f})")

    # 变量 gap：float，两个分布的均值差
    #   这是本图的「一个 claim」：gap 越大，说明对比学习把正负样本拉得越开
    gap = float(neg_dist.mean() - pos_dist.mean())

    ax.set_xlabel("cosine distance  (1 - cos similarity)")
    ax.set_ylabel("density")
    ax.set_title(f"Positive vs negative pair distance  (gap = {gap:.3f})")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==== 5. 主流程 ====


def main():
    parser = argparse.ArgumentParser(description="W2-B: t-SNE visualization")
    parser.add_argument("--ckpt", default=str(PROJECT_ROOT / "runs" / "ckpt_tau0.5.pth"),
                        help="训练好的 checkpoint 路径（D1 产出）")
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS, help="每类取几张测试图")
    parser.add_argument("--perplexity", type=float, default=PERPLEXITY, help="t-SNE 困惑度")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs"), help="产物目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 {ckpt_path}。t-SNE 必须用「训练好的权重」和「随机初始化的权重」两份对比 —— "
            "随机那份本脚本现场新建，训练那份必须先跑完 D1（train.py）"
        )

    # --- 数据：干净测试图，类别均衡 ---
    subset, labels, classes = build_eval_subset(args.n_per_class)
    print(f"[data] {len(subset)} 张测试图（{len(classes)} 类 × {args.n_per_class}），不带增强")

    loader = DataLoader(subset, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    # --- 两份模型 ---
    model_trained = load_trained_model(ckpt_path, device)
    # 变量 model_rand：SimCLR，随机初始化
    #   变量 set_seed 已经在 main 开头设过，这里再设一次是为了让「随机初始化」这件事本身可复现
    set_seed(args.seed)
    model_rand = make_random_model(device)

    # --- 提特征（★ 两份模型提的是同一批图，这是对比成立的前提）---
    feats_rand = extract_features(model_rand, loader, device)
    feats_trained = extract_features(model_trained, loader, device)
    print(f"[feat] z_rand {feats_rand.shape}   z_trained {feats_trained.shape}")

    # --- t-SNE ---
    print(f"[tsne] perplexity={args.perplexity}  max_iter={TSNE_ITER}  init=pca ...")
    xy_rand = run_tsne(feats_rand, args.perplexity, args.seed)
    xy_trained = run_tsne(feats_trained, args.perplexity, args.seed)

    # --- 图 (a)：随机初始化 ---
    # 变量 p1：Path，随机初始化那张的输出路径
    #   编号 _1_ / _2_ / _3_ 是刻意排的：文件名顺序 = 「该看的顺序」
    p1 = out_dir / "W2_tsne_1_random_init.png"
    plot_single_tsne(
        xy_rand, labels, classes,
        f"(a) random initialization  (untrained)\n"
        f"t-SNE of projected features z — CIFAR-10 test, {len(labels)} images, no augmentation",
        p1,
    )
    print(f"[saved] {p1}")

    # --- 图 (b)：训练后 ---
    p2 = out_dir / "W2_tsne_2_trained.png"
    plot_single_tsne(
        xy_trained, labels, classes,
        f"(b) after InfoNCE training  (15 epochs)\n"
        f"t-SNE of projected features z — CIFAR-10 test, {len(labels)} images, no augmentation",
        p2,
    )
    print(f"[saved] {p2}")

    # --- 图 3：正/负样本对距离分布 ---
    pos_dist, neg_dist = compute_pos_neg_distance(model_trained, device)
    p3 = out_dir / "W2_tsne_3_pos_neg_dist.png"
    plot_pos_neg_distance(pos_dist, neg_dist, p3)
    print(f"[saved] {p3}")

    print("-" * 62)
    print(f"正样本对余弦距离  mean = {pos_dist.mean():.4f}  std = {pos_dist.std():.4f}")
    print(f"负样本对余弦距离  mean = {neg_dist.mean():.4f}  std = {neg_dist.std():.4f}")
    print(f"分离度 gap = {neg_dist.mean() - pos_dist.mean():+.4f}")

    # --- 自检 ---
    # 变量 embed_std：float，训练后特征的「各向异性」指标
    #   退化解（representation collapse）时，所有 z 会挤到同一个方向，各维标准差趋近于 0。
    #   所以它同时是「训练有效」的证据和「没崩」的体检
    embed_std = float(feats_trained.std(axis=0).mean())
    print(f"训练后特征各维标准差均值 = {embed_std:.4f}（退化解时趋近 0）")

    assert neg_dist.mean() - pos_dist.mean() > 0.2, (
        f"正负样本对没拉开（gap = {neg_dist.mean() - pos_dist.mean():.4f}）。"
        "检查：① ckpt 是不是训练好的 ② 是不是误用了干净图当视图（图 2 必须用增强）"
    )
    assert embed_std > 0.05, f"特征各维标准差只有 {embed_std:.4f}，疑似退化解（z 全塌到一个方向）"
    print("[PASS] W2-B：t-SNE 与正负距离图均已生成，正负样本对明显分离")


if __name__ == "__main__":
    main()


# ==== 怎么验证跑对了 ====
#
# 前置：D1 的 train.py 已经跑完，runs/ckpt_tau0.5.pth 存在
#
# 命令：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\tsne_vis.py
#
# 期望输出：
#   [data] 2000 张测试图（10 类 × 200），不带增强
#   [load] ...\runs\ckpt_tau0.5.pth
#          tau=0.5  epochs=15  init_loss=6.17xx  end_loss=4.xxx
#   [feat] z_rand (2000, 128)   z_trained (2000, 128)
#   [tsne] perplexity=30.0  max_iter=1000  init=pca ...
#   [saved] ...\runs\W2_tsne_1_random_init.png
#   [saved] ...\runs\W2_tsne_2_trained.png
#   [saved] ...\runs\W2_tsne_3_pos_neg_dist.png
#   正样本对余弦距离  mean = 0.1x ~ 0.3x
#   负样本对余弦距离  mean = 0.7x ~ 1.0x
#   分离度 gap = +0.5x
#   训练后特征各维标准差均值 = 0.3x
#   [PASS] W2-B：...
#
# 两张 t-SNE 怎么读（按编号顺序看）：
#   图 1（_1_random_init）—— 10 个颜色完全混在一起（特征还没学，纯噪声）
#   图 2（_2_trained）—— 按类别聚成簇。★ 不会像监督学习那样干净，这是正常的：
#       SimCLR 无监督，它只学「同一张图的两个视图要像」，压根不知道「猫」和「狗」的区别。
#       「比随机那张有序很多」即达标
#   ★ 两张图用【同一套 COLORS 配色】，所以看图时颜色含义一一对应（红永远是 airplane）
#
# 异常对照表（对应 W2 计划书 F 节）：
#   FileNotFoundError: runs/ckpt_tau0.5.pth
#       → D1 没跑完，或用 --ckpt 指定别的路径
#   TypeError: TSNE.__init__() got an unexpected keyword argument 'n_iter'
#       → 有人把 max_iter 改回旧名了。sklearn ≥1.5 只认 max_iter
#   t-SNE 图是一大团分不开
#       → ① 先调 perplexity（30→50 或 30→15）② 确认加载的是训练后的权重
#         ③ 确认用了【不带增强】的干净图（用增强图会让簇变糊）
#   图 2 两条分布重叠在一起（gap ≈ 0）
#       → ① ckpt 没训练好（回去看 loss 曲线有没有降到 4.5~5.5）
#         ② 误把干净图当视图用了（图 2 必须用 CIFAR10Pair 的增强视图）
#   内存不足
#       → --n-per-class 200 降到 100（t-SNE 是 O(N²) 内存）
