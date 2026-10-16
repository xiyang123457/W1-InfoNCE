"""W2 · 从 metrics JSON 画 loss 曲线（独立于训练，秒级）

【用途】
    把「训练」和「画图」彻底解耦：
      - train.py 只负责训练，并把指标每 epoch 落盘成 runs/metrics_tau{τ}.json
      - 本脚本只负责读那个 JSON 画图
    于是改标题、换横轴、加参考线、把两组 τ 叠在一起对比 —— 全都是秒级操作，
    **不需要重跑 26 分钟的训练**。

【输入】
    runs/metrics_tau{τ}.json —— 由 train.py 产出，结构见文件末尾「JSON 结构」一节
【输出】
    runs/W2_curve_tau{τ}_loss.png（或 --out 指定的路径）

【用法】
    # 单组：从 JSON 重画（秒级）
    python src/plot_curve.py --metrics runs/metrics_tau0.5.json

    # 换横轴为 epoch（20 个点，适合给导师/周报看整体趋势）
    python src/plot_curve.py --metrics runs/metrics_tau0.5.json --x-axis epoch

    # 两组 τ 叠在一张图里对比
    python src/plot_curve.py --metrics runs/metrics_tau0.5.json runs/metrics_tau0.07.json

【跑完应看到的自检数字】
    终端：
      [load] runs/metrics_tau0.5.json   tau=0.5  records=195  epochs=20
             init=6.1753  end=4.8xxx
      [saved] runs/W2_curve_tau0.5_loss.png
    图上（单组）：
      一条细的半透明 raw 曲线 + 一条粗的 moving avg
      + 灰色虚线 = 随机初始化起点（≈6.175）
      + 浅灰点线 = 理论 ln(2B−1) = 6.2364
"""

# --- 标准库 ---
# argparse：命令行参数（--metrics / --x-axis / --out / --window）
import argparse
# json：读写指标文件。★ 用 JSON 而不是 .pth 就是为了「人和 Excel 都能直接看」
import json
# math：算理论基线 ln(2B−1)
import math
# pathlib.Path：跨平台路径拼接
from pathlib import Path

# --- 第三方 ---
# matplotlib：画图。★ 必须在 pyplot 之前切 Agg 后端，否则无显示器环境下会卡住
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==== 1. 常量 ====

# 常量 DEFAULT_WINDOW：int，移动平均窗口
#   示例值：50（个记录点）
#   为什么需要平滑：单点 loss 抖动大（batch 内负样本偶然好区分），趋势要靠平滑才看得清。
#   为什么在画图侧而不是训练侧做：平滑是「呈现」决策，原始数据必须原样保留在 JSON 里 ——
#   否则以后想看原始抖动就得重训
DEFAULT_WINDOW = 50


# ==== 2. 读取与工具 ====


def load_metrics(path: Path) -> dict:
    """读一个 metrics JSON，返回 dict。

    关键
        encoding="utf-8" —— Windows 默认是 GBK，JSON 里若有中文注释字段会读崩
    """
    # 调用 Path.open(mode, encoding)：签名 `-> TextIOWrapper`
    #   作用：以文本方式打开
    #   坑：★ 不写 encoding 会用系统默认编码（中文 Windows 上是 gbk），
    #       而这个 JSON 是 train.py 用 utf-8 写的 —— 这就是经典的「自己写的文件自己读不了」
    with open(path, "r", encoding="utf-8") as f:
        # 调用 json.load(fp)：签名 `-> dict | list`
        #   作用：反序列化
        #   坑：JSON 里没有 NaN，但 Python 的 json.dump 默认会把 float('nan') 写成 NaN
        #       （非标准），读回来仍能解析 —— 本项目 loss 正常时不会出现
        return json.load(f)


def moving_average(values: list[float], window: int) -> list[float]:
    """前向滑动平均。

    参数
        values：原始序列
        window：窗口长度（不足时用已有长度）
    返回
        list[float]，与输入等长（前几个点是「不完整窗口」的平均）
    关键
        用「前向」而不是「居中」：居中平滑会让曲线末端出现一段没有数据的尾巴，
        而 loss 曲线最右端的值恰恰是「收敛到什么水平」的结论，不能被平滑吃掉
    """
    window = max(1, min(window, len(values)))
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo : i + 1]
        out.append(sum(seg) / len(seg))
    return out


def _series_of(m: dict, x_axis: str) -> tuple[list, list]:
    """按横轴类型取出 (x, y)，并把「未训练的起点」接到序列最前面。

    返回
        (xs, ys)：xs[0] = 0（步数 0 / epoch 0），ys[0] = init_loss
    关键
        ★ 接起点是必须的：训练中的记录点从第 20 步（≈5.69）才开始，
        只画它们的话图上完全看不到「从 6.175 掉下来」这一段 ——
        而那正是 P1 要证明的事（「能从 ln(2B−1) 降到明显更低」）
    """
    # 变量 init：float，随机初始化（未训练）的 loss
    #   为什么信任它：train.py 是在训练【之前】单独前向一次测出来的，不是训练中的值
    init = m["init_loss"]

    if x_axis == "epoch":
        # epoch 轴：点数少（20 个），适合看整体趋势
        return [0] + list(m["epoch_index"]), [init] + list(m["epoch_mean"])

    # step 轴（默认）：点数多（195 个），能看到「前 20 步陡降」这种细节
    return [0] + list(m["steps"]), [init] + list(m["losses"])


# ==== 3. 画图 ====


def plot(metrics_list: list[dict], out_path: Path, x_axis: str, window: int) -> None:
    """把 1 个或多个 metrics 画成一张 PNG。

    参数
        metrics_list：读出来的 metrics dict 列表。长度 1 = 单组曲线；≥2 = 对比图
        out_path：输出 PNG 路径
        x_axis："step" 或 "epoch"
        window：移动平均窗口
    关键
        单组与对比共用这一条通路 —— 对比只是「多画几条线」，没必要写两份代码
    """
    # 变量 single：bool，是否单组模式
    #   为什么区分：单组时不需要在图例里加 τ 前缀（图里只有一条曲线，加前缀是噪声）
    single = len(metrics_list) == 1

    # 变量 colors：list[str]，不同 τ 用不同颜色
    #   为什么不用 matplotlib 默认轮转色：默认色在黑白打印/灰度下难区分，
    #   这里显式指定 tab10 前几个，对比图更清楚
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for k, m in enumerate(metrics_list):
        # 变量 mu：float，该组的温度系数（从 config 里读，不从文件名猜）
        mu = m["config"]["tau"]
        # 变量 prefix：str，图例前缀
        prefix = "" if single else f"tau={mu} "
        color = colors[k % len(colors)]

        xs, ys = _series_of(m, x_axis)
        smooth = moving_average(ys, window)

        # 原始曲线：细、半透明。作用：保留抖动信息
        #   （抖动本身也是结论 —— 例如 τ=0.07 那组会明显更抖）
        ax.plot(xs, ys, alpha=0.35, linewidth=0.8, color=color, label=f"{prefix}raw")

        # 移动平均：粗线。作用：趋势判断的主线
        ax.plot(xs, smooth, linewidth=2.0, color=color, label=f"{prefix}moving avg")

        # 参考线：该组自己实测的随机初始化起点
        #   为什么每组各画一条：不同 τ 的起点不同（0.5 ≈ 6.17，0.07 ≈ 7.07），
        #   只画一条会误导 —— 对比图的意义就在于「各自的起点 vs 各自的收敛值」
        ax.axhline(m["init_loss"], color=color, linestyle="--", linewidth=0.9, alpha=0.55,
                   label=f"{prefix}init loss = {m['init_loss']:.3f}")

    # 参考线：理论基线 ln(2B−1)
    #   作用：把 W1 查明的那件事摆进图里 —— 真实数据的起点【低于】理论基线，
    #   因为同一张图的两个视图天然比随机两张图更相似（见 PROGRESS.md §4.4）
    b = metrics_list[0]["config"]["batch_size"]
    ax.axhline(math.log(2 * b - 1), color="black", linestyle=":", linewidth=0.9, alpha=0.35,
               label=f"ln(2B-1) = {math.log(2 * b - 1):.4f}")

    # 变量 xlabel：str，横轴标签
    #   为什么两个轴都存在：step 看细节，epoch 看趋势（见 README/PROGRESS 里 step vs epoch 说明）
    xlabel = "training step" if x_axis == "step" else "epoch"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"SimCLR-style InfoNCE on CIFAR-10  (B={b})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 调用 tight_layout()：自动调整边距，防止图例/标签被裁掉
    fig.tight_layout()
    # 调用 savefig(path, dpi=150)：dpi 150 保证在论文/周报里放大不糊
    fig.savefig(out_path, dpi=150)
    # 调用 close(fig)：释放 figure，避免批量重画时堆积
    plt.close(fig)


# ==== 4. 主流程 ====


def main():
    parser = argparse.ArgumentParser(description="W2: plot loss curve from metrics JSON")
    # 变量 --metrics：一个或多个 JSON 路径
    #   用 nargs="+" 让「单组」和「多组对比」走同一个入口，不必学两套命令
    parser.add_argument("--metrics", nargs="+", required=True, help="metrics JSON 路径（可给多个做对比）")
    parser.add_argument("--x-axis", choices=["step", "epoch"], default="step", help="横轴类型")
    parser.add_argument("--out", default=None, help="输出 PNG 路径（默认按 tau 自动命名）")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="移动平均窗口（记录点数）")
    args = parser.parse_args()

    # 变量 metrics_list：list[dict]，读进来的各组指标
    metrics_list = []
    for p in args.metrics:
        path = Path(p)
        m = load_metrics(path)
        metrics_list.append(m)
        print(f"[load] {path}   tau={m['config']['tau']}  "
              f"records={len(m['losses'])}  epochs={len(m['epoch_index'])}")
        print(f"       init={m['init_loss']:.4f}  "
              f"end={(m['end_loss'] if m['end_loss'] is not None else float('nan')):.4f}")

    # 变量 out_path：Path，输出路径
    #   自动命名规则：单组带 tau（两个 τ 是两次独立实验，各自一张 PNG）；
    #   对比图用 compare 后缀，并 Explicit 要求用户知道自己在做对比
    if args.out:
        out_path = Path(args.out)
    else:
        first = Path(args.metrics[0]).parent
        if len(metrics_list) == 1:
            out_path = first / f"W2_curve_tau{metrics_list[0]['config']['tau']}_loss.png"
        else:
            out_path = first / "W2_curve_tau_compare_loss.png"

    # 调用 plot(...)：画图
    #   关键参数：x_axis 决定横轴；window 决定平滑强度
    plot(metrics_list, out_path, args.x_axis, args.window)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()


# ==== JSON 结构（train.py 产出，本脚本消费）====
#
# {
#   "config": {                      # 超参 —— 读它就能复现这次训练
#     "tau": 0.5, "batch_size": 256, "lr": 0.0003,
#     "weight_decay": 0.0001, "epochs": 20, "seed": 0
#   },
#   "log_every": 20,                 # 每多少步记一个点
#   "init_loss": 6.1753,             # 训练【前】测的随机初始化 loss（= 曲线起点）
#   "end_loss": 4.81,                # 训练结束时的收敛值；训练未完成时为 null
#   "steps":    [20, 40, ...],       # 与 losses 一一对应（全局连续步号）
#   "losses":   [5.69, ...],         # step 级记录（每 log_every 步的区间均值）
#   "epoch_index": [1, 2, ...],      # 已完成的 epoch 序号
#   "epoch_mean":  [5.29, ...],      # 与 epoch_index 对应（epoch 轴用）
#   "elapsed_s":   [77.3, ...]       # 每个 epoch 的耗时（判断要不要调 num_workers）
# }
#
# ★ 为什么训练中途也要写：torch.save 只在训练【完全结束】时执行一次，
#   若跑 20 epoch 到第 18 个崩了，什么都留不下。每 epoch 落一次盘，
#   崩了也还有前 18 个 epoch 的曲线和一段可用的权重。


# ==== 怎么验证跑对了 ====
#
# 命令（在项目根目录）：
#   cd "d:\learning project\W1 InfoNCE"
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json
#
# 期望输出：
#   [load] runs\metrics_tau0.5.json   tau=0.5  records=9  epochs=1
#          init=6.1753  end=5.1752
#   [saved] runs\W2_curve_tau0.5_loss.png
#
# 换 epoch 轴：
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json --x-axis epoch
#
# 两组对比（等 τ=0.07 那组跑完）：
#   & "D:\anaconda\envs\pytorch\python.exe" src\plot_curve.py --metrics runs\metrics_tau0.5.json runs\metrics_tau0.07.json
#   期望：一张图里两组曲线 + 两条各自颜色的 init 虚线，能直接看出
#         「τ=0.07 起点更高（≈7.07）、收敛更差、抖动更大」—— 这就是 P4 的结论图
#
# 异常对照表：
#   FileNotFoundError: runs/metrics_tau0.5.json
#       → 训练还没跑过，或 tau 标签不匹配（文件名里的 tau 值来自 --tau 参数）
#   json.decoder.JSONDecodeError
#       → JSON 写坏了（进程在写盘瞬间被强杀）。删掉该文件重跑训练即可
#   UnicodeDecodeError
#       → 有人在没写 encoding 的情况下改过 JSON。本脚本读写都是 utf-8
#   图上只有 9 个点（不是 195）
#       → 你读的是只训了 1 epoch 的那次产出。正式跑 --epochs 20 会覆盖成 195 个点
