"""从 step_04 生成的样本 patch 中画“三联图”（原始 SSI 版）。

每个样本包含：
  - fy4: LR SSI 32×32（FY-4A 粗网格，4 km/像元）；
  - fy3: HR SSI 128×128（FY-3C 细网格，1 km/格）。

三联图布局（1 行 3 列）：
  1) LR SSI：最近邻放大到 128×128，保留粗像元的块状结构；
  2) HR SSI：128×128 真值；
  3) |HR - LR↑| 绝对误差图。
前两列共用同一色标范围（0~1200 W/m²），保证“低分放大 vs 高分真值”可比。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端：只负责把图保存成 PNG

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ---- 中文字体设置（Windows）----
# matplotlib 默认字体不含中文，标题会显示成方框；
# 这里自动注册微软雅黑/黑体，找不到时退回默认字体。
for font_path in (
    r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",  # 微软雅黑粗体
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
):
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=font_path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False  # 让负号正常显示

# step_04 样本输出的默认根目录。
SAMPLES_ROOT = Path(r"D:\Super-Resolution\outputs\fy\samples")

# SSI 显示范围：与 plot_two_swaths.py 一致，取 0~1200 W/m²。
SSI_VMIN, SSI_VMAX = 0.0, 1200.0


def parse_args() -> argparse.Namespace:
    """解析命令行参数：日期、样本编号、输出路径。"""
    parser = argparse.ArgumentParser(description="切块样本三联图：LR CSI | HR CSI | 误差")
    parser.add_argument("--date", default="20180501", help="日期 YYYYMMDD（step_04 输出目录后缀）。")
    parser.add_argument("--sample-id", type=int, default=0, help="要绘制的样本编号（sample_index.csv 中的 sample_id）。")
    parser.add_argument("--all", action="store_true", help="把当天全部样本拼成一张 N×3 大图。")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="--all 模式下最多画前 N 个样本（默认全部）。")
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径；默认写到样本目录。")
    return parser.parse_args()


def load_sample(samples_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """按 sample_id 从索引 CSV 定位 batch npz，并取出 fy4 / fy3（原始 SSI）。

    返回:
      lr: 32×32 LR SSI（FY-4A）；
      hr: 128×128 HR SSI（FY-3C）；
      row: 该样本的索引行（含时次、中心经纬度、QC 统计）。
    """
    index_path = samples_dir / "sample_index.csv"
    with index_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((r for r in rows if int(r["sample_id"]) == sample_id), None)
    if row is None:
        raise KeyError(f"sample_id={sample_id} 不在 {index_path} 中")

    # batch npz 里可能打包多个样本，用 batch_index 定位具体第几个。
    batch = np.load(samples_dir / "patches" / row["batch_file"])
    idx = int(row["batch_index"])
    return batch["fy4"][idx], batch["fy3"][idx], row


def nearest_upsample_4x(lr: np.ndarray) -> np.ndarray:
    """把 32×32 LR 最近邻放大成 128×128（每格复制成 4×4 块）。

    这样能保留 FY-4A 粗像元的真实块状结构，便于和 HR 直接目视对比。
    """
    return np.kron(lr, np.ones((4, 4), dtype=lr.dtype))


def render_triple(
    axes: np.ndarray,
    lr: np.ndarray,
    hr: np.ndarray,
    row: dict[str, str],
    sample_id: int,
    compact: bool = False,
) -> tuple[object, object, object, float]:
    """在给定的 3 个坐标轴上画一个样本的三联图（LR SSI | HR SSI | 误差）。

    参数:
      axes: 长度 3 的坐标轴数组（单样本为一行，全样本模式为其中一行）。
      lr / hr: FY-4A 32×32 与 FY-3C 128×128 的 SSI。
      row: 样本索引行（时次、中心经纬度等）。
      sample_id: 样本编号，用于标题。
      compact: True 时用简短标题（适合拼大图），False 时用完整标题。

    返回:
      im0 / im1 / im2: 三个 imshow 对象（供外部添加色标）；
      err_max: 误差图最大值。
    """
    # 低分图最近邻放大到与高分同尺寸，保留 4 km 粗像元的块状结构。
    lr_up = nearest_upsample_4x(lr)
    err = np.abs(hr - lr_up)
    err_max = float(np.nanmax(err)) if np.any(np.isfinite(err)) else 0.0

    # 三个面板：LR（放大）、HR、绝对误差。
    im0 = axes[0].imshow(lr_up, cmap="viridis", vmin=SSI_VMIN, vmax=SSI_VMAX)
    im1 = axes[1].imshow(hr, cmap="viridis", vmin=SSI_VMIN, vmax=SSI_VMAX)
    im2 = axes[2].imshow(err, cmap="magma")

    if compact:
        axes[0].set_title(f"#{sample_id} LR SSI 32×32\nslot {row['dominant_slot_label']} UTC")
        axes[1].set_title(f"#{sample_id} HR SSI 128×128")
        axes[2].set_title(f"#{sample_id} |HR−LR↑|\nmax={err_max:.0f}")
    else:
        axes[0].set_title(
            f"LR SSI 32×32 (FY-4A) 最近邻放大\n"
            f"slot={row['dominant_slot_label']} UTC / {row['dominant_slot_label_cst']} CST\n"
            f"center=({float(row['center_lon']):.2f}°E, {float(row['center_lat']):.2f}°N)"
        )
        axes[1].set_title("HR SSI 128×128 (FY-3C)\n1 km/格真值")
        axes[2].set_title(f"|HR − LR↑| 绝对误差\nmax={err_max:.0f}")

    # 关闭刻度，画面更干净；数值信息都在标题里。
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return im0, im1, im2, err_max


def main() -> None:
    args = parse_args()
    samples_dir = SAMPLES_ROOT / f"{args.date}_utc"

    # ---- 单样本模式：1×3 三联图 ----
    if not args.all:
        lr, hr, row = load_sample(samples_dir, args.sample_id)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        im0, im1, im2, err_max = render_triple(axes, lr, hr, row, args.sample_id)
        # 前两列共用 SSI 色标，误差图独立色标。
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"step_04 切块样本 #{args.sample_id}（{args.date}）", fontsize=14)
        fig.tight_layout()
        out_path = args.output or (samples_dir / f"sample_triple_ssi_{args.sample_id:04d}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"已保存: {out_path}")
        # 注意：控制台可能是 GBK 编码，这里用 "W/m2" 避免特殊字符报错。
        print(f"  LR SSI 范围: {np.nanmin(lr):.1f} ~ {np.nanmax(lr):.1f} W/m2, "
              f"HR SSI 范围: {np.nanmin(hr):.1f} ~ {np.nanmax(hr):.1f} W/m2")
        return

    # ---- 全部样本模式：N 行 × 3 列大图 ----
    with (samples_dir / "sample_index.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    n = len(rows)
    if n == 0:
        raise RuntimeError(f"{samples_dir} 没有样本（sample_index.csv 为空）")
    # 样本少时行高一点，样本多时压缩行高，保证整张图不至于过长。
    row_h = 4.5 if n <= 8 else 2.6
    fig, axes = plt.subplots(n, 3, figsize=(15, row_h * n))
    axes = np.atleast_2d(axes)  # n=1 时 axes 退化成 1 维，统一成二维方便索引
    im_ssi = None
    for i, row in enumerate(rows):
        sid = int(row["sample_id"])
        lr, hr, _ = load_sample(samples_dir, sid)
        im0, im1, im2, err_max = render_triple(axes[i], lr, hr, row, sid, compact=True)
        if i == 0:
            im_ssi = im0
    # 所有行的 LR/HR 共用同一个 SSI 色标，误差图各列色标由 imshow 自动定。
    fig.colorbar(im_ssi, ax=axes[:, :2], fraction=0.025, pad=0.02)
    cap_note = f"，前 {n} 个" if args.max_samples is not None else ""
    fig.suptitle(f"step_04 切块样本（{args.date}，共 {n} 个{cap_note}）", fontsize=16)
    fig.tight_layout()
    out_path = args.output or (samples_dir / "sample_triple_ssi_all.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"已保存: {out_path}（{n} 个样本）")


if __name__ == "__main__":
    main()
