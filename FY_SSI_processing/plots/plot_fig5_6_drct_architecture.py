"""Draw thesis Figure 5-6 in the compact employed-network style.

The diagram is a new vector redraw of the DRCT main path for the project's
single-channel FY-4A-to-FY-3C ×4 task.  Its thumbnails are produced directly
from the raw 32×32/128×128 CSI arrays of sample 6 in the final ten-sample
paper figure; the SR array is obtained by an inference-only checkpoint pass.
No element is copied from a paper.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


FY_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FY_PROCESSING_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fy" / "figures"
NATIVE_SAMPLE_HELPER = Path(__file__).with_name("prepare_fig5_6_native_sample.py")
DRCT_PYTHON = Path(r"D:\ana\envs\drct\python.exe")
NATIVE_SAMPLE_CACHE = PROJECT_ROOT / "tmp" / "fig5_6_native_sample6.npz"
CSI_MAX = 1.5
# The SVG is authored for direct placement in Word at 16 cm wide.  The main
# labels are therefore 小五 (9 pt), rather than relying on later Word scaling.
LABEL_SIZE = 9.0
OPERATION_SIZE = 9.0
GROUP_LABEL_SIZE = 9.0


# Academic, restrained colour system: blue feature stacks, pale grey operation
# boxes, light cyan module regions, and black dashed grouping outlines.
INK = "#111111"
TEXT = "#111111"
BLUE = "#84BDEB"
BLUE_DARK = "#4D8ECB"
CYAN = "#BDE8EC"
LILAC = "#D4D4F4"
GREY = "#F0F2F4"
GREEN = "#DFF2E6"


def configure_style() -> None:
    """Keep SVG text editable and use a paper-style serif face."""
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "Arial"],
            "font.serif": ["Times New Roman", "SimSun", "Arial"],
            "font.size": 9.0,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    width: float = 0.85,
    connection: str = "arc3,rad=0",
    color: str = INK,
    mutation: float = 8.5,
) -> None:
    """Draw a compact, consistent directional connection."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "-",
            color=color,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


def operation_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str = GREY,
    fontsize: float = OPERATION_SIZE,
    rotate: bool = True,
) -> None:
    """Draw one vertical operation box, matching the reference figure style."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.45",
            facecolor=fill,
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        rotation=90 if rotate else 0,
        linespacing=0.9,
        zorder=7,
    )


def dashed_group(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    """Draw a non-filled module group with its label placed below."""
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=INK,
            linewidth=0.72,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y - 1.3,
        label,
        ha="center",
        va="top",
        fontsize=GROUP_LABEL_SIZE,
        color=TEXT,
        linespacing=0.9,
    )


def feature_stack(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    colour: str = BLUE,
    sheets: int = 4,
    offset: float = 1.15,
) -> None:
    """Draw a small stack of isometric feature-map sheets as vector paths."""
    for index in range(sheets - 1, -1, -1):
        dx = index * offset
        dy = index * offset * 0.56
        front = [(x + dx, y + dy), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
        side = [(x + w + dx, y + dy), (x + w + dx + 1.35, y + dy + 1.15), (x + w + dx + 1.35, y + h + dy + 1.15), (x + w + dx, y + h + dy)]
        top = [(x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w + dx + 1.35, y + h + dy + 1.15), (x + dx + 1.35, y + h + dy + 1.15)]
        alpha = 0.52 + 0.10 * (sheets - 1 - index)
        ax.add_patch(Polygon(front, closed=True, facecolor=colour, edgecolor=INK, linewidth=0.38, alpha=alpha, zorder=3 + index * 0.01))
        ax.add_patch(Polygon(side, closed=True, facecolor=BLUE_DARK, edgecolor=INK, linewidth=0.32, alpha=alpha, zorder=3 + index * 0.01))
        ax.add_patch(Polygon(top, closed=True, facecolor="#DCEEFF", edgecolor=INK, linewidth=0.32, alpha=alpha, zorder=3 + index * 0.01))


def plus_node(ax: plt.Axes, x: float, y: float) -> None:
    """Draw an addition node for global residual fusion."""
    ax.add_patch(Circle((x, y), radius=1.45, facecolor="white", edgecolor=INK, linewidth=0.75, zorder=6))
    ax.text(x, y, "+", ha="center", va="center", fontsize=9.0, color=INK, zorder=7)


def slanted_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str = LILAC,
    fontsize: float = OPERATION_SIZE,
) -> None:
    """Draw a restrained trapezoidal block in the reference-paper style."""
    bevel = min(1.1, w * 0.26)
    ax.add_patch(
        Polygon(
            [(x, y), (x + w, y + bevel), (x + w, y + h - bevel), (x, y + h)],
            closed=True,
            facecolor=fill,
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        rotation=90,
        linespacing=0.9,
        zorder=7,
    )


def load_real_csi_thumbnails() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load native 32×32 LR/128×128 HR CSI and infer native 128×128 SR."""
    if not NATIVE_SAMPLE_HELPER.exists():
        raise FileNotFoundError(f"Missing inference helper: {NATIVE_SAMPLE_HELPER}")
    python = DRCT_PYTHON if DRCT_PYTHON.exists() else Path("python")
    NATIVE_SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(python), str(NATIVE_SAMPLE_HELPER), "--output", str(NATIVE_SAMPLE_CACHE)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    try:
        with np.load(NATIVE_SAMPLE_CACHE, allow_pickle=False) as archive:
            lr = np.asarray(archive["lr_csi"], dtype=np.float32)
            sr = np.asarray(archive["sr_csi"], dtype=np.float32)
            hr = np.asarray(archive["hr_csi"], dtype=np.float32)
    finally:
        NATIVE_SAMPLE_CACHE.unlink(missing_ok=True)
    if lr.shape != (32, 32) or sr.shape != (128, 128) or hr.shape != (128, 128):
        raise ValueError(f"Expected native 32×32/128×128/128×128 arrays, got {lr.shape}/{sr.shape}/{hr.shape}")
    return lr, sr, hr


def sample_tile(ax: plt.Axes, image: np.ndarray, x: float, y: float, w: float, h: float, label: str) -> None:
    """Embed one actual CSI field as an image thumbnail inside the SVG."""
    ax.imshow(
        image,
        extent=(x, x + w, y, y + h),
        cmap="viridis",
        vmin=0.0,
        vmax=CSI_MAX,
        interpolation="nearest",
        zorder=1,
        aspect="auto",
    )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK, linewidth=0.9, zorder=5))
    if label:
        ax.text(x + w / 2, y - 2.2, label, ha="center", va="top", fontsize=LABEL_SIZE, color=TEXT, linespacing=1.02)


def draw_figure() -> tuple[Path, Path]:
    """Generate compact SVG and 300 dpi PNG architecture files."""
    configure_style()
    lr, sr, hr = load_real_csi_thumbnails()

    # Preserve the physical aspect of the canvas: all native CSI inputs and
    # outputs are square fields and must never be stretched for layout fill.
    # 6.30 in = 16.0 cm: the intended Word placement width.  Matching the
    # canvas and data aspect leaves no need to resize the figure in Word.
    fig, ax = plt.subplots(figsize=(6.30, 2.67))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 123.0)
    # The artwork occupies y=17–63; crop the formerly empty lower canvas.
    ax.set_ylim(15.0, 67.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Main path: every forward arrow has the same 3-unit edge-to-edge gap.
    centre_y = 33.0
    gap = 3.0

    # Native fields stay square and use the sixth paper-figure sample.
    sample_tile(ax, lr, 3.0, 27.0, 12.0, 12.0, "FY-4A CSI\n32×32×1")
    # The SR tile is placed after the last Conv once its exact grid location
    # has been computed below.

    # Shallow extraction: Conv followed by the first feature-map stack.
    dashed_group(ax, 16.5, 20.0, 18.0, 26.0, "Shallow feature\nextraction")
    operation_box(ax, 18.0, 23.0, 4.0, 20.0, "3×3 Conv", fill=CYAN, fontsize=OPERATION_SIZE)
    feature_stack(ax, 25.0, 28.2, 6.0, 8.5, colour=BLUE, sheets=4, offset=0.70)
    shallow_stack_right = 25.0 + 6.0 + 3 * 0.70 + 1.35
    arrow(ax, (15.0, centre_y), (18.0, centre_y))
    arrow(ax, (22.0, centre_y), (25.0, centre_y))

    # Compact paper-level DRCT body: 12 × RDG is not over-expanded, while the
    # trapezoid makes its Transformer-derived role visually distinct.
    deep_start = shallow_stack_right + gap
    dashed_group(ax, deep_start - 1.45, 20.0, 30.45, 26.0, "Deep feature extraction")
    slanted_block(ax, deep_start, 23.0, 4.2, 20.0, "12 × RDG", fill=LILAC)
    deep_conv_x = deep_start + 4.2 + gap
    operation_box(ax, deep_conv_x, 23.0, 4.0, 20.0, "3×3 Conv", fill=GREY, fontsize=OPERATION_SIZE)
    deep_stack_x = deep_conv_x + 4.0 + gap
    feature_stack(ax, deep_stack_x, 28.2, 5.4, 8.5, colour=BLUE, sheets=3, offset=0.70)
    deep_stack_right = deep_stack_x + 5.4 + 2 * 0.70 + 1.35
    plus_x = deep_stack_right + gap + 1.45
    arrow(ax, (shallow_stack_right, centre_y), (deep_start, centre_y))
    arrow(ax, (deep_start + 4.2, centre_y), (deep_conv_x, centre_y))
    arrow(ax, (deep_conv_x + 4.0, centre_y), (deep_stack_x, centre_y))
    arrow(ax, (deep_stack_right, centre_y), (plus_x - 1.45, centre_y))
    plus_node(ax, plus_x, centre_y)

    # Exact DRCT global residual: shallow feature F0 is fused after the body.
    residual_y = 52.0
    ax.plot(
        [shallow_stack_right, shallow_stack_right, plus_x, plus_x],
        [39.8, residual_y, residual_y, centre_y + 1.45],
        color=INK,
        linewidth=0.78,
        zorder=3,
    )
    arrow(ax, (plus_x, centre_y + 1.45), (plus_x, centre_y + 1.40), width=0.78)
    ax.text(
        (shallow_stack_right + plus_x) / 2,
        residual_y + 1.3,
        "Global residual connection (F₀)",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=TEXT,
    )

    # Actual reconstruction: Conv → LeakyReLU → [Conv + PixelShuffle×2] ×2
    # → Conv.  Each outer stage is compacted into one vertical box only for
    # presentation; it still explicitly shows the two successive ×2 stages.
    rec_first_x = plus_x + 1.45 + gap
    dashed_group(ax, rec_first_x - 0.7, 20.0, 35.5, 26.0, "Image reconstruction")
    operation_box(ax, rec_first_x, 23.0, 3.3, 20.0, "3×3 Conv", fill=GREEN, fontsize=OPERATION_SIZE)
    relu_x = rec_first_x + 3.3 + gap
    operation_box(ax, relu_x, 23.0, 3.3, 20.0, "LeakyReLU", fill=GREEN, fontsize=OPERATION_SIZE)
    upsample_stage_1_x = relu_x + 3.3 + gap
    operation_box(
        ax, upsample_stage_1_x, 23.0, 6.3, 20.0,
        "3×3 Conv\nPixelShuffle ×2", fill=GREEN, fontsize=9.0,
    )
    upsample_stage_2_x = upsample_stage_1_x + 6.3 + gap
    operation_box(
        ax, upsample_stage_2_x, 23.0, 6.3, 20.0,
        "3×3 Conv\nPixelShuffle ×2", fill=GREEN, fontsize=9.0,
    )
    final_conv_x = upsample_stage_2_x + 6.3 + gap
    operation_box(ax, final_conv_x, 23.0, 3.15, 20.0, "3×3 Conv", fill=GREEN, fontsize=9.0)
    arrow(ax, (plus_x + 1.45, centre_y), (rec_first_x, centre_y))
    arrow(ax, (rec_first_x + 3.3, centre_y), (relu_x, centre_y))
    arrow(ax, (relu_x + 3.3, centre_y), (upsample_stage_1_x, centre_y))
    arrow(ax, (upsample_stage_1_x + 6.3, centre_y), (upsample_stage_2_x, centre_y))
    arrow(ax, (upsample_stage_2_x + 6.3, centre_y), (final_conv_x, centre_y))
    sr_tile_x = final_conv_x + 3.15 + gap
    arrow(ax, (final_conv_x + 3.15, centre_y), (sr_tile_x, centre_y))
    sample_tile(ax, sr, sr_tile_x, 27.0, 12.0, 12.0, "SR CSI\n128×128×1")

    # FY-3C is not an inference input: it enters only through supervision.
    sample_tile(ax, hr, sr_tile_x + 2.25, 51.5, 7.5, 7.5, "")
    ax.text(
        sr_tile_x + 6.0,
        60.0,
        "FY-3C CSI\n128×128×1",
        ha="center",
        va="bottom",
        fontsize=LABEL_SIZE,
        color=TEXT,
        linespacing=1.02,
    )
    arrow(ax, (sr_tile_x + 3.25, 51.5), (sr_tile_x + 6.0, 39.0), dashed=True, width=0.78, connection="arc3,rad=-0.06")
    ax.text(sr_tile_x - 4.0, 48.7, "HR reference", ha="center", va="center", fontsize=9.0, color=TEXT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / "fig5_6_drct_architecture.svg"
    png_path = OUTPUT_DIR / "fig5_6_drct_architecture.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = draw_figure()
    print("========== FIGURE 5-6 COMPLETE ==========")
    print("Style: compact main network with real FY-4A LR, DRCT SR, and FY-3C HR thumbnails")
    print("Architecture: 3×3 Conv → 12 × RDG → Conv + global residual → reconstruction ×4")
    print(f"SVG: {svg}")
    print(f"PNG (300 dpi): {png}")
