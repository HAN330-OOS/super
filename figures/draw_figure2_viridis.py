"""Redraw formal 2018-2019 GHI Figure 2 at its final paper size.

Requires Python, NumPy, Matplotlib, PyMuPDF, and Pillow. Example:
    python draw_figure2_viridis.py --output-dir .
This horizontal layout was checked with Python 3.10, NumPy 1.23.5,
Matplotlib 3.7.0 and PyMuPDF 1.28.2. Use Times New Roman.

The plot reads the original formal run's saved paired GHI scatter values.
It performs no inference, scaling, clipping, fitting or new sampling. All
geometry and typography are set in this code before PDF export; no existing
PDF is cropped, transformed or rescaled. Only the two panel positions
and page dimensions use the horizontal layout: 346 x 184 pt, each plot 134 x 134 pt.
Points use KDE-density percentile viridis colors following the supplied R example;
marker size, alpha, coordinates, labels, limits, fonts, and vector styles are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_SCATTER_SHA256 = "ef0b76c1a9c6f9f432ff2d336ee94ed28660834539d682545c36da3942b85875"
EXPECTED_TEST_CSV_SHA256 = "3f3a72f49919a93228f6673932397316dbd6ed5989769f044253e4a21882d710"
EXPECTED_METRICS = {
    "bicubic": {"r2": "0.670413", "rmse": "154.304341", "mae": "117.707811", "mbe": "24.069970"},
    "drct": {"r2": "0.892074", "rmse": "88.299011", "mae": "57.286309", "mbe": "1.095085"},
}
PAGE_W, PAGE_H = 346.0, 184.0
PLOT_LEFT, PLOT_SIDE = 34.0, 134.0
PLOT_LEFTS = (34.0, 210.0)
PLOT_TOPS = (16.0, 16.0)  # PDF coordinates measured downward from page top.
BAR_HEIGHT = 10.0
FONT_SIZE, LABEL_SIZE = 8.0, 8.2
AXIS_LIMITS, TICKS = (0.0, 1200.0), (0, 400, 800, 1200)
COLORS = ("viridis", "viridis")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_formal_data(run: Path, test_csv: Path, samples_root: Path | None):
    data_path = run / "ghi_scatter_data.npz"
    metric_path, config_path = run / "metrics.json", run / "run_config.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert metrics["is_full_test"] and metrics["samples"] == 35476
    assert metrics["valid_pixels"] == 581238784
    assert config["numerics"]["normalization"] == "none" and config["numerics"]["clipping"] is False
    assert config["numerics"]["csi_multiplier"] == 1.0
    assert json.loads((run / "verification_report.json").read_text(encoding="utf-8"))["passed"]
    for method, expected in EXPECTED_METRICS.items():
        assert {key: f"{metrics['ghi'][method][key]:.6f}" for key in expected} == expected
    assert file_hash(data_path) == EXPECTED_SCATTER_SHA256
    csv_path = test_csv
    assert file_hash(csv_path) == EXPECTED_TEST_CSV_SHA256 == config["data_identity"]["split_csv_sha256"]["test"]
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 35476
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    x, bicubic, drct = (arrays[name] for name in ("fy3_reference", "bicubic_ghi", "drct_ghi"))
    for values in (x, bicubic, drct):
        assert values.shape == (150000,) and values.dtype == np.float32 and np.isfinite(values).all()
        assert np.all((values >= AXIS_LIMITS[0]) & (values <= AXIS_LIMITS[1]))
    indices = arrays["csv_row_index"].astype(np.int64)
    pixels = arrays["pixel_flat_index"].astype(np.int64)
    assert np.array_equal(arrays["global_pixel_index"], indices * 16384 + pixels)
    assert np.array_equal(pixels, arrays["pixel_row"].astype(np.int64) * 128 + arrays["pixel_col"])
    assert np.all((indices >= 0) & (indices < len(records))) and np.all((pixels >= 0) & (pixels < 16384))
    assert np.array_equal(arrays["bicubic_csi"] * arrays["clear_ghi"], bicubic)
    assert np.array_equal(arrays["drct_csi"] * arrays["clear_ghi"], drct)
    assert np.isfinite(arrays["clear_ghi"]).all() and (arrays["clear_ghi"] > 0).all()
    assert all(str(int(value)) == records[int(position)]["sample_id"] for value, position in zip(arrays["sample_id"], indices))
    # Reuse saved points unchanged; only verify their original selection.
    expected_indices = np.sort(np.random.default_rng(metrics["scatter"]["seed"]).choice(581238784, size=150000, replace=False))
    assert np.array_equal(arrays["global_pixel_index"], expected_indices)
    # Read actual FY3 references for deterministic spot checks, without changing any arrays.
    samples_root = samples_root or Path(config["arguments"]["samples_root"])
    for point in np.linspace(0, len(x) - 1, 12, dtype=int):
        record = records[int(indices[point])]
        source = samples_root / record["date"] / record["npz_file"]
        with np.load(source, allow_pickle=False) as original:
            value = original["fy3"][int(record["array_index"])].reshape(-1)[int(pixels[point])]
        assert value == x[point]
    sources = {str(path): file_hash(path) for path in (data_path, metric_path, config_path, csv_path)}
    return x, bicubic, drct, metrics, config, sources


def verify_pdf(path: Path, expected_rectangles: list[list[float]]) -> dict:
    import pymupdf

    doc = pymupdf.open(path)
    assert len(doc) == 1
    page = doc[0]
    assert abs(page.rect.width - PAGE_W) < 1e-6 and abs(page.rect.height - PAGE_H) < 1e-6
    spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    assert spans
    for span in spans:
        assert "TimesNewRoman" in span["font"].replace(" ", "")
        expected = LABEL_SIZE if "Result" in span["text"] or "Observation" in span["text"] else FONT_SIZE
        assert abs(span["size"] - expected) < 1e-4, span
        rect = pymupdf.Rect(span["bbox"])
        assert rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= PAGE_W and rect.y1 <= PAGE_H, span
    overlaps = []
    for index, first in enumerate(spans):
        for second in spans[index + 1:]:
            intersection = pymupdf.Rect(first["bbox"]) & pymupdf.Rect(second["bbox"])
            if intersection.width > 0.1 and intersection.height > 0.1:
                overlaps.append([first["text"], second["text"], list(intersection)])
    assert not overlaps, overlaps
    embedded = []
    for font in page.get_fonts(full=True):
        extracted = doc.extract_font(font[0])
        assert extracted[3] and "TimesNewRoman" in font[3].replace(" ", "")
        embedded.append({"name": font[3], "type": font[2], "embedded_bytes": len(extracted[3])})
    drawings = page.get_drawings()
    plot_rectangles = [drawing["rect"] for drawing in drawings if drawing["fill"] == (1.0, 1.0, 1.0) and abs(drawing["rect"].width - PLOT_SIDE) < 1e-3 and abs(drawing["rect"].height - PLOT_SIDE) < 1e-3]
    assert len(plot_rectangles) == 2
    plot_rectangles.sort(key=lambda rect: rect.y0)
    assert all(np.allclose(list(actual), expected, atol=1e-3, rtol=0) for actual, expected in zip(plot_rectangles, expected_rectangles))
    bars = [drawing["rect"] for drawing in drawings if drawing["fill"] == (0.0, 0.0, 0.0) and abs(drawing["rect"].width - PLOT_SIDE) < 1e-3 and abs(drawing["rect"].height - BAR_HEIGHT) < 1e-3]
    assert len(bars) == 2
    bars.sort(key=lambda rect: rect.y0)
    titles = sorted([span for span in spans if "km to" in span["text"]], key=lambda span: span["bbox"][1])
    assert len(titles) == 2 and all(bar.contains(pymupdf.Rect(title["bbox"])) for bar, title in zip(bars, titles))
    spine_widths = []
    diagonal_widths = []
    for rectangle in plot_rectangles:
        edges = [(rectangle.x0, rectangle.y0, rectangle.x0, rectangle.y1), (rectangle.x1, rectangle.y0, rectangle.x1, rectangle.y1), (rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y0), (rectangle.x0, rectangle.y1, rectangle.x1, rectangle.y1)]
        for edge in edges:
            matches = [drawing for drawing in drawings if drawing["fill"] is None and drawing["color"] == (0.0, 0.0, 0.0) and np.allclose(list(drawing["rect"]), edge, atol=1e-3, rtol=0)]
            assert len(matches) == 1 and abs(matches[0]["width"] - 0.65) < 1e-5
            spine_widths.append(matches[0]["width"])
        diagonals = [drawing for drawing in drawings if drawing["fill"] is None and drawing["color"] == (0.0, 0.0, 0.0) and np.allclose(list(drawing["rect"]), list(rectangle), atol=1e-3, rtol=0)]
        assert len(diagonals) == 1 and abs(diagonals[0]["width"] - 0.8) < 1e-5 and diagonals[0]["dashes"] != "[] 0"
        diagonal_widths.append(diagonals[0]["width"])
    images = page.get_image_info()
    assert len(images) == 2, f"Expected only the two rasterized scatter layers, got {len(images)}"
    assert len(drawings) > 10
    png = path.with_name("figure3_rendered_300dpi.png")
    page.get_pixmap(dpi=300, alpha=False).save(png)
    report = {"page_size_pt": [page.rect.width, page.rect.height], "page_size_mm": [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72], "font_sizes_pt": sorted(set(round(span["size"], 4) for span in spans)), "embedded_fonts": embedded, "all_text_inside_page": True, "text_overlap_count": len(overlaps), "equal_square_panel_size_pt": [PLOT_SIDE, PLOT_SIDE], "panel_rectangles_pdf_pt": [list(rect) for rect in plot_rectangles], "two_equal_title_bars": True, "title_text_inside_black_bars": True, "equal_border_width_pt": round(spine_widths[0], 4), "equal_reference_line_width_pt": round(diagonal_widths[0], 4), "rasterized_scatter_layers": len(images), "vector_drawing_count": len(drawings), "text_spans": spans}
    doc.close()
    return report


def compare_reference(reference: Path, final_pdf: Path, output: Path):
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    pictures = []
    for path in (reference, final_pdf):
        doc = pymupdf.open(path)
        assert abs(doc[0].rect.width - PAGE_W) < 1e-6 and abs(doc[0].rect.height - PAGE_H) < 1e-6
        pix = doc[0].get_pixmap(dpi=300, alpha=False)
        pictures.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
    # Both references are rendered at the same 300 dpi and pasted at native size.
    assert pictures[0].size == pictures[1].size
    width, height = pictures[0].size
    canvas = Image.new("RGB", (width * 2 + 60, height + 65), "white")
    canvas.paste(pictures[0], (10, 60))
    canvas.paste(pictures[1], (width + 50, 60))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 30)
    draw.text((width / 2 + 10, 25), "Teacher reference", font=font, fill="black", anchor="mm")
    draw.text((width * 1.5 + 50, 25), "Formal 2018-2019 redraw", font=font, fill="black", anchor="mm")
    canvas.save(output, dpi=(300, 300))


def effective_mass_nrd_bandwidth(values):
    """Effective Gaussian sd in MASS::kde2d: bandwidth.nrd(values) / 4.

    Formula source: https://raw.githubusercontent.com/cran/MASS/master/R/hist.scott.R
    This uses sample sd and the type-7 (linear) quartiles, as in the R helper.
    """
    values = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    scale = min(float(values.std(ddof=1)), float((q3 - q1) / 1.34))
    bandwidth = 1.06 * scale * values.size ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("MASS-style KDE bandwidth must be finite and positive")
    return bandwidth


def percent_rank_with_min_ties(values):
    """Equivalent to dplyr::percent_rank: (min_rank(x)-1)/(n-1)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Percentile ranks require at least two finite values")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first_indices = np.cumsum(counts) - counts
    return first_indices[inverse].astype(np.float64) / (len(values) - 1)


def mass_kde2d_point_density(x, y, grid_size=200, chunk_size=4096):
    """MASS::kde2d product-normal KDE, then R findInterval grid lookup.

    Reference algorithm: https://raw.githubusercontent.com/cran/MASS/master/R/kde2d.R
    Uses all existing plotting pairs, the same data-range grid and separate-axis
    default bandwidths. Chunking bounds the temporary arrays; no point is sampled
    away, jittered, clipped, reordered, fitted to model errors, or reweighted.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("KDE pairs must be finite one-dimensional arrays")
    hx, hy = effective_mass_nrd_bandwidth(x), effective_mass_nrd_bandwidth(y)
    gx = np.linspace(float(x.min()), float(x.max()), grid_size)
    gy = np.linspace(float(y.min()), float(y.max()), grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    normalizer = np.sqrt(2.0 * np.pi)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        ax = (gx[:, None] - x[None, start:stop]) / hx
        ay = (gy[:, None] - y[None, start:stop]) / hy
        kx = np.exp(-0.5 * ax * ax) / normalizer
        ky = np.exp(-0.5 * ay * ay) / normalizer
        grid += kx @ ky.T
    grid /= len(x) * hx * hy
    # R findInterval returns 1..n here, including n for the right endpoint;
    # searchsorted(right)-1 gives the corresponding zero-based grid indices.
    ix = np.searchsorted(gx, x, side="right") - 1
    iy = np.searchsorted(gy, y, side="right") - 1
    if (ix < 0).any() or (ix >= grid_size).any() or (iy < 0).any() or (iy >= grid_size).any():
        raise ValueError("Density grid lookup is outside its data-range grid")
    density = grid[ix, iy]
    if not np.isfinite(density).all() or (density <= 0).any():
        raise ValueError("Nonpositive/nonfinite density at a plotted point")
    ranks = percent_rank_with_min_ties(density)
    metadata = {
        "method": "MASS::kde2d product-normal KDE + findInterval + dplyr percent_rank(min ties)",
        "grid_size": [grid_size, grid_size], "gaussian_bandwidth_sd": [hx, hy],
        "grid_x_limits": [float(gx[0]), float(gx[-1])],
        "grid_y_limits": [float(gy[0]), float(gy[-1])],
        "point_count": len(x), "color_scale": "viridis", "point_order_changed": False,
        "density_min": float(density.min()), "density_max": float(density.max()),
        "percentile_min": float(ranks.min()), "percentile_max": float(ranks.max()),
        "color_normalization": "Observed percentile extent in this panel, matching R continuous-scale defaults",
        "color_interpretation": "Within-panel density percentile, not absolute density shared between methods",
        "density_payload_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
        "percentile_payload_sha256": hashlib.sha256(ranks.tobytes()).hexdigest(),
        "R_runtime_executed": False,
    }
    return density, ranks, metadata


def check_density_implementation():
    """Independent small Gaussian-kernel sum and tied-rank/endpoint checks."""
    from scipy.stats import norm
    tx = np.array([-3., -1.2, -0.1, 0.7, 1.5, 2.8, 4.])
    ty = np.array([1., -2., 0.5, 2.7, -0.9, 1.9, 4.4])
    actual, ranks, meta = mass_kde2d_point_density(tx, ty, grid_size=7, chunk_size=2)
    hx, hy = meta["gaussian_bandwidth_sd"]
    gx = np.linspace(tx.min(), tx.max(), 7)
    gy = np.linspace(ty.min(), ty.max(), 7)
    expected = []
    for a, b in zip(tx, ty):
        u = gx[np.searchsorted(gx, a, side="right") - 1]
        v = gy[np.searchsorted(gy, b, side="right") - 1]
        expected.append(np.mean(norm.pdf(u, loc=tx, scale=hx) * norm.pdf(v, loc=ty, scale=hy)))
    assert np.allclose(actual, expected, atol=1e-15, rtol=2e-14)
    assert np.array_equal(percent_rank_with_min_ties([10., 10., 20., 30.]), [0., 0., 2/3, 1.])
    assert (ranks >= 0).all() and (ranks <= 1).all()
    return {"independent_scipy_kernel_sum": True, "min_tie_ranks": True,
            "data_range_endpoint_grid_lookup": True, "R_runtime_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path(__file__).resolve().parents[1] / "splits/test_samples.csv")
    parser.add_argument("--samples-root", type=Path, help="Override the archived samples path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-pdf", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "figure2_viridis.pdf"
    preview = output / "figure2_viridis_preview.png"
    for path in (pdf, preview):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Use a new output directory or explicitly pass --overwrite: {path}")
    references = [path.resolve(strict=True) for path in args.reference_pdf]
    assert pdf.resolve() not in references
    reference_hashes = {str(path): file_hash(path) for path in references}
    x, bicubic, drct, metrics, config, source_hashes = load_formal_data(args.run_dir, args.test_csv, args.samples_root)
    density_selfcheck = check_density_implementation()
    coloring = []
    density_arrays = {}
    for method, values in (("bicubic", bicubic), ("drct", drct)):
        print(f"Computing {method} density with all {len(x)} saved plotting pairs ...", flush=True)
        density, ranks, metadata = mass_kde2d_point_density(x, values)
        coloring.append((ranks, metadata))
        density_arrays[method + "_density"] = density
        density_arrays[method + "_density_percentile"] = ranks
    np.savez_compressed(output / "density_color_data.npz", **density_arrays)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import Rectangle

    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": FONT_SIZE, "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9, "savefig.bbox": None, "text.usetex": False})
    figure = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=150, facecolor="white")
    axes, rectangles = [], []
    for number, (left, top, y, method, color) in enumerate(zip(PLOT_LEFTS, PLOT_TOPS, (bicubic, drct), ("Bicubic", "DRCT"), COLORS)):
        bottom = PAGE_H - top - PLOT_SIDE
        axis = figure.add_axes([left / PAGE_W, bottom / PAGE_H, PLOT_SIDE / PAGE_W, PLOT_SIDE / PAGE_H])
        axes.append(axis)
        rectangles.append([left, top, left + PLOT_SIDE, top + PLOT_SIDE])
        axis.scatter(x, y, s=0.22, alpha=0.075, c=coloring[number][0], cmap="viridis", edgecolors="none", rasterized=True)
        axis.plot(AXIS_LIMITS, AXIS_LIMITS, color="black", linestyle=(0, (6, 5)), linewidth=0.8, zorder=3)
        axis.set_xlim(*AXIS_LIMITS)
        axis.set_ylim(*AXIS_LIMITS)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks(TICKS)
        axis.set_yticks(TICKS)
        # Keep the 1200 endpoint label inside the fixed 170-pt page, without
        # reducing the requested font size or shrinking the drawing afterward.
        axis.get_xticklabels()[-1].set_horizontalalignment("right")
        axis.tick_params(axis="x", labelsize=FONT_SIZE, width=0.6, length=2, pad=2.5)
        axis.tick_params(axis="y", labelsize=FONT_SIZE, width=0.6, length=2, pad=1)
        axis.grid(True, color="#9E9E9E", linewidth=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("black")
        axis.add_patch(Rectangle((0, 1), 1, BAR_HEIGHT / PLOT_SIDE, transform=axis.transAxes, facecolor="black", edgecolor="none", clip_on=False, zorder=4))
        axis.text(0.5, 1 + (BAR_HEIGHT / 2 - 0.75) / PLOT_SIDE, f"{method}: 4 km to 1 km", transform=axis.transAxes, color="white", ha="center", va="center", fontsize=FONT_SIZE, zorder=5)
        figure.text((3.5 + left - PLOT_LEFT) / PAGE_W, (PAGE_H - top + BAR_HEIGHT / 2 - 0.75) / PAGE_H, "(a)" if number == 0 else "(b)", fontsize=FONT_SIZE, ha="left", va="center", fontweight="normal")
        axis.set_xlabel("FY-3C Observation [W/m²]", fontsize=LABEL_SIZE, labelpad=4)
        figure.text((7.5 + left - PLOT_LEFT) / PAGE_W, (bottom + PLOT_SIDE / 2) / PAGE_H, f"{method} Result [W/m²]", fontsize=LABEL_SIZE, rotation=90, ha="center", va="center")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text_boxes = []
    for artist in [*figure.texts, *[text for axis in axes for text in [*axis.texts, *axis.get_xticklabels(), *axis.get_yticklabels(), axis.xaxis.label]]]:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= figure.bbox.width and bounds.y1 <= figure.bbox.height, (artist.get_text(), list(bounds.extents))
        text_boxes.append({"text": artist.get_text(), "bounds_figure_px": list(bounds.extents)})
    figure.savefig(pdf, dpi=600, facecolor="white", bbox_inches=None, metadata={"Title": "2018-2019 formal independent-test GHI scatter", "Subject": "Bicubic and DRCT-L; final-size figure; only scatter layers rasterized"})
    figure.savefig(preview, dpi=300, facecolor="white", bbox_inches=None)
    plt.close(figure)
    checks = verify_pdf(pdf, rectangles)
    # The old reference is vertical; geometry/color-only comparisons are checked separately.
    for name, digest in {**source_hashes, **reference_hashes}.items():
        assert file_hash(Path(name)) == digest, f"Input changed: {name}"
    report = {"passed": True, "created_by": "fresh Matplotlib render; no PDF cropping or page scaling", "font_path": font_path, "matplotlib_version": matplotlib.__version__, "formal_run": str(args.run_dir), "formal_test_samples": metrics["samples"], "formal_valid_pixels": metrics["valid_pixels"], "plotted_paired_pixels": len(x), "sampling": "The original formal run's saved uniform global-pixel selection with seed 60; no new drawing sample", "data_transform": "none; original float32 GHI values used as stored", "axis_limits": list(AXIS_LIMITS), "ticks": list(TICKS), "points_outside_plot": {"fy3": 0, "bicubic": 0, "drct": 0}, "colors": list(COLORS), "scatter_raster_dpi": 600, "preview_dpi": 300, "full_test_metrics_for_source_verification_only": metrics["ghi"], "source_sha256": source_hashes, "reference_sha256_unchanged": reference_hashes, "clear_ghi_provenance_as_formal_run": config["ghi"]["clear_provenance"], "pdf_checks": checks}
    report.update(density_encoding=[item[1] for item in coloring], density_math_selfcheck=density_selfcheck, original_marker_size_pt2=0.22, original_point_alpha=0.075, colorbar_added=False)
    (output / "figure2_viridis_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "pdf": str(pdf), "preview": str(preview), "page_size_pt": checks["page_size_pt"], "font_sizes_pt": checks["font_sizes_pt"], "panel_size_pt": checks["equal_square_panel_size_pt"], "plotted_pixels": len(x), "overlaps": checks["text_overlap_count"], "embedded_fonts": checks["embedded_fonts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
