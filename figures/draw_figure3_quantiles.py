"""Redraw formal CSI examples using a continuous, shared quantile colour map.

Requirements: numpy, matplotlib, pymupdf; Times New Roman installed.
Example: python draw_figure3_quantiles.py --output-dir <new-output-folder>

The saved arrays and their original sample selection are read-only. The original
PDF is a geometry/font reference only: its CSI imagery and 0--1.5 norm are NEVER
used as data. No model inference, resampling of the data set, metric calculation,
PDF scaling, or PDF editing is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

CURRENT_SHA256 = '070824e33f67f5c8533edc1a7d7103d23c0e1d152ae4a2ef29577b7719701707'
INDICES = [20433, 11091, 11327, 6637, 12327, 24422, 7999, 32354, 12567, 11270]
KEYS = ('low', 'bicubic', 'prediction', 'target')
LABELS = ('FY-4A', 'Bicubic', 'DRCT', 'FY-3C')
QUANTILE_PROBABILITIES = np.linspace(0.0, 1.0, 6)
STEM = 'figure3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def payload(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def archived_hash(manifest, relative):
    suffix = "/" + relative
    matches = [digest for name, digest in manifest["source_sha256"].items()
               if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify archived source: {relative}")
    return matches[0]


def load_and_verify(selected, current, teacher, test_csv, samples_root, run_dir):
    manifest = json.loads((selected / 'selection_manifest.json').read_text(encoding='utf-8'))
    figure_manifest = json.loads((selected / 'selected_figure_manifest.json').read_text(encoding='utf-8'))
    for name in ('selected_csi_data.npz', 'selected_samples.csv', 'selection_manifest.json'):
        if sha(selected / name) != figure_manifest['output_sha256'][name]:
            raise ValueError(f'Saved CSI source differs from the original verified figure: {name}')
    if sha(current) != CURRENT_SHA256:
        raise ValueError('The backed-up supplied figure differs from the inspected source')
    original_figure = selected / 'figure3_csi_comparison_formal.pdf'
    if sha(original_figure) != figure_manifest['output_sha256']['figure3_csi_comparison_formal.pdf']:
        raise ValueError('The original formal figure does not match the saved CSI data')
    if manifest['frozen_csv_row_indices'] != INDICES or manifest['full_test_size'] != 35476:
        raise ValueError('The original ten-example selection or formal test size changed')
    if manifest['checkpoint_epoch'] != 65 or manifest['raw_csi_scaling_or_clipping']:
        raise ValueError('Unexpected formal experiment identity or CSI transformation')
    # Verify and preserve the source's outer display limits (printed as 0--2.0).
    # The internal colour mapping is intentionally replaced by shared quantiles.
    lo, hi = manifest['display_minimum'], manifest['display_maximum']
    if lo != 0 or hi != 1.999814510345459:
        raise ValueError('The original new-experiment CSI color mapping changed')
    data_file = selected / 'selected_csi_data.npz'
    with np.load(data_file, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in KEYS}
    for k, a in arrays.items():
        shape = (10, 32, 32) if k == 'low' else (10, 128, 128)
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'Invalid original CSI field: {k}')
        if payload(a) != figure_manifest['array_metadata'][k]['raw_payload_sha256']:
            raise ValueError(f'CSI payload changed: {k}')
        a.flags.writeable = False
    # Independently verify that the supplied PDF contains these exact existing
    # arrays in its old order/linear mapping, before changing row order or colour.
    import pymupdf as fitz
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    expected_rows = []
    for k in ('low', 'target', 'prediction', 'bicubic'):
        a = arrays[k]
        display = np.repeat(np.repeat(a, 4, axis=1), 4, axis=2) if k == 'low' else a
        expected_rows.append(np.concatenate([colormaps['viridis'](Normalize(lo, hi)(v), bytes=True)
                                              for v in display], axis=1))
    expected_rgb = np.concatenate(expected_rows, axis=0)[..., :3]
    with fitz.open(current) as doc:
        image = [v for v in doc[0].get_images(full=True) if v[2:4] == (1280, 512)]
        if len(image) != 1:
            raise ValueError('Cannot uniquely identify the original forty-image CSI grid')
        pix = fitz.Pixmap(doc, image[0][0])
        actual_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        info = next(v for v in doc[0].get_image_info(xrefs=True) if v['xref'] == image[0][0])
        a, b, c, d, _, _ = info['transform']
        if a <= 0 or abs(b) > 1e-6 or abs(c) > 1e-6 or d == 0:
            raise ValueError('Unexpected source image transformation')
        # PDF image storage order differs from visible order when d is negative.
        if d < 0:
            actual_rgb = actual_rgb[::-1]
        if not np.array_equal(expected_rgb, actual_rgb):
            raise ValueError('The supplied PDF imagery differs from the verified saved CSI arrays')
    rows, order = read_csv(test_csv), read_csv(run_dir / 'test_order.csv')
    if len(rows) != 35476 or len(order) != 35476:
        raise ValueError('Wrong formal test CSV')
    if sha(test_csv) != archived_hash(manifest, 'splits/test_samples.csv'):
        raise ValueError('The formal test CSV hash changed')
    if sha(run_dir / 'metrics.json') != archived_hash(manifest, 'metrics.json'):
        raise ValueError('The formal test result identity changed')
    prediction = np.load(run_dir / 'drct_prediction_csi.npy', mmap_mode='r', allow_pickle=False)
    if prediction.shape != (35476, 128, 128) or prediction.dtype != np.float32:
        raise ValueError('Unexpected formal prediction file')
    source_hashes = {str(p): sha(p) for p in (teacher, current, original_figure, data_file, selected / 'selection_manifest.json',
                                             selected / 'selected_figure_manifest.json', test_csv)}
    for col, idx in enumerate(INDICES):
        row = rows[idx]
        saved = manifest['selected_samples'][col]
        if any(saved[k] != v for k, v in row.items()) or order[idx]['sample_id'] != row['sample_id']:
            raise ValueError(f'Example order mismatch at column {col + 1}')
        source = samples_root / row['date'] / row['npz_file']
        digest = sha(source)
        if digest != archived_hash(manifest, f"samples/{row['date']}/{row['npz_file']}"):
            raise ValueError(f'Original NPZ hash changed: {source}')
        source_hashes[str(source)] = digest
        with np.load(source, allow_pickle=False) as npz:
            i = int(row['array_index'])
            if not np.array_equal(npz['lr_csi'][i], arrays['low'][col]):
                raise ValueError(f'LR source mismatch: {idx}')
            if not np.array_equal(npz['hr_csi'][i], arrays['target'][col]):
                raise ValueError(f'HR source mismatch: {idx}')
        if not np.array_equal(prediction[idx], arrays['prediction'][col]):
            raise ValueError(f'DRCT-L saved prediction mismatch: {idx}')
    del prediction
    return arrays, manifest, figure_manifest, source_hashes


def template_geometry(path):
    import pymupdf as fitz
    with fitz.open(path) as doc:
        if len(doc) != 1:
            raise ValueError('Expected a one-page layout template')
        p = doc[0]
        typ, media = doc.xref_get_key(p.xref, 'MediaBox')
        if typ != 'array':
            raise ValueError('Template MediaBox must be explicit')
        box = [float(v) for v in re.findall(r'[-+]?\d+(?:\.\d+)?', media)]
        # MuPDF returns normalized decimal strings with fewer digits. Read the
        # one explicit original MediaBox as well, retaining its exact decimals.
        raw_boxes = re.findall(rb'/MediaBox\s*\[([^\]]+)\]', path.read_bytes())
        if len(raw_boxes) != 1:
            raise ValueError('Expected one unambiguous explicit template MediaBox')
        exact_box = [float(v) for v in raw_boxes[0].split()]
        if len(exact_box) != 4 or max(abs(a-b) for a,b in zip(exact_box,box)) > 1e-4:
            raise ValueError('Template MediaBox representations disagree')
        box = exact_box
        if box[:2] != [0., 0.]:
            raise ValueError('Unexpected template page origin')
        images = p.get_image_info()
        grid = max(images, key=lambda i: i['bbox'][2] * i['bbox'][3])['bbox']
        bar = min(images, key=lambda i: i['bbox'][3] - i['bbox'][1])['bbox']
        spans = [s for b in p.get_text('dict')['blocks'] if b['type'] == 0
                 for l in b['lines'] for s in l['spans']]
        if not all(abs(s['size'] - 8.75) < 1e-6 and 'TimesNewRoman' in s['font'] for s in spans):
            raise ValueError('Template typography differs from the inspected source')
        line_width = p.get_drawings()[0]['width']
        sample_baseline = next(s['origin'][1] for s in spans if s['text'] == 'Sample 1')
        label = next(s for s in spans if s['text'] == 'Clear-sky index (CSI)')
        tick_baseline = next(s['origin'][1] for s in spans if s['text'] == '0.0')
    return dict(width=box[2], height=box[3], grid=list(grid), bar=list(bar), font_size=8.75,
                line_width=line_width, sample_baseline=sample_baseline,
                bar_label_right=label['bbox'][2], bar_label_baseline=label['origin'][1],
                tick_baseline=tick_baseline, tick_length=2.83464)


def shared_quantile_scale(arrays, lo, hi):
    # Every displayed image has equal weight: LR pixels are repeated only for
    # display, as in the existing figure, so each row contributes 163,840 values.
    values = []
    for key in KEYS:
        a = arrays[key]
        display = np.repeat(np.repeat(a, 4, axis=1), 4, axis=2) if key == 'low' else a
        values.append(display.reshape(-1))
    pooled = np.concatenate(values)
    if not np.isfinite(pooled).all() or pooled.size != 655360:
        raise ValueError('Unexpected/non-finite displayed CSI pixels')
    quantiles = np.quantile(pooled, QUANTILE_PROBABILITIES, method='linear')
    boundaries = quantiles.copy()
    # Retain the previously approved 0--2.0 outer display limits. Internal
    # 20/40/60/80th percentiles anchor a continuous nonlinear colour mapping.
    if quantiles[0] < lo or quantiles[-1] > hi:
        raise ValueError('Existing display limits would exclude original CSI values')
    boundaries[0], boundaries[-1] = lo, hi
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError('Duplicate/non-increasing quantile boundaries; do not silently invent bins')
    counts = np.bincount(np.searchsorted(boundaries[1:-1], pooled, side='right'), minlength=5)
    return boundaries, dict(probabilities=QUANTILE_PROBABILITIES.tolist(),
        raw_sample_quantiles=quantiles.tolist(), shared_boundaries=boundaries.tolist(),
        pixel_count=int(pooled.size), pixels_per_method=163840,
        quantile_interval_pixel_counts=counts.tolist(),
        quantile_method='numpy linear interpolation',
        endpoint_policy='Existing display endpoints retained; internal boundaries are pooled quintiles',
        reference='All forty displayed images pooled, equal pixel-area weight per image',
        same_boundaries_for_every_image=True, sample_values_modified=False)


def draw(arrays, lo, hi, g, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, FuncNorm
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    font = FontProperties(family='Times New Roman', size=g['font_size'])
    font_path = findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.bbox': None, 'image.composite_image': False,
                         # Use explicit 8-bit RGB: this Matplotlib version omits
                         # BitsPerComponent from the PNG predictor for 4-bit
                         # indexed images, corrupting five-colour PDF rasters.
                         'pdf.compression': 0})
    w, h = g['width'], g['height']
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=300, facecolor='white')
    class ExactExtentImage(AxesImage):
        # Matplotlib otherwise rounds native image extents up to integer PDF
        # points. Disable that renderer rounding, so raster and vector cell
        # boundaries share the same measured coordinates (no PDF resizing).
        def _make_image(self, *args, **kwargs):
            kwargs['round_to_pixel_border'] = False
            return super()._make_image(*args, **kwargs)

    def exact_image(axis, data, **kwargs):
        image = ExactExtentImage(axis, interpolation='none', origin='upper', **kwargs)
        image.set_data(data)
        image.set_extent((-.5, data.shape[1]-.5, data.shape[0]-.5, -.5))
        axis.add_image(image)
        axis.set_aspect('auto')
        return image
    x0, y0, x1, y1 = g['grid']
    tile = (x1 - x0) / 10
    if abs((y1 - y0) / 4 - tile) > 1e-4:
        raise ValueError('Template cells are not square')
    boundaries, quantile_info = shared_quantile_scale(arrays, lo, hi)
    cmap = plt.get_cmap('viridis')
    probabilities = QUANTILE_PROBABILITIES
    norm = FuncNorm((lambda v: np.interp(v, boundaries, probabilities),
                     lambda v: np.interp(v, probabilities, boundaries)),
                    vmin=lo, vmax=hi, clip=False)
    if not np.allclose(norm(boundaries), probabilities, atol=1e-12):
        raise ValueError('Continuous colour mapping does not pass through the shared quantile anchors')
    if not np.allclose(norm.inverse(norm(boundaries)), boundaries, atol=1e-12):
        raise ValueError('Continuous colour mapping inverse failed')
    # Rendering only: nearest-neighbor repeats retain the original LR block colors.
    # No values in the saved float32 arrays are changed.
    tiles = []
    for k in KEYS:
        a = arrays[k]
        display = np.repeat(np.repeat(a, 4, axis=1), 4, axis=2) if k == 'low' else a
        tiles.append(np.concatenate([cmap(norm(v), bytes=True) for v in display], axis=1))
    rgba = np.concatenate(tiles, axis=0)
    rendered_colour_count = len(np.unique(rgba[..., :3].reshape(-1, 3), axis=0))
    if rendered_colour_count < 100:
        raise ValueError('Expected continuous Viridis detail, not a small set of flat colour bins')

    def axes_rect(rect):
        l, top, r, bottom = rect
        return [l / w, (h - bottom) / h, (r - l) / w, (bottom - top) / h]

    ax = fig.add_axes(axes_rect(g['grid']))
    exact_image(ax, rgba)
    ax.set_axis_off()

    def rectangle(rect, color, z=3):
        l, top, r, bottom = rect
        patch = Rectangle((l / w, (h - bottom) / h), (r-l) / w, (bottom-top) / h,
                          transform=fig.transFigure, facecolor=color, edgecolor='none', zorder=z)
        fig.add_artist(patch)

    def line(xa, ya, xb, yb, color='black'):
        artist = Line2D([xa / w, xb / w], [(h - ya) / h, (h - yb) / h],
                        transform=fig.transFigure, color=color, linewidth=g['line_width'],
                        solid_capstyle='butt', zorder=4)
        artist.set_clip_box(fig.bbox)
        fig.add_artist(artist)

    texts = []
    def text(x, y, s, ha='center', va='baseline', color='black', rotation=0):
        t = fig.text(x / w, (h - y) / h, s, fontproperties=font, ha=ha, va=va,
                     color=color, rotation=rotation, zorder=5)
        texts.append(t)
        return t

    rectangle([x0, 0, x1, y0], 'black')
    rectangle([x1, y0, w, y1], 'black')
    for j in range(10):
        text(x0 + (j + .5) * tile, g['sample_baseline'], f'Sample {j+1}', color='white')
        if j:
            line(x0 + j*tile, 0, x0 + j*tile, y0, 'white')
            line(x0 + j*tile, y0, x0 + j*tile, y1)
    # Outer strokes are kept inside the page; all cell centers and dividers
    # remain exactly at the measured template coordinates.
    half = g['line_width'] / 2
    line(max(half, x0), y0, x1, y0)
    line(max(half, x0), y1, x1, y1)
    line(max(half, x0), y0, max(half, x0), y1)
    line(x1, y0, x1, y1)
    for i, label in enumerate(LABELS):
        if i:
            line(x0, y0 + i*tile, x1, y0 + i*tile)
        text((x1 + w) / 2, y0 + (i + .5)*tile, label, va='center', color='white', rotation=270)

    bx0, by0, bx1, by1 = g['bar']
    cax = fig.add_axes(axes_rect(g['bar']))
    # A smooth Viridis ramp; the same shared quantile anchors label its positions.
    exact_image(cax, np.arange(256, dtype=np.float32)[None, :], cmap=cmap,
                norm=Normalize(0,255,clip=False))
    cax.set_axis_off()
    for xa, ya, xb, yb in [(bx0,by0,bx1,by0),(bx0,by1,bx1,by1),
                           (bx0,by0,bx0,by1),(bx1,by0,bx1,by1)]:
        line(xa,ya,xb,yb)
    ticks = boundaries
    for j, value in enumerate(ticks):
        x = bx0 + j / 5 * (bx1 - bx0)
        line(x, by1, x, by1 + g['tick_length'])
        text(x, g['tick_baseline'], f'{value:.2f}')
    text(g['bar_label_right'], g['bar_label_baseline'], 'Clear-sky index (CSI)', ha='right')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in texts:
        bounds = t.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
            raise ValueError(f'Text outside page: {t.get_text()}')
    fig.savefig(out / (STEM + '.pdf'), dpi=600, bbox_inches=None, pad_inches=0,
                metadata={'Title': 'Formal 2018-2019 CSI comparison, template-matched layout',
                          'Subject': 'Existing ten formal examples; FY-4A/Bicubic/DRCT/FY-3C; continuous shared quantile normalization; raw CSI unchanged'})
    plt.close(fig)
    return dict(font_path=font_path, matplotlib_version=matplotlib.__version__,
                grid_render_rgba_sha256=payload(rgba), grid_render_shape=list(rgba.shape),
                grid_render_rgb_sha256=payload(rgba[..., :3]),
                colorbar_ticks=ticks.tolist(), display_tick_labels=[f'{v:.2f}' for v in ticks],
                tile_size_pt=tile, row_labels=LABELS, source_cmap='viridis', data_scaled=False,
                data_clipped=False, shared_norm_min=lo, shared_norm_max=hi,
                quantile_scale=quantile_info, row_data_keys=KEYS,
                colour_mapping='continuous piecewise-linear interpolation through shared quantile anchors',
                palette_entries=cmap.N, rendered_colour_count=rendered_colour_count,
                display_note='Continuous quantile Viridis gradient; original outer display range retained (0.00--2.00)')


def verify_output(pdf, template, g, qa, style):
    import pymupdf as fitz
    from PIL import Image, ImageDraw, ImageFont
    with fitz.open(pdf) as d:
        p = d[0]
        actual = list(p.rect)
        raw_box = re.findall(rb'/MediaBox\s*\[([^\]]+)\]', pdf.read_bytes())
        if len(raw_box) != 1 or [float(v) for v in raw_box[0].split()] != [0.,0.,g['width'],g['height']]:
            raise ValueError('The exact MediaBox decimals differ from template')
        if len(d) != 1 or abs(p.rect.width-g['width']) > 2e-5 or abs(p.rect.height-g['height']) > 2e-5:
            raise ValueError('Final PDF page size differs from template')
        spans = [s for b in p.get_text('dict')['blocks'] if b['type'] == 0
                 for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        text = [s['text'] for s in spans]
        expected = ([f'Sample {j}' for j in range(1,11)] + list(LABELS)
                    + ['Clear-sky index (CSI)'] + style['display_tick_labels'])
        if sorted(text) != sorted(expected):
            raise ValueError(f'Output labels differ: {text}')
        if not all(abs(s['size']-8.75) < 1e-6 and 'TimesNewRoman' in s['font'] for s in spans):
            raise ValueError('Output font or point size differs from template')
        outside = [s['text'] for s in spans if not p.rect.contains(fitz.Rect(s['bbox']))]
        overlaps = [(a['text'],b['text']) for i,a in enumerate(spans) for b in spans[i+1:]
                    if (fitz.Rect(a['bbox']) & fitz.Rect(b['bbox'])).get_area() > .01]
        if outside or overlaps:
            raise ValueError(f'Text clipping/overlap: {outside}, {overlaps}')
        row_spans = [next(s for s in spans if s['text'] == label) for label in LABELS]
        row_centres = [(s['bbox'][1] + s['bbox'][3]) / 2 for s in row_spans]
        if row_centres != sorted(row_centres):
            raise ValueError('The requested row-label order is incorrect')
        fonts = []
        for f in p.get_fonts(full=True):
            _, ext, _, data = d.extract_font(f[0])
            if not data or ext != 'ttf' or 'TimesNewRoman' not in f[3]:
                raise ValueError('Times New Roman must be embedded as TrueType')
            fonts.append(dict(name=f[3], extension=ext, embedded_bytes=len(data)))
        images = p.get_image_info(xrefs=True)
        grid = max(images, key=lambda i:i['width']*i['height'])
        if (grid['width'],grid['height']) != (1280,512):
            raise ValueError('Native CSI raster dimensions changed')
        if max(abs(a-b) for a,b in zip(grid['bbox'],g['grid'])) > 1e-4:
            raise ValueError('CSI grid rectangle differs from teacher template')
        pix = fitz.Pixmap(d, grid['xref'])
        decoded = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        if grid['transform'][3] < 0:
            decoded = decoded[::-1]
        if payload(decoded) != style['grid_render_rgb_sha256']:
            raise ValueError('PDF-decoded grid pixels differ from the intended quantile colour mapping')
        preview = p.get_pixmap(dpi=300)
        preview.save(str(qa / 'new_render.png'))
        preview.save(str(pdf.with_name(STEM + '_preview.png')))
    with fitz.open(template) as d:
        d[0].get_pixmap(dpi=300).save(str(qa / 'teacher_render.png'))
    teacher = Image.open(qa / 'teacher_render.png').convert('RGB')
    new = Image.open(qa / 'new_render.png').convert('RGB')
    if teacher.size != new.size:
        raise ValueError('Teacher and output page render dimensions differ')
    gap, top = 32, 58
    both = Image.new('RGB',(teacher.width*2+gap,teacher.height+top),'white')
    both.paste(teacher,(0,top));both.paste(new,(teacher.width+gap,top))
    draw = ImageDraw.Draw(both)
    font = ImageFont.truetype(r'C:\Windows\Fonts\times.ttf', 30)
    draw.text((12,10),'Teacher layout (old experiment)',fill='black',font=font)
    draw.text((teacher.width+gap+12,10),'Matched layout (formal 2018-2019 data)',fill='black',font=font)
    both.save(qa / (STEM + '_layout_comparison.png'))
    return dict(page_pt=[g['width'],g['height']], mupdf_page_pt=actual[2:],
                page_mm=[g['width']/72*25.4,g['height']/72*25.4],
                same_page_size_as_template=True, fonts=fonts, font_size_pt=8.75,
                labels_complete=True, text_outside_page=outside, text_overlaps=overlaps,
                native_grid_pixels=[1280,512], grid_rectangle=list(grid['bbox']),
                equal_square_panels=True, image_panel_count=40, glyphs_and_borders_vector=True,
                pdf_decoded_pixels_exactly_match_quantile_mapped_arrays=True,
                geometry=g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--qa-dir', type=Path, help='Optional separate QA folder')
    ap.add_argument('--selected-dir', type=Path, required=True)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--current-pdf', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--samples-root', type=Path, required=True)
    ap.add_argument('--test-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'splits/test_samples.csv')
    args = ap.parse_args()
    out = args.output_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    qa = args.qa_dir.resolve() if args.qa_dir else out
    qa.mkdir(parents=True, exist_ok=True)
    for name in (STEM+'.pdf', STEM+'_preview.png', STEM+'_checks.json'):
        if (out/name).exists():
            raise FileExistsError(f'Refusing to overwrite: {out/name}; choose a new output directory')
    os.environ.setdefault('MPLCONFIGDIR', str(qa / 'mpl_cache'))
    arrays, manifest, figure_manifest, protected = load_and_verify(args.selected_dir, args.current_pdf, args.template,
                                                                  args.test_csv, args.samples_root, args.run_dir)
    before = {k:payload(a) for k,a in arrays.items()}
    g = template_geometry(args.template)
    style = draw(arrays, manifest['display_minimum'], manifest['display_maximum'], g, out)
    checks = verify_output(out/(STEM+'.pdf'),args.template,g,qa,style)
    if before != {k:payload(a) for k,a in arrays.items()}:
        raise ValueError('CSI arrays changed during plotting')
    if protected != {p:sha(p) for p in protected}:
        raise ValueError('A protected source file changed')
    report = dict(passed=True, source_current_pdf=str(args.current_pdf), template=str(args.template),
                  original_source_sha256=protected, selected_csv_row_indices=INDICES,
                  selection_type=manifest['selection_type'], raw_array_payload_sha256=before,
                  formal_sample_size=35476, model_epoch=65, style=style, pdf_checks=checks,
                  source_files_unchanged=True, csi_values_unchanged=True,
                  prediction_and_raw_npz_selected_samples_bitwise_equal=True,
                  bicubic_source='Original verified selected_csi_data.npz, bitwise payload and file hash checked',
                  latex_or_caption_modified=False,
                  output_sha256={n:sha(out/n) for n in (STEM+'.pdf', STEM+'_preview.png')})
    (out/(STEM+'_checks.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True, output=str(out/(STEM+'.pdf')), page_pt=checks['page_pt'],
                          font_size=8.75, unchanged_sample_indices=INDICES,
                          row_order=LABELS, shared_colour_boundaries=style['colorbar_ticks'],
                          original_outer_display_limits_preserved=True),indent=2),flush=True)


if __name__ == '__main__':
    main()
