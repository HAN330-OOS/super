"""Apply the requested annotation/layout changes without resampling the diagram.

The existing figure was produced by outputs/merge_figures.py from the original
vector assets. Its two component streams are reused unchanged except that the
six replaced label objects are removed. New labels are authored as embedded
Times New Roman text, not painted over existing labels. The network paths,
connections, colours, example images, and component scale stay unchanged.

Run with the bundled Python (pypdf, reportlab, Pillow, PyMuPDF required).
The immutable input is kept alongside this script for repeatable regeneration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

TOP_WHITESPACE_REMOVED_PT = 43.0
INTERPANEL_GAP_REDUCTION_PT = 22.0
FONT_SIZE = 8.75
INK = (17 / 255,) * 3
REMOVED_LABELS = {
    "FY-3C CSI",  # moved to the left of its unchanged reference image
    "Global residual connection (F0)",
    "HR reference",
    "Residual Dense Group (RDG)",
    "Swin Transformer Layer (STL)",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown_text(operations):
    text = []
    for args, op in operations:
        if op in (b"Tj", b"TJ"):
            values = args[0] if op == b"TJ" else args
            text.extend(str(v) for v in values if isinstance(v, str))
    return "".join(text)


def graphics_groups(operations):
    """Return balanced q/Q groups without relying on hard-coded PDF offsets."""
    stack, groups = [], []
    for i, (_, op) in enumerate(operations):
        if op == b"q":
            stack.append(i)
        elif op == b"Q":
            start = stack.pop()
            groups.append((start, i, len(stack)))
    assert not stack
    return groups


def text_blocks(operations):
    start = None
    for i, (_, op) in enumerate(operations):
        if op == b"BT":
            assert start is None
            start = i
        elif op == b"ET":
            assert start is not None
            yield start, i, shown_text(operations[start:i + 1])
            start = None
    assert start is None


def edit_component(operations, is_a):
    removed = Counter()
    drop = set()
    for start, end, text in text_blocks(operations):
        # The second occurrence belongs to FY-3C; the first is the SR output.
        reference_dimensions = is_a and text == "128 × 128 × 1" and removed[
            "FY-3C CSI"
        ] == 1
        if text in REMOVED_LABELS or reference_dimensions:
            key = "moved reference dimensions" if reference_dimensions else text
            removed[key] += 1
            drop.update(range(start, end + 1))
    return [v for i, v in enumerate(operations) if i not in drop], removed


def translated(operations, dy):
    return [([], b"q"),
            ([FloatObject(v) for v in [1, 0, 0, 1, 0, dy]], b"cm"),
            *operations, ([], b"Q")]


def xobject_payloads(resources):
    values = []
    for ref in resources.get("/XObject", {}).values():
        obj = ref.get_object()
        values.append((str(obj.get("/Subtype")),
                       hashlib.sha256(obj.get_data()).hexdigest()))
        values.extend(xobject_payloads(obj.get("/Resources", {})))
    return sorted(values)


def draw_labels(width, height, font, bold_font):
    pdfmetrics.registerFont(TTFont("TNR", str(font)))
    pdfmetrics.registerFont(TTFont("TNRBold", str(bold_font)))
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height), pageCompression=1,
                      initialFontName="TNR", initialFontSize=FONT_SIZE)
    c.setFillColorRGB(*INK)
    c.setFont("TNR", FONT_SIZE)

    def baseline(x, y_from_top, text, align="center"):
        method = {"center": c.drawCentredString, "right": c.drawRightString,
                  "left": c.drawString}[align]
        method(x, height - y_from_top, text)

    # Same centre and baseline as the original label; remove the entire (F0).
    baseline(423.0 * width / 1051.5, 68.643158 - 43.0,
             "Global residual connection")

    # Right-aligned two-line label to the left of the original FY-3C image.
    # Original thumbnail rectangle: x=447.4575..478.7559, y=48.7470..79.3175.
    baseline(439.45752, 61.600 - 43.0, "FY-3C CSI", "right")
    baseline(439.45752, 73.245905 - 43.0, "128 × 128 × 1", "right")

    # Preserve heading wording, size, baseline and centre; use normal weight.
    offset = TOP_WHITESPACE_REMOVED_PT + INTERPANEL_GAP_REDUCTION_PT
    baseline(163.7835, 278.415527 - offset, "Residual dense group (RDG)")
    baseline(419.83, 278.716736 - offset, "Swin Transformer layer (STL)")

    # Keep the original tag font/size, but anchor each at its panel's upper left.
    c.setFont("TNRBold", FONT_SIZE)
    ascent = pdfmetrics.getAscent("TNRBold") * FONT_SIZE / 1000.0
    baseline(2.834646, 20.838158 + ascent, "(a)", "left")
    baseline(2.834646, 196.550 + ascent, "(b)", "left")
    c.showPage()
    c.save()
    stream.seek(0)
    return PdfReader(stream)


def spans(page):
    return [s for b in page.get_text("dict")["blocks"] if "lines" in b
            for line in b["lines"] for s in line["spans"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/times.ttf"))
    parser.add_argument("--bold-font", type=Path, default=Path("C:/Windows/Fonts/timesbd.ttf"))
    args = parser.parse_args()
    source = args.source_pdf.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "figure1.pdf"
    reader = PdfReader(source)
    original = reader.pages[0]
    old_height, width = float(original.mediabox.height), float(original.mediabox.width)
    height = old_height - TOP_WHITESPACE_REMOVED_PT - INTERPANEL_GAP_REDUCTION_PT
    content = ContentStream(original.get_contents(), reader)
    ops = content.operations
    groups = graphics_groups(ops)
    # The existing compositor has one depth-1 group per full content panel.
    candidates = [(start, end) for start, end, depth in groups if depth == 1
                  and any(op == b"Do" for _, op in ops[start:end + 1])]
    assert len(candidates) == 2, candidates
    components = []
    audit = Counter()
    for i, (start, end) in enumerate(candidates):
        edited, removed = edit_component(ops[start:end + 1], is_a=i == 0)
        audit.update(removed)
        components.extend(translated(edited, -INTERPANEL_GAP_REDUCTION_PT
                                     if i == 0 else 0.0))
    assert audit == Counter({**{s: 1 for s in REMOVED_LABELS},
                             "moved reference dimensions": 1}), audit
    content.operations = components
    writer = PdfWriter()
    writer.pdf_header = b"%PDF-1.4"
    page = writer.add_blank_page(width=width, height=height)
    page[NameObject("/Resources")] = original["/Resources"].clone(writer)
    page.replace_contents(content.clone(writer))
    page.merge_page(draw_labels(width, height, args.font, args.bold_font).pages[0])
    writer.add_metadata({"/Title": "DRCT architecture: overall model and internal blocks",
                         "/Subject": "Annotation and spacing revision; original vectors and images preserved"})
    writer.write(output)

    final_reader = PdfReader(output)
    assert xobject_payloads(original["/Resources"]) == xobject_payloads(
        final_reader.pages[0]["/Resources"]), "Original image/vector payload changed"

    source_doc, doc = fitz.open(source), fitz.open(output)
    p, old_p = doc[0], source_doc[0]
    ss = spans(p)
    text = p.get_text()
    assert "HR reference" not in text
    assert "Global residual connection (" not in text
    assert "Global residual connection" in text
    assert text.count("FY-3C CSI") == 1
    assert text.count("(a)") == 1 and text.count("(b)") == 1
    # The shallow feature F0 must remain: only the one in the connection label goes.
    assert any(s["text"] == "F" and "Italic" in s["font"] for s in ss)
    for title in ("Residual dense group (RDG)", "Swin Transformer layer (STL)"):
        title_span = next(s for s in ss if s["text"] == title)
        assert "Bold" not in title_span["font"]
        assert abs(title_span["size"] - FONT_SIZE) < 0.001
    assert all(0 <= s["bbox"][0] < s["bbox"][2] <= width + 0.01
               and 0 <= s["bbox"][1] < s["bbox"][3] <= height + 0.01 for s in ss)
    # Compare every untouched text span after the exact component translation.
    old_spans = spans(old_p)
    excluded = {"FY-3C CSI", "HR reference", "Residual Dense Group (RDG)",
                "Swin Transformer Layer (STL)", "(a)", "(b)"}
    keep_count = 0
    for s in old_spans:
        x0, y0, x1, y1 = s["bbox"]
        if s["text"] in excluded or (y0 < 40 and x0 > 400) or 60 < y0 < 75:
            continue
        dy = -43.0 if y0 < 240 else -65.0
        matches = [t for t in ss if t["text"] == s["text"]
                   and t["font"] == s["font"]
                   and abs(t["size"] - s["size"]) < 0.001
                   and max(abs(a - b) for a, b in zip(t["bbox"],
                       (x0, y0 + dy, x1, y1 + dy))) < 0.003]
        assert len(matches) == 1, (s["text"], s["bbox"], matches)
        keep_count += 1

    # The same native path geometry must remain, apart from panel translations.
    old_drawings, new_drawings = old_p.get_drawings(), p.get_drawings()
    assert len(old_drawings) == len(new_drawings)
    for a, b in zip(old_drawings, new_drawings):
        expected = fitz.Rect(a["rect"])
        dy = -43.0 if expected.y1 < 240 else -65.0
        expected.y0 += dy
        expected.y1 += dy
        assert max(abs(x - y) for x, y in zip(expected, b["rect"])) < 0.003
        for key in ("width", "color", "fill", "dashes", "type"):
            assert a[key] == b[key], (key, a[key], b[key])

    p.get_pixmap(dpi=200, alpha=False).save(output_dir / "figure1_preview.png")
    old_p.get_pixmap(dpi=140, alpha=False).save(output_dir / "before_preview.png")
    p.get_pixmap(dpi=140, alpha=False).save(output_dir / "after_preview.png")
    left, right = Image.open(output_dir / "before_preview.png"), Image.open(output_dir / "after_preview.png")
    comparison = Image.new("RGB", (left.width + right.width + 30,
                                    max(left.height, right.height)), "white")
    comparison.paste(left, (0, 0))
    comparison.paste(right, (left.width + 30, 0))
    comparison.save(output_dir / "before_after.png")
    fonts = []
    for item in p.get_fonts(full=True):
        name, ext, kind, data = doc.extract_font(item[0])
        assert data, (name, "not embedded")
        assert "TimesNewRoman" in name, name
        fonts.append({"name": name, "embedded": True})
    result = {
        "source": str(source), "output": str(output),
        "source_sha256": sha256(source), "output_sha256": sha256(output),
        "page_pt_before": [width, old_height], "page_pt_after": [width, height],
        "page_mm_after": [width * 25.4 / 72, height * 25.4 / 72],
        "network_component_scale_unchanged": True,
        "vector_paths_preserved": len(old_drawings),
        "all_original_vector_and_image_xobject_payloads_identical": True,
        "unchanged_text_spans_position_checked": keep_count,
        "all_text_inside_page": True, "fonts": fonts,
        "interpanel_gap_reduction_pt": INTERPANEL_GAP_REDUCTION_PT,
        "replaced_text_objects": dict(audit),
        "requested_edits": ["remove (F0) beside Global residual connection",
                            "remove HR reference", "regular-weight RDG/STL headers",
                            "move FY-3C CSI and dimensions to left of reference image",
                            "move panel tags to upper left near each panel",
                            "reduce interpanel gap without scaling components"],
    }
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
