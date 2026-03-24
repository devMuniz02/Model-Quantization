from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with `pip install pillow`.") from exc


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "quantization-flow.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_box(draw, box, radius, fill, outline, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_text(draw, box, text, font, fill, spacing=6):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def arrow(draw, start, end, fill, width=7):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    size = 16
    draw.polygon([(ex, ey), (ex - size, ey - size / 2), (ex - size, ey + size / 2)], fill=fill)


def main():
    ASSETS.mkdir(exist_ok=True)
    img = Image.new("RGB", (1500, 900), "#0f172a")
    draw = ImageDraw.Draw(img)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(22)
    box_title = load_font(28, bold=True)
    box_body = load_font(18)
    metric_font = load_font(22, bold=True)

    draw.text((60, 50), "Model Quantization", font=title_font, fill="#f8fafc")
    draw.text(
        (60, 108),
        "Programmatic view of the repo workflow: source model -> quantization pipeline -> smaller, faster artifact.",
        font=subtitle_font,
        fill="#cbd5e1",
    )

    left_box = (70, 230, 450, 650)
    mid_box = (560, 200, 940, 680)
    right_box = (1050, 230, 1430, 650)

    rounded_box(draw, left_box, 24, "#172554", "#60a5fa")
    rounded_box(draw, mid_box, 24, "#1e293b", "#22d3ee")
    rounded_box(draw, right_box, 24, "#1f3a2d", "#34d399")

    fit_text(draw, (90, 265, 430, 360), "Source Hugging Face Model", box_title, "#f8fafc")
    fit_text(draw, (100, 360, 420, 560), "Full-precision weights\nlarger memory footprint\nbaseline inference cost", box_body, "#dbeafe")

    fit_text(draw, (590, 230, 910, 320), "Quantization Pipeline", box_title, "#f8fafc")
    fit_text(
        draw,
        (600, 320, 900, 560),
        "Terminal or scripted quantization\nNF4 / FP4 support\nconfigurable compute + storage dtype\nlocal save or hub upload",
        box_body,
        "#e2e8f0",
    )
    fit_text(draw, (595, 560, 905, 650), "Reduced size | lower memory | efficient inference", metric_font, "#67e8f9")

    fit_text(draw, (1070, 265, 1410, 360), "Deployment Artifact", box_title, "#f8fafc")
    fit_text(draw, (1080, 360, 1400, 560), "Quantized model package\nportable upload target\nresource-constrained friendly", box_body, "#dcfce7")

    arrow(draw, (450, 440), (540, 440), "#64748b")
    arrow(draw, (940, 440), (1030, 440), "#64748b")

    footer = (80, 735, 1420, 835)
    rounded_box(draw, footer, 18, "#111827", "#334155", width=2)
    fit_text(draw, footer, "Designed for reproducible optimization workflows, not manual notebook screenshots.", metric_font, "#e5e7eb")

    img.save(OUTPUT)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
