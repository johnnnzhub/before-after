"""Motor de composição de imagens — renderização PIL com tipografia Inter."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Font system
# ---------------------------------------------------------------------------
FONTS_DIR = Path(__file__).parent / "fonts"

_WEIGHT_MAP = {
    300: "Inter-Light.ttf",
    400: "Inter-Regular.ttf",
    500: "Inter-Medium.ttf",
    600: "Inter-SemiBold.ttf",
    700: "Inter-Bold.ttf",
}

_font_cache: dict[tuple[int, int], ImageFont.FreeTypeFont] = {}


def _load_font(weight: int = 400, size: int = 32) -> ImageFont.FreeTypeFont:
    key = (weight, size)
    if key in _font_cache:
        return _font_cache[key]
    path = FONTS_DIR / _WEIGHT_MAP.get(weight, "Inter-Regular.ttf")
    try:
        font = ImageFont.truetype(str(path), size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
CANVAS_BG = (10, 10, 10)
TEXT_PRIMARY = (255, 255, 255, 235)   # 92% opacity
TEXT_SECONDARY = (160, 160, 160, 100) # watermark
ACCENT = (200, 255, 0)               # #C8FF00
PILL_BG = (26, 26, 26, 220)
PILL_BORDER = (51, 51, 51, 153)

BF_COLORS = {
    "blue":   (0, 212, 170),   # atleta/essencial
    "green":  (200, 255, 0),   # fitness
    "yellow": (255, 184, 0),   # normal
    "red":    (255, 68, 68),   # acima
}


class OutputTemplate(Enum):
    LABELED = "labeled"
    CLEAN = "clean"
    SQUARE = "square"
    PORTRAIT = "portrait"
    WITH_DATES = "with_dates"
    STORY = "story"


# ---------------------------------------------------------------------------
# Text helpers (letter-spacing support)
# ---------------------------------------------------------------------------
def _measure_spaced(text: str, font: ImageFont.FreeTypeFont, spacing_em: float = 0.0) -> int:
    spacing_px = spacing_em * font.size
    total = 0.0
    for i, ch in enumerate(text):
        bbox = font.getbbox(ch)
        total += bbox[2] - bbox[0]
        if i < len(text) - 1:
            total += spacing_px
    return int(total)


def _draw_spaced(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    spacing_em: float = 0.0,
):
    x, y = pos
    spacing_px = spacing_em * font.size
    for i, ch in enumerate(text):
        draw.text((x, y), ch, font=font, fill=fill)
        bbox = font.getbbox(ch)
        x += (bbox[2] - bbox[0]) + (spacing_px if i < len(text) - 1 else 0)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------
def _gradient_bar(width: int, height: int, alpha_top: int = 0, alpha_bottom: int = 200) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    alphas = np.linspace(alpha_top, alpha_bottom, height, dtype=np.uint8)
    arr[:, :, 3] = alphas[:, np.newaxis]
    return Image.fromarray(arr, "RGBA")


def _gradient_bar_inverted(width: int, height: int, alpha_top: int = 220, alpha_bottom: int = 0) -> Image.Image:
    return _gradient_bar(width, height, alpha_top, alpha_bottom)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _resize_match_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    scale = target_h / img.height
    new_w = int(img.width * scale)
    return img.resize((new_w, target_h), Image.LANCZOS)


def _round_corners(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, img.width, img.height], radius=radius, fill=255)
    result = img.copy()
    result.putalpha(mask)
    return result


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def _draw_label_bar(photo: Image.Image, label: str) -> Image.Image:
    """Add gradient label bar at bottom of a single photo."""
    w, h = photo.size
    bar_h = max(40, int(h * 0.12))

    rgba = photo.convert("RGBA")
    gradient = _gradient_bar(w, bar_h)
    rgba.paste(Image.alpha_composite(
        Image.new("RGBA", (w, bar_h), (0, 0, 0, 0)),
        gradient,
    ), (0, h - bar_h), gradient)

    draw = ImageDraw.Draw(rgba)
    font_size = max(16, int(h * 0.028))
    font = _load_font(600, font_size)
    text = label.upper()
    spacing = 0.15

    tw = _measure_spaced(text, font, spacing)
    tx = (w - tw) // 2
    ty = h - bar_h + int(bar_h * 0.6) - font_size // 2

    _draw_spaced(draw, (tx, ty), text, font, TEXT_PRIMARY, spacing)
    return rgba


def _draw_watermark(canvas: Image.Image, text: str = "@cobaiateam") -> Image.Image:
    w, h = canvas.size
    rgba = canvas.convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_size = max(12, int(h * 0.012))
    font = _load_font(400, font_size)
    spacing = 0.05

    tw = _measure_spaced(text, font, spacing)
    x = w - tw - int(w * 0.025)
    y = h - font_size - int(h * 0.015)

    _draw_spaced(draw, (x, y), text, font, TEXT_SECONDARY, spacing)
    return Image.alpha_composite(rgba, layer).convert("RGB")


def _draw_date_pill(photo: Image.Image, date_text: str, y_ratio: float = 0.18) -> Image.Image:
    w, h = photo.size
    rgba = photo.convert("RGBA") if photo.mode != "RGBA" else photo.copy()
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_size = max(12, int(h * 0.016))
    font = _load_font(500, font_size)

    bbox = font.getbbox(date_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pill_h = max(24, int(h * 0.035))
    pill_w = tw + int(w * 0.036)
    px = (w - pill_w) // 2
    py = h - int(h * y_ratio) - pill_h

    draw.rounded_rectangle(
        [px, py, px + pill_w, py + pill_h],
        radius=int(pill_h * 0.45),
        fill=PILL_BG,
        outline=PILL_BORDER,
        width=1,
    )

    text_x = px + (pill_w - tw) // 2
    text_y = py + (pill_h - th) // 2 - 1
    draw.text((text_x, text_y), date_text, font=font, fill=(255, 255, 255, 217))

    return Image.alpha_composite(rgba, layer)


# ---------------------------------------------------------------------------
# BF banner
# ---------------------------------------------------------------------------
def _draw_bf_banner(
    canvas: Image.Image,
    bf_after: Optional[dict],
) -> Image.Image:
    """Draw BF% banner at the top — only 'depois' value."""
    if not bf_after:
        return canvas

    w, h = canvas.size
    rgba = canvas.convert("RGBA")
    banner_h = max(60, int(h * 0.08))

    gradient = _gradient_bar_inverted(w, banner_h)
    rgba.paste(Image.alpha_composite(
        Image.new("RGBA", (w, banner_h), (0, 0, 0, 0)),
        gradient,
    ), (0, 0), gradient)

    draw = ImageDraw.Draw(rgba)

    val_size = max(18, int(h * 0.032))
    cat_size = max(10, int(h * 0.014))
    val_font = _load_font(700, val_size)
    cat_font = _load_font(500, cat_size)

    center_y = banner_h // 2

    # Main value
    bf_text = f"{bf_after['ensemble_bf']}%"
    cat_label = bf_after["category"]["label"].upper()
    cat_color_key = bf_after["category"]["color"]
    cat_color = BF_COLORS.get(cat_color_key, (255, 255, 255))

    # Measure for centering
    val_bb = val_font.getbbox(bf_text)
    val_w = val_bb[2] - val_bb[0]
    val_h = val_bb[3] - val_bb[1]

    cat_bb = cat_font.getbbox(cat_label)
    cat_w = cat_bb[2] - cat_bb[0]

    gap = int(w * 0.015)
    total_w = val_w + gap + cat_w
    x = (w - total_w) // 2

    # Draw value
    y_val = center_y - val_h // 2
    draw.text((x, y_val), bf_text, font=val_font, fill=(255, 255, 255, 235))

    # Draw category tag next to value
    x_cat = x + val_w + gap
    y_cat = center_y - (cat_bb[3] - cat_bb[1]) // 2
    draw.text((x_cat, y_cat), cat_label, font=cat_font, fill=(*cat_color, 220))

    return rgba.convert("RGB")


# ---------------------------------------------------------------------------
# Story layout (9:16 vertical stack)
# ---------------------------------------------------------------------------
def _compose_story(
    before_pil: Image.Image,
    after_pil: Image.Image,
    label_before: str,
    label_after: str,
) -> Image.Image:
    cw, ch = 1080, 1920
    pad_x = 40
    photo_w = cw - 2 * pad_x  # 1000px
    photo_h = int(ch * 0.42)  # ~806px
    gap = 12
    corner_r = 12

    canvas = Image.new("RGB", (cw, ch), CANVAS_BG)

    # Resize and crop photos to fit
    for img, label, y_offset in [
        (before_pil, label_before, int(ch * 0.05)),
        (after_pil, label_after, int(ch * 0.05) + photo_h + gap),
    ]:
        resized = img.resize((photo_w, photo_h), Image.LANCZOS)
        with_label = _draw_label_bar(resized, label)
        rounded = _round_corners(with_label, corner_r)
        canvas.paste(rounded, (pad_x, y_offset), rounded)

    # Optional accent line between photos
    accent_w = int(cw * 0.3)
    accent_x = (cw - accent_w) // 2
    accent_y = int(ch * 0.05) + photo_h + gap // 2
    line_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(line_layer)
    draw.line([(accent_x, accent_y), (accent_x + accent_w, accent_y)],
              fill=(*ACCENT, 77), width=2)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), line_layer).convert("RGB")

    return canvas


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------
def compose(
    before: np.ndarray,
    after: np.ndarray,
    template: OutputTemplate = OutputTemplate.LABELED,
    label_before: str = "Antes",
    label_after: str = "Depois",
    date_before: Optional[str] = None,
    date_after: Optional[str] = None,
    bf_after: Optional[dict] = None,
) -> Image.Image:
    """Compose final output image. Returns PIL Image (RGB)."""
    before_pil = _bgr_to_pil(before)
    after_pil = _bgr_to_pil(after)

    # Story has its own layout
    if template == OutputTemplate.STORY:
        canvas = _compose_story(before_pil, after_pil, label_before, label_after)
        canvas = _draw_watermark(canvas)
        if bf_after:
            canvas = _draw_bf_banner(canvas, bf_after)
        return canvas

    # --- Side-by-side templates ---
    # Match heights
    target_h = min(before_pil.height, after_pil.height)
    before_pil = _resize_match_height(before_pil, target_h)
    after_pil = _resize_match_height(after_pil, target_h)

    total_w = before_pil.width + after_pil.width
    gap_w = max(2, int(total_w * 0.004))
    canvas_w = before_pil.width + gap_w + after_pil.width
    canvas_h = target_h

    # Instagram fixed sizes
    if template == OutputTemplate.SQUARE:
        canvas_w, canvas_h = 1080, 1080
    elif template == OutputTemplate.PORTRAIT:
        canvas_w, canvas_h = 1080, 1350

    canvas = Image.new("RGB", (canvas_w, canvas_h), CANVAS_BG)

    if template in (OutputTemplate.SQUARE, OutputTemplate.PORTRAIT):
        # Resize photos to fit within fixed canvas
        avail_w = (canvas_w - gap_w) // 2
        before_fit = before_pil.copy()
        after_fit = after_pil.copy()

        # Scale to fit width
        scale_b = avail_w / before_fit.width
        scale_a = avail_w / after_fit.width
        scale = min(scale_b, scale_a)

        new_w = int(before_fit.width * scale)
        new_h_b = int(before_fit.height * scale)
        new_h_a = int(after_fit.height * scale)
        new_h = min(new_h_b, new_h_a)

        before_fit = before_fit.resize((new_w, new_h), Image.LANCZOS)
        after_fit = after_fit.resize((new_w, new_h), Image.LANCZOS)

        # Center crop if taller than canvas
        if new_h > canvas_h:
            crop_top = (new_h - canvas_h) // 2
            before_fit = before_fit.crop((0, crop_top, new_w, crop_top + canvas_h))
            after_fit = after_fit.crop((0, crop_top, new_w, crop_top + canvas_h))
            new_h = canvas_h

        # Center on canvas
        x_before = (canvas_w // 2 - gap_w // 2 - new_w)
        x_after = canvas_w // 2 + gap_w // 2
        if x_before < 0:
            x_before = 0
            x_after = new_w + gap_w
        y_offset = (canvas_h - new_h) // 2

        # Apply labels before pasting
        if template != OutputTemplate.CLEAN:
            before_labeled = _draw_label_bar(before_fit, label_before)
            after_labeled = _draw_label_bar(after_fit, label_after)
            canvas.paste(before_labeled.convert("RGB"), (x_before, y_offset))
            canvas.paste(after_labeled.convert("RGB"), (x_after, y_offset))
        else:
            canvas.paste(before_fit, (x_before, y_offset))
            canvas.paste(after_fit, (x_after, y_offset))
    else:
        # Full-size side-by-side
        if template in (OutputTemplate.LABELED, OutputTemplate.WITH_DATES):
            before_labeled = _draw_label_bar(before_pil, label_before)
            after_labeled = _draw_label_bar(after_pil, label_after)
            canvas.paste(before_labeled.convert("RGB"), (0, 0))
            canvas.paste(after_labeled.convert("RGB"), (before_pil.width + gap_w, 0))
        else:
            canvas.paste(before_pil, (0, 0))
            canvas.paste(after_pil, (before_pil.width + gap_w, 0))

    # Date pills
    if template == OutputTemplate.WITH_DATES:
        if date_before:
            # Draw pill on the left photo area
            left_area = canvas.crop((0, 0, before_pil.width, canvas_h))
            left_with_pill = _draw_date_pill(left_area, date_before)
            canvas.paste(left_with_pill.convert("RGB"), (0, 0))
        if date_after:
            right_x = before_pil.width + gap_w
            right_area = canvas.crop((right_x, 0, canvas_w, canvas_h))
            right_with_pill = _draw_date_pill(right_area, date_after)
            canvas.paste(right_with_pill.convert("RGB"), (right_x, 0))

    # Watermark (skip for CLEAN)
    if template != OutputTemplate.CLEAN:
        canvas = _draw_watermark(canvas)

    # BF banner (only "depois" reference on the photo)
    if bf_after:
        canvas = _draw_bf_banner(canvas, bf_after)

    return canvas
