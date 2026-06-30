"""
HEYBIKE / TrackFlow shipping-label compositor
Improved version for the warehouse-floor HEYBIKE base photo.

This version:
- loads base.png from the same folder as label_gen.py
- replaces the full small/blurry label with a cleaner slightly larger label
- keeps the same realistic perspective on the box
- keeps barcode static across generations
- generates clean SHIP FROM / SHIP TO / tracking content
- avoids the ugly patch look
"""

import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.png")

# Original label position in the current base photo.
# Used only to crop the original/static barcode area.
ORIGINAL_LABEL_CORNERS = np.array([
    [640, 201],
    [889, 199],
    [918, 315],
    [608, 314],
], dtype=np.float32)

# New slightly larger label position.
# This makes the label cleaner and easier to read.
LABEL_CORNERS = np.array([
    [594, 182],
    [935, 181],
    [968, 338],
    [562, 337],
], dtype=np.float32)

LABEL_W = 1000
LABEL_H = 520

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 28)
MUTED_TEXT = (80, 80, 85)
LINE_COLOR = (85, 85, 90)

# Fixed sender
SHIP_FROM_LINES = [
    "TrackFlow LTD",
    "2820 N Pulaski Rd",
    "Chicago, IL 60641",
    "United States",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_font(path: str, size: int):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)

    fallback_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]

    for fp in fallback_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)

    return ImageFont.load_default()


def _flat_to_image_matrix(corners: np.ndarray) -> np.ndarray:
    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, corners)


def _image_to_flat_matrix(corners: np.ndarray) -> np.ndarray:
    dst = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst)


def _sample_paper_color(base_bgr: np.ndarray):
    flat = cv2.warpPerspective(
        base_bgr,
        _image_to_flat_matrix(ORIGINAL_LABEL_CORNERS),
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC
    )

    points = [
        (70, 70),
        (930, 70),
        (80, 250),
        (930, 250),
        (930, 470),
        (500, 500),
    ]

    samples = []
    for x, y in points:
        b, g, r = flat[y, x]
        samples.append((int(r), int(g), int(b)))

    arr = np.array(samples, dtype=np.float32)
    med = np.median(arr, axis=0)

    gray = med.mean()
    final = med * 0.88 + gray * 0.12

    # keep it close to white paper
    final = np.clip(final + 8, 210, 245)

    return tuple(int(v) for v in final)


def normalize_tracking(tracking_number: str | None) -> str:
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)
    if not core:
        core = "Q8ZH7PDVRH"

    return f"TF-{core}"


def _draw_wrapped(draw, xy, text, font, fill, max_width, line_h):
    x, y = xy
    words = text.split(" ")
    line = ""

    for word in words:
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += line_h
            line = word

    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h

    return y


def _crop_static_barcode(base_bgr: np.ndarray) -> Image.Image:
    """
    Crop barcode from the original base photo after flattening the original label.
    This makes the barcode static across all generated images.
    """
    flat = cv2.warpPerspective(
        base_bgr,
        _image_to_flat_matrix(ORIGINAL_LABEL_CORNERS),
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC
    )

    # Barcode source area inside the original flattened label.
    # Crops mostly the barcode bars, not the whole label.
    x0, y0, x1, y1 = 105, 370, 930, 510
    crop = flat[y0:y1, x0:x1]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def _make_fallback_barcode(width: int, height: int) -> Image.Image:
    """
    Fallback if the source barcode crop is unusable.
    Static fake barcode, same every time.
    """
    img = Image.new("RGB", (width, height), (235, 236, 238))
    draw = ImageDraw.Draw(img)

    rng = np.random.default_rng(22)
    x = 8
    while x < width - 8:
        bar_w = int(rng.integers(2, 6))
        gap = int(rng.integers(1, 4))
        if rng.random() > 0.18:
            draw.rectangle([x, 6, x + bar_w, height - 8], fill=(25, 25, 25))
        x += bar_w + gap

    return img


# ---------------------------------------------------------------------------
# BUILD FULL LABEL
# ---------------------------------------------------------------------------

def build_full_label(base_bgr: np.ndarray, recipient: dict, tracking_number: str) -> Image.Image:
    paper_rgb = _sample_paper_color(base_bgr)

    rng = np.random.default_rng(5)

    # paper texture
    paper = np.zeros((LABEL_H, LABEL_W, 3), dtype=np.uint8)
    paper[:, :, 0] = paper_rgb[0]
    paper[:, :, 1] = paper_rgb[1]
    paper[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 1.0, (LABEL_H, LABEL_W, 1))
    paper = np.clip(paper.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    label = Image.fromarray(paper, "RGB")
    draw = ImageDraw.Draw(label)

    f_brand = _load_font(FONT_BOLD, 42)
    f_small = _load_font(FONT_REG, 18)
    f_label = _load_font(FONT_BOLD, 24)
    f_text = _load_font(FONT_REG, 23)
    f_text_bold = _load_font(FONT_BOLD, 23)
    f_tracking_label = _load_font(FONT_BOLD, 23)
    f_tracking = _load_font(FONT_BOLD, 23)

    margin = 46
    right = LABEL_W - margin

    # Header
    draw.text((margin, 30), "TrackFlow", font=f_brand, fill=TEXT_COLOR)

    one = "1 of 1"
    one_w = draw.textlength(one, font=f_small)
    draw.text((right - one_w, 44), one, font=f_small, fill=MUTED_TEXT)

    draw.line((margin, 88, right, 88), fill=LINE_COLOR, width=3)

    # Two columns
    col_top = 112
    col_bottom = 292
    split_x = LABEL_W // 2

    draw.line((split_x, col_top - 8, split_x, col_bottom), fill=LINE_COLOR, width=2)
    draw.line((margin, col_bottom, right, col_bottom), fill=LINE_COLOR, width=2)

    # SHIP FROM
    x_from = margin
    y = col_top
    draw.text((x_from, y), "SHIP FROM:", font=f_label, fill=TEXT_COLOR)
    y += 34

    for line in SHIP_FROM_LINES:
        draw.text((x_from, y), line, font=f_text, fill=TEXT_COLOR)
        y += 29

    # SHIP TO
    x_to = split_x + 34
    y = col_top
    draw.text((x_to, y), "SHIP TO:", font=f_label, fill=TEXT_COLOR)
    y += 34

    fields = [
        recipient.get("name", ""),
        recipient.get("line1", ""),
    ]

    postal = recipient.get("line3", "")
    city = recipient.get("line2", "")

    if postal and city:
        fields.append(f"{postal} {city}")
    elif postal or city:
        fields.append(postal or city)

    country = recipient.get("line4", "")
    if country:
        fields.append(country)

    phone = recipient.get("phone", "")
    if phone:
        fields.append(phone)

    max_w = right - x_to
    line_h = 29

    for i, line in enumerate(fields):
        if not line:
            continue

        font = f_text_bold if i == 0 else f_text
        y = _draw_wrapped(draw, (x_to, y), line, font, TEXT_COLOR, max_w, line_h)

    # Tracking section
    y_track = 320
    draw.text((margin, y_track), "TRACKING NUMBER:", font=f_tracking_label, fill=TEXT_COLOR)

    trk = normalize_tracking(tracking_number)
    trk_w = draw.textlength(trk, font=f_tracking)
    draw.text((right - trk_w, y_track), trk, font=f_tracking, fill=TEXT_COLOR)

    # Barcode
    barcode_target = (80, 365, 790, 470)
    barcode_w = barcode_target[2] - barcode_target[0]
    barcode_h = barcode_target[3] - barcode_target[1]

    try:
        barcode = _crop_static_barcode(base_bgr)
        barcode = barcode.resize((barcode_w, barcode_h), Image.LANCZOS)
    except Exception:
        barcode = _make_fallback_barcode(barcode_w, barcode_h)

    label.paste(barcode, (barcode_target[0], barcode_target[1]))

    # Bottom subtle line
    draw.line((margin, 492, right, 492), fill=(125, 125, 128), width=2)

    return label


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_and_composite(base_bgr: np.ndarray, label_img: Image.Image) -> np.ndarray:
    h, w = base_bgr.shape[:2]

    label_rgb = np.array(label_img)
    label_bgr = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2BGR)

    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, LABEL_CORNERS)

    warped_label = cv2.warpPerspective(
        label_bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

    mask = np.ones((LABEL_H, LABEL_W), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # subtle soft edge, not too much
    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 1.1)
    alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]

    # subtle shadow under label
    shadow_mask = cv2.GaussianBlur(warped_mask, (13, 13), 4)
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0)[:, :, None] * 0.10

    darkened = base_bgr.astype(np.float32) * (1 - shadow_alpha)
    base_shadow = np.clip(darkened, 0, 255).astype(np.uint8)

    out = base_shadow.astype(np.float32) * (1 - alpha) + warped_label.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def apply_final_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Light realism only. Not heavy blur.
    """
    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.12)

    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_label(
    recipient: dict,
    tracking_number: str | None = None,
    out_path: str | None = None
):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)

    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    label_img = build_full_label(base_bgr, recipient, tracking_number or "")
    result_bgr = warp_and_composite(base_bgr, label_img)
    result_bgr = apply_final_realism(result_bgr)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if out_path:
        ext = os.path.splitext(out_path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            result_img.save(out_path, quality=95, subsampling=0)
        else:
            result_img.save(out_path)

    return result_img


if __name__ == "__main__":
    img = generate_label(
        {
            "name": "Helen Deli",
            "line1": "Rosenweg 7",
            "line2": "Bad Mergentheim",
            "line3": "97980",
            "line4": "DE",
            "phone": ""
        },
        tracking_number="TF-Q8ZH7PDVRH",
        out_path="test_heybike_output.png",
    )

    print("done", img.size)
