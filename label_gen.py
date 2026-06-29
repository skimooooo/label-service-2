"""
HEYBIKE / TrackFlow shipping-label image compositor
Updated for the NEW warehouse-floor base photo.

This version:
- uses base.png
- is calibrated for the new base image
- keeps the barcode locked from the base image
- keeps SHIP FROM fixed from the base photo
- only redraws:
  1) SHIP TO block
  2) tracking row text
- applies a soft low-quality phone-photo look to hide AI-like flaws
"""

import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_IMAGE_PATH = "base.png"

# New label position for the latest base photo
# Order: top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [648, 164],
    [999, 166],
    [1008, 315],
    [641, 313],
], dtype=np.float32)

# Internal flat label coordinate system
LABEL_W = 1000
LABEL_H = 540

# Editable regions inside the flat label coordinates
# Right address block
SHIP_TO_RECT = (535, 92, 885, 235)

# Tracking row above barcode
TRACKING_ROW_RECT = (60, 250, 935, 322)

# Locked barcode area copied from base photo as final step
BARCODE_LOCK_RECT = (118, 334, 895, 492)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (28, 28, 32)
SUBTLE_TEXT_COLOR = (65, 65, 70)

APPLY_REALISM_PASS = True


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


def _label_to_image_matrix() -> np.ndarray:
    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, LABEL_CORNERS)


def _image_to_label_matrix() -> np.ndarray:
    dst = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(LABEL_CORNERS, dst)


def _sample_paper_color(base_bgr: np.ndarray):
    """
    Sample the white paper tone from clean blank areas of the label
    in flattened label coordinates.
    """
    M = _image_to_label_matrix()
    flat = cv2.warpPerspective(base_bgr, M, (LABEL_W, LABEL_H), flags=cv2.INTER_CUBIC)

    sample_points = [
        (135, 60),
        (910, 60),
        (85, 185),
        (930, 185),
        (90, 510),
        (920, 510),
        (955, 290),
    ]

    samples = []
    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (228, 229, 232)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    gray = median.mean()
    final = median * 0.85 + gray * 0.15
    return tuple(int(v) for v in final)


def _paper_patch(size, paper_rgb, seed=7):
    """
    Create a realistic paper patch with light texture and soft feathered edges.
    """
    w, h = size
    rng = np.random.default_rng(seed)

    patch = np.zeros((h, w, 4), dtype=np.uint8)
    patch[:, :, 0] = paper_rgb[0]
    patch[:, :, 1] = paper_rgb[1]
    patch[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 1.15, (h, w, 1))
    rgb = np.clip(patch[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    patch[:, :, :3] = rgb

    alpha = np.ones((h, w), dtype=np.float32) * 255.0
    feather = 7
    for i in range(feather):
        a = 255.0 * ((i + 1) / feather)
        alpha[i, :] = np.minimum(alpha[i, :], a)
        alpha[h - 1 - i, :] = np.minimum(alpha[h - 1 - i, :], a)
        alpha[:, i] = np.minimum(alpha[:, i], a)
        alpha[:, w - 1 - i] = np.minimum(alpha[:, w - 1 - i], a)

    patch[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(patch, "RGBA")


def _paste_paper_rect(overlay, rect, paper_rgb, seed):
    x0, y0, x1, y1 = rect
    patch = _paper_patch((x1 - x0, y1 - y0), paper_rgb, seed=seed)
    overlay.alpha_composite(patch, (x0, y0))


def _draw_wrapped(draw, xy, text, font, fill, max_width, line_h):
    x, y = xy
    words = text.split(" ")
    line = ""

    for word in words:
        test_line = (line + " " + word).strip()
        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += line_h
            line = word

    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h

    return y


def normalize_tracking(tracking_number: str | None) -> str:
    """
    Normalize to TF-XXXXXXXXXX style but supports alphanumeric codes too.
    """
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)
    if not core:
        core = "Q8ZH7PDVRH"

    return f"TF-{core}"


# ---------------------------------------------------------------------------
# OVERLAY BUILDING
# ---------------------------------------------------------------------------

def build_label_overlay(recipient: dict, tracking_number: str, paper_rgb):
    """
    Build transparent overlay that patches only the dynamic areas.
    """
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_label = _load_font(FONT_BOLD, 19)
    f_text = _load_font(FONT_REG, 17)
    f_text_bold = _load_font(FONT_BOLD, 17)
    f_tracking_label = _load_font(FONT_BOLD, 17)
    f_tracking = _load_font(FONT_REG, 15)

    # -------------------------------------------------------------
    # SHIP TO block
    # -------------------------------------------------------------
    _paste_paper_rect(overlay, SHIP_TO_RECT, paper_rgb, seed=11)

    tx = 555
    ty = 105
    draw.text((tx, ty), "SHIP TO:", font=f_label, fill=TEXT_COLOR + (255,))
    ty += 24

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

    if recipient.get("phone"):
        fields.append(recipient.get("phone", ""))

    total_chars = sum(len(x) for x in fields)
    if total_chars > 100 or len(fields) > 5:
        f_text_use = _load_font(FONT_REG, 15)
        f_text_first = _load_font(FONT_BOLD, 15)
        line_h = 18
    else:
        f_text_use = f_text
        f_text_first = f_text_bold
        line_h = 20

    max_w = 290
    for i, line in enumerate(fields):
        if not line:
            continue
        font = f_text_first if i == 0 else f_text_use
        ty = _draw_wrapped(draw, (tx, ty), line, font, TEXT_COLOR + (255,), max_w, line_h)

    # -------------------------------------------------------------
    # TRACKING ROW
    # -------------------------------------------------------------
    _paste_paper_rect(overlay, TRACKING_ROW_RECT, paper_rgb, seed=12)

    trk = normalize_tracking(tracking_number)

    x0, y0, x1, y1 = TRACKING_ROW_RECT
    row_y = y0 + 18

    draw.text((82, row_y), "TRACKING NO.:", font=f_tracking_label, fill=TEXT_COLOR + (255,))

    # draw the tracking code to the right, subtle printed style
    trk_x = 310
    draw.text((trk_x, row_y), trk, font=f_tracking, fill=SUBTLE_TEXT_COLOR + (255,))

    return overlay


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_overlay_to_image(overlay_rgba: Image.Image, base_shape):
    h, w = base_shape[:2]
    overlay_np = np.array(overlay_rgba)

    M = _label_to_image_matrix()
    warped = cv2.warpPerspective(
        overlay_np,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Tiny blur so the text looks printed/photo-like
    warped = cv2.GaussianBlur(warped, (3, 3), 0.3)
    return warped


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    overlay_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Slightly degrade the image so it feels like a real warehouse phone photo.
    """
    if not APPLY_REALISM_PASS:
        return img_bgr

    h, w = img_bgr.shape[:2]

    # blur a little
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0.75)

    # flatten contrast slightly
    img_f = img.astype(np.float32)
    img_f = img_f * 0.978 + 128 * 0.022
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # subtle scale softness
    small = cv2.resize(img, (int(w * 0.94), int(h * 0.94)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # JPEG recompression adds realistic harshness on zoom
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 74])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def paste_locked_barcode_last(result_bgr: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    """
    Copy the original barcode pixels from the base image as the FINAL step.
    """
    M = _label_to_image_matrix()
    x0, y0, x1, y1 = BARCODE_LOCK_RECT

    flat_pts = np.array([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]],
    ], dtype=np.float32)

    img_pts = cv2.perspectiveTransform(flat_pts, M).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]
    return out


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_label(recipient: dict, tracking_number: str | None = None, out_path: str | None = None):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_rgb = _sample_paper_color(base_bgr)
    overlay = build_label_overlay(recipient, tracking_number or "", paper_rgb)
    warped_overlay = warp_overlay_to_image(overlay, base_bgr.shape)

    result_bgr = alpha_composite_bgr(base_bgr, warped_overlay)

    # Apply global realism before locking barcode
    result_bgr = apply_photo_realism(result_bgr)

    # Final step: restore original barcode
    result_bgr = paste_locked_barcode_last(result_bgr, base_bgr)

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
