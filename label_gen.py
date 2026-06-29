"""
HEYBIKE / TrackFlow shipping-label compositor
Fixed for the NEW warehouse-floor base photo.

Fixes:
- Correct label corners
- No giant wrong patch over the label
- Only patches SHIP TO text and tracking code
- Keeps barcode locked from base image
- Keeps base.png path compatible with Render
- Applies slight blur/compression for realistic phone-photo look
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

# Correct label corners for the latest base.png
# Order: top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [640, 201],
    [889, 199],
    [918, 315],
    [608, 314],
], dtype=np.float32)

LABEL_W = 1000
LABEL_H = 540

# Only patch the right SHIP TO area
SHIP_TO_RECT = (490, 145, 930, 340)

# Only patch the old tracking code text, not the whole barcode section
TRACKING_CODE_RECT = (250, 398, 660, 445)

# Barcode area copied from the original base photo as the final step
BARCODE_LOCK_RECT = (85, 452, 930, 540)

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
    """Sample label paper color from clean areas of the flattened label."""
    flat = cv2.warpPerspective(
        base_bgr,
        _image_to_label_matrix(),
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC
    )

    sample_points = [
        (930, 80),
        (920, 370),
        (460, 135),
        (430, 375),
        (75, 360),
        (760, 390),
        (950, 430),
    ]

    samples = []

    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (230, 230, 233)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    gray = median.mean()
    final = median * 0.88 + gray * 0.12

    return tuple(int(v) for v in final)


def _paper_patch(size, paper_rgb, seed=7):
    """Create a realistic paper patch with light texture and soft edges."""
    w, h = size
    rng = np.random.default_rng(seed)

    patch = np.zeros((h, w, 4), dtype=np.uint8)

    patch[:, :, 0] = paper_rgb[0]
    patch[:, :, 1] = paper_rgb[1]
    patch[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 0.9, (h, w, 1))
    patch[:, :, :3] = np.clip(
        patch[:, :, :3].astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    alpha = np.ones((h, w), dtype=np.float32) * 255
    feather = 6

    for i in range(feather):
        a = 255 * ((i + 1) / feather)

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
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_label = _load_font(FONT_BOLD, 18)
    f_text = _load_font(FONT_REG, 17)
    f_text_bold = _load_font(FONT_BOLD, 17)
    f_tracking = _load_font(FONT_REG, 15)

    # -----------------------------
    # SHIP TO block only
    # -----------------------------
    _paste_paper_rect(overlay, SHIP_TO_RECT, paper_rgb, seed=11)

    tx = 520
    ty = 160

    draw.text((tx, ty), "SHIP TO:", font=f_label, fill=TEXT_COLOR + (255,))
    ty += 26

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

    total_chars = sum(len(x) for x in fields)

    if total_chars > 100 or len(fields) > 5:
        f_text_use = _load_font(FONT_REG, 15)
        f_first_use = _load_font(FONT_BOLD, 15)
        line_h = 18
    else:
        f_text_use = f_text
        f_first_use = f_text_bold
        line_h = 21

    max_w = 340

    for i, line in enumerate(fields):
        if not line:
            continue

        font = f_first_use if i == 0 else f_text_use

        ty = _draw_wrapped(
            draw,
            (tx, ty),
            line,
            font,
            TEXT_COLOR + (255,),
            max_w,
            line_h
        )

    # -----------------------------
    # Tracking code only
    # -----------------------------
    _paste_paper_rect(overlay, TRACKING_CODE_RECT, paper_rgb, seed=12)

    trk = normalize_tracking(tracking_number)

    draw.text(
        (270, 414),
        trk,
        font=f_tracking,
        fill=SUBTLE_TEXT_COLOR + (255,)
    )

    return overlay


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_overlay_to_image(overlay_rgba: Image.Image, base_shape):
    h, w = base_shape[:2]
    overlay_np = np.array(overlay_rgba)

    warped = cv2.warpPerspective(
        overlay_np,
        _label_to_image_matrix(),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

    warped = cv2.GaussianBlur(warped, (3, 3), 0.25)

    return warped


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    overlay_bgr = cv2.cvtColor(
        rgb.astype(np.uint8),
        cv2.COLOR_RGB2BGR
    ).astype(np.float32)

    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    if not APPLY_REALISM_PASS:
        return img_bgr

    h, w = img_bgr.shape[:2]

    img = cv2.GaussianBlur(img_bgr, (5, 5), 0.75)

    img_f = img.astype(np.float32)
    img_f = img_f * 0.978 + 128 * 0.022
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    small = cv2.resize(
        img,
        (int(w * 0.94), int(h * 0.94)),
        interpolation=cv2.INTER_AREA
    )

    img = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 74]
    )

    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def paste_locked_barcode_last(result_bgr: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = BARCODE_LOCK_RECT

    flat_pts = np.array([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]],
    ], dtype=np.float32)

    img_pts = cv2.perspectiveTransform(
        flat_pts,
        _label_to_image_matrix()
    ).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]

    return out


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

    paper_rgb = _sample_paper_color(base_bgr)

    overlay = build_label_overlay(
        recipient,
        tracking_number or "",
        paper_rgb
    )

    warped_overlay = warp_overlay_to_image(overlay, base_bgr.shape)

    result_bgr = alpha_composite_bgr(base_bgr, warped_overlay)

    # Apply global realism first
    result_bgr = apply_photo_realism(result_bgr)

    # Barcode restored as final operation
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
