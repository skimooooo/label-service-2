import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.png")

# Correct label corners on the new base image
# order: TL, TR, BR, BL
LABEL_CORNERS = np.array([
    [640, 201],
    [889, 199],
    [918, 315],
    [608, 314],
], dtype=np.float32)

LABEL_W = 1000
LABEL_H = 540

# Dynamic editable regions in flat-label space
SHIP_TO_RECT = (540, 120, 905, 290)
TRACKING_CODE_RECT = (340, 392, 670, 430)

# Static regions to restore exactly from the original base photo
HEADER_LOCK_RECT = (0, 0, 1000, 95)          # TrackFlow header strip
SHIP_FROM_LOCK_RECT = (0, 95, 505, 310)      # Left sender block
BARCODE_LOCK_RECT = (110, 440, 930, 540)     # Barcode area

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (28, 28, 32)
SUBTLE_TEXT_COLOR = (70, 70, 74)


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


def _label_to_image_matrix():
    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, LABEL_CORNERS)


def _image_to_label_matrix():
    dst = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(LABEL_CORNERS, dst)


def _sample_paper_color(base_bgr: np.ndarray):
    flat = cv2.warpPerspective(
        base_bgr,
        _image_to_label_matrix(),
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC
    )

    sample_points = [
        (950, 80),
        (940, 360),
        (520, 350),
        (960, 470),
        (90, 385),
    ]

    samples = []
    for x, y in sample_points:
        b, g, r = flat[y, x]
        samples.append((int(r), int(g), int(b)))

    arr = np.array(samples, dtype=np.float32)
    med = np.median(arr, axis=0)
    gray = med.mean()
    final = med * 0.9 + gray * 0.1
    return tuple(int(v) for v in final)


def _paper_patch(size, paper_rgb, seed=7):
    w, h = size
    rng = np.random.default_rng(seed)

    patch = np.zeros((h, w, 4), dtype=np.uint8)
    patch[:, :, 0] = paper_rgb[0]
    patch[:, :, 1] = paper_rgb[1]
    patch[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 0.7, (h, w, 1))
    patch[:, :, :3] = np.clip(
        patch[:, :, :3].astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    alpha = np.ones((h, w), dtype=np.float32) * 252
    feather = 5
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


def normalize_tracking(tracking_number: str | None) -> str:
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)
    if not core:
        core = "Q8ZH7PDVRH"

    return f"TF-{core}"


def build_label_overlay(recipient: dict, tracking_number: str, paper_rgb):
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_label = _load_font(FONT_BOLD, 18)
    f_text = _load_font(FONT_REG, 16)
    f_text_bold = _load_font(FONT_BOLD, 16)
    f_tracking = _load_font(FONT_REG, 15)

    # SHIP TO only
    _paste_paper_rect(overlay, SHIP_TO_RECT, paper_rgb, seed=11)

    tx = 565
    ty = 145

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

    phone = recipient.get("phone", "")
    if phone:
        fields.append(phone)

    max_w = 300
    line_h = 20

    for i, line in enumerate(fields):
        if not line:
            continue
        font = f_text_bold if i == 0 else f_text
        ty = _draw_wrapped(draw, (tx, ty), line, font, TEXT_COLOR + (255,), max_w, line_h)

    # TRACKING CODE only (not barcode, not label title)
    _paste_paper_rect(overlay, TRACKING_CODE_RECT, paper_rgb, seed=12)
    trk = normalize_tracking(tracking_number)

    draw.text(
        (365, 403),
        trk,
        font=f_tracking,
        fill=SUBTLE_TEXT_COLOR + (255,)
    )

    return overlay


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

    # Very small blur only on overlay edges/text
    warped = cv2.GaussianBlur(warped, (3, 3), 0.15)
    return warped


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    overlay_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _restore_flat_region(result_bgr: np.ndarray, base_bgr: np.ndarray, rect):
    """
    Restore a region from the original base photo by:
    1. flattening current result and base into label space
    2. copying the original pixels for that rect
    3. warping back to image space
    """
    to_flat = _image_to_label_matrix()
    to_img = _label_to_image_matrix()

    result_flat = cv2.warpPerspective(result_bgr, to_flat, (LABEL_W, LABEL_H), flags=cv2.INTER_CUBIC)
    base_flat = cv2.warpPerspective(base_bgr, to_flat, (LABEL_W, LABEL_H), flags=cv2.INTER_CUBIC)

    x0, y0, x1, y1 = rect
    result_flat[y0:y1, x0:x1] = base_flat[y0:y1, x0:x1]

    restored = cv2.warpPerspective(result_flat, to_img, (base_bgr.shape[1], base_bgr.shape[0]), flags=cv2.INTER_CUBIC)

    # Build mask for only this restored region
    flat_mask = np.zeros((LABEL_H, LABEL_W), dtype=np.uint8)
    flat_mask[y0:y1, x0:x1] = 255
    mask = cv2.warpPerspective(flat_mask, to_img, (base_bgr.shape[1], base_bgr.shape[0]), flags=cv2.INTER_NEAREST)

    out = result_bgr.copy()
    out[mask > 0] = restored[mask > 0]
    return out


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Light realism only.
    We do NOT heavily blur the whole image anymore.
    """
    h, w = img_bgr.shape[:2]

    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.22)

    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def generate_label(recipient: dict, tracking_number: str | None = None, out_path: str | None = None):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_rgb = _sample_paper_color(base_bgr)
    overlay = build_label_overlay(recipient, tracking_number or "", paper_rgb)
    warped_overlay = warp_overlay_to_image(overlay, base_bgr.shape)

    result_bgr = alpha_composite_bgr(base_bgr, warped_overlay)

    # Restore static regions exactly
    result_bgr = _restore_flat_region(result_bgr, base_bgr, HEADER_LOCK_RECT)
    result_bgr = _restore_flat_region(result_bgr, base_bgr, SHIP_FROM_LOCK_RECT)
    result_bgr = _restore_flat_region(result_bgr, base_bgr, BARCODE_LOCK_RECT)

    # Very light realism after restore
    result_bgr = apply_photo_realism(result_bgr)

    # Restore static regions AGAIN after realism so they stay crisp/static
    result_bgr = _restore_flat_region(result_bgr, base_bgr, HEADER_LOCK_RECT)
    result_bgr = _restore_flat_region(result_bgr, base_bgr, SHIP_FROM_LOCK_RECT)
    result_bgr = _restore_flat_region(result_bgr, base_bgr, BARCODE_LOCK_RECT)

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
