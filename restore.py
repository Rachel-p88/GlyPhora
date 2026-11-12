"""Adaptive document restoration. Returns (restored_bgr, steps). Never mutates input."""
import cv2, numpy as np

def analyze(gray):
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=.25, fy=.25), 31)
    return dict(contrast=float(gray.std()), mean=float(gray.mean()),
                sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                shadow=float(bg.std()),
                noise=float(np.median(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.5)))))

def is_document(gray):
    """True when most pixels are plain paper background (text on a light page), not a photograph."""
    sm = cv2.resize(gray, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return float(((bg.astype(np.int16) - sm) < 35).mean()) > 0.70 and np.median(sm) > 130


def deskew(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    m = cv2.threshold(
        cv2.GaussianBlur(g, (5, 5), 0),
        0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    m = cv2.dilate(
        m,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    )

    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        _, (w, h), angle = cv2.minAreaRect(c)

        if max(w, h) < 0.15 * img.shape[1]:
            continue

        if max(w, h) / max(1, min(w, h)) < 4:
            continue

        if w < h:
            angle -= 90

        angle = (angle + 90) % 180 - 90
        candidates.append((angle, max(w, h)))

    if len(candidates) < 3:
        return img, 0.0

    angles = np.array([a for a, _ in candidates])
    median_angle = float(np.median(angles))

    consistent = [
        (a, weight)
        for a, weight in candidates
        if abs(a - median_angle) <= 1.2
    ]

    if len(consistent) < 3:
        return img, 0.0

    if len(consistent) / len(candidates) < 0.6:
        return img, 0.0

    angle = float(np.average(
        [a for a, _ in consistent],
        weights=[weight for _, weight in consistent]
    ))

    if abs(angle) < 0.4 or abs(angle) > 12:
        return img, 0.0

    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2), angle, 1
    )

    corrected = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected, angle


def flatten(img):
    """Divide by estimated background to remove shadows/uneven light."""
    k = max(31, (min(img.shape[:2]) // 20) | 1)
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return cv2.divide(img, bg, scale=240)

SHARP_FLOOR = 0.0
# Ablation on real HCCD images (20/condition): sharpen and CLAHE raised MSE on blur images and gave no
# measurable benefit elsewhere, so both are off unless requested. Flatten and brightness normalization helped.
DEFAULT_SKIP = frozenset({"sharpen", "clahe"})

def stretch(g):
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

def restore(img, do_deskew=True, skip=DEFAULT_SKIP):
    steps = []
    out = img.copy()
    g0 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # Document gate is evaluated on a brightness-stretched copy so dark (low-light) pages are not mistaken for photos.
    if not is_document(stretch(g0)): return out, ["not a text-on-paper page: preserved"]
    p95 = float(np.percentile(g0, 95))
    if p95 < 190 and "brightness" not in skip:  # under-exposed: scale paper brightness up
        out = np.clip(out.astype(np.float32) * (235.0 / max(p95, 1)), 0, 255).astype(np.uint8)
        steps.append("brightness normalization")
    g0 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    a = analyze(g0)
    if do_deskew:
        out, ang = deskew(out)
        if ang: steps.append(f"deskew {ang:.1f} deg")
    if "flatten" not in skip and a["shadow"] > 12 and np.percentile(g0, 90) > 170:
        out = flatten(out); steps.append("illumination correction")
    if "denoise" not in skip and a["noise"] > 2.2 and a["sharp"] < 3000:
        out = cv2.fastNlMeansDenoisingColored(out, None, 4, 4, 7, 21); steps.append("denoise")
    g2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if "clahe" not in skip and np.percentile(g2, 95) - np.percentile(g2, 2) < 80:  # low dynamic range only
        l, A, B = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, A, B]), cv2.COLOR_LAB2BGR); steps.append("local contrast (CLAHE)")
    # Unsharp mask helps light/moderate blur; very strong blur cannot be recovered by filtering and is left alone.
    if "sharpen" not in skip and a["sharp"] < 150 and a["sharp"] > SHARP_FLOOR:
        out = cv2.addWeighted(out, 2.5, cv2.GaussianBlur(out, (0, 0), 2.0), -1.5, 0); steps.append("sharpen")
    if not steps: steps.append("already clear: preserved")
    return out, steps

def ocr_prep(img, target_w=2000):
    """OCR-only representation (grayscale, normalised, resized). Never shown to users."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = target_w / g.shape[1]
    if s > 1.15 or s < 0.7:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)


def _glyph_height(g):
    b = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    n, _, st, _ = cv2.connectedComponentsWithStats(b)
    hs = [st[i, 3] for i in range(1, n) if 3 < st[i, 3] < g.shape[0] / 8 and st[i, 4] > 8]
    return float(np.median(hs)) if hs else None

def lighting_unevenness(g):
    """Std-dev of the estimated paper background (same measure the restorer uses for shadows)."""
    return float(cv2.medianBlur(cv2.resize(g, None, fx=.25, fy=.25), 31).std())

def ocr_prep_scaled(img, target=36, bgnorm="auto", binarize=False, lanczos=True):
    """OCR-only copy scaled so the median glyph is ~target px tall (Tesseract works best near 25-35 px)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if bgnorm == "auto": bgnorm = lighting_unevenness(g) > 12   # normalise only when lighting is uneven
    if bgnorm:
        k = max(31, (min(g.shape) // 25) | 1)
        bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
        g = cv2.divide(g, bg, scale=245)
    th = _glyph_height(g); s = float(np.clip(target / th, .5, 6)) if th else 1.0
    s = min(s, max(1.0, 4500 / g.shape[1]))  # keep output <= ~4500 px wide
    if abs(s - 1) > .1: g = cv2.resize(g, None, fx=s, fy=s, interpolation=(cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_CUBIC) if s > 1 else cv2.INTER_AREA)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    if binarize: g = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return g