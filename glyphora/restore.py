import cv2, numpy as np

def analyze(gray):
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=.25, fy=.25), 31)
    return dict(contrast=float(gray.std()), mean=float(gray.mean()),
                sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                shadow=float(bg.std()),
                noise=float(np.median(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.5)))))

def is_document(gray):
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
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    k = max(31, (min(img.shape[:2]) // 20) | 1)

    bg = cv2.morphologyEx(
        l,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)

    corrected = cv2.divide(l, bg, scale=200)

    corrected = np.clip(
        corrected.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(
        cv2.merge([corrected, a, b]),
        cv2.COLOR_LAB2BGR
    )

def correct_uneven_illumination(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    h, w = l.shape
    sigma = max(30, min(h, w) // 12)

    background = cv2.GaussianBlur(
        l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    background = np.maximum(background, 1)

    corrected = cv2.divide(
        l,
        background,
        scale=200
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    corrected = cv2.addWeighted(
        l, 0.35,
        corrected, 0.65,
        0
    )

    return cv2.cvtColor(
        cv2.merge([corrected, a, b]),
        cv2.COLOR_LAB2BGR
    )

SHARP_FLOOR = 0.0
DEFAULT_SKIP = frozenset()

def stretch(g):
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    return np.clip((g.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

def restore(img, do_deskew=True, skip=DEFAULT_SKIP):
    steps = []
    out = img.copy()
    g0 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    document_like = is_document(stretch(g0))
    p95 = float(np.percentile(g0, 95))

    if p95 < 150 and "brightness" not in skip:
        factor = min(1.10, 165.0 / max(p95, 1))

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a_channel, b_channel = cv2.split(lab)

        l = np.clip(
            l.astype(np.float32) * factor,
            0, 255
        ).astype(np.uint8)

        out = cv2.cvtColor(
            cv2.merge([l, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR
        )

        steps.append("mild brightness correction")
    g0 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    a = analyze(g0)
    print("Image analysis:", a)
    print("Document-like:", document_like)
    if do_deskew and document_like:
        out, ang = deskew(out)
        if ang:
            steps.append(f"deskew {ang:.1f} deg")
    if "flatten" not in skip and a["shadow"] > 18 and (
        document_like or a["contrast"] < 45
    ):
        out = flatten(out)
        steps.append("illumination correction")
    if "denoise" not in skip and a["noise"] > 2.2 and a["sharp"] < 3000:
        out = cv2.fastNlMeansDenoisingColored(out, None, 4, 4, 7, 21); steps.append("denoise")
    g2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if "clahe" not in skip:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, A, B = cv2.split(lab)

        local_contrast = cv2.Laplacian(l, cv2.CV_64F).var()

        if local_contrast < 150:
            clahe = cv2.createCLAHE(
                clipLimit=1.5,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(l)

            l = cv2.addWeighted(l, 0.65, enhanced, 0.35, 0)

            out = cv2.cvtColor(
                cv2.merge([l, A, B]),
                cv2.COLOR_LAB2BGR
            )

            steps.append("adaptive local contrast")

    if "sharpen" not in skip and 100 < a["sharp"] < 500:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.3, blurred, -0.3, 0)
        steps.append("mild sharpening")
    if not steps: steps.append("already clear: preserved")
    return out, steps

def ocr_prep(img, target_w=2000):
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
    return float(cv2.medianBlur(cv2.resize(g, None, fx=.25, fy=.25), 31).std())

def ocr_prep_scaled(img, target=36, bgnorm="auto", binarize=False, lanczos=True):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if bgnorm == "auto": bgnorm = lighting_unevenness(g) > 12
    if bgnorm:
        k = max(31, (min(g.shape) // 25) | 1)
        bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
        g = cv2.divide(g, bg, scale=245)
    th = _glyph_height(g); s = float(np.clip(target / th, .5, 6)) if th else 1.0
    s = min(s, max(1.0, 4500 / g.shape[1]))
    if abs(s - 1) > .1: g = cv2.resize(g, None, fx=s, fy=s, interpolation=(cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_CUBIC) if s > 1 else cv2.INTER_AREA)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    if binarize: g = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return g
