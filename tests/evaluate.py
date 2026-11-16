
"""Repeatable OCR and image restoration evaluation.
Run: python -m tests.evaluate
"""

import cv2
import numpy as np
import time
import os

from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from glyphora.restore import restore, ocr_prep
from glyphora.ocr import extract


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(
                p[j] + 1,
                c[-1] + 1,
                p[j - 1] + (x != y)
            ))
        p = c
    return p[-1]


def cer(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return lev(reference, hypothesis) / len(reference)


def wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()

    if not r:
        return 0.0 if not h else float("inf")

    return lev(r, h) / len(r)


def exact_match(reference, hypothesis):
    return int(reference == hypothesis)


def normalize(text):
    return " ".join(text.split())


def image_metrics(reference, restored):
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)

    if ref.shape != out.shape:
        raise ValueError(
            "Reference and restored images must have identical dimensions."
        )

    if np.array_equal(ref, out):
        psnr = float("inf")
    else:
        psnr = peak_signal_noise_ratio(
            ref, out, data_range=255
        )

    ssim = structural_similarity(
        ref, out, data_range=255
    )

    return {
        "PSNR": psnr,
        "SSIM": ssim
    }


REF = [
    "Annual Report 2023",
    "",
    "The committee met on 14 March 2023 and approved a budget of $4,250.75 for",
    "the library renovation. Total attendance was 128 (up 12% from 2022).",
    "",
    "Section 2: Findings",
    "",
    "Students reported higher satisfaction, particularly in reading rooms,",
    "where noise levels fell by 3.5 dB. See Table 4, p. 17 for details."
]



def load_font(size=30):
    candidates = [
        r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arial.ttf"
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def page():
    f = load_font(30)

    im = Image.new("RGB", (1500, 560), "white")
    d = ImageDraw.Draw(im)

    for i, line in enumerate(REF):
        d.text(
            (60, 40 + i * 52),
            line,
            fill=(20, 20, 20),
            font=f
        )

    return cv2.cvtColor(
        np.array(im),
        cv2.COLOR_RGB2BGR
    )


def degrade(img, kind):
    h, w = img.shape[:2]
    rng = np.random.RandomState(1)
    out = img.astype(np.float32)

    if kind == "clean":
        return img.copy()

    elif kind == "blur":
        out = cv2.GaussianBlur(out, (0, 0), 2.2)

    elif kind == "faded":
        out = 255 - (255 - out) * 0.25 - 8

    elif kind == "shadow":
        illumination = np.tile(
            np.linspace(1, 0.38, w, dtype=np.float32),
            (h, 1)
        )
        out = out * illumination[..., None]

    elif kind == "noisy_skew":
        out = out + rng.normal(0, 18, out.shape)
        matrix = cv2.getRotationMatrix2D(
            (w / 2, h / 2), 3, 1
        )
        out = cv2.warpAffine(
            out,
            matrix,
            (w, h),
            borderValue=(255, 255, 255)
        )

    elif kind == "lowres":
        small = cv2.resize(
            out,
            (w // 3, h // 3),
            interpolation=cv2.INTER_AREA
        )
        out = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    return np.clip(out, 0, 255).astype(np.uint8)


def evaluate():
    reference = normalize("\n".join(REF))
    base = page()

    variants = {
        "raw": lambda im, rest: cv2.cvtColor(
            im, cv2.COLOR_BGR2GRAY
        ),
        "prep": lambda im, rest: ocr_prep(im),
        "restored": lambda im, rest: cv2.cvtColor(
            rest, cv2.COLOR_BGR2GRAY
        ),
        "restored_prep": lambda im, rest: ocr_prep(rest)
    }

    totals = {
        name: {
            "char_errors": 0,
            "characters": 0,
            "word_errors": 0,
            "words": 0,
            "exact": 0,
            "count": 0,
            "ocr_time": 0.0
        }
        for name in variants
    }

    print("\nGLYPHORA OCR EVALUATION")
    print("=" * 90)

    print(
        f"{'Case':14}"
        f"{'Variant':18}"
        f"{'CER %':>9}"
        f"{'WER %':>9}"
        f"{'Exact %':>10}"
        f"{'OCR sec':>10}"
        f"{'Total sec':>11}"
    )

    for kind in [
        "clean",
        "blur",
        "faded",
        "shadow",
        "noisy_skew",
        "lowres"
    ]:
        image = degrade(base, kind)

        t = time.perf_counter()
        restored, steps = restore(image)
        restoration_time = time.perf_counter() - t

        degraded_metrics = image_metrics(base, image)
        restored_metrics = image_metrics(base, restored)

        print("\n" + "-" * 90)

        for name, prepare in variants.items():
            processed = prepare(image, restored)

            t = time.perf_counter()
            hypothesis = normalize(extract(processed))
            ocr_time = time.perf_counter() - t

            char_errors = lev(reference, hypothesis)
            word_errors = lev(
                reference.split(),
                hypothesis.split()
            )

            char_count = len(reference)
            word_count = len(reference.split())

            c = cer(reference, hypothesis)
            w = wer(reference, hypothesis)
            exact = exact_match(reference, hypothesis)

            total_time = ocr_time

            if name in ("restored", "restored_prep"):
                total_time += restoration_time

            result = totals[name]

            result["char_errors"] += char_errors
            result["characters"] += char_count
            result["word_errors"] += word_errors
            result["words"] += word_count
            result["exact"] += exact
            result["count"] += 1
            result["ocr_time"] += ocr_time

            print(
                f"{kind:14}"
                f"{name:18}"
                f"{100*c:9.2f}"
                f"{100*w:9.2f}"
                f"{100*exact:10.1f}"
                f"{ocr_time:10.2f}"
                f"{total_time:11.2f}"
            )

        print(f"Restoration steps: {', '.join(steps)}")

        print(
            f"Image quality | "
            f"Degraded PSNR: {degraded_metrics['PSNR']:.2f} | "
            f"Restored PSNR: {restored_metrics['PSNR']:.2f} | "
            f"Degraded SSIM: {degraded_metrics['SSIM']:.4f} | "
            f"Restored SSIM: {restored_metrics['SSIM']:.4f}"
        )

    print("\nAGGREGATE OCR RESULTS")
    print("=" * 75)

    print(
        f"{'Variant':18}"
        f"{'Corpus CER %':>16}"
        f"{'Corpus WER %':>16}"
        f"{'Exact Match %':>17}"
    )

    for name, result in totals.items():
        corpus_cer = (
            result["char_errors"] /
            result["characters"]
        )

        corpus_wer = (
            result["word_errors"] /
            result["words"]
        )

        exact_rate = (
            result["exact"] /
            result["count"]
        )

        print(
            f"{name:18}"
            f"{100*corpus_cer:16.2f}"
            f"{100*corpus_wer:16.2f}"
            f"{100*exact_rate:17.2f}"
        )


if __name__ == "__main__":
    evaluate()
