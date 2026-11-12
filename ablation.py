
import time
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from glyphora.restore import restore, deskew, flatten, ocr_prep
from glyphora.ocr import extract
from tests.evaluate import REF, page, degrade


OUTPUT = Path("ablation_results")
OUTPUT.mkdir(exist_ok=True)


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def levenshtein(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))

        previous = current

    return previous[-1]


def error_rate(reference, prediction):
    reference = normalize(reference)
    prediction = normalize(prediction)

    if not reference:
        return 0.0 if not prediction else 100.0

    return 100 * levenshtein(reference, prediction) / len(reference)


def word_error_rate(reference, prediction):
    ref = normalize(reference).split()
    pred = normalize(prediction).split()

    if not ref:
        return 0.0 if not pred else 100.0

    return 100 * levenshtein(ref, pred) / len(ref)


def apply_variant(img, variant):
    if variant == "original":
        return img.copy(), "none"

    if variant == "deskew":
        out, angle = deskew(img)
        return out, f"deskew {angle:.2f} deg"

    if variant == "illumination":
        return flatten(img), "illumination correction"

    if variant == "denoise":
        out = cv2.fastNlMeansDenoisingColored(
            img, None, 4, 4, 7, 21
        )
        return out, "denoise"

    if variant == "deskew_illumination":
        out, angle = deskew(img)
        out = flatten(out)
        return out, f"deskew {angle:.2f} deg + illumination correction"

    if variant == "deskew_denoise":
        out, angle = deskew(img)
        out = cv2.fastNlMeansDenoisingColored(
            out, None, 4, 4, 7, 21
        )
        return out, f"deskew {angle:.2f} deg + denoise"

    if variant == "full":
        out, steps = restore(img)
        return out, ", ".join(steps)

    raise ValueError(f"Unknown variant: {variant}")


def measure(reference, restored):
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)

    psnr = (
        float("inf")
        if np.array_equal(ref, out)
        else peak_signal_noise_ratio(ref, out, data_range=255)
    )

    ssim = structural_similarity(ref, out, data_range=255)

    return psnr, ssim


def run():
    clean = page()
    reference_text = "\n".join(REF)

    cases = {
        "shadow": degrade(clean, "shadow"),
        "noisy_skew": degrade(clean, "noisy_skew")
    }

    variants = [
        "original",
        "deskew",
        "illumination",
        "denoise",
        "deskew_illumination",
        "deskew_denoise",
        "full"
    ]

    print("\nGLYPHORA RESTORATION ABLATION")
    print("=" * 112)

    print(
        f"{'Case':<13} {'Variant':<23} {'CER %':>9} "
        f"{'WER %':>9} {'PSNR':>10} {'SSIM':>9} {'Time':>9}"
    )

    print("-" * 112)

    for case_name, degraded in cases.items():
        print(f"\n{case_name.upper()}")

        for variant in variants:
            start = time.perf_counter()

            restored, steps = apply_variant(degraded, variant)

            elapsed = time.perf_counter() - start

            ocr_start = time.perf_counter()
            prediction = extract(ocr_prep(restored))
            ocr_time = time.perf_counter() - ocr_start

            cer = error_rate(reference_text, prediction)
            wer = word_error_rate(reference_text, prediction)

            psnr, ssim = measure(clean, restored)

            filename = OUTPUT / f"{case_name}_{variant}.png"
            cv2.imwrite(str(filename), restored)

            psnr_text = "inf" if np.isinf(psnr) else f"{psnr:.2f}"

            print(
                f"{case_name:<13} {variant:<23} "
                f"{cer:>8.2f} {wer:>8.2f} "
                f"{psnr_text:>10} {ssim:>9.4f} "
                f"{elapsed + ocr_time:>8.2f}s"
            )

            print(f"    Operations: {steps}")

    print("\n" + "=" * 112)
    print(f"Restored images saved in: {OUTPUT.resolve()}")


if __name__ == "__main__":
    run()
