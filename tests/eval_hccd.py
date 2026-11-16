import argparse
import csv
import os
import re
import time
import statistics as st
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity as SSIM,
)

from glyphora.restore import restore


FOLDER_MAP = {
    ("BLUR", "LEVEL_1"): Path("Blur") / "Blur_level_1",
    ("BLUR", "LEVEL_2"): Path("Blur") / "Blur_level_2",
    ("BLUR", "LEVEL_3"): Path("Blur") / "Blur_Level_3",

    ("LOW_ILLUMINATION", "LEVEL_1"):
        Path("Low illumination") / "Low_level_1",
    ("LOW_ILLUMINATION", "LEVEL_2"):
        Path("Low illumination") / "Low_Level_2",
    ("LOW_ILLUMINATION", "LEVEL_3"):
        Path("Low illumination") / "Low_level_3",

    ("SHADOW", "LEVEL_1"):
        Path("Shadow") / "Shadow level 1",
    ("SHADOW", "LEVEL_2"):
        Path("Shadow") / "Shadow level 2",
}

EXT = (
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
)

DEFAULT_SKIP = frozenset({"sharpen", "clahe"})

VARIANTS = {
    "default": DEFAULT_SKIP,
    "+clahe": frozenset({"sharpen"}),
    "+sharpen": frozenset({"clahe"}),
}


def normalize_id(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_condition(value):
    value = re.sub(r"[^a-z]", "", str(value).lower())

    aliases = {
        "blur": "BLUR",
        "lowillumination": "LOW_ILLUMINATION",
        "shadow": "SHADOW",
    }

    return aliases.get(value, value.upper())


def normalize_level(value):
    value = re.sub(r"[^a-z0-9]", "", str(value).lower())

    if value.startswith("level"):
        return "LEVEL_" + value[5:]

    return value.upper()


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def resolve(base, relative):
    """Resolve folder names without case sensitivity."""
    current = Path(base)

    for part in relative.parts:
        if not current.is_dir():
            return None

        hit = next(
            (
                item for item in current.iterdir()
                if item.is_dir() and item.name.lower() == part.lower()
            ),
            None,
        )

        if hit is None:
            return None

        current = hit

    return current


def image_files(folder):
    return sorted(
        (
            path for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in EXT
        ),
        key=natural_key,
    )


def build_file_index(paths):
    """Map normalized filename stems to all matching files."""
    index = {}

    for path in paths:
        index.setdefault(normalize_id(path.stem), []).append(path)

    return index


def ink(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )[1]

    return binary > 0


def metrics(image, reference):
    gx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    a = ink(gx)
    b = ink(gr)

    union = np.logical_or(a, b).sum()

    mse = float(
        np.mean(
            (gx.astype(np.float64) - gr.astype(np.float64)) ** 2
        )
    )

    psnr = float("inf") if mse == 0 else float(PSNR(gr, gx, data_range=255))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": float(SSIM(gr, gx, data_range=255)),
        "iou": (
            float(np.logical_and(a, b).sum() / union)
            if union else 1.0
        ),
    }



def fit(image, reference_shape, max_side):
    height, width = reference_shape

    if image.shape[:2] != reference_shape:
        interpolation = (
            cv2.INTER_AREA
            if image.shape[0] > height or image.shape[1] > width
            else cv2.INTER_CUBIC
        )

        image = cv2.resize(
            image,
            (width, height),
            interpolation=interpolation,
        )

    if max_side and max(height, width) > max_side:
        scale = max_side / max(height, width)

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    return image



def aspect_ratio_difference(a, b):
    ratio_a = a.shape[1] / a.shape[0]
    ratio_b = b.shape[1] / b.shape[0]

    return abs(ratio_a - ratio_b) / max(ratio_a, ratio_b)


def load_metadata(path):
    if not path.is_file():
        raise SystemExit(f"Metadata CSV not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "Doc_ID",
            "Degradation_Type",
            "Degradation_Level",
            "Paired_Doc_ID",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise SystemExit(
                f"Metadata is missing required columns: {sorted(missing)}"
            )

        records = list(reader)

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--metadata", default="")

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum images per condition; 0 means all.",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N metadata records per condition.",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Maximum image side for metrics; 0 means full resolution.",
    )

    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Enable deskewing. Disabled by default for aligned HCCD pairs.",
    )

    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Compare default, CLAHE-enabled and sharpening-enabled variants.",
    )

    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated degradation types: BLUR,LOW_ILLUMINATION,SHADOW.",
    )

    parser.add_argument("--out", default="results_hccd")

    options = parser.parse_args()

    if options.limit < 0 or options.skip < 0 or options.max_side < 0:
        raise SystemExit("--limit, --skip and --max-side cannot be negative.")

    gt_root = Path(options.gt)
    synthetic_root = Path(options.synthetic)

    metadata_path = (
        Path(options.metadata)
        if options.metadata
        else synthetic_root.parent / "Metadata" / "Metadata.csv"
    )

    for label, folder in (
        ("Ground truth", gt_root),
        ("Synthetic", synthetic_root),
    ):
        if not folder.is_dir():
            raise SystemExit(f"{label} folder does not exist: {folder}")

    os.makedirs(options.out, exist_ok=True)

    print(f"Ground truth: {gt_root}")
    print(f"Synthetic:   {synthetic_root}")
    print(f"Metadata:    {metadata_path}")


    gt_files = image_files(gt_root)
    gt_index = build_file_index(gt_files)

    print(f"\nGround-truth image files: {len(gt_files)}")
    print(f"Unique normalized ground-truth IDs: {len(gt_index)}")

    duplicate_gt = {
        key: paths for key, paths in gt_index.items()
        if len(paths) > 1
    }

    if duplicate_gt:
        print(
            f"WARNING: {len(duplicate_gt)} duplicate normalized "
            "ground-truth IDs found."
        )

    if not gt_files:
        raise SystemExit("No ground-truth images were found.")


    metadata = load_metadata(metadata_path)

    print(f"Metadata records: {len(metadata)}")

    metadata_index = {}

    for record in metadata:
        doc_id = normalize_id(record["Doc_ID"])
        metadata_index.setdefault(doc_id, []).append(record)

    duplicate_metadata = {
        key: records for key, records in metadata_index.items()
        if len(records) > 1
    }

    print(f"Duplicate metadata IDs: {len(duplicate_metadata)}")


    requested_conditions = {
        normalize_condition(value)
        for value in options.conditions.split(",")
        if value.strip()
    }

    rows = []
    summaries = []
    validation = []

    deskew_enabled = options.deskew

    print(
        "\nDeskew:",
        "ON" if deskew_enabled else "OFF",
    )

    print(
        "Metadata-driven pairing: ENABLED"
    )

    print(
        "Sorted-order fallback: DISABLED"
    )


    for (degradation, level), relative_folder in FOLDER_MAP.items():

        if requested_conditions and degradation not in requested_conditions:
            continue

        folder = resolve(synthetic_root, relative_folder)

        if folder is None:
            message = (
                f"MISSING FOLDER: {degradation} {level}: "
                f"{relative_folder}"
            )
            print(message)
            validation.append(message)
            continue

        degraded_files = image_files(folder)
        degraded_index = build_file_index(degraded_files)

        condition_records = []

        for record in metadata:
            record_deg = normalize_condition(record["Degradation_Type"])
            record_lvl = normalize_level(record["Degradation_Level"])

            if record_deg == degradation and record_lvl == level:
                condition_records.append(record)

        condition_records.sort(
            key=lambda record: natural_key(
                Path(record["Doc_ID"] + ".jpg")
            )
        )

        metadata_ids = {
            normalize_id(record["Doc_ID"])
            for record in condition_records
        }

        orphan_files = [
            path for path in degraded_files
            if normalize_id(path.stem) not in metadata_ids
        ]

        selected_records = condition_records[
            options.skip:
            options.skip + options.limit if options.limit else None
        ]

        condition_rows = []

        counts = {
            "paired": 0,
            "missing_synthetic": 0,
            "missing_gt": 0,
            "ambiguous": 0,
            "unreadable": 0,
            "resized": 0,
        }

        for record in selected_records:

            doc_id = record["Doc_ID"]
            paired_id = record["Paired_Doc_ID"]

            doc_key = normalize_id(doc_id)
            gt_key = normalize_id(paired_id)

            synthetic_matches = degraded_index.get(doc_key, [])

            if len(synthetic_matches) == 0:
                counts["missing_synthetic"] += 1
                continue

            if len(synthetic_matches) > 1:
                counts["ambiguous"] += 1
                continue

            gt_matches = gt_index.get(gt_key, [])

            if len(gt_matches) == 0:
                counts["missing_gt"] += 1
                continue

            if len(gt_matches) > 1:
                counts["ambiguous"] += 1
                continue

            synthetic_path = synthetic_matches[0]
            gt_path = gt_matches[0]

            image = cv2.imread(str(synthetic_path))
            reference = cv2.imread(str(gt_path))

            if image is None or reference is None:
                counts["unreadable"] += 1
                continue

            if image.shape[:2] != reference.shape[:2]:
                counts["resized"] += 1

            start = time.perf_counter()

            restored, steps = restore(
                image,
                do_deskew=deskew_enabled,
                skip=DEFAULT_SKIP,
            )

            elapsed = time.perf_counter() - start

            if restored is None:
                counts["unreadable"] += 1
                continue

            reference_fit = fit(
                reference,
                reference.shape[:2],
                options.max_side,
            )

            degraded_fit = fit(
                image,
                reference.shape[:2],
                options.max_side,
            )

            restored_fit = fit(
                restored,
                reference.shape[:2],
                options.max_side,
            )

            row = {
                "degradation": degradation,
                "level": level,
                "doc_id": doc_id,
                "paired_doc_id": paired_id,
                "file": synthetic_path.name,
                "ground_truth_file": gt_path.name,
                "seconds": elapsed,
                "steps": "; ".join(steps),
            }

            for key, value in metrics(
                degraded_fit, reference_fit
            ).items():
                row["deg_" + key] = value

            for key, value in metrics(
                restored_fit, reference_fit
            ).items():
                row["rest_" + key] = value

            if options.ablate:

                for variant_name, skip_set in VARIANTS.items():

                    if variant_name == "default":
                        variant_image = restored_fit

                    else:
                        variant_result, _ = restore(
                            image,
                            do_deskew=deskew_enabled,
                            skip=skip_set,
                        )

                        if variant_result is None:
                            continue

                        variant_image = fit(
                            variant_result,
                            reference.shape[:2],
                            options.max_side,
                        )

                    for key, value in metrics(
                        variant_image, reference_fit
                    ).items():
                        row[f"v{variant_name}_{key}"] = value

            rows.append(row)
            condition_rows.append(row)
            counts["paired"] += 1

            if len(condition_rows) <= 2 and options.skip == 0:

                preview_images = []

                for current in (
                    degraded_fit,
                    restored_fit,
                    reference_fit,
                ):
                    height, width = current.shape[:2]
                    preview_width = 500
                    preview_height = max(
                        1,
                        int(height * preview_width / width),
                    )

                    preview_images.append(
                        cv2.resize(
                            current,
                            (preview_width, preview_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    )

                preview = np.hstack(preview_images)

                preview_name = (
                    f"sample_{degradation}_{level}_"
                    f"{len(condition_rows)}.jpg"
                )

                cv2.imwrite(
                    os.path.join(options.out, preview_name),
                    preview,
                )

        summary = {
            "degradation": degradation,
            "level": level,
            "records": len(condition_records),
            "folder_files": len(degraded_files),
            "orphan_files": len(orphan_files),
            **counts,
        }

        summaries.append((degradation, level, condition_rows, summary))

        message = (
            f"{degradation} {level}: "
            f"{counts['paired']} paired | "
            f"{counts['missing_synthetic']} missing synthetic | "
            f"{counts['missing_gt']} missing GT | "
            f"{counts['ambiguous']} ambiguous | "
            f"{counts['unreadable']} unreadable | "
            f"{counts['resized']} resized | "
            f"{len(orphan_files)} files without metadata"
        )

        print(message)
        validation.append(message)


    if rows:
        csv_path = os.path.join(options.out, "per_image.csv")

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            columns = sorted({
                key for row in rows for key in row
            })

            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"\nPer-image results saved: {csv_path}")

    def mean(records, key):
        values = [record[key] for record in records if key in record]
        return st.mean(values) if values else float("nan")

    report = [
        "# Glyphora on HCCD Synthetic Degradations",
        "",
        f"Metadata: `{metadata_path}`",
        f"Ground truth: `{gt_root}`",
        f"Synthetic: `{synthetic_root}`",
        "",
        f"Deskew: {'ON' if deskew_enabled else 'OFF'}",
        f"Maximum metric side: {options.max_side or 'Full resolution'}",
        f"Metadata-driven pairing: Yes",
        f"Sorted-order fallback: No",
        "",
        "Images are paired using the document IDs and Paired_Doc_ID "
        "specified in the dataset metadata.",
        "Images with differing dimensions are resized to the paired "
        "ground-truth dimensions before pixel-level metric calculation.",
        "Geometric normalization may affect pixel-level metrics when "
        "aspect ratios differ.",
        "",
        "## Restoration results",
        "",
        "| Condition | n | MSE | PSNR (dB) | SSIM | Ink IoU | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for degradation, level, condition_rows, summary in summaries:

        if not condition_rows:
            continue

        report.append(
            f"| {degradation} {level} "
            f"| {len(condition_rows)} "
            f"| {mean(condition_rows, 'deg_mse'):.1f} "
            f"→ {mean(condition_rows, 'rest_mse'):.1f} "
            f"| {mean(condition_rows, 'deg_psnr'):.2f} "
            f"→ {mean(condition_rows, 'rest_psnr'):.2f} "
            f"| {mean(condition_rows, 'deg_ssim'):.4f} "
            f"→ {mean(condition_rows, 'rest_ssim'):.4f} "
            f"| {mean(condition_rows, 'deg_iou'):.4f} "
            f"→ {mean(condition_rows, 'rest_iou'):.4f} "
            f"| {1 / mean(condition_rows, 'seconds'):.2f} |"
        )

    if options.ablate:

        report += [
            "",
            "## Ablation results",
            "",
            "Mean PSNR (dB) / SSIM / MSE against ground truth.",
            "",
            "| Condition | Degraded | Default | +CLAHE | +Sharpen |",
            "|---|---|---|---|---|",
        ]

        for degradation, level, condition_rows, summary in summaries:

            if not condition_rows:
                continue

            def ablation_cell(variant, records):
                return (
                    f"{mean(records, f'v{variant}_psnr'):.2f} / "
                    f"{mean(records, f'v{variant}_ssim'):.4f} / "
                    f"{mean(records, f'v{variant}_mse'):.1f}"
                )

            degraded_cell = (
                f"{mean(condition_rows, 'deg_psnr'):.2f} / "
                f"{mean(condition_rows, 'deg_ssim'):.4f} / "
                f"{mean(condition_rows, 'deg_mse'):.1f}"
            )

            report.append(
                f"| {degradation} {level} "
                f"| {degraded_cell} "
                f"| {ablation_cell('default', condition_rows)} "
                f"| {ablation_cell('+clahe', condition_rows)} "
                f"| {ablation_cell('+sharpen', condition_rows)} |"
            )


    report += [
        "",
        "## Pairing and validation report",
        "",
        f"- Total metadata records: {len(metadata)}",
        f"- Ground-truth image files: {len(gt_files)}",
        f"- Duplicate metadata IDs: {len(duplicate_metadata)}",
        f"- Duplicate normalized ground-truth IDs: {len(duplicate_gt)}",
        f"- Total evaluated images: {len(rows)}",
        "",
        "| Condition | Metadata | Folder files | Paired | Missing synthetic | Missing GT | Ambiguous | Unreadable | Resized | Orphan files |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for degradation, level, condition_rows, summary in summaries:

        report.append(
            f"| {degradation} {level} "
            f"| {summary['records']} "
            f"| {summary['folder_files']} "
            f"| {summary['paired']} "
            f"| {summary['missing_synthetic']} "
            f"| {summary['missing_gt']} "
            f"| {summary['ambiguous']} "
            f"| {summary['unreadable']} "
            f"| {summary['resized']} "
            f"| {summary['orphan_files']} |"
        )

    if rows:
        average_time = mean(rows, "seconds")

        report += [
            "",
            "## Overall performance",
            "",
            f"- Evaluated images: {len(rows)}",
            f"- Mean restoration time: {average_time:.4f} seconds/image",
            f"- Restoration throughput: {1 / average_time:.2f} images/second",
        ]

    report += [
        "",
        "## Metric interpretation",
        "",
        "- Lower MSE is better.",
        "- Higher PSNR is better.",
        "- Higher SSIM is better.",
        "- Higher ink IoU is better.",
        "- FPS measures restoration throughput, not OCR speed.",
        "- CER/WER are not calculated because transcription ground truth is unavailable.",
    ]

    summary_path = os.path.join(options.out, "summary.md")

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))

    validation_path = os.path.join(options.out, "validation.txt")

    with open(validation_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(validation))

    print("\n" + "\n".join(report))
    print(f"\nSummary saved: {summary_path}")
    print(f"Validation saved: {validation_path}")


if __name__ == "__main__":
    main()
