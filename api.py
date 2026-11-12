import base64
import io
import os
import zipfile
import logging

import cv2
import numpy as np
import pytesseract

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles

from .restore import restore, ocr_prep_scaled
from .ocr import extract


log = logging.getLogger("glyphora")
app = FastAPI(title="Glyphora")

MAX_BYTES, MAX_PIXELS, MAX_BATCH = 15 << 20, 40_000_000, 20

POOL = ThreadPoolExecutor(2)


@app.exception_handler(Exception)
async def _err(_, e):
    log.exception("unhandled")
    return JSONResponse({"error": "Internal error"}, 500)


def decode(data: bytes):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES >> 20} MB")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(415, "Unsupported or corrupted image")

    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "Image dimensions too large")

    return img


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def safe(name):
    return "".join(
        c for c in os.path.splitext(os.path.basename(name or "doc"))[0]
        if c.isalnum() or c in "-_"
    )[:60] or "doc"


def process(data):
    img = decode(data)
    rest, steps = restore(img)

    out = {
        "restored": png(rest),
        "steps": steps
    }

    try:
        out["text"] = extract(ocr_prep_scaled(rest))
        out["ocr_error"] = None

    except pytesseract.TesseractNotFoundError:
        log.exception("tesseract missing")
        out["text"] = ""
        out["ocr_error"] = (
            "Tesseract OCR is not installed or not found. "
            "Install it (see README) and restart the server."
        )

    except Exception as e:
        log.exception("ocr failed")
        out["text"] = ""
        out["ocr_error"] = (
            f"Text extraction failed ({type(e).__name__})"
        )

    return out


@app.post("/api/process")
async def process_one(file: UploadFile = File(...)):
    r = process(await file.read())

    return {
        "image_png_b64": base64.b64encode(r["restored"]).decode(),
        "text": r["text"],
        "steps": r["steps"],
        "ocr_error": r["ocr_error"]
    }


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"Max {MAX_BATCH} files")

    items = [
        (safe(f.filename), await f.read())
        for f in files
    ]

    def run(it):
        try:
            return it[0], process(it[1]), None

        except HTTPException as e:
            return it[0], None, e.detail

        except Exception:
            log.exception("batch item")
            return it[0], None, "Processing failed"

    buf, report = io.BytesIO(), []

    with zipfile.ZipFile(buf, "w") as z:
        for i, (n, r, err) in enumerate(POOL.map(run, items), 1):

            if r:
                z.writestr(
                    f"{i:02d}_{n}_restored.png",
                    r["restored"]
                )

                z.writestr(
                    f"{i:02d}_{n}.txt",
                    r["text"].encode("utf-8")
                )

            else:
                report.append(f"{n}: FAILED - {err}")

        if report:
            z.writestr("failures.txt", "\n".join(report))

    return Response(
        buf.getvalue(),
        media_type="application/zip"
    )


app.mount(
    "/",
    StaticFiles(
        directory=os.path.join(
            os.path.dirname(__file__), "..", "web"
        ),
        html=True
    ),
    name="web"
)
