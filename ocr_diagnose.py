"""Run on ONE of your real pages to see which OCR preprocessing works best for it.
  python -m tests.ocr_diagnose page.png --truth page.txt [--tessdata C:\\path\\to\\tessdata_best] [--verbose]
Prints image size, glyph height, then a compact table (CER/WER per variant if --truth is given). Saves OCR inputs to ocr_diag/."""
import argparse, os, time, cv2
from glyphora.restore import restore, ocr_prep, ocr_prep_scaled, _glyph_height, lighting_unevenness, lighting_unevenness
from glyphora import ocr
from tests.evaluate import lev
a = argparse.ArgumentParser(); a.add_argument("image"); a.add_argument("--truth"); a.add_argument("--tessdata"); a.add_argument("--verbose", action="store_true"); o = a.parse_args()
if o.tessdata: ocr.TESSDATA = o.tessdata
im = cv2.imread(o.image); assert im is not None, "cannot read image"
g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); rest, steps = restore(im)
print(f"size {im.shape[1]}x{im.shape[0]} px | median glyph height {_glyph_height(g)} px (Tesseract prefers ~25-35) | lighting unevenness {lighting_unevenness(g):.1f} (bg-norm auto-applies above 12) | restore: {steps} | tessdata: {ocr.TESSDATA or 'default'}\n")
n = lambda t: " ".join(t.split()); ref = n(open(o.truth, encoding="utf-8").read()) if o.truth else None
C = dict(bgnorm=False, lanczos=False)  # explicit: plain cubic, no bg-norm
V = {"OLD pipeline (fixed 2000 px width)": (lambda: ocr_prep(rest), 3), "raw gray": (lambda: g, 3)}
for t in (20, 28, 36, 44): V[f"scaled to {t}px (cubic)"] = (lambda t=t: ocr_prep_scaled(rest, target=t, **C), 3)
V["36px, lanczos"] = (lambda: ocr_prep_scaled(rest, target=36, bgnorm=False, lanczos=True), 3)
V["36px, lanczos + bg-norm (always)"] = (lambda: ocr_prep_scaled(rest, target=36, bgnorm=True), 3)
V["APP DEFAULT (36px, lanczos, bg-norm auto)"] = (lambda: ocr_prep_scaled(rest), 3)
V["36px lanczos, psm 6"] = (lambda: ocr_prep_scaled(rest, bgnorm=False), 6)
os.makedirs("ocr_diag", exist_ok=True); print(f"{'variant':38}{'CER%':>7}{'WER%':>7}{'sec':>6}")
for k, (fn, psm) in V.items():
    img = fn(); t = time.time(); txt = ocr.extract(img, psm); dt = time.time() - t
    cv2.imwrite("ocr_diag/" + "".join(c if c.isalnum() else "_" for c in k) + ".png", img)
    c, w = (f"{100*lev(ref,n(txt))/len(ref):7.1f}", f"{100*lev(ref.split(),n(txt).split())/len(ref.split()):7.1f}") if ref else ("    n/a", "    n/a")
    print(f"{k:38}{c}{w}{dt:6.1f}")
    if o.verbose: print(txt, "\n")