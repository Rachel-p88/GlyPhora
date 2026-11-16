import os, shutil, pytesseract
from pytesseract import Output

def _locate():
    if shutil.which("tesseract"): return
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")):
        if os.path.exists(c): pytesseract.pytesseract.tesseract_cmd = c; return
_locate()

TESSDATA = os.environ.get("GLYPHORA_TESSDATA")

def extract(gray, psm=3):
    if TESSDATA: os.environ["TESSDATA_PREFIX"] = TESSDATA
    d = pytesseract.image_to_data(gray, config=f"--oem 1 --psm {psm}", output_type=Output.DICT)
    lines, key, cur, par = [], None, [], None
    for i, t in enumerate(d["text"]):
        if float(d["conf"][i]) < 0 or not t.strip(): continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key:
            if cur: lines.append((par, " ".join(cur)))
            key, cur, par = k, [], k[:2]
        cur.append(t)
    if cur: lines.append((par, " ".join(cur)))
    out, last = [], None
    for p, l in lines:
        if last is not None and p != last: out.append("")
        out.append(l); last = p
    return "\n".join(out)
