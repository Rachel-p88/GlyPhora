# GlyPhora – Intelligent Document Restoration using IVP & OCR

GlyPhora is a web-based application designed to enhance degraded document images and improve readability. It uses image processing techniques to address issues such as blur, poor illumination, shadows, and noise.

The application also extracts text from document images using OCR, providing users with both the restored image and extracted text.

## Features

- Adaptive image enhancement and restoration.
- Brightness and illumination correction.
- Noise reduction and image preprocessing.
- Document deskewing.
- Text extraction using OCR.
- Web-based user interface.

## Tech Stack

- **Backend:** Python, FastAPI
- **Image Processing:** OpenCV, NumPy
- **OCR:** Tesseract OCR
- **Frontend:** HTML, CSS, JavaScript

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

Install Tesseract OCR 5 and ensure it is accessible through your system PATH.

### 3. Run the Application

```bash
python -m uvicorn glyphora.api:app --reload
```

### 4. Open the Application

Visit http://127.0.0.1:8000 in your browser.

