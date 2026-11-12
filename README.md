# GlyPhora
# GlyPhora – Adaptive Image Restoration & OCR

GlyPhora is a web-based application designed to enhance degraded images and improve document readability. It uses adaptive image processing techniques to identify and address degradation such as blur, poor illumination, and shadows.

The application also extracts text from images using Tesseract OCR and provides users with an enhanced image and extracted text.

## Features

- Adaptive image enhancement and restoration.
- Brightness and illumination correction.
- Noise reduction and image preprocessing.
- Document deskewing.
- Text extraction using Tesseract OCR.
- Simple web-based interface.

## Tech Stack

- **Backend:** Python, FastAPI
- **Image Processing:** OpenCV, NumPy
- **OCR:** Tesseract OCR
- **Frontend:** HTML, CSS, JavaScript

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CIVP
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv venv
```

Activate it in the VS Code terminal (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR 5 and ensure that it is accessible from your system PATH.

### 5. Run GlyPhora

Open the project folder in VS Code, activate the virtual environment, and execute:

```bash
python -m uvicorn glyphora.api:app --reload
```

### 6. Open the Application

Visit the following address in your browser:

```text
http://127.0.0.1:8000
```

The application is now ready to use!

---

**Note:** Run the command from the project's root directory, where the `glyphora` folder and `requirements.txt` are located.
