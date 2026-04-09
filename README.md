
> **Praxis II — Automated Screw Sorting with Computer Vision & AI**
ScrewSorter is a computer-vision pipeline that automatically **identifies screw head types** and **estimates screw lengths** from images using Google Gemini AI and OpenCV. It was built for the Praxis II engineering design course and provides real-time screw analysis through a camera interface.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Supported Screw Head Types](#supported-screw-head-types)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Features

- 🔩 **Three-class screw head classification** — Flat Head, Oval Head, Round Washer Head using Google Gemini AI
- 🤖 **AI-powered classification** — Zero-shot screw head identification with Google's Gemini vision model
- 📏 **Screw length estimator** — Uses OpenCV contour analysis and QR code calibration to measure physical length in millimetres
- 📷 **Real-time camera interface** — Live video feed with capture functionality
- 🔒 **Secure API key management** — Environment variables with .env file support
- 💾 **Result visualization** — Saves annotated images with measurements and classifications

---

## How It Works

```
Camera feed  ──►  QR detection (scale calibration)  ──►  Screw contour analysis
                        │                                       │
                        ▼                                       ▼
                 Physical measurements (mm)          Head shape classification
                        │                                       │
                        └───────────────────►  Gemini AI analysis  ──►  Results
```

1. **QR Code Detection** — Detects a QR code in the image to establish a scale reference (calibrated to known size)
2. **Screw Detection** — Uses OpenCV thresholding and contour analysis to identify screw shapes in the image
3. **Length Measurement** — Calculates screw shaft length using the QR scale and fitted bounding rectangles
4. **Head Classification** — Sends the image to Google Gemini AI with a prompt to identify the screw head type
5. **Result Display** — Shows measurements, classification, and saves an annotated image

---

## Supported Screw Head Types

| Class | Description |
|---|---|
| **Flat_Head** | Countersunk head that sits flush with the surface |
| **Oval_Head** | Partially countersunk with a rounded, decorative top |
| **Round_Washer** | Dome-shaped head with a built-in washer bearing surface |

---

## Prerequisites

| Package | Purpose |
|---|---|
| `google-generativeai` | Gemini AI API client |
| `opencv-python` | Image processing and computer vision |
| `Pillow` | Image I/O and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Result visualization |
| `python-dotenv` | Environment variable loading |

> Install all dependencies in a virtual environment: `pip install google-generativeai opencv-python Pillow numpy matplotlib python-dotenv`

---

## Getting Started

1. **Clone or download the repository**

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install google-generativeai opencv-python Pillow numpy matplotlib python-dotenv
   ```

3. **Get a Google Gemini API key:**
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key

4. **Configure the API key:**
   - Create a `.env` file in the project root
   - Add your API key: `GEMINI_API_KEY=your_api_key_here`

5. **Run the application:**
   ```bash
   python mains.py
   ```

6. **Usage:**
   - Place a screw next to a QR code sheet for scale reference
   - Press SPACEBAR to capture and analyze
   - Press Q to quit
   - Results are displayed in the console and saved as `result.png`

---

## Configuration

The following constants at the top of `mains.py` can be tuned:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `1` | Camera device index (0 for default webcam) |
| `QR_SIZE_MM` | `21.9` | Physical size of QR code in mm for calibration |
| `SAVE_PATH` | `result.png` | Path to save the annotated result image |
| `MODEL_ID` | `gemini-2.5-flash` | Gemini model to use for classification |

---

## Project Structure

```
ScrewSorter/
├── mains.py                 # Main application script
├── .env                     # Environment variables (API key)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── LICENSE                 # License information
├── demo.py                 # Demo/test script
└── Older-Backup Files/     # Archived files
```

---

## Limitations & Future Work

- **Camera calibration** — The QR size (`QR_SIZE_MM`) needs to be measured accurately for precise measurements
- **Lighting conditions** — Performance may vary with different lighting; optimal results with even, bright lighting
- **Screw orientation** — Best results when screws are positioned with heads clearly visible
- **API rate limits** — Gemini API has usage limits; includes retry logic for quota exceeded errors
- **Real-time performance** — Processing time depends on image complexity and API response time

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
