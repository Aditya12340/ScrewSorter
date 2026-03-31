# ScrewSorter

> **Praxis II — Automated Screw Sorting with Computer Vision & AI**

ScrewSorter is a computer-vision pipeline that automatically **identifies screw head types** and **estimates screw lengths** from images. It was built for the Praxis II engineering design course and offers two classification approaches: a custom-trained deep-learning model (MobileNetV2) and a Google Gemini AI wrapper — so you can pick the method that fits your workflow.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
  - [Approach 1 — Custom CNN Classifier (`main.py`)](#approach-1--custom-cnn-classifier-mainpy)
  - [Approach 2 — Gemini AI Wrapper (`ScrewClassifierAIWrapper.ipynb`)](#approach-2--gemini-ai-wrapper-screwclassifieraiwrapperipynb)
- [Supported Screw Head Types](#supported-screw-head-types)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Running the CNN Classifier on Google Colab](#running-the-cnn-classifier-on-google-colab)
  - [Running the Gemini AI Wrapper on Google Colab](#running-the-gemini-ai-wrapper-on-google-colab)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Features

- 🔩 **Three-class screw head classification** — Flat Head, Oval Head, Round Washer Head
- 🤖 **Two interchangeable classifiers** — a fine-tuned MobileNetV2 CNN and a Google Gemini vision model
- 🖼️ **Few-shot synthetic dataset generation** — trains a production-ready model from as few as **3 seed images** using aggressive on-the-fly augmentation
- 📏 **Screw length estimator** — uses OpenCV edge detection and contour analysis to measure physical length in millimetres directly from an image
- ☁️ **Runs entirely on Google Colab** — no local GPU required

---

## How It Works

### Approach 1 — Custom CNN Classifier (`main.py`)

```
3 seed images  ──►  Augmentation  ──►  Synthetic dataset
                                          │
                                    MobileNetV2 (fine-tuned)
                                          │
                               Classification + confidence
                                          │
                               Optional: Length Estimator
```

1. **Upload 3 seed images** (one per class) via the Colab file picker.
2. **Synthetic dataset creation** — each seed image is augmented 100× for training and 5× for validation using random rotation (±25°), random crop, brightness/contrast jitter, Gaussian blur, and horizontal flip. All images are resized to 224 × 224 px.
3. **Model training** — a pretrained MobileNetV2 backbone (ImageNet weights) is fine-tuned with a new 3-class head using the Adam optimiser and CrossEntropyLoss for 6 epochs (batch size 32, LR 1e-4).
4. **Inference** — upload any screw image and receive the predicted class together with class probabilities.
5. **Length estimation** — a separate OpenCV widget detects screw contours via Canny edge detection, fits a bounding rectangle, and converts the largest dimension to millimetres using a calibrated scale factor (default: `0.192 mm/pixel`).
6. **Model export** — the best checkpoint (`screw_classifier.pt`) is automatically downloaded to your local machine.

### Approach 2 — Gemini AI Wrapper (`ScrewClassifierAIWrapper.ipynb`)

```
Screw image  ──►  Google Gemini Flash (vision LLM)  ──►  Head type label
```

Uses the `gemini-flash-latest` multimodal model as a zero-shot mechanical-fastener expert. Upload any screw image and the model returns exactly one of: **Round Head**, **Oval Head**, or **Flat Head** — no training required.

---

## Supported Screw Head Types

| Class | Description |
|---|---|
| **Flat Head** | Countersunk head that sits flush with the surface |
| **Oval Head** | Partially countersunk with a rounded, decorative top |
| **Round Washer Head** | Dome-shaped head with a built-in washer bearing surface |

---

## Prerequisites

### CNN Classifier (`main.py`)
| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training & inference |
| `opencv-python` (`cv2`) | Image augmentation & length estimation |
| `Pillow` | Image I/O |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation |
| `google-colab` | File upload/download helpers (Colab built-in) |

### Gemini AI Wrapper (`ScrewClassifierAIWrapper.ipynb`)
| Package | Purpose |
|---|---|
| `google-generativeai` | Gemini API client (installed automatically in the notebook) |
| `Pillow` | Image loading |
| `google-colab` | File upload helpers (Colab built-in) |

> All packages except `google-generativeai` are pre-installed in the standard Google Colab runtime.

---

## Getting Started

### Running the CNN Classifier on Google Colab

1. **Open `main.py` in Colab**

   Because `main.py` is a plain Python script (not a notebook), open a new Colab session, upload `main.py` via the file panel, and run it with:
   ```python
   %run main.py
   ```
   Alternatively, copy-paste the cells sequentially into a new Colab notebook.

2. **Upload your 3 seed images** when prompted. One image per screw head class.

3. **Update the `class_map` dictionary** to match your uploaded filenames:

   ```python
   class_map = {
       "Flat_Head_Screws.png": "Flat_Head",
       "Oval_Head_Screw.jpg":  "Oval_Head",
       "Round_Washer_Head.jpg": "Round_Washer"
   }
   ```

4. **Run all cells** — the script will:
   - Generate the synthetic dataset
   - Train the model (≈ 2–5 minutes on a free Colab GPU)
   - Print per-epoch training loss and validation accuracy
   - Save and download `screw_classifier.pt`

5. **Test inference** — upload a new screw image when prompted to see the prediction and confidence scores.

6. **Optional — Length Estimator** — the final section renders an interactive file-upload widget. Upload a screw image to get the estimated length in mm overlaid on the image.

### Running the Gemini AI Wrapper on Google Colab

1. **Store your Gemini API key** as a Colab secret named `ScrewClassifier`:
   - Open *Runtime → Manage secrets* and add the key.
   - The notebook reads it via `userdata.get("ScrewClassifier")`.

2. **Open and run the notebook**

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aditya12340/ScrewSorter/blob/main/ScrewClassifierAIWrapper.ipynb)

3. **Upload a screw image** when prompted — the identified head type is printed to the console.

---

## Configuration

The following constants at the top of `main.py` can be tuned without changing any other code:

| Constant | Default | Description |
|---|---|---|
| `SEED` | `42` | Global random seed for reproducibility |
| `TRAIN_PER_CLASS` | `100` | Augmented training images generated per class |
| `VAL_PER_CLASS` | `5` | Augmented validation images generated per class |
| `BATCH_SIZE` | `32` | Mini-batch size during training |
| `EPOCHS` | `6` | Number of training epochs |
| `LR` | `1e-4` | Adam learning rate |
| `scale` (length estimator) | `0.192` | Millimetres per pixel — calibrate to your camera/setup |

---

## Project Structure

```
ScrewSorter/
├── main.py                        # CNN training, inference & length estimator (Colab script)
├── ScrewClassifierAIWrapper.ipynb # Gemini AI-powered classifier (Colab notebook)
├── LICENSE
└── README.md
```

**Generated at runtime (not committed):**

```
screw_data/
├── train/
│   ├── Flat_Head/     (100 augmented images)
│   ├── Oval_Head/     (100 augmented images)
│   └── Round_Washer/  (100 augmented images)
└── val/
    ├── Flat_Head/     (5 augmented images)
    ├── Oval_Head/     (5 augmented images)
    └── Round_Washer/  (5 augmented images)
screw_classifier.pt                # Best model checkpoint
```

---

## Limitations & Future Work

- **Length estimator calibration** — the `scale` factor (`0.192 mm/pixel`) assumes a fixed camera distance and sensor. For accurate measurements, calibrate against a known reference object in the same image plane.
- **Small validation set** — with only 5 augmented validation images per class (all derived from a single seed), validation accuracy is a rough guide rather than a statistically robust metric. Expanding to real diverse images would improve reliability.
- **Three classes only** — the current model is hard-coded for three screw head types. Adding new classes requires re-uploading seed images and retraining.
- **Colab dependency** — `main.py` uses `google.colab.files` for uploads/downloads and is therefore tied to the Colab environment. Refactoring to accept local file paths would enable standalone execution.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
