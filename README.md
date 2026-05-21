# Document Scanner

A computer vision tool that transforms photos of documents into clean, readable scanned copies — built with Python and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv)

---

## Problem Statement

Taking photos of documents (notes, receipts, pages) with a phone often results in skewed, poorly lit, hard-to-read images. This project solves that by automatically:

1. **Detecting** the document boundary in the photo
2. **Correcting** the perspective to produce a flat, top-down view
3. **Enhancing** the image for clean, readable output

## Features

- **Automatic document detection** using Canny edge detection + contour analysis
- **Perspective correction** (4-point transform) to flatten skewed documents
- **3 enhancement modes:**
  - `bw` — Clean black & white scan (adaptive thresholding)
  - `color` — Enhanced color scan (CLAHE histogram equalization)
  - `sharp` — High-contrast sharpened output
- **Batch processing** — process all modes at once with `--all-modes`
- **Visual pipeline** — saves intermediate steps (edges, contours, warped) for analysis

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/zetroretron/document-scanner.git
cd document-scanner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample test images (optional)
python create_samples.py

# 4. Run the scanner
python main.py sample_images/document.jpg
```

## Usage

### Basic Usage

```bash
# Scan a document (default: black & white mode)
python main.py path/to/photo.jpg

# Use color enhancement mode
python main.py path/to/photo.jpg --mode color

# Use sharpened mode
python main.py path/to/photo.jpg --mode sharp

# Generate all 3 modes at once
python main.py path/to/photo.jpg --all-modes

# Save without displaying windows
python main.py path/to/photo.jpg --no-display

# Specify output directory
python main.py path/to/photo.jpg --output results/
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `image` | Path to document photo | *(required)* |
| `--mode` | Enhancement: `bw`, `color`, or `sharp` | `bw` |
| `--output` | Output directory | `output/` |
| `--no-display` | Don't show OpenCV windows | off |
| `--all-modes` | Generate all 3 enhancement modes | off |

### Output Files

For an input `photo.jpg`, the scanner produces:

```
output/
├── photo_scanned_bw.jpg      # Final scanned output
├── photo_edges.jpg            # Edge detection result
├── photo_contour.jpg          # Detected document boundary
├── photo_warped.jpg           # Perspective-corrected image
```

## Project Structure

```
├── scanner.py          # Core scanning & enhancement pipeline
├── main.py             # CLI entry point
├── create_samples.py   # Generates synthetic test images
├── requirements.txt    # Dependencies (opencv-python, numpy)
├── sample_images/      # Test input images
├── output/             # Generated scan results
├── README.md           # This file
├── REPORT.md           # Project report
└── .gitignore
```

## How It Works

```
Input Photo → Grayscale → Gaussian Blur → Canny Edges → Find Contours
    → Largest 4-Point Contour → Perspective Transform → Enhancement → Clean Scan
```

1. **Load & Resize** — Read image, resize for fast processing
2. **Preprocess** — Convert to grayscale, blur to remove noise, detect edges with Canny
3. **Find Document** — Find contours, pick the largest 4-sided one (= document boundary)
4. **Perspective Transform** — Warp the document to a flat rectangle
5. **Enhance** — Apply thresholding/CLAHE/sharpening based on chosen mode

## License

MIT License
