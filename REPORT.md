# Project Report — Document Scanner & Enhancer

**Course:** SE2005 — Computer Vision  
**Project Type:** BYOP (Build Your Own Project)  
**Date:** March 2026

---

## 1. Problem Statement

### The Problem

When students take photos of handwritten notes, receipts, or printed documents using their phone cameras, the resulting images are often:

- **Skewed** due to camera angle
- **Poorly lit** with shadows and uneven brightness
- **Hard to read** because of low contrast and noise

Popular scanner apps (CamScanner, Adobe Scan) solve this, but they are proprietary and use complex deep learning models.

### Why It Matters

As students, we take dozens of document photos daily — notes, assignments, ID cards, bills. Having a simple, transparent tool that uses classical computer vision techniques (the ones taught in this course) to produce clean scans is both practical and educational.

### Goal

Build a Python tool using OpenCV that:
1. Detects a document in a photo
2. Corrects the perspective
3. Enhances the output for readability

---

## 2. Approach

### Pipeline Design

The solution follows a linear processing pipeline:

```
Photo → Preprocess → Detect Document → Perspective Correction → Enhancement → Output
```

Each stage maps directly to course concepts:

**Stage 1 — Preprocessing (Module 1, 2)**
- Convert to grayscale (Module 1)
- Apply Gaussian blur to reduce noise (Module 2, Session 1)
- The blur kernel size (5×5) was chosen to balance noise removal and edge preservation

**Stage 2 — Edge Detection (Module 3)**
- Apply Canny edge detection with thresholds (50, 200)
- Dilate edges using a 3×3 kernel to close small gaps (Module 2, Morphological Ops)
- This makes contour detection more reliable

**Stage 3 — Document Detection (Module 3)**
- Find contours in the edge image
- Sort by area (largest first)
- Approximate each contour and select the first one with exactly 4 vertices
- If no 4-point contour is found, fall back to the full image

**Stage 4 — Perspective Transform**
- Order the 4 corner points (top-left, top-right, bottom-right, bottom-left)
- Compute the perspective transform matrix
- Warp the image to a flat, rectangular view

**Stage 5 — Enhancement (Module 2, 4)**
Three modes are provided:
- **B&W Scan:** Morphological opening → Adaptive thresholding → Morphological closing
- **Color Scan:** Convert to LAB space → CLAHE on L channel → Gamma correction
- **Sharp Scan:** CLAHE → Sharpening convolution filter

---

## 3. Key Decisions

| Decision | Rationale |
|----------|-----------|
| Used Canny over Sobel for edge detection | Canny provides cleaner, thinner edges with automatic non-maximum suppression |
| Adaptive thresholding over global | Documents have uneven lighting; adaptive handles this better |
| CLAHE over standard histogram equalization | CLAHE prevents over-amplification of contrast in already-bright regions |
| LAB color space for color enhancement | L channel separates luminance from color, allowing contrast enhancement without color distortion |
| Resize to 500px for processing | Speeds up contour detection without affecting final output quality (transform is applied to original resolution) |

---

## 4. Challenges Faced

### 1. Contour Detection Accuracy
**Problem:** The document contour was sometimes not the largest contour (e.g., a desk boundary was detected instead).  
**Solution:** Limited search to top-5 contours sorted by area, and required exactly 4 vertices.

### 2. Point Ordering
**Problem:** The 4 corner points from contour detection are unordered, causing incorrect warping.  
**Solution:** Used the sum/difference method to reliably sort points as TL, TR, BR, BL.

### 3. Handling Missing Documents
**Problem:** When no 4-point contour is found (e.g., document fills the frame), the pipeline crashed.  
**Solution:** Added a fallback that uses the full image boundaries as the contour.

### 4. Enhancement Quality
**Problem:** Global thresholding produced poor results on documents with shadows.  
**Solution:** Switched to Gaussian adaptive thresholding with a block size of 11.

---

## 5. What I Learned

1. **Classical CV is powerful** — Many tasks that seem to need deep learning can be solved well with traditional techniques like edge detection, contour analysis, and morphological operations.

2. **Preprocessing matters most** — The quality of the final output depends heavily on how well noise is removed and edges are detected before contour analysis.

3. **Parameter tuning is an art** — Canny thresholds, blur kernel sizes, and adaptive threshold block sizes all needed careful tuning through experimentation.

4. **The pipeline pattern** — Breaking the problem into discrete, testable stages (load → preprocess → detect → transform → enhance) made development and debugging much easier.

5. **Multiple output modes** — Different use cases need different enhancements. B&W works best for printed text, color for photos/diagrams, and sharp for low-contrast documents.

---

## 6. Course Concepts Used

| Module | Session | Concept | Application in Project |
|--------|---------|---------|----------------------|
| 1 | 1-5 | Image I/O, representation, OpenCV | Loading, saving, displaying images |
| 2 | 1 | Noise removal (Gaussian blur) | Preprocessing before edge detection |
| 2 | 3 | Power-law (gamma) correction | Brightness adjustment in color mode |
| 2 | 4 | Morphological operations | Opening/closing for noise cleanup |
| 2 | 5 | Histogram equalization (CLAHE) | Contrast enhancement |
| 3 | 1-2 | Edge detection (Canny) | Finding document edges |
| 3 | 3-4 | Corner/contour detection | Identifying document boundary |
| 4 | 1 | Image segmentation | Adaptive thresholding for B&W scan |

---

## 7. Results

The scanner successfully processes document photos and produces clean scanned output in three modes. The pipeline handles:
- Documents on cluttered backgrounds
- Tilted/skewed photos
- Various document types (text, receipts, handwritten notes)

Intermediate outputs (edges, contours, warped) are saved alongside the final result, allowing visual inspection of each pipeline stage.

---

## 8. References

- OpenCV Documentation: https://docs.opencv.org/
- Canny Edge Detection: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- Perspective Transform: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- CLAHE: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- Morphological Operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
