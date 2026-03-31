"""
Document Scanner & Enhancer
===========================
A computer vision tool that transforms photos of documents
into clean, readable scanned copies using OpenCV.

Applies concepts from:
- Module 1: Image representation, grayscale conversion
- Module 2: Gaussian blur, histogram equalization (CLAHE), morphological ops, gamma correction
- Module 3: Canny edge detection, contour detection
- Module 4: Adaptive thresholding for segmentation
"""

import cv2
import numpy as np
import os


class DocumentScanner:
    """Main class that handles the full document scanning pipeline."""

    def __init__(self):
        """Initialize scanner with default parameters."""
        self.original = None
        self.processed = None
        self.scan_modes = ["bw", "color", "sharp"]

    # ------------------------------------------------------------------ #
    #  Step 1 — Load and Resize                         (Module 1: Basics)
    # ------------------------------------------------------------------ #
    def load_image(self, image_path):
        """
        Load an image from disk and resize it for processing.
        Uses OpenCV's imread (Module 1 - Image I/O).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Store original dimensions for later
        self.orig_h, self.orig_w = self.original.shape[:2]

        # Resize for faster processing (keep aspect ratio)
        self.ratio = self.orig_h / 500.0
        self.resized = self._resize(self.original, height=500)

        print(f"[+] Loaded image: {image_path} ({self.orig_w}x{self.orig_h})")
        return self.resized

    # ------------------------------------------------------------------ #
    #  Step 2 — Preprocess                    (Module 2: Enhancement,
    #                                          Module 3: Edge Detection)
    # ------------------------------------------------------------------ #
    def preprocess(self, image):
        """
        Convert to grayscale, apply blur, and detect edges.
        - Grayscale conversion (Module 1)
        - Gaussian blur for noise removal (Module 2 - Session 1)
        - Canny edge detection (Module 3 - Session 2)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise (Module 2)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection (Module 3)
        edges = cv2.Canny(blurred, 50, 200)

        # Dilate edges to close small gaps (Module 2 - Morphological Ops)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        print("[+] Preprocessing complete: grayscale → blur → Canny edges → dilation")
        return edges

    # ------------------------------------------------------------------ #
    #  Step 3 — Find Document Boundary        (Module 3: Contour Detection)
    # ------------------------------------------------------------------ #
    def find_document(self, edges):
        """
        Find the largest 4-sided contour — this is likely the document.
        Uses contour detection and approximation (Module 3).
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for contour in contours:
            # Approximate the contour to reduce number of points
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # If it has 4 points, it's likely a document
            if len(approx) == 4:
                doc_contour = approx
                print("[+] Document boundary found (4-point contour)")
                break

        if doc_contour is None:
            print("[!] No document boundary found — using full image")
            h, w = edges.shape[:2]
            doc_contour = np.array([
                [[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]
            ])

        return doc_contour

    # ------------------------------------------------------------------ #
    #  Step 4 — Perspective Transform               (Geometric Transform)
    # ------------------------------------------------------------------ #
    def perspective_transform(self, image, contour):
        """
        Warp the document region to a flat, top-down view using
        a 4-point perspective transform.
        """
        # Scale contour points back to original image size
        points = contour.reshape(4, 2) * self.ratio
        rect = self._order_points(points)

        (tl, tr, br, bl) = rect

        # Compute new width and height
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        # Destination points for the transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Compute and apply perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        print(f"[+] Perspective transform applied ({max_width}x{max_height})")
        return warped

    # ------------------------------------------------------------------ #
    #  Step 5 — Enhance the Scanned Image     (Module 2: Enhancement,
    #                                          Module 4: Segmentation)
    # ------------------------------------------------------------------ #
    def enhance(self, image, mode="bw"):
        """
        Apply enhancement to make the scan look clean and readable.

        Modes:
            - 'bw'    : Black & white scan (adaptive threshold)
            - 'color' : Enhanced color scan (CLAHE + sharpening)
            - 'sharp' : High contrast sharpened output

        Uses:
            - CLAHE histogram equalization (Module 2 - Session 5)
            - Adaptive thresholding (Module 4 - Segmentation)
            - Morphological operations (Module 2 - Session 4)
            - Gamma correction / power-law transform (Module 2 - Session 3)
        """
        if mode == "bw":
            return self._enhance_bw(image)
        elif mode == "color":
            return self._enhance_color(image)
        elif mode == "sharp":
            return self._enhance_sharp(image)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bw', 'color', or 'sharp'.")

    def _enhance_bw(self, image):
        """Black & white scan using adaptive thresholding (Module 4)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove noise with morphological opening (Module 2 - Session 4)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Adaptive threshold for clean B&W output (Module 4 - Segmentation)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 8
        )

        # Close small holes (Module 2 - Morphological Closing)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        print("[+] Enhancement applied: B&W scan (adaptive threshold + morphology)")
        return thresh

    def _enhance_color(self, image):
        """Enhanced color scan using CLAHE (Module 2 - Session 5)."""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel (Module 2 - Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Gamma correction / Power-law transform (Module 2 - Session 3)
        enhanced = self._gamma_correction(enhanced, gamma=1.2)

        print("[+] Enhancement applied: Color scan (CLAHE + gamma correction)")
        return enhanced

    def _enhance_sharp(self, image):
        """High contrast sharpened output."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast (Module 2)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpening kernel
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1,  5, -1],
            [0, -1, 0]
        ])
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

        print("[+] Enhancement applied: Sharpened scan (CLAHE + sharpen filter)")
        return enhanced

    # ------------------------------------------------------------------ #
    #  Full Pipeline — Scan                                               
    # ------------------------------------------------------------------ #
    def scan(self, image_path, mode="bw", output_dir="output"):
        """
        Run the complete scanning pipeline:
        1. Load image
        2. Preprocess (blur + edge detection)
        3. Find document boundary
        4. Perspective transform
        5. Enhance

        Args:
            image_path: Path to input image
            mode: Enhancement mode ('bw', 'color', 'sharp')
            output_dir: Directory to save results

        Returns:
            Dictionary with original, warped, and enhanced images
        """
        print("=" * 50)
        print(f"  Document Scanner — Processing: {os.path.basename(image_path)}")
        print("=" * 50)

        # Step 1: Load
        resized = self.load_image(image_path)

        # Step 2: Preprocess
        edges = self.preprocess(resized)

        # Step 3: Find document
        contour = self.find_document(edges)

        # Draw contour on a copy for visualization
        contour_img = resized.copy()
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)

        # Step 4: Perspective transform (on original resolution)
        warped = self.perspective_transform(self.original, contour)

        # Step 5: Enhance
        enhanced = self.enhance(warped, mode=mode)

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        results = {
            "original": self.original,
            "edges": edges,
            "contour": contour_img,
            "warped": warped,
            "enhanced": enhanced,
        }

        # Save the enhanced (final) output
        output_path = os.path.join(output_dir, f"{base_name}_scanned_{mode}.jpg")
        cv2.imwrite(output_path, enhanced)
        print(f"\n[✓] Saved scanned output: {output_path}")

        # Also save intermediate steps for the report
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_edges.jpg"), edges)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_contour.jpg"), contour_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_warped.jpg"), warped)

        print(f"[✓] Saved intermediate steps to {output_dir}/")
        print("=" * 50)

        return results

    # ------------------------------------------------------------------ #
    #  Helper Methods                                                      
    # ------------------------------------------------------------------ #
    def _resize(self, image, height=500):
        """Resize image while maintaining aspect ratio."""
        ratio = height / image.shape[0]
        dim = (int(image.shape[1] * ratio), height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def _order_points(self, pts):
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.
        Needed for correct perspective transform.
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
        rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference

        return rect

    def _gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma (power-law) correction (Module 2 - Session 3).
        gamma < 1 = brighter, gamma > 1 = darker
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        return cv2.LUT(image, table)
