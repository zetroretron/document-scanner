"""
Generate sample document images for testing the scanner.
Creates synthetic document photos with perspective distortion.
"""

import cv2
import numpy as np
import os


def create_sample_document(filename, doc_type="text"):
    """Create a synthetic document photo for testing."""

    # Create a white canvas (simulating a desk/background)
    bg = np.ones((800, 1000, 3), dtype=np.uint8) * 180  # gray background

    # Add some texture to simulate a desk surface
    noise = np.random.randint(0, 30, bg.shape, dtype=np.uint8)
    bg = cv2.add(bg, noise)

    # Create a white document
    doc = np.ones((500, 380, 3), dtype=np.uint8) * 255

    if doc_type == "text":
        # Add fake text lines
        y = 40
        for i in range(15):
            width = np.random.randint(150, 340)
            cv2.rectangle(doc, (30, y), (30 + width, y + 8), (40, 40, 40), -1)
            y += 28

        # Add a title
        cv2.rectangle(doc, (30, 10), (250, 28), (20, 20, 20), -1)

    elif doc_type == "receipt":
        # Add receipt-like content
        cv2.rectangle(doc, (100, 15), (280, 35), (30, 30, 30), -1)  # Store name
        cv2.line(doc, (30, 50), (350, 50), (100, 100, 100), 1)

        y = 70
        for i in range(10):
            w1 = np.random.randint(100, 200)
            cv2.rectangle(doc, (30, y), (30 + w1, y + 7), (50, 50, 50), -1)
            cv2.rectangle(doc, (280, y), (350, y + 7), (50, 50, 50), -1)
            y += 22

        cv2.line(doc, (30, y), (350, y), (100, 100, 100), 1)
        y += 15
        cv2.rectangle(doc, (200, y), (350, y + 10), (20, 20, 20), -1)

    elif doc_type == "notes":
        # Add handwriting-like wavy lines
        for i in range(18):
            y = 30 + i * 25
            pts = []
            for x in range(30, 350, 5):
                offset = np.random.randint(-3, 4)
                pts.append([x, y + offset])
            pts = np.array(pts, dtype=np.int32)
            end_x = np.random.randint(200, 350)
            pts = pts[pts[:, 0] <= end_x]
            if len(pts) > 1:
                cv2.polylines(doc, [pts], False, (60, 60, 120), 1)

    # Apply perspective warp to simulate a tilted phone photo
    h, w = doc.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Random perspective distortion
    offset = 40
    dst_pts = np.float32([
        [200 + np.random.randint(-offset, offset),
         100 + np.random.randint(-offset, offset)],
        [650 + np.random.randint(-offset, offset),
         120 + np.random.randint(-offset, offset)],
        [680 + np.random.randint(-offset, offset),
         680 + np.random.randint(-offset, offset)],
        [180 + np.random.randint(-offset, offset),
         650 + np.random.randint(-offset, offset)]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(doc, matrix, (bg.shape[1], bg.shape[0]),
                                  dst=bg.copy(),
                                  borderMode=cv2.BORDER_TRANSPARENT)

    # Blend document onto background
    mask = cv2.warpPerspective(
        np.ones_like(doc) * 255, matrix, (bg.shape[1], bg.shape[0])
    )
    mask = mask.astype(bool)
    output = bg.copy()
    output[mask] = result[mask]

    # Add slight blur to simulate camera
    output = cv2.GaussianBlur(output, (3, 3), 0)

    # Add slight brightness variation
    output = cv2.convertScaleAbs(output, alpha=0.95, beta=10)

    os.makedirs("sample_images", exist_ok=True)
    path = os.path.join("sample_images", filename)
    cv2.imwrite(path, output)
    print(f"Created: {path}")
    return path


if __name__ == "__main__":
    create_sample_document("document.jpg", "text")
    create_sample_document("receipt.jpg", "receipt")
    create_sample_document("notes.jpg", "notes")
    print("\nDone! Sample images saved to sample_images/")
