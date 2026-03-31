"""
Document Scanner & Enhancer — Main Entry Point
================================================
Usage:
    python main.py <image_path> [--mode bw|color|sharp] [--no-display]

Examples:
    python main.py sample_images/receipt.jpg
    python main.py sample_images/notes.jpg --mode color
    python main.py sample_images/document.jpg --mode sharp --no-display
"""

import sys
import os
import argparse
import cv2
from scanner import DocumentScanner


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Document Scanner & Enhancer — Turn photos into clean scans"
    )
    parser.add_argument(
        "image",
        help="Path to the document image"
    )
    parser.add_argument(
        "--mode",
        choices=["bw", "color", "sharp"],
        default="bw",
        help="Enhancement mode: bw (default), color, or sharp"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip displaying the result windows"
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Generate output in all 3 modes (bw, color, sharp)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    # Create scanner
    scanner = DocumentScanner()

    if args.all_modes:
        # Run all 3 enhancement modes
        for mode in ["bw", "color", "sharp"]:
            print()
            scanner.scan(args.image, mode=mode, output_dir=args.output)
    else:
        # Run single mode
        results = scanner.scan(args.image, mode=args.mode, output_dir=args.output)

        # Display results if not suppressed
        if not args.no_display:
            display_results(results)


def display_results(results):
    """Show original vs scanned side by side using OpenCV windows."""
    print("\n[i] Displaying results. Press any key to close.\n")

    # Resize for display
    def resize_for_display(img, max_height=600):
        h, w = img.shape[:2]
        if h > max_height:
            ratio = max_height / h
            img = cv2.resize(img, (int(w * ratio), max_height))
        return img

    original = resize_for_display(results["original"])
    cv2.imshow("1 - Original", original)

    edges = resize_for_display(results["edges"])
    cv2.imshow("2 - Edge Detection", edges)

    contour = resize_for_display(results["contour"])
    cv2.imshow("3 - Document Detected", contour)

    enhanced = resize_for_display(results["enhanced"])
    cv2.imshow("4 - Final Scanned Output", enhanced)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
