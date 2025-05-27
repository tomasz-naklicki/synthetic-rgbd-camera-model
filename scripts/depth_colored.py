#!/usr/bin/env python3
"""
enhance_depth_image.py

Skrypt do zwiększania widoczności ciemnych map głębokości.
Ładuje obraz głębi (8- lub 16-bitowy), normalizuje, wyrównuje histogram
(z użyciem CLAHE), opcjonalnie nakłada kolorową paletę i zapisuje wynik.

Użycie:
    python enhance_depth_image.py --input path/to/depth.png --output path/to/enhanced.png
"""
import argparse
import cv2
import numpy as np


def enhance_depth(
    clip_limit: float = 2.0, grid_size: int = 8, apply_colormap: bool = True
):
    # Wczytaj obraz głębi
    depth = cv2.imread("img/depth_0000_aligned.png", cv2.IMREAD_UNCHANGED)

    # Konwersja do 8-bitów poprzez normalizację
    if depth.dtype == np.uint16:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_8u = depth_norm.astype(np.uint8)
    elif depth.dtype == np.uint8:
        depth_8u = depth.copy()
    else:
        depth_float = depth.astype(np.float32)
        depth_norm = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX)
        depth_8u = depth_norm.astype(np.uint8)

    # Wyrównanie histogramu CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(depth_8u)

    cv2.imwrite("output/depth_aligned_real_color.png", enhanced)


if __name__ == "__main__":

    enhance_depth()
