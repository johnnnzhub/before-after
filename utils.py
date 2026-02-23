"""Utility functions for before-after comparison."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Optional


def fix_orientation(image: np.ndarray, pil_image: Image.Image) -> np.ndarray:
    """Fix image orientation based on EXIF data (common with phone photos)."""
    try:
        exif = pil_image._getexif()
        if exif is None:
            return image

        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break

        if orientation_key is None or orientation_key not in exif:
            return image

        orientation = exif[orientation_key]

        if orientation == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 6:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except (AttributeError, KeyError):
        pass

    return image


def resize_to_max(image: np.ndarray, max_dim=2400) -> np.ndarray:
    """Resize image if larger than max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def match_brightness(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match brightness of source image to target using LAB L-channel transfer.

    Only adjusts luminance (L channel). Preserves original color hue (A, B channels).
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    s_l, s_a, s_b = cv2.split(source_lab)
    t_l, _, _ = cv2.split(target_lab)

    s_mean, s_std = float(s_l.mean()), float(s_l.std())
    t_mean, t_std = float(t_l.mean()), float(t_l.std())

    if s_std < 1:
        s_std = 1.0

    matched_l = ((s_l.astype(np.float64) - s_mean) * (t_std / s_std) + t_mean)
    matched_l = np.clip(matched_l, 0, 255).astype(np.uint8)

    result_lab = cv2.merge([matched_l, s_a, s_b])
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def load_image(uploaded_file) -> Optional[np.ndarray]:
    """Load image from Streamlit upload, handling HEIC and EXIF orientation."""
    try:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        pil_img = Image.open(uploaded_file)
        pil_img = pil_img.convert("RGB")

        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Fix EXIF orientation
        uploaded_file.seek(0)
        pil_raw = Image.open(uploaded_file)
        img = fix_orientation(img, pil_raw)

        # Limit size for performance
        img = resize_to_max(img, max_dim=2400)

        return img
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None
