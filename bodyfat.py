"""Estimativa de percentual de gordura corporal via landmarks MediaPipe."""

import math

import cv2
import numpy as np

# MediaPipe landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
NOSE = 0
L_ANKLE = 27
R_ANKLE = 28
L_EAR = 7
R_EAR = 8


def _to_px(landmark, img_w, img_h):
    return np.array([landmark.x * img_w, landmark.y * img_h])


def _dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def _ellipse_circumference(width_cm, depth_ratio=0.7):
    """Approximate circumference from frontal width using ellipse formula."""
    a = width_cm / 2  # semi-major (frontal half-width)
    b = a * depth_ratio  # semi-minor (estimated depth)
    # Ramanujan approximation
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def estimate_body_fat(landmarks, img_w, img_h, height_cm, weight_kg, age, sex):
    """
    Estimate body fat % using Navy Formula + Deurenberg ensemble.

    Args:
        landmarks: MediaPipe pose landmarks
        img_w, img_h: image dimensions in pixels
        height_cm: user height in cm
        weight_kg: user weight in kg
        age: user age in years
        sex: 'M' or 'F'

    Returns:
        dict with navy_bf, deurenberg_bf, ensemble_bf, measurements, category
    """
    # Step 1: Calibrate pixel scale
    nose = _to_px(landmarks[NOSE], img_w, img_h)
    l_ankle = _to_px(landmarks[L_ANKLE], img_w, img_h)
    r_ankle = _to_px(landmarks[R_ANKLE], img_w, img_h)
    ankle_mid = (l_ankle + r_ankle) / 2

    body_height_px = _dist(nose, ankle_mid)
    if body_height_px < 50:
        return None

    # Scale: add ~8% for top of head above nose
    estimated_full_height_px = body_height_px * 1.08
    px_per_cm = estimated_full_height_px / height_cm

    # Step 2: Extract measurements from landmarks
    ls = _to_px(landmarks[L_SHOULDER], img_w, img_h)
    rs = _to_px(landmarks[R_SHOULDER], img_w, img_h)
    lh = _to_px(landmarks[L_HIP], img_w, img_h)
    rh = _to_px(landmarks[R_HIP], img_w, img_h)

    shoulder_width_px = _dist(ls, rs)
    hip_width_px = _dist(lh, rh)

    # Waist estimated at ~85% of hip width (above hip bone level)
    waist_width_px = hip_width_px * 0.85

    # Neck estimated from shoulder width
    neck_width_px = shoulder_width_px * 0.35

    # Convert to cm
    shoulder_width_cm = shoulder_width_px / px_per_cm
    hip_width_cm = hip_width_px / px_per_cm
    waist_width_cm = waist_width_px / px_per_cm
    neck_width_cm = neck_width_px / px_per_cm

    # Step 3: Estimate circumferences (ellipse approximation)
    waist_circ = _ellipse_circumference(waist_width_cm, depth_ratio=0.70)
    hip_circ = _ellipse_circumference(hip_width_cm, depth_ratio=0.65)
    neck_circ = _ellipse_circumference(neck_width_cm, depth_ratio=0.85)

    # Step 4: Navy Formula
    navy_bf = None
    try:
        if sex == 'M':
            if waist_circ > neck_circ:
                navy_bf = 86.010 * math.log10(waist_circ - neck_circ) - 70.041 * math.log10(height_cm) + 36.76
        else:
            if (waist_circ + hip_circ) > neck_circ:
                navy_bf = 163.205 * math.log10(waist_circ + hip_circ - neck_circ) - 97.684 * math.log10(height_cm) - 78.387
    except (ValueError, ZeroDivisionError):
        navy_bf = None

    # Clamp Navy result
    if navy_bf is not None:
        navy_bf = max(3.0, min(60.0, navy_bf))

    # Step 5: Deurenberg (BMI-based)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    sex_factor = 1 if sex == 'M' else 0
    deurenberg_bf = 1.20 * bmi + 0.23 * age - 10.8 * sex_factor - 5.4
    deurenberg_bf = max(3.0, min(60.0, deurenberg_bf))

    # Step 6: Ensemble
    if navy_bf is not None:
        ensemble_bf = navy_bf * 0.6 + deurenberg_bf * 0.4
    else:
        ensemble_bf = deurenberg_bf

    # Category classification
    category = _classify_bf(ensemble_bf, sex, age)

    return {
        "navy_bf": round(navy_bf, 1) if navy_bf else None,
        "deurenberg_bf": round(deurenberg_bf, 1),
        "ensemble_bf": round(ensemble_bf, 1),
        "bmi": round(bmi, 1),
        "measurements": {
            "waist_circ_cm": round(waist_circ, 1),
            "hip_circ_cm": round(hip_circ, 1),
            "neck_circ_cm": round(neck_circ, 1),
            "shoulder_width_cm": round(shoulder_width_cm, 1),
        },
        "category": category,
    }


def _classify_bf(bf, sex, age):
    """Classify body fat percentage into category."""
    if sex == 'M':
        if bf < 6:
            return {"label": "Essencial", "color": "blue"}
        elif bf < 14:
            return {"label": "Atleta", "color": "green"}
        elif bf < 18:
            return {"label": "Fitness", "color": "green"}
        elif bf < 25:
            return {"label": "Normal", "color": "yellow"}
        else:
            return {"label": "Acima", "color": "red"}
    else:
        if bf < 14:
            return {"label": "Essencial", "color": "blue"}
        elif bf < 21:
            return {"label": "Atleta", "color": "green"}
        elif bf < 25:
            return {"label": "Fitness", "color": "green"}
        elif bf < 32:
            return {"label": "Normal", "color": "yellow"}
        else:
            return {"label": "Acima", "color": "red"}


def draw_measurement_overlay(image, landmarks, img_w, img_h):
    """Draw measurement lines on image for visualization."""
    overlay = image.copy()

    ls = _to_px(landmarks[L_SHOULDER], img_w, img_h)
    rs = _to_px(landmarks[R_SHOULDER], img_w, img_h)
    lh = _to_px(landmarks[L_HIP], img_w, img_h)
    rh = _to_px(landmarks[R_HIP], img_w, img_h)

    # Shoulder line (cyan)
    cv2.line(overlay, tuple(ls.astype(int)), tuple(rs.astype(int)), (255, 255, 0), 3)

    # Hip line (magenta)
    cv2.line(overlay, tuple(lh.astype(int)), tuple(rh.astype(int)), (255, 0, 255), 3)

    # Waist line (estimated, yellow) — between shoulders and hips
    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2
    waist_y = shoulder_mid + (hip_mid - shoulder_mid) * 0.65
    waist_half = (rh - lh) * 0.85 / 2
    waist_l = (waist_y - waist_half).astype(int)
    waist_r = (waist_y + waist_half).astype(int)
    cv2.line(overlay, tuple(waist_l), tuple(waist_r), (0, 255, 255), 3)

    # Neck line (green)
    neck_center = _to_px(landmarks[NOSE], img_w, img_h) * 0.3 + shoulder_mid * 0.7
    neck_half = (rs - ls) * 0.35 / 2
    neck_l = (neck_center - neck_half).astype(int)
    neck_r = (neck_center + neck_half).astype(int)
    cv2.line(overlay, tuple(neck_l), tuple(neck_r), (0, 255, 0), 3)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(0.7, img_w / 1500))
    th = max(1, int(scale * 2))

    cv2.putText(overlay, "Ombros", tuple((rs + np.array([10, 0])).astype(int)), font, scale, (255, 255, 0), th)
    cv2.putText(overlay, "Quadril", tuple((rh + np.array([10, 0])).astype(int)), font, scale, (255, 0, 255), th)
    cv2.putText(overlay, "Cintura", tuple((waist_r + np.array([10, 0])).astype(int)), font, scale, (0, 255, 255), th)
    cv2.putText(overlay, "Pescoco", tuple((neck_r + np.array([10, 0])).astype(int)), font, scale, (0, 255, 0), th)

    return overlay


BF_REFERENCE_TABLE = {
    "M": [
        ("Essencial", "2-5%"),
        ("Atleta", "6-13%"),
        ("Fitness", "14-17%"),
        ("Normal", "18-24%"),
        ("Acima", "25%+"),
    ],
    "F": [
        ("Essencial", "10-13%"),
        ("Atleta", "14-20%"),
        ("Fitness", "21-24%"),
        ("Normal", "25-31%"),
        ("Acima", "32%+"),
    ],
}
