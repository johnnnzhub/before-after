"""Motor de alinhamento corporal usando MediaPipe Pose + OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_DIR = Path(__file__).parent / "models"

# Landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
NOSE = 0
L_ANKLE = 27
R_ANKLE = 28
L_KNEE = 25
R_KNEE = 26


@dataclass
class TorsoMetrics:
    shoulder_mid: np.ndarray
    hip_mid: np.ndarray
    torso_center: np.ndarray
    torso_length: float
    torso_angle: float  # degrees from vertical


@dataclass
class AlignmentResult:
    before_aligned: np.ndarray
    after_aligned: np.ndarray
    composite: np.ndarray
    confidence: float
    landmarks_before: list | None = None
    landmarks_after: list | None = None
    warnings: list = field(default_factory=list)


def _download_model() -> str:
    """Download pose model if not cached. Returns path."""
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "pose_landmarker_heavy.task"
    if not model_path.exists():
        print("Baixando modelo MediaPipe Pose (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return str(model_path)


def _to_px(landmark, img_w, img_h):
    """Convert normalized landmark to pixel coordinates."""
    return np.array([landmark.x * img_w, landmark.y * img_h])


class BodyAligner:
    def __init__(self):
        model_path = _download_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def _detect(self, image: np.ndarray):
        """Detect pose landmarks. Returns list of landmarks or None."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        return result.pose_landmarks[0]

    def _get_torso_metrics(self, landmarks, img_w, img_h) -> TorsoMetrics | None:
        """Extract torso reference frame from landmarks."""
        ls = _to_px(landmarks[L_SHOULDER], img_w, img_h)
        rs = _to_px(landmarks[R_SHOULDER], img_w, img_h)
        lh = _to_px(landmarks[L_HIP], img_w, img_h)
        rh = _to_px(landmarks[R_HIP], img_w, img_h)

        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        torso_center = (shoulder_mid + hip_mid) / 2
        torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))

        if torso_length < 10:
            return None

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        torso_angle = float(np.degrees(np.arctan2(dx, dy)))

        return TorsoMetrics(shoulder_mid, hip_mid, torso_center, torso_length, torso_angle)

    def _compute_affine(self, src_lm, dst_lm, src_shape, dst_shape):
        """Compute affine transform to align src image onto dst reference frame."""
        sh, sw = src_shape[:2]
        dh, dw = dst_shape[:2]

        src_sm = (_to_px(src_lm[L_SHOULDER], sw, sh) + _to_px(src_lm[R_SHOULDER], sw, sh)) / 2
        src_hm = (_to_px(src_lm[L_HIP], sw, sh) + _to_px(src_lm[R_HIP], sw, sh)) / 2
        dst_sm = (_to_px(dst_lm[L_SHOULDER], dw, dh) + _to_px(dst_lm[R_SHOULDER], dw, dh)) / 2
        dst_hm = (_to_px(dst_lm[L_HIP], dw, dh) + _to_px(dst_lm[R_HIP], dw, dh)) / 2

        # Third point: perpendicular to torso axis at shoulder level
        def perp(sm, hm):
            axis = hm - sm
            perp_vec = np.array([-axis[1], axis[0]])
            return sm + perp_vec * 0.5

        src_pts = np.float32([src_sm, src_hm, perp(src_sm, src_hm)])
        dst_pts = np.float32([dst_sm, dst_hm, perp(dst_sm, dst_hm)])

        return cv2.getAffineTransform(src_pts, dst_pts)

    def _compute_crop(self, landmarks, img_w, img_h, torso_length, padding_factor=0.3):
        """Compute body bounding box for cropping."""
        # Get all visible landmark positions
        points = []
        for lm in landmarks:
            if lm.visibility > 0.3:
                points.append(_to_px(lm, img_w, img_h))

        if len(points) < 4:
            return 0, 0, img_w, img_h

        points = np.array(points)
        padding = torso_length * padding_factor

        x_min = max(0, int(np.min(points[:, 0]) - padding))
        y_min = max(0, int(np.min(points[:, 1]) - padding))
        x_max = min(img_w, int(np.max(points[:, 0]) + padding))
        y_max = min(img_h, int(np.max(points[:, 1]) + padding))

        return x_min, y_min, x_max, y_max

    def _compute_confidence(self, landmarks) -> float:
        """Average visibility of torso landmarks."""
        indices = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        visibilities = [landmarks[i].visibility for i in indices]
        return float(np.mean(visibilities))

    def align(self, before_img: np.ndarray, after_img: np.ndarray) -> AlignmentResult:
        """Align after_img to before_img's body reference frame."""
        warnings = []

        # Detect landmarks
        before_lm = self._detect(before_img)
        after_lm = self._detect(after_img)

        if before_lm is None and after_lm is None:
            warnings.append("Pose nao detectada em nenhuma foto. Mostrando sem alinhamento.")
            return self._fallback_result(before_img, after_img, warnings)

        if before_lm is None:
            warnings.append("Pose nao detectada na foto 'Antes'. Mostrando sem alinhamento.")
            return self._fallback_result(before_img, after_img, warnings)

        if after_lm is None:
            warnings.append("Pose nao detectada na foto 'Depois'. Mostrando sem alinhamento.")
            return self._fallback_result(before_img, after_img, warnings)

        # Confidence check
        conf_before = self._compute_confidence(before_lm)
        conf_after = self._compute_confidence(after_lm)
        avg_conf = (conf_before + conf_after) / 2

        if avg_conf < 0.3:
            warnings.append(f"Confianca baixa ({avg_conf:.0%}). O alinhamento pode nao ser preciso.")

        # Torso metrics
        bh, bw = before_img.shape[:2]
        ah, aw = after_img.shape[:2]

        before_metrics = self._get_torso_metrics(before_lm, bw, bh)
        after_metrics = self._get_torso_metrics(after_lm, aw, ah)

        if before_metrics is None or after_metrics is None:
            warnings.append("Tronco nao detectado adequadamente. Mostrando sem alinhamento.")
            return self._fallback_result(before_img, after_img, warnings)

        # Scale check
        scale_factor = before_metrics.torso_length / after_metrics.torso_length
        if scale_factor > 2.5 or scale_factor < 0.4:
            warnings.append(f"Diferenca de escala muito grande ({scale_factor:.1f}x). Qualidade pode ser afetada.")

        # Compute and apply affine transform (align after to before's reference)
        M = self._compute_affine(after_lm, before_lm, after_img.shape, before_img.shape)
        after_aligned = cv2.warpAffine(
            after_img, M, (bw, bh),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(40, 40, 40),
        )

        # Re-detect landmarks on aligned image for crop
        after_aligned_lm = self._detect(after_aligned)
        crop_lm = after_aligned_lm if after_aligned_lm else before_lm

        # Compute crop region using before landmarks as reference
        x1, y1, x2, y2 = self._compute_crop(before_lm, bw, bh, before_metrics.torso_length)

        # Also consider aligned after landmarks for broader crop
        if after_aligned_lm:
            ax1, ay1, ax2, ay2 = self._compute_crop(after_aligned_lm, bw, bh, before_metrics.torso_length)
            x1 = min(x1, ax1)
            y1 = min(y1, ay1)
            x2 = max(x2, ax2)
            y2 = max(y2, ay2)

        # Apply crop
        before_cropped = before_img[y1:y2, x1:x2]
        after_cropped = after_aligned[y1:y2, x1:x2]

        # Ensure same dimensions
        target_h = min(before_cropped.shape[0], after_cropped.shape[0])
        target_w = min(before_cropped.shape[1], after_cropped.shape[1])
        before_cropped = before_cropped[:target_h, :target_w]
        after_cropped = after_cropped[:target_h, :target_w]

        # Create composite
        composite = self._create_composite(before_cropped, after_cropped)

        return AlignmentResult(
            before_aligned=before_cropped,
            after_aligned=after_cropped,
            composite=composite,
            confidence=avg_conf,
            landmarks_before=before_lm,
            landmarks_after=after_lm,
            warnings=warnings,
        )

    def _fallback_result(self, before_img, after_img, warnings):
        """Simple side-by-side without alignment."""
        # Resize both to same height
        target_h = min(before_img.shape[0], after_img.shape[0])
        scale_b = target_h / before_img.shape[0]
        scale_a = target_h / after_img.shape[0]

        before_resized = cv2.resize(before_img, None, fx=scale_b, fy=scale_b, interpolation=cv2.INTER_LANCZOS4)
        after_resized = cv2.resize(after_img, None, fx=scale_a, fy=scale_a, interpolation=cv2.INTER_LANCZOS4)

        # Ensure same height after rounding
        h = min(before_resized.shape[0], after_resized.shape[0])
        before_resized = before_resized[:h]
        after_resized = after_resized[:h]

        composite = self._create_composite(before_resized, after_resized)

        return AlignmentResult(
            before_aligned=before_resized,
            after_aligned=after_resized,
            composite=composite,
            confidence=0.0,
            warnings=warnings,
        )

    def _create_composite(self, before: np.ndarray, after: np.ndarray, divider_width=3):
        """Create side-by-side composite image."""
        h = before.shape[0]
        divider = np.full((h, divider_width, 3), 255, dtype=np.uint8)
        return np.hstack([before, divider, after])


def draw_landmarks_overlay(image: np.ndarray, landmarks) -> np.ndarray:
    """Draw pose landmarks on image for debugging."""
    overlay = image.copy()
    h, w = overlay.shape[:2]

    # Connections for skeleton
    connections = [
        (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
        (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
        (L_SHOULDER, 13), (13, 15),  # left arm
        (R_SHOULDER, 14), (14, 16),  # right arm
        (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
        (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
    ]

    # Draw connections
    for i, j in connections:
        if i < len(landmarks) and j < len(landmarks):
            p1 = tuple(int(v) for v in _to_px(landmarks[i], w, h))
            p2 = tuple(int(v) for v in _to_px(landmarks[j], w, h))
            if landmarks[i].visibility > 0.3 and landmarks[j].visibility > 0.3:
                cv2.line(overlay, p1, p2, (0, 255, 0), 2)

    # Draw key points
    for idx in [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]:
        if idx < len(landmarks) and landmarks[idx].visibility > 0.3:
            pt = tuple(int(v) for v in _to_px(landmarks[idx], w, h))
            cv2.circle(overlay, pt, 5, (0, 200, 255), -1)
            cv2.circle(overlay, pt, 5, (0, 0, 0), 1)

    # Draw torso center
    sm = (_to_px(landmarks[L_SHOULDER], w, h) + _to_px(landmarks[R_SHOULDER], w, h)) / 2
    hm = (_to_px(landmarks[L_HIP], w, h) + _to_px(landmarks[R_HIP], w, h)) / 2
    center = ((sm + hm) / 2).astype(int)
    cv2.drawMarker(overlay, tuple(center), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

    return overlay
