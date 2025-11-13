"""
Image Quality Assessment Module
Evaluates face image quality for preprocessing pipeline
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from skimage import exposure


class QualityChecker:
    """
    Assesses face image quality based on multiple criteria:
    - Face size
    - Blur detection
    - Brightness/Contrast
    - Occlusion (optional)
    """

    def __init__(self, config: Dict):
        """
        Initialize quality checker

        Args:
            config: Quality assessment configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)

        # Size constraints
        self.min_face_size = config.get('min_face_size', 50)
        self.max_face_size = config.get('max_face_size', 2000)

        # Blur threshold (Laplacian variance)
        self.blur_threshold = config.get('blur_threshold', 100.0)

        # Brightness/Contrast thresholds
        self.min_brightness = config.get('min_brightness', 30)
        self.max_brightness = config.get('max_brightness', 225)
        self.min_contrast = config.get('min_contrast', 20)

        # Occlusion detection
        self.check_occlusion = config.get('check_occlusion', True)
        self.occlusion_threshold = config.get('occlusion_threshold', 0.3)

        print(f"[QualityChecker] Initialized (enabled: {self.enabled})")

    def check_quality(self, image: np.ndarray, detection_info: Dict) -> Dict:
        """
        Perform comprehensive quality check

        Args:
            image: Face image (H, W, 3)
            detection_info: Detection information (bbox, landmarks, confidence)

        Returns:
            Dictionary containing:
                - is_valid: Overall quality flag (bool)
                - scores: Individual quality scores
                - reasons: List of failure reasons if invalid
        """
        if not self.enabled:
            return {
                'is_valid': True,
                'scores': {},
                'reasons': []
            }

        scores = {}
        reasons = []

        # 1. Face size check
        size_valid, size_score = self._check_face_size(detection_info['bbox'])
        scores['face_size'] = size_score
        if not size_valid:
            reasons.append(f"Invalid face size: {size_score:.1f}px")

        # 2. Blur detection
        blur_valid, blur_score = self._check_blur(image)
        scores['blur'] = blur_score
        if not blur_valid:
            reasons.append(f"Image too blurry: {blur_score:.2f}")

        # 3. Brightness check
        bright_valid, bright_score = self._check_brightness(image)
        scores['brightness'] = bright_score
        if not bright_valid:
            reasons.append(f"Invalid brightness: {bright_score:.1f}")

        # 4. Contrast check
        contrast_valid, contrast_score = self._check_contrast(image)
        scores['contrast'] = contrast_score
        if not contrast_valid:
            reasons.append(f"Low contrast: {contrast_score:.1f}")

        # 5. Occlusion check (if enabled)
        if self.check_occlusion:
            occl_valid, occl_score = self._check_occlusion(image, detection_info['landmarks'])
            scores['occlusion'] = occl_score
            if not occl_valid:
                reasons.append(f"Face occlusion detected: {occl_score:.2f}")

        # 6. Detection confidence
        scores['detection_confidence'] = detection_info['confidence']

        # Overall validity
        is_valid = len(reasons) == 0

        # Compute overall quality score (0-1)
        quality_weights = {
            'face_size': 0.15,
            'blur': 0.25,
            'brightness': 0.15,
            'contrast': 0.15,
            'occlusion': 0.15,
            'detection_confidence': 0.15
        }

        overall_score = sum(
            self._normalize_score(k, v) * quality_weights.get(k, 0)
            for k, v in scores.items()
        )

        return {
            'is_valid': is_valid,
            'overall_score': overall_score,
            'scores': scores,
            'reasons': reasons
        }

    def _check_face_size(self, bbox: np.ndarray) -> Tuple[bool, float]:
        """
        Check if face size is within acceptable range

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            (is_valid, face_size)
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        face_size = min(width, height)

        is_valid = self.min_face_size <= face_size <= self.max_face_size

        return is_valid, float(face_size)

    def _check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect blur using Laplacian variance method

        Args:
            image: Input image (H, W, 3)

        Returns:
            (is_valid, blur_score) - Higher score means sharper
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        is_valid = blur_score >= self.blur_threshold

        return is_valid, float(blur_score)

    def _check_brightness(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check image brightness

        Args:
            image: Input image (H, W, 3)

        Returns:
            (is_valid, brightness_score)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Mean brightness
        brightness = gray.mean()

        is_valid = self.min_brightness <= brightness <= self.max_brightness

        return is_valid, float(brightness)

    def _check_contrast(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check image contrast

        Args:
            image: Input image (H, W, 3)

        Returns:
            (is_valid, contrast_score)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Standard deviation as contrast measure
        contrast = gray.std()

        is_valid = contrast >= self.min_contrast

        return is_valid, float(contrast)

    def _check_occlusion(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Detect face occlusion using landmark-based regions
        Simple heuristic: check if regions around landmarks are uniform (potentially occluded)

        Args:
            image: Input image (H, W, 3)
            landmarks: (5, 2) facial landmarks

        Returns:
            (is_valid, occlusion_score) - Lower score means more occlusion
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Define regions around each landmark
        region_size = min(h, w) // 10
        occlusion_scores = []

        for x, y in landmarks.astype(int):
            # Extract region around landmark
            x1 = max(0, x - region_size // 2)
            y1 = max(0, y - region_size // 2)
            x2 = min(w, x + region_size // 2)
            y2 = min(h, y + region_size // 2)

            region = gray[y1:y2, x1:x2]

            if region.size == 0:
                continue

            # Compute variance (low variance might indicate occlusion or uniform area)
            variance = region.var()
            occlusion_scores.append(variance)

        if len(occlusion_scores) == 0:
            return False, 0.0

        # Average variance across landmarks
        avg_variance = np.mean(occlusion_scores)

        # Normalize to [0, 1] range (higher is better)
        # Typical variance range: 0-5000
        normalized_score = min(avg_variance / 1000.0, 1.0)

        is_valid = normalized_score >= self.occlusion_threshold

        return is_valid, float(normalized_score)

    def _normalize_score(self, score_name: str, score_value: float) -> float:
        """
        Normalize different scores to [0, 1] range

        Args:
            score_name: Name of the score
            score_value: Raw score value

        Returns:
            Normalized score [0, 1]
        """
        if score_name == 'face_size':
            # Optimal range: 100-500 pixels
            optimal_min, optimal_max = 100, 500
            if score_value < optimal_min:
                return score_value / optimal_min
            elif score_value > optimal_max:
                return max(0, 1 - (score_value - optimal_max) / optimal_max)
            else:
                return 1.0

        elif score_name == 'blur':
            # Higher is better, normalize with sigmoid-like function
            return min(score_value / (self.blur_threshold * 2), 1.0)

        elif score_name == 'brightness':
            # Optimal range: 100-150
            optimal = 127.5
            deviation = abs(score_value - optimal)
            return max(0, 1 - deviation / optimal)

        elif score_name == 'contrast':
            # Higher is better
            return min(score_value / (self.min_contrast * 5), 1.0)

        elif score_name == 'occlusion':
            # Already normalized
            return score_value

        elif score_name == 'detection_confidence':
            # Already normalized
            return score_value

        else:
            return 0.5  # Default

    def visualize_quality(self, image: np.ndarray, quality_result: Dict) -> np.ndarray:
        """
        Visualize quality assessment results

        Args:
            image: Input image
            quality_result: Quality check result dictionary

        Returns:
            Image with quality information overlay
        """
        vis_image = image.copy()

        # Determine border color based on validity
        border_color = (0, 255, 0) if quality_result['is_valid'] else (255, 0, 0)

        # Draw border
        cv2.rectangle(vis_image, (0, 0), (vis_image.shape[1] - 1, vis_image.shape[0] - 1),
                      border_color, 3)

        # Prepare text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Overall score
        overall_text = f"Quality: {quality_result['overall_score']:.2f}"
        cv2.putText(vis_image, overall_text, (10, y_offset),
                    font, font_scale, (255, 255, 255), thickness)
        y_offset += 20

        # Individual scores
        for score_name, score_value in quality_result['scores'].items():
            score_text = f"{score_name}: {score_value:.2f}"
            cv2.putText(vis_image, score_text, (10, y_offset),
                        font, font_scale * 0.8, (200, 200, 200), thickness)
            y_offset += 18

        # Failure reasons
        if quality_result['reasons']:
            y_offset += 10
            cv2.putText(vis_image, "Issues:", (10, y_offset),
                        font, font_scale, (255, 100, 100), thickness)
            y_offset += 20

            for reason in quality_result['reasons']:
                cv2.putText(vis_image, f"- {reason}", (10, y_offset),
                            font, font_scale * 0.7, (255, 150, 150), thickness)
                y_offset += 18

        return vis_image