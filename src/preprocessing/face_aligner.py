"""
Face Alignment Module
Aligns detected faces using landmark-based transformation
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from skimage import transform as trans


class FaceAligner:
    """
    Aligns face images using similarity or affine transformation
    Based on 5-point facial landmarks
    """

    def __init__(self, config: Dict):
        """
        Initialize face aligner

        Args:
            config: Alignment configuration dictionary
        """
        self.config = config
        self.output_size = tuple(config['output_size'])
        self.method = config.get('method', 'similarity')
        self.border_mode = self._get_border_mode(config.get('border_mode', 'constant'))
        self.border_value = config.get('border_value', 0)

        # Reference landmarks (standard positions in output image)
        self.reference_landmarks = self._get_reference_landmarks(config)

        print(f"[FaceAligner] Initialized with output size: {self.output_size}")

    def _get_reference_landmarks(self, config: Dict) -> np.ndarray:
        """
        Get reference landmark positions (normalized coordinates)

        Returns:
            (5, 2) array of reference landmark positions
        """
        ref_dict = config.get('reference_landmarks', {})

        # Default reference positions (normalized)
        default_refs = {
            'left_eye': [0.31, 0.32],
            'right_eye': [0.69, 0.32],
            'nose': [0.50, 0.55],
            'left_mouth': [0.35, 0.75],
            'right_mouth': [0.65, 0.75]
        }

        # Use provided or default values
        refs = {k: ref_dict.get(k, v) for k, v in default_refs.items()}

        # Convert to absolute coordinates
        w, h = self.output_size
        reference = np.array([
            refs['left_eye'],
            refs['right_eye'],
            refs['nose'],
            refs['left_mouth'],
            refs['right_mouth']
        ], dtype=np.float32)

        reference[:, 0] *= w
        reference[:, 1] *= h

        return reference

    def _get_border_mode(self, mode: str) -> int:
        """Convert border mode string to OpenCV constant"""
        modes = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP
        }
        return modes.get(mode, cv2.BORDER_CONSTANT)

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face image using landmarks

        Args:
            image: Input face image (H, W, 3)
            landmarks: (5, 2) array of landmark coordinates

        Returns:
            Tuple of (aligned_image, transformation_matrix)
        """
        if self.method == 'similarity':
            aligned, tform = self._align_similarity(image, landmarks)
        elif self.method == 'affine':
            aligned, tform = self._align_affine(image, landmarks)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")

        return aligned, tform

    def _align_similarity(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align using similarity transformation (rotation, scale, translation)
        Preserves angles and aspect ratios

        Args:
            image: Input image
            landmarks: Source landmarks (5, 2)

        Returns:
            Aligned image and transformation matrix
        """
        # Estimate similarity transform
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.reference_landmarks)

        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            tform.params[:2],
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )

        return aligned, tform.params

    def _align_affine(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align using affine transformation (allows shearing)
        More flexible but may distort

        Args:
            image: Input image
            landmarks: Source landmarks (5, 2)

        Returns:
            Aligned image and transformation matrix
        """
        # Use first 3 points for affine transformation
        src_pts = landmarks[:3].astype(np.float32)
        dst_pts = self.reference_landmarks[:3].astype(np.float32)

        # Get affine transformation matrix
        tform_matrix = cv2.getAffineTransform(src_pts, dst_pts)

        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            tform_matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )

        return aligned, tform_matrix

    def align_batch(self, images: list, landmarks_list: list) -> Tuple[list, list]:
        """
        Align batch of images

        Args:
            images: List of images
            landmarks_list: List of landmark arrays

        Returns:
            List of aligned images and transformation matrices
        """
        aligned_images = []
        tform_matrices = []

        for image, landmarks in zip(images, landmarks_list):
            aligned, tform = self.align(image, landmarks)
            aligned_images.append(aligned)
            tform_matrices.append(tform)

        return aligned_images, tform_matrices

    def get_aligned_landmarks(self, landmarks: np.ndarray, tform_matrix: np.ndarray) -> np.ndarray:
        """
        Transform landmarks to aligned image coordinates

        Args:
            landmarks: Original landmarks (5, 2)
            tform_matrix: Transformation matrix (2, 3) or (3, 3)

        Returns:
            Transformed landmarks (5, 2)
        """
        # Convert to homogeneous coordinates
        landmarks_homo = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])

        # Apply transformation
        if tform_matrix.shape[0] == 2:
            # Affine matrix (2, 3)
            aligned_landmarks = landmarks_homo @ tform_matrix.T
        else:
            # Full transformation matrix (3, 3)
            aligned_landmarks = (tform_matrix @ landmarks_homo.T).T
            aligned_landmarks = aligned_landmarks[:, :2]

        return aligned_landmarks.astype(np.float32)

    def visualize_alignment(self, original: np.ndarray, aligned: np.ndarray,
                            src_landmarks: np.ndarray, dst_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize original and aligned images side by side

        Args:
            original: Original image
            aligned: Aligned image
            src_landmarks: Original landmarks
            dst_landmarks: Aligned landmarks (optional, will use reference if None)

        Returns:
            Concatenated visualization image
        """
        # Resize original to match aligned size for visualization
        original_resized = cv2.resize(original, self.output_size)

        # Draw landmarks on original
        vis_original = original_resized.copy()
        for x, y in src_landmarks.astype(int):
            # Scale landmarks if needed
            scale_x = self.output_size[0] / original.shape[1]
            scale_y = self.output_size[1] / original.shape[0]
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            cv2.circle(vis_original, (x_scaled, y_scaled), 2, (0, 255, 0), -1)

        # Draw landmarks on aligned
        vis_aligned = aligned.copy()
        ref_landmarks = dst_landmarks if dst_landmarks is not None else self.reference_landmarks
        for x, y in ref_landmarks.astype(int):
            cv2.circle(vis_aligned, (x, y), 2, (0, 255, 0), -1)

        # Concatenate horizontally
        vis = np.hstack([vis_original, vis_aligned])

        # Add labels
        cv2.putText(vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Aligned", (self.output_size[0] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def compute_alignment_quality(self, src_landmarks: np.ndarray,
                                  dst_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute alignment quality metrics

        Args:
            src_landmarks: Original landmarks
            dst_landmarks: Aligned landmarks (should be close to reference)

        Returns:
            Dictionary of quality metrics
        """
        # Mean distance from reference
        distances = np.linalg.norm(dst_landmarks - self.reference_landmarks, axis=1)
        mean_distance = distances.mean()
        max_distance = distances.max()

        # Inter-eye distance (IED) for normalization
        ied = np.linalg.norm(self.reference_landmarks[1] - self.reference_landmarks[0])

        # Normalized metrics
        normalized_mean_dist = mean_distance / ied
        normalized_max_dist = max_distance / ied

        return {
            'mean_distance': float(mean_distance),
            'max_distance': float(max_distance),
            'normalized_mean_distance': float(normalized_mean_dist),
            'normalized_max_distance': float(normalized_max_dist),
            'inter_eye_distance': float(ied)
        }


class NormalizationProcessor:
    """
    Handles image normalization for neural network input
    """

    def __init__(self, config: Dict):
        """
        Initialize normalization processor

        Args:
            config: Normalization configuration
        """
        self.enabled = config.get('enabled', True)
        self.mean = np.array(config.get('mean', [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(config.get('std', [0.229, 0.224, 0.225]), dtype=np.float32)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean and std

        Args:
            image: Input image (H, W, 3) in range [0, 255] or [0, 1]

        Returns:
            Normalized image (H, W, 3)
        """
        if not self.enabled:
            return image

        # Convert to float and scale to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Apply normalization
        normalized = (image - self.mean) / self.std

        return normalized.astype(np.float32)

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization

        Args:
            image: Normalized image (H, W, 3)

        Returns:
            Denormalized image (H, W, 3) in range [0, 255]
        """
        if not self.enabled:
            return image

        # Reverse normalization
        denormalized = (image * self.std) + self.mean

        # Clip and convert to uint8
        denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)

        return denormalized