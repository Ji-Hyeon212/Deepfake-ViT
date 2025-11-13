"""
Face Detection Module using RetinaFace
Detects faces and extracts 5-point landmarks for alignment
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from facenet_pytorch import MTCNN
import cv2


class FaceDetector:
    """
    RetinaFace-based face detector wrapper
    Outputs: bounding boxes, landmarks, confidence scores
    """

    def __init__(self, config: Dict):
        """
        Initialize face detector

        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = config['confidence_threshold']
        self.keep_top_k = config.get('keep_top_k', 1)

        # Initialize detector
        # Using MTCNN from facenet-pytorch as a reliable alternative
        # For production, use insightface's RetinaFace
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, 0.8],  # MTCNN thresholds
            post_process=False
        )

        print(f"[FaceDetector] Initialized on device: {self.device}")

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect faces in image and extract landmarks

        Args:
            image: Input image (H, W, 3) in RGB format

        Returns:
            Dictionary containing:
                - bbox: [x1, y1, x2, y2] bounding box
                - landmarks: (5, 2) array of landmark coordinates
                - confidence: detection confidence score
            Returns None if no face detected
        """
        # Convert to RGB if needed
        if image.shape[2] == 3 and self._is_bgr(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)

        # No face detected
        if boxes is None or len(boxes) == 0:
            return None

        # Filter by confidence
        valid_indices = probs >= self.confidence_threshold
        if not valid_indices.any():
            return None

        boxes = boxes[valid_indices]
        probs = probs[valid_indices]
        landmarks = landmarks[valid_indices]

        # Keep only top-k faces (by confidence)
        if len(boxes) > self.keep_top_k:
            top_indices = np.argsort(probs)[-self.keep_top_k:]
            boxes = boxes[top_indices]
            probs = probs[top_indices]
            landmarks = landmarks[top_indices]

        # Get the most confident face
        best_idx = np.argmax(probs)

        result = {
            'bbox': boxes[best_idx].astype(np.float32),
            'landmarks': landmarks[best_idx].astype(np.float32),
            'confidence': float(probs[best_idx]),
            'num_faces': len(boxes)
        }

        return result

    def batch_detect(self, images: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Detect faces in batch of images

        Args:
            images: List of images (H, W, 3)

        Returns:
            List of detection results (same format as detect())
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results

    def _is_bgr(self, image: np.ndarray) -> bool:
        """
        Heuristic to check if image is in BGR format
        Assumes most images have more red/blue than green
        """
        mean_channels = image.mean(axis=(0, 1))
        # If green channel is significantly lower, likely RGB
        # Otherwise might be BGR
        return mean_channels[1] < min(mean_channels[0], mean_channels[2]) * 0.9

    def visualize_detection(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw detection results on image for visualization

        Args:
            image: Original image
            detection: Detection result dictionary

        Returns:
            Image with drawn bounding box and landmarks
        """
        vis_image = image.copy()

        if detection is None:
            return vis_image

        # Draw bounding box
        bbox = detection['bbox'].astype(int)
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 2)

        # Draw landmarks
        landmarks = detection['landmarks'].astype(int)
        for i, (x, y) in enumerate(landmarks):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                     (255, 255, 0), (255, 0, 255)][i]
            cv2.circle(vis_image, (x, y), 3, color, -1)

        # Draw confidence
        conf_text = f"Conf: {detection['confidence']:.3f}"
        cv2.putText(vis_image, conf_text, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image

    def get_face_roi(self, image: np.ndarray, detection: Dict,
                     margin: float = 0.2) -> np.ndarray:
        """
        Extract face region with margin

        Args:
            image: Original image
            detection: Detection result
            margin: Margin ratio to add around bbox

        Returns:
            Cropped face region
        """
        bbox = detection['bbox']
        h, w = image.shape[:2]

        # Calculate margins
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)

        # Apply margins
        x1 = max(0, int(bbox[0] - margin_x))
        y1 = max(0, int(bbox[1] - margin_y))
        x2 = min(w, int(bbox[2] + margin_x))
        y2 = min(h, int(bbox[3] + margin_y))

        face_roi = image[y1:y2, x1:x2]

        # Update landmarks to ROI coordinates
        adjusted_landmarks = detection['landmarks'].copy()
        adjusted_landmarks[:, 0] -= x1
        adjusted_landmarks[:, 1] -= y1

        return face_roi, adjusted_landmarks, (x1, y1, x2, y2)


class RetinaFaceDetector(FaceDetector):
    """
    Alternative implementation using InsightFace's RetinaFace
    Use this for production with better performance
    """

    def __init__(self, config: Dict):
        """Initialize RetinaFace detector"""
        self.config = config
        self.device = config['device']
        self.confidence_threshold = config['confidence_threshold']

        try:
            from insightface.app import FaceAnalysis

            # Initialize InsightFace
            self.app = FaceAnalysis(
                name='buffalo_l',  # or 'antelopev2'
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            print(f"[RetinaFace] Initialized with InsightFace")

        except ImportError:
            raise ImportError(
                "InsightFace not installed. Install with: "
                "pip install insightface onnxruntime-gpu"
            )

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect faces using InsightFace RetinaFace

        Args:
            image: Input image (H, W, 3) in RGB or BGR

        Returns:
            Detection dictionary or None
        """
        # InsightFace expects BGR
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Detect faces
        faces = self.app.get(image_bgr)

        if len(faces) == 0:
            return None

        # Filter by confidence
        faces = [f for f in faces if f.det_score >= self.confidence_threshold]

        if len(faces) == 0:
            return None

        # Sort by confidence and take best
        faces.sort(key=lambda x: x.det_score, reverse=True)
        best_face = faces[0]

        # Extract information
        result = {
            'bbox': best_face.bbox.astype(np.float32),
            'landmarks': best_face.kps.astype(np.float32),  # 5 points
            'confidence': float(best_face.det_score),
            'num_faces': len(faces)
        }

        return result


# Factory function
def create_face_detector(config: Dict) -> FaceDetector:
    """
    Factory function to create appropriate face detector

    Args:
        config: Detection configuration

    Returns:
        FaceDetector instance
    """
    model_type = config.get('model', 'retinaface')

    if model_type == 'retinaface':
        try:
            return RetinaFaceDetector(config)
        except ImportError:
            print("[Warning] InsightFace not available, falling back to MTCNN")
            return FaceDetector(config)
    else:
        return FaceDetector(config)


