"""
Main Preprocessing Pipeline
Integrates face detection, alignment, and quality assessment
Outputs data ready for feature extraction stage
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict

from .face_detector import create_face_detector
from .face_aligner import FaceAligner, NormalizationProcessor
from .quality_checker import QualityChecker


@dataclass
class PreprocessingOutput:
    """
    Data class for preprocessing pipeline output
    Designed for seamless transfer to feature extraction stage
    """
    # Primary outputs
    aligned_face: np.ndarray  # (224, 224, 3) RGB image
    landmarks: np.ndarray  # (5, 2) landmark coordinates in aligned space

    # Quality metrics
    quality_score: float
    is_valid: bool
    quality_metrics: Dict

    # Metadata
    original_bbox: np.ndarray
    detection_confidence: float
    transformation_matrix: np.ndarray

    # Source information
    image_id: str
    dataset_name: str
    label: str  # 'real' or 'fake'

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'aligned_face_shape': self.aligned_face.shape,
            'landmarks': self.landmarks.tolist(),
            'quality_score': float(self.quality_score),
            'is_valid': bool(self.is_valid),
            'quality_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in self.quality_metrics.items()},
            'original_bbox': self.original_bbox.tolist(),
            'detection_confidence': float(self.detection_confidence),
            'transformation_matrix': self.transformation_matrix.tolist(),
            'image_id': self.image_id,
            'dataset_name': self.dataset_name,
            'label': self.label
        }

    def to_tensor(self, normalize: bool = True) -> torch.Tensor:
        """
        Convert aligned face to PyTorch tensor
        Ready for feature extraction model input

        Args:
            normalize: Apply ImageNet normalization

        Returns:
            Tensor of shape (3, 224, 224)
        """
        # Convert to float and normalize to [0, 1]
        image = self.aligned_face.astype(np.float32) / 255.0

        if normalize:
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std

        # Convert to CHW format
        tensor = torch.from_numpy(image.transpose(2, 0, 1))

        return tensor


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for deepfake detection

    Pipeline stages:
    1. Face Detection (RetinaFace)
    2. Face Alignment (Similarity Transform)
    3. Quality Assessment
    4. Normalization

    Output: PreprocessingOutput object ready for feature extraction
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline

        Args:
            config: Complete configuration dictionary
        """
        self.config = config

        # Initialize components
        self.detector = create_face_detector(config['detection'])
        self.aligner = FaceAligner(config['alignment'])
        self.quality_checker = QualityChecker(config['quality'])
        self.normalizer = NormalizationProcessor(config['pipeline']['normalize'])

        # Output settings
        self.output_config = config['output']
        self.save_intermediate = config['pipeline'].get('save_intermediate', True)

        print("[PreprocessingPipeline] Initialized successfully")

    def process_image(self,
                      image: np.ndarray,
                      image_id: str,
                      dataset_name: str,
                      label: str) -> Optional[PreprocessingOutput]:
        """
        Process single image through complete pipeline

        Args:
            image: Input image (H, W, 3) in RGB
            image_id: Unique image identifier
            dataset_name: Source dataset name
            label: 'real' or 'fake'

        Returns:
            PreprocessingOutput object or None if processing failed
        """
        # Stage 1: Face Detection
        detection = self.detector.detect(image)

        if detection is None:
            print(f"[Warning] No face detected in {image_id}")
            return None

        # Stage 2: Quality Check (on original detection)
        quality_result = self.quality_checker.check_quality(image, detection)

        if not quality_result['is_valid']:
            print(f"[Warning] Quality check failed for {image_id}: {quality_result['reasons']}")
            # Note: We still process but mark as invalid

        # Stage 3: Face Alignment
        aligned_face, tform_matrix = self.aligner.align(image, detection['landmarks'])

        # Stage 4: Get aligned landmarks
        aligned_landmarks = self.aligner.get_aligned_landmarks(
            detection['landmarks'],
            tform_matrix
        )

        # Create output object
        output = PreprocessingOutput(
            aligned_face=aligned_face,
            landmarks=aligned_landmarks,
            quality_score=quality_result['overall_score'],
            is_valid=quality_result['is_valid'],
            quality_metrics=quality_result['scores'],
            original_bbox=detection['bbox'],
            detection_confidence=detection['confidence'],
            transformation_matrix=tform_matrix,
            image_id=image_id,
            dataset_name=dataset_name,
            label=label
        )

        return output

    def process_batch(self,
                      images: List[np.ndarray],
                      image_ids: List[str],
                      dataset_names: List[str],
                      labels: List[str]) -> List[Optional[PreprocessingOutput]]:
        """
        Process batch of images

        Args:
            images: List of images
            image_ids: List of image IDs
            dataset_names: List of dataset names
            labels: List of labels

        Returns:
            List of PreprocessingOutput objects (None for failed images)
        """
        results = []

        for img, img_id, ds_name, label in zip(images, image_ids, dataset_names, labels):
            result = self.process_image(img, img_id, ds_name, label)
            results.append(result)

        return results

    def save_output(self, output: PreprocessingOutput, output_dir: Path) -> Dict[str, Path]:
        """
        Save preprocessing output to disk

        Args:
            output: PreprocessingOutput object
            output_dir: Base output directory

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)

        # Create subdirectories
        faces_dir = output_dir / self.output_config['faces_dir']
        landmarks_dir = output_dir / self.output_config['landmarks_dir']
        metadata_dir = output_dir / self.output_config['metadata_dir']

        for directory in [faces_dir, landmarks_dir, metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename_base = f"{output.dataset_name}_{output.label}_{output.image_id}"

        saved_paths = {}

        # 1. Save aligned face image
        face_path = faces_dir / f"{filename_base}.png"
        cv2.imwrite(str(face_path), cv2.cvtColor(output.aligned_face, cv2.COLOR_RGB2BGR))
        saved_paths['face'] = face_path

        # 2. Save landmarks
        landmarks_path = landmarks_dir / f"{filename_base}_landmarks.npy"
        np.save(landmarks_path, output.landmarks)
        saved_paths['landmarks'] = landmarks_path

        # 3. Save metadata
        metadata_path = metadata_dir / f"{filename_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(output.to_dict(), f, indent=2)
        saved_paths['metadata'] = metadata_path

        return saved_paths

    def load_output(self, output_dir: Path, filename_base: str) -> PreprocessingOutput:
        """
        Load saved preprocessing output

        Args:
            output_dir: Base output directory
            filename_base: Base filename without extension

        Returns:
            PreprocessingOutput object
        """
        output_dir = Path(output_dir)

        # Load aligned face
        face_path = output_dir / self.output_config['faces_dir'] / f"{filename_base}.png"
        aligned_face = cv2.imread(str(face_path))
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Load landmarks
        landmarks_path = output_dir / self.output_config['landmarks_dir'] / f"{filename_base}_landmarks.npy"
        landmarks = np.load(landmarks_path)

        # Load metadata
        metadata_path = output_dir / self.output_config['metadata_dir'] / f"{filename_base}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Reconstruct PreprocessingOutput
        output = PreprocessingOutput(
            aligned_face=aligned_face,
            landmarks=landmarks,
            quality_score=metadata['quality_score'],
            is_valid=metadata['is_valid'],
            quality_metrics=metadata['quality_metrics'],
            original_bbox=np.array(metadata['original_bbox']),
            detection_confidence=metadata['detection_confidence'],
            transformation_matrix=np.array(metadata['transformation_matrix']),
            image_id=metadata['image_id'],
            dataset_name=metadata['dataset_name'],
            label=metadata['label']
        )

        return output

    def visualize_pipeline(self,
                           image: np.ndarray,
                           output: PreprocessingOutput) -> np.ndarray:
        """
        Create visualization of pipeline processing

        Args:
            image: Original input image
            output: PreprocessingOutput object

        Returns:
            Visualization image showing pipeline stages
        """
        # Create canvas
        canvas_height = max(image.shape[0], output.aligned_face.shape[0])
        canvas_width = image.shape[1] + output.aligned_face.shape[0] + 400  # Extra space for info
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # 1. Original image with detection
        img_with_detection = image.copy()
        bbox = output.original_bbox.astype(int)
        cv2.rectangle(img_with_detection, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 2)

        # Draw original landmarks
        for x, y in output.landmarks:
            # Transform landmarks back to original space (approximate)
            cv2.circle(img_with_detection, (int(x), int(y)), 3, (255, 0, 0), -1)

        canvas[:image.shape[0], :image.shape[1]] = img_with_detection

        # 2. Aligned face
        x_offset = image.shape[1] + 20
        aligned_resized = cv2.resize(output.aligned_face, (image.shape[0], image.shape[0]))
        canvas[:aligned_resized.shape[0], x_offset:x_offset + aligned_resized.shape[1]] = aligned_resized

        # Draw aligned landmarks
        for x, y in output.landmarks:
            cv2.circle(canvas, (x_offset + int(x), int(y)), 2, (0, 255, 0), -1)

        # 3. Information panel
        info_x = x_offset + aligned_resized.shape[1] + 20
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        info_texts = [
            f"Image ID: {output.image_id}",
            f"Dataset: {output.dataset_name}",
            f"Label: {output.label}",
            "",
            f"Quality Score: {output.quality_score:.3f}",
            f"Valid: {output.is_valid}",
            f"Detection Conf: {output.detection_confidence:.3f}",
            "",
            "Quality Metrics:",
        ]

        for key, value in output.quality_metrics.items():
            info_texts.append(f"  {key}: {value:.2f}")

        for text in info_texts:
            cv2.putText(canvas, text, (info_x, y_pos), font, 0.4, (0, 0, 0), 1)
            y_pos += 20

        # Add stage labels
        cv2.putText(canvas, "1. Detection", (10, 30), font, 0.6, (255, 0, 0), 2)
        cv2.putText(canvas, "2. Alignment", (x_offset + 10, 30), font, 0.6, (0, 255, 0), 2)

        return canvas

    def get_statistics(self, outputs: List[PreprocessingOutput]) -> Dict:
        """
        Compute statistics over processed outputs

        Args:
            outputs: List of PreprocessingOutput objects

        Returns:
            Statistics dictionary
        """
        valid_outputs = [o for o in outputs if o is not None]

        if len(valid_outputs) == 0:
            return {'error': 'No valid outputs'}

        stats = {
            'total_processed': len(outputs),
            'successful': len(valid_outputs),
            'success_rate': len(valid_outputs) / len(outputs),
            'valid_quality': sum(o.is_valid for o in valid_outputs),
            'quality_pass_rate': sum(o.is_valid for o in valid_outputs) / len(valid_outputs),
            'avg_quality_score': np.mean([o.quality_score for o in valid_outputs]),
            'avg_detection_confidence': np.mean([o.detection_confidence for o in valid_outputs]),
            'quality_score_std': np.std([o.quality_score for o in valid_outputs]),
        }

        # Per-metric statistics
        all_metrics = {}
        for output in valid_outputs:
            for key, value in output.quality_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        stats['quality_metrics'] = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for key, values in all_metrics.items()
        }

        return stats


def create_pipeline_from_config(config_path: str) -> PreprocessingPipeline:
    """
    Factory function to create pipeline from config file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Initialized PreprocessingPipeline
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = PreprocessingPipeline(config)

    return pipeline