"""
Preprocessing Module
전처리 단계 (Stage 1) 모듈

주요 기능:
- 얼굴 검출 (Face Detection)
- 얼굴 정렬 (Face Alignment)
- 품질 평가 (Quality Assessment)
- 전처리 파이프라인 통합
"""

from .face_detector import (
    FaceDetector,
    RetinaFaceDetector,
    create_face_detector
)

from .face_aligner import (
    FaceAligner,
    NormalizationProcessor
)

from .quality_checker import (
    QualityChecker
)

from .pipeline import (
    PreprocessingOutput,
    PreprocessingPipeline,
    create_pipeline_from_config
)

__all__ = [
    # Face Detection
    'FaceDetector',
    'RetinaFaceDetector',
    'create_face_detector',

    # Face Alignment
    'FaceAligner',
    'NormalizationProcessor',

    # Quality Assessment
    'QualityChecker',

    # Pipeline
    'PreprocessingOutput',
    'PreprocessingPipeline',
    'create_pipeline_from_config',
]

__version__ = '1.0.0'