"""
전처리 단계와 특징 추출 단계 간 데이터 전달 인터페이스
위치: src/data/interface.py
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.pipeline import PreprocessingOutput


@dataclass
class FeatureExtractionInput:
    """
    특징 추출 단계(Stage 2)의 입력 포맷
    전처리 단계(Stage 1)의 출력을 변환
    """
    images: torch.Tensor  # (B, 3, 224, 224) 정규화된 이미지
    landmarks: torch.Tensor  # (B, 5, 2) 얼굴 랜드마크
    quality_scores: torch.Tensor  # (B,) 품질 점수
    labels: torch.Tensor  # (B,) 레이블 (0=real, 1=fake)
    image_ids: List[str]  # 이미지 ID 리스트
    batch_metadata: List[Dict]  # 배치 메타데이터

    def to(self, device: torch.device) -> 'FeatureExtractionInput':
        """GPU/CPU로 이동"""
        return FeatureExtractionInput(
            images=self.images.to(device),
            landmarks=self.landmarks.to(device),
            quality_scores=self.quality_scores.to(device),
            labels=self.labels.to(device),
            image_ids=self.image_ids,
            batch_metadata=self.batch_metadata
        )


class PreprocessingToFeatureInterface:
    """
    전처리 출력 → 특징 추출 입력 변환 인터페이스

    역할:
    1. PreprocessingOutput 객체들을 배치로 변환
    2. PyTorch Tensor 포맷으로 변환
    3. 특징 추출 모델이 요구하는 형식으로 구성
    """

    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ImageNet 정규화 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocessing_outputs_to_batch(
            self,
            outputs: List[PreprocessingOutput]
    ) -> FeatureExtractionInput:
        """
        PreprocessingOutput 리스트 → 배치 텐서 변환

        Args:
            outputs: PreprocessingOutput 객체 리스트

        Returns:
            FeatureExtractionInput 객체 (배치 형태)
        """
        batch_size = len(outputs)

        # 1. 이미지 배치 생성
        images = []
        for output in outputs:
            # (224, 224, 3) NumPy → (3, 224, 224) Tensor
            img = torch.from_numpy(output.aligned_face.transpose(2, 0, 1)).float()
            img = img / 255.0  # [0, 255] → [0, 1]
            images.append(img)

        images = torch.stack(images)  # (B, 3, 224, 224)

        # 2. 정규화 적용
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)

        # 3. 랜드마크 배치
        landmarks = torch.stack([
            torch.from_numpy(output.landmarks).float()
            for output in outputs
        ])  # (B, 5, 2)

        # 4. 품질 점수 배치
        quality_scores = torch.tensor([
            output.quality_score for output in outputs
        ], dtype=torch.float32)  # (B,)

        # 5. 레이블 배치
        labels = torch.tensor([
            1 if output.label == 'fake' else 0
            for output in outputs
        ], dtype=torch.long)  # (B,)

        # 6. 메타데이터
        image_ids = [output.image_id for output in outputs]
        batch_metadata = [output.to_dict() for output in outputs]

        return FeatureExtractionInput(
            images=images,
            landmarks=landmarks,
            quality_scores=quality_scores,
            labels=labels,
            image_ids=image_ids,
            batch_metadata=batch_metadata
        )

    def dataloader_batch_to_feature_input(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> FeatureExtractionInput:
        """
        DataLoader 배치 → FeatureExtractionInput 변환

        Args:
            batch: DataLoader에서 반환된 배치 딕셔너리

        Returns:
            FeatureExtractionInput 객체
        """
        return FeatureExtractionInput(
            images=batch['image'],
            landmarks=batch['landmarks'],
            quality_scores=batch['quality_score'],
            labels=batch['label'],
            image_ids=batch['image_id'],
            batch_metadata=[]  # DataLoader는 메타데이터를 로드하지 않음
        )

    def prepare_for_efficientnet(
            self,
            feature_input: FeatureExtractionInput,
            use_landmarks: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        EfficientNet-B4 입력 준비

        Args:
            feature_input: FeatureExtractionInput 객체
            use_landmarks: 랜드마크 정보 사용 여부

        Returns:
            (images, landmarks) 또는 (images, None)
            - images: (B, 3, 224, 224) 정규화된 이미지
            - landmarks: (B, 5, 2) 또는 None
        """
        images = feature_input.images.to(self.device)

        if use_landmarks:
            landmarks = feature_input.landmarks.to(self.device)
            return images, landmarks
        else:
            return images, None

    def create_landmark_attention_map(
            self,
            landmarks: torch.Tensor,
            feature_map_size: Tuple[int, int] = (7, 7)
    ) -> torch.Tensor:
        """
        랜드마크 기반 어텐션 맵 생성

        Args:
            landmarks: (B, 5, 2) 랜드마크 좌표 (224x224 기준)
            feature_map_size: EfficientNet feature map 크기 (7x7)

        Returns:
            attention_map: (B, 1, H, W) 어텐션 맵
        """
        B = landmarks.shape[0]
        H, W = feature_map_size

        # 224x224 좌표를 7x7 feature map 좌표로 변환
        scale_x = W / 224.0
        scale_y = H / 224.0

        landmarks_scaled = landmarks.clone()
        landmarks_scaled[:, :, 0] *= scale_x
        landmarks_scaled[:, :, 1] *= scale_y

        # 가우시안 어텐션 맵 생성
        attention_map = torch.zeros(B, 1, H, W, device=landmarks.device)

        # 각 랜드마크 주변에 가우시안 분포
        sigma = 1.5  # 표준편차

        y_coords = torch.arange(H, device=landmarks.device).float().view(1, 1, H, 1)
        x_coords = torch.arange(W, device=landmarks.device).float().view(1, 1, 1, W)

        for i in range(5):  # 5개 랜드마크
            lm_x = landmarks_scaled[:, i:i + 1, 0:1].view(B, 1, 1, 1)
            lm_y = landmarks_scaled[:, i:i + 1, 1:2].view(B, 1, 1, 1)

            # 가우시안 분포 계산
            gaussian = torch.exp(
                -((x_coords - lm_x) ** 2 + (y_coords - lm_y) ** 2) / (2 * sigma ** 2)
            )

            attention_map += gaussian

        # 정규화 [0, 1]
        attention_map = attention_map / attention_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return attention_map

    def visualize_batch(
            self,
            feature_input: FeatureExtractionInput,
            num_samples: int = 4
    ):
        """
        배치 시각화 (디버깅용)

        Args:
            feature_input: FeatureExtractionInput 객체
            num_samples: 시각화할 샘플 수
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib이 필요합니다: pip install matplotlib")
            return

        num_samples = min(num_samples, feature_input.images.shape[0])

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

        for i in range(num_samples):
            # 역정규화
            img = feature_input.images[i].cpu()
            img = img * self.std.squeeze() + self.mean.squeeze()
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            # 이미지 표시
            axes[0, i].imshow(img)
            label = "Fake" if feature_input.labels[i] == 1 else "Real"
            axes[0, i].set_title(f"{label}\nQ: {feature_input.quality_scores[i]:.2f}")
            axes[0, i].axis('off')

            # 랜드마크 표시
            axes[1, i].imshow(img)
            landmarks = feature_input.landmarks[i].cpu().numpy()
            axes[1, i].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50, marker='x')
            axes[1, i].set_title("Landmarks")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('batch_visualization.png', dpi=150, bbox_inches='tight')
        print("배치 시각화 저장: batch_visualization.png")
        plt.close()


# 편의 함수들
def batch_to_device(
        batch: Dict[str, torch.Tensor],
        device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    배치의 모든 텐서를 device로 이동

    Args:
        batch: DataLoader 배치
        device: 타겟 디바이스

    Returns:
        device로 이동된 배치
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def collate_preprocessing_outputs(
        outputs: List[PreprocessingOutput]
) -> FeatureExtractionInput:
    """
    커스텀 collate 함수
    PreprocessingOutput 리스트를 배치로 변환

    DataLoader에서 사용:
    loader = DataLoader(dataset, collate_fn=collate_preprocessing_outputs)
    """
    interface = PreprocessingToFeatureInterface()
    return interface.preprocessing_outputs_to_batch(outputs)


# 테스트 코드
if __name__ == "__main__":
    """
    인터페이스 테스트
    실행: python src/data/interface.py
    """
    print("인터페이스 테스트 시작...\n")

    # DataLoader에서 배치 가져오기
    from src.data.dataset import create_dataloaders

    train_loader, _, _ = create_dataloaders(
        processed_dir="data/processed",
        batch_size=4,
        num_workers=0
    )

    # 첫 배치 가져오기
    batch = next(iter(train_loader))
    print("DataLoader 배치:")
    print(f"  Keys: {batch.keys()}")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Label shape: {batch['label'].shape}")
    print()

    # 인터페이스 생성
    interface = PreprocessingToFeatureInterface(device='cpu')

    # FeatureExtractionInput으로 변환
    feature_input = interface.dataloader_batch_to_feature_input(batch)

    print("FeatureExtractionInput:")
    print(f"  Images shape: {feature_input.images.shape}")
    print(f"  Landmarks shape: {feature_input.landmarks.shape}")
    print(f"  Quality scores: {feature_input.quality_scores}")
    print(f"  Labels: {feature_input.labels}")
    print(f"  Image IDs: {feature_input.image_ids}")
    print()

    # EfficientNet 입력 준비
    images, landmarks = interface.prepare_for_efficientnet(feature_input)
    print("EfficientNet 입력:")
    print(f"  Images shape: {images.shape}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print()

    # 어텐션 맵 생성
    attention_map = interface.create_landmark_attention_map(landmarks, (7, 7))
    print("Attention Map:")
    print(f"  Shape: {attention_map.shape}")
    print(f"  Range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
    print()

    # 시각화 (선택)
    # interface.visualize_batch(feature_input, num_samples=4)

    print("✅ 인터페이스 테스트 성공!")