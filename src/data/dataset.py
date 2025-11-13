"""
전처리된 데이터를 로드하는 PyTorch Dataset
위치: src/data/dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


class PreprocessedFaceDataset(Dataset):
    """
    전처리 완료된 얼굴 이미지 데이터셋

    전처리 단계(1단계)의 출력을 로드하여
    특징 추출 단계(2단계)로 전달
    """

    def __init__(
            self,
            csv_file: str,
            processed_dir: str,
            transform: Optional[callable] = None,
            load_landmarks: bool = True,
            load_metadata: bool = False,
            normalize: bool = True
    ):
        """
        Args:
            csv_file: train.csv, val.csv, test.csv 경로
            processed_dir: data/processed 디렉토리 경로
            transform: 추가 augmentation (선택)
            load_landmarks: landmark 로드 여부
            load_metadata: 메타데이터 로드 여부
            normalize: ImageNet 정규화 적용 여부
        """
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.load_landmarks = load_landmarks
        self.load_metadata = load_metadata
        self.normalize = normalize

        # CSV 로드 (전처리 결과)
        self.data = pd.read_csv(csv_file)

        # 성공적으로 처리된 데이터만 사용
        self.data = self.data[self.data['processed'] == True].reset_index(drop=True)

        print(f"[Dataset] Loaded {len(self.data)} samples from {csv_file}")
        print(f"  - Real: {len(self.data[self.data['label'] == 'real'])}")
        print(f"  - Fake: {len(self.data[self.data['label'] == 'fake'])}")

        # ImageNet 정규화 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'image': Tensor (3, 224, 224) - 정규화된 얼굴 이미지
                'label': int - 0(real) or 1(fake)
                'landmarks': Tensor (5, 2) - 얼굴 랜드마크 좌표
                'image_id': str - 이미지 ID
                'dataset': str - 데이터셋 이름
                'quality_score': float - 품질 점수
                'metadata': dict - 추가 메타데이터 (선택)
            }
        """
        row = self.data.iloc[idx]

        # 1. 얼굴 이미지 로드
        face_path = self.processed_dir / row['face_path']
        image = cv2.imread(str(face_path))

        if image is None:
            raise FileNotFoundError(f"Image not found: {face_path}")

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NumPy (H, W, C) -> Tensor (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # [0, 255] -> [0, 1]
        image = image / 255.0

        # ImageNet 정규화
        if self.normalize:
            image = (image - self.mean) / self.std

        # 2. 레이블 (real=0, fake=1)
        label = 1 if row['label'] == 'fake' else 0

        # 기본 반환값
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': row['image_id'],
            'dataset': row['dataset'],
            'quality_score': torch.tensor(row['quality_score'], dtype=torch.float32)
        }

        # 3. 랜드마크 로드 (선택)
        if self.load_landmarks:
            landmarks_path = self.processed_dir / row['landmarks_path']
            landmarks = np.load(landmarks_path)
            sample['landmarks'] = torch.from_numpy(landmarks).float()

        # 4. 메타데이터 로드 (선택)
        if self.load_metadata:
            metadata_path = self.processed_dir / row['metadata_path']
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            sample['metadata'] = metadata

        # 5. 추가 augmentation (선택)
        if self.transform is not None:
            # Augmentation은 정규화 전에 적용하는 것이 일반적
            # 하지만 여기서는 이미 정규화된 상태이므로 조심해야 함
            sample = self.transform(sample)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        클래스 불균형 처리를 위한 가중치 계산

        Returns:
            weights: Tensor [weight_real, weight_fake]
        """
        num_real = len(self.data[self.data['label'] == 'real'])
        num_fake = len(self.data[self.data['label'] == 'fake'])
        total = len(self.data)

        weight_real = total / (2 * num_real) if num_real > 0 else 1.0
        weight_fake = total / (2 * num_fake) if num_fake > 0 else 1.0

        return torch.tensor([weight_real, weight_fake], dtype=torch.float32)

    def get_quality_distribution(self) -> Dict[str, float]:
        """품질 점수 분포 통계"""
        return {
            'mean': self.data['quality_score'].mean(),
            'std': self.data['quality_score'].std(),
            'min': self.data['quality_score'].min(),
            'max': self.data['quality_score'].max(),
            'median': self.data['quality_score'].median()
        }


def create_dataloaders(
        processed_dir: str = "data/processed",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train/Val/Test DataLoader 생성

    Args:
        processed_dir: 전처리된 데이터 디렉토리
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 전송 최적화
        shuffle_train: 학습 데이터 셔플 여부

    Returns:
        (train_loader, val_loader, test_loader)
    """
    processed_path = Path(processed_dir)

    # CSV 파일 경로
    train_csv = processed_path / "splits" / "train.csv"
    val_csv = processed_path / "splits" / "val.csv"
    test_csv = processed_path / "splits" / "test.csv"

    # 파일 존재 확인
    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(f"Split file not found: {csv_file}")

    # Dataset 생성
    train_dataset = PreprocessedFaceDataset(
        csv_file=str(train_csv),
        processed_dir=str(processed_path),
        load_landmarks=True,
        load_metadata=False,
        normalize=True
    )

    val_dataset = PreprocessedFaceDataset(
        csv_file=str(val_csv),
        processed_dir=str(processed_path),
        load_landmarks=True,
        load_metadata=False,
        normalize=True
    )

    test_dataset = PreprocessedFaceDataset(
        csv_file=str(test_csv),
        processed_dir=str(processed_path),
        load_landmarks=True,
        load_metadata=False,
        normalize=True
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 마지막 불완전한 배치 제거
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # 정보 출력
    print("\n" + "=" * 60)
    print("DataLoader 생성 완료")
    print("=" * 60)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")

    # 클래스 가중치 출력
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")

    # 품질 분포 출력
    quality_dist = train_dataset.get_quality_distribution()
    print(f"\nQuality distribution (train):")
    print(f"  Mean: {quality_dist['mean']:.3f}")
    print(f"  Std: {quality_dist['std']:.3f}")
    print(f"  Range: [{quality_dist['min']:.3f}, {quality_dist['max']:.3f}]")
    print("=" * 60 + "\n")

    return train_loader, val_loader, test_loader


# 테스트 코드
if __name__ == "__main__":
    """
    DataLoader 테스트
    실행: python src/data/dataset.py
    """
    print("DataLoader 테스트 시작...\n")

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_dir="data/processed",
        batch_size=8,
        num_workers=0  # 테스트시 0 (디버깅 쉬움)
    )

    # 첫 번째 배치 확인
    print("첫 번째 배치 확인:")
    batch = next(iter(train_loader))

    print(f"\nBatch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")  # (B, 3, 224, 224)
    print(f"Label shape: {batch['label'].shape}")  # (B,)
    print(f"Landmarks shape: {batch['landmarks'].shape}")  # (B, 5, 2)
    print(f"Quality scores: {batch['quality_score'][:5]}")

    print(f"\nImage stats:")
    print(f"  Mean: {batch['image'].mean():.3f}")
    print(f"  Std: {batch['image'].std():.3f}")
    print(f"  Min: {batch['image'].min():.3f}")
    print(f"  Max: {batch['image'].max():.3f}")

    print(f"\nLabels: {batch['label']}")
    print(f"  Real (0): {(batch['label'] == 0).sum()}")
    print(f"  Fake (1): {(batch['label'] == 1).sum()}")

    print("\n✅ DataLoader 테스트 성공!")