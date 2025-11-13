"""
DataLoader 및 인터페이스 통합 테스트
위치: scripts/test_dataloader.py
"""

import sys
from pathlib import Path
import torch

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    create_dataloaders,
    PreprocessingToFeatureInterface,
    batch_to_device
)


def test_dataloader_basic():
    """기본 DataLoader 테스트"""
    print("\n" + "=" * 70)
    print("1. 기본 DataLoader 테스트")
    print("=" * 70)

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_dir="data/processed",
        batch_size=16,
        num_workers=0  # Windows에서는 0 권장
    )

    # 첫 배치 확인
    batch = next(iter(train_loader))

    print("\n배치 구조:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key:20s}: list of {len(value)} items")

    print("\n이미지 통계:")
    print(f"  Mean: {batch['image'].mean():.4f}")
    print(f"  Std: {batch['image'].std():.4f}")
    print(f"  Min: {batch['image'].min():.4f}")
    print(f"  Max: {batch['image'].max():.4f}")

    print("\n레이블 분포:")
    print(f"  Real (0): {(batch['label'] == 0).sum().item()}")
    print(f"  Fake (1): {(batch['label'] == 1).sum().item()}")

    return train_loader, val_loader, test_loader


def test_interface(train_loader):
    """인터페이스 변환 테스트"""
    print("\n" + "=" * 70)
    print("2. 인터페이스 변환 테스트")
    print("=" * 70)

    # 인터페이스 생성
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    interface = PreprocessingToFeatureInterface(device=device)

    # 배치 가져오기
    batch = next(iter(train_loader))

    # FeatureExtractionInput으로 변환
    feature_input = interface.dataloader_batch_to_feature_input(batch)
    feature_input = feature_input.to(torch.device(device))

    print("\nFeatureExtractionInput:")
    print(f"  Images: {feature_input.images.shape} on {feature_input.images.device}")
    print(f"  Landmarks: {feature_input.landmarks.shape} on {feature_input.landmarks.device}")
    print(f"  Quality scores: {feature_input.quality_scores.shape}")
    print(f"  Labels: {feature_input.labels.shape}")
    print(f"  Image IDs: {len(feature_input.image_ids)}")

    # EfficientNet 입력 준비
    images, landmarks = interface.prepare_for_efficientnet(feature_input)

    print("\nEfficientNet 입력:")
    print(f"  Images: {images.shape} on {images.device}")
    print(f"  Landmarks: {landmarks.shape} on {landmarks.device}")

    # 어텐션 맵 생성
    attention_map = interface.create_landmark_attention_map(landmarks, (7, 7))

    print("\nAttention Map:")
    print(f"  Shape: {attention_map.shape}")
    print(f"  Range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")

    return interface, feature_input


def test_full_pipeline(train_loader):
    """전체 파이프라인 시뮬레이션"""
    print("\n" + "=" * 70)
    print("3. 전체 파이프라인 시뮬레이션")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interface = PreprocessingToFeatureInterface(device=device)

    print(f"\nDevice: {device}")
    print("시뮬레이션: 전처리 → 특징 추출 → 분류\n")

    # 여러 배치 처리
    num_batches = min(5, len(train_loader))

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        # 1. DataLoader 배치 → FeatureExtractionInput
        feature_input = interface.dataloader_batch_to_feature_input(batch)
        feature_input = feature_input.to(device)

        # 2. EfficientNet 입력 준비
        images, landmarks = interface.prepare_for_efficientnet(feature_input)

        # 3. (여기서 EfficientNet 모델에 전달)
        # features = efficientnet(images, landmarks)

        print(f"Batch {i + 1}/{num_batches}:")
        print(f"  Input shape: {images.shape}")
        print(f"  Labels: Real={((feature_input.labels == 0).sum().item())}, "
              f"Fake={(feature_input.labels == 1).sum().item()}")
        print(f"  Avg quality: {feature_input.quality_scores.mean():.3f}")

    print("\n✅ 전체 파이프라인 시뮬레이션 성공!")


def test_performance(train_loader):
    """성능 측정"""
    print("\n" + "=" * 70)
    print("4. 데이터 로딩 성능 측정")
    print("=" * 70)

    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_batches = min(50, len(train_loader))

    # 성능 측정
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        # GPU로 전송
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # 간단한 연산 (실제 모델 시뮬레이션)
        _ = images.mean()

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n{num_batches} 배치 처리 시간: {elapsed:.2f}초")
    print(f"배치당 평균 시간: {elapsed / num_batches * 1000:.1f}ms")
    print(f"초당 처리 샘플: {(num_batches * train_loader.batch_size) / elapsed:.1f}")


def main():
    """전체 테스트 실행"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "DataLoader 통합 테스트" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # 1. 기본 테스트
        train_loader, val_loader, test_loader = test_dataloader_basic()

        # 2. 인터페이스 테스트
        interface, feature_input = test_interface(train_loader)

        # 3. 전체 파이프라인
        test_full_pipeline(train_loader)

        # 4. 성능 측정
        test_performance(train_loader)

        print("\n" + "=" * 70)
        print("모든 테스트 통과! ✅")
        print("=" * 70)
        print("\n다음 단계:")
        print("1. EfficientNet-B4 특징 추출 모델 구현")
        print("2. 학습 스크립트 작성")
        print("3. 모델 학습 시작")
        print()

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n해결 방법:")
        print("1. 전처리를 먼저 실행하세요:")
        print("   python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets all")
        print()
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()