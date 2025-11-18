"""
특징 추출 모듈 통합 테스트
위치: scripts/test_feature_extraction.py

실행: python scripts/test_feature_extraction.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction import (
    EfficientNetB4Backbone,
    LandmarkAttention,
    DeepfakeFeatureExtractor,
    DeepfakeDetectionModel
)
from src.data import create_dataloaders


def test_backbone():
    """EfficientNet-B4 백본 테스트"""
    print("\n" + "=" * 70)
    print("1. EfficientNet-B4 Backbone 테스트")
    print("=" * 70)

    model = EfficientNetB4Backbone(pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)

    features, intermediate = model(dummy_input, return_intermediate=True)

    print(f"✅ 입력: {dummy_input.shape}")
    print(f"✅ 출력: {features.shape}")
    print(f"✅ 특징 차원: {model.feature_dim}")

    if intermediate:
        print("\n중간 특징:")
        for name, feat in intermediate.items():
            print(f"  {name}: {feat.shape}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터: {total_params:,}")


def test_attention():
    """랜드마크 어텐션 테스트"""
    print("\n" + "=" * 70)
    print("2. Landmark Attention 테스트")
    print("=" * 70)

    feature_maps = torch.randn(4, 1792, 7, 7)
    landmarks = torch.rand(4, 5, 2) * 224

    attention = LandmarkAttention(feature_size=(7, 7))
    attended = attention(feature_maps, landmarks)

    print(f"✅ 입력 특징: {feature_maps.shape}")
    print(f"✅ 랜드마크: {landmarks.shape}")
    print(f"✅ 출력 특징: {attended.shape}")
    print(f"✅ 어텐션 가중치: {attention.attention_weights.data}")


def test_feature_extractor():
    """특징 추출기 테스트"""
    print("\n" + "=" * 70)
    print("3. Feature Extractor 테스트")
    print("=" * 70)

    model = DeepfakeFeatureExtractor(
        pretrained=False,
        use_attention=True
    )

    images = torch.randn(4, 3, 224, 224)
    landmarks = torch.rand(4, 5, 2) * 224

    features, attention_map = model(images, landmarks, return_attention=True)

    print(f"✅ 입력 이미지: {images.shape}")
    print(f"✅ 출력 특징: {features.shape}")
    if attention_map is not None:
        print(f"✅ 어텐션 맵: {attention_map.shape}")

    # 임베딩
    embeddings = model.get_embedding(images, landmarks, normalize=True)
    print(f"✅ 정규화된 임베딩: {embeddings.shape}")
    print(f"✅ L2 norm: {torch.norm(embeddings[0]):.4f}")


def test_full_model():
    """전체 모델 테스트"""
    print("\n" + "=" * 70)
    print("4. Full Detection Model 테스트")
    print("=" * 70)

    model = DeepfakeDetectionModel(
        num_classes=2,
        pretrained=False
    )

    images = torch.randn(4, 3, 224, 224)
    landmarks = torch.rand(4, 5, 2) * 224

    logits, features = model(images, landmarks, return_features=True)

    print(f"✅ 입력 이미지: {images.shape}")
    print(f"✅ 로짓: {logits.shape}")
    print(f"✅ 특징: {features.shape}")

    # 예측
    probs = model.predict(images, landmarks)
    preds = probs.argmax(dim=1)

    print(f"✅ 확률: {probs.shape}")
    print(f"✅ 예측: {preds}")

    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n모델 정보:")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능: {trainable_params:,}")
    print(f"  크기 (추정): {total_params * 4 / (1024 ** 2):.2f} MB")


def test_with_dataloader():
    """실제 데이터로더와 통합 테스트"""
    print("\n" + "=" * 70)
    print("5. DataLoader 통합 테스트")
    print("=" * 70)

    try:
        # 데이터로더 생성
        train_loader, _, _ = create_dataloaders(
            processed_dir="data/processed",
            batch_size=8,
            num_workers=0
        )

        # 모델
        model = DeepfakeDetectionModel(num_classes=2, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # 배치 처리
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        landmarks = batch['landmarks'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            logits, features = model(images, landmarks, return_features=True)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        print(f"✅ 배치 크기: {images.shape[0]}")
        print(f"✅ 로짓: {logits.shape}")
        print(f"✅ 특징: {features.shape}")
        print(f"✅ 예측: {preds}")
        print(f"✅ 실제 레이블: {labels}")

        # 정확도
        acc = (preds == labels).float().mean()
        print(f"✅ 정확도 (랜덤): {acc.item() * 100:.2f}%")

        print("\n✅ DataLoader 통합 테스트 성공!")

    except FileNotFoundError as e:
        print(f"⚠️  데이터를 찾을 수 없습니다: {e}")
        print("   전처리를 먼저 실행하세요:")
        print("   python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets all")


def test_forward_backward():
    """Forward/Backward 테스트"""
    print("\n" + "=" * 70)
    print("6. Forward/Backward 테스트")
    print("=" * 70)

    model = DeepfakeDetectionModel(num_classes=2, pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    images = torch.randn(4, 3, 224, 224)
    landmarks = torch.rand(4, 5, 2) * 224
    labels = torch.randint(0, 2, (4,))

    # Forward
    logits, _ = model(images, landmarks)
    loss = criterion(logits, labels)

    print(f"✅ Forward pass 성공")
    print(f"   Loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✅ Backward pass 성공")

    # Gradient 확인
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"✅ Gradients 존재: {has_grad}")


def main():
    """전체 테스트 실행"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "특징 추출 모듈 통합 테스트" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")

    # 테스트 실행
    test_backbone()
    test_attention()
    test_feature_extractor()
    test_full_model()
    test_with_dataloader()
    test_forward_backward()

    # 요약
    print("\n" + "=" * 70)
    print("테스트 요약")
    print("=" * 70)
    print("✅ 모든 컴포넌트 정상 작동")
    print("\n다음 단계:")
    print("1. 학습 시작:")
    print("   python scripts/train.py --config config/training_config.yaml")
    print("\n2. 평가:")
    print("   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()