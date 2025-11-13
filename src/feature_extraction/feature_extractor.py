"""
통합 특징 추출기
위치: src/feature_extraction/feature_extractor.py

EfficientNet-B4 + 랜드마크 어텐션 결합
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .efficientnet import EfficientNetB4Backbone
from .landmark_attention import HybridAttention


class DeepfakeFeatureExtractor(nn.Module):
    """
    Deepfake 탐지용 특징 추출 모델

    구조:
    1. EfficientNet-B4 백본
    2. 랜드마크 기반 하이브리드 어텐션
    3. 1792-dim 특징 벡터 출력
    """

    def __init__(
            self,
            pretrained: bool = True,
            freeze_bn: bool = False,
            dropout_rate: float = 0.4,
            use_attention: bool = True,
            attention_config: Optional[Dict] = None
    ):
        """
        Args:
            pretrained: ImageNet 사전학습 가중치 사용
            freeze_bn: BatchNorm 레이어 고정
            dropout_rate: Dropout 비율
            use_attention: 어텐션 사용 여부
            attention_config: 어텐션 설정
        """
        super(DeepfakeFeatureExtractor, self).__init__()

        # EfficientNet-B4 백본
        self.backbone = EfficientNetB4Backbone(
            pretrained=pretrained,
            freeze_bn=freeze_bn,
            dropout_rate=dropout_rate,
            extract_features=True
        )

        self.use_attention = use_attention
        self.feature_dim = self.backbone.feature_dim  # 1792

        # 하이브리드 어텐션
        if use_attention:
            if attention_config is None:
                attention_config = {
                    'use_landmark': True,
                    'use_spatial': True,
                    'use_channel': True
                }

            self.attention = HybridAttention(
                channels=self.feature_dim,
                feature_size=(7, 7),
                **attention_config
            )
            print("✅ 하이브리드 어텐션 활성화")
        else:
            self.attention = None
            print("⚠️  어텐션 비활성화")

    def forward(
            self,
            images: torch.Tensor,
            landmarks: Optional[torch.Tensor] = None,
            return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            images: (B, 3, 224, 224) 입력 이미지
            landmarks: (B, 5, 2) 랜드마크 좌표 (선택)
            return_attention: 어텐션 맵 반환 여부

        Returns:
            features: (B, 1792) 특징 벡터
            attention_map: (B, 1, 7, 7) 어텐션 맵 (선택)
        """
        # 1. Feature maps 추출 (7x7x1792)
        feature_maps = self.backbone.get_feature_maps(images)

        # 2. 어텐션 적용
        attention_map = None
        if self.use_attention and self.attention is not None:
            # 어텐션 맵 저장 (시각화용)
            if return_attention and landmarks is not None:
                attention_map = self.attention.landmark_attn._create_attention_map(
                    landmarks, (7, 7), images.device
                )

            # 어텐션 적용
            feature_maps = self.attention(feature_maps, landmarks)

        # 3. Global Average Pooling
        features = F.adaptive_avg_pool2d(feature_maps, 1)  # (B, 1792, 1, 1)
        features = features.flatten(1)  # (B, 1792)

        # 4. Dropout
        features = self.backbone.dropout(features)

        if return_attention:
            return features, attention_map
        else:
            return features, None

    def extract_multi_scale_features(
            self,
            images: torch.Tensor,
            landmarks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        다중 스케일 특징 추출

        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 5, 2)

        Returns:
            features_dict: {
                'low': (B, D_low),
                'mid': (B, D_mid),
                'high': (B, D_high),
                'final': (B, 1792)
            }
        """
        # 중간 특징 추출
        _, intermediate = self.backbone(images, return_intermediate=True)

        features_dict = {}

        # 중간 레이어 특징 pooling
        if intermediate:
            for name, feat in intermediate.items():
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                features_dict[name] = pooled

        # 최종 특징
        final_features, _ = self.forward(images, landmarks)
        features_dict['final'] = final_features

        return features_dict

    def get_embedding(
            self,
            images: torch.Tensor,
            landmarks: Optional[torch.Tensor] = None,
            normalize: bool = True
    ) -> torch.Tensor:
        """
        L2 정규화된 임베딩 추출

        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 5, 2)
            normalize: L2 정규화 적용

        Returns:
            embeddings: (B, 1792) 정규화된 특징
        """
        features, _ = self.forward(images, landmarks)

        if normalize:
            features = F.normalize(features, p=2, dim=1)

        return features


import torch.nn.functional as F


class DeepfakeDetectionModel(nn.Module):
    """
    전체 Deepfake 탐지 모델 (특징 추출 + 분류)

    End-to-end 학습 가능
    """

    def __init__(
            self,
            num_classes: int = 2,
            pretrained: bool = True,
            feature_extractor_config: Optional[Dict] = None,
            classifier_hidden_dims: list = [512, 128, 32],
            dropout_rate: float = 0.4
    ):
        """
        Args:
            num_classes: 출력 클래스 수 (2: real/fake)
            pretrained: 사전학습 가중치 사용
            feature_extractor_config: 특징 추출기 설정
            classifier_hidden_dims: 분류기 은닉층 차원
            dropout_rate: Dropout 비율
        """
        super(DeepfakeDetectionModel, self).__init__()

        # 특징 추출기
        if feature_extractor_config is None:
            feature_extractor_config = {
                'pretrained': pretrained,
                'use_attention': True
            }

        self.feature_extractor = DeepfakeFeatureExtractor(
            **feature_extractor_config
        )

        # 분류기
        feature_dim = self.feature_extractor.feature_dim

        layers = []
        input_dim = feature_dim

        for hidden_dim in classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 최종 분류 레이어
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        self.num_classes = num_classes

    def forward(
            self,
            images: torch.Tensor,
            landmarks: Optional[torch.Tensor] = None,
            return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 5, 2)
            return_features: 특징도 반환할지

        Returns:
            logits: (B, num_classes)
            features: (B, 1792) - return_features=True인 경우
        """
        # 특징 추출
        features, _ = self.feature_extractor(images, landmarks)

        # 분류
        logits = self.classifier(features)

        if return_features:
            return logits, features
        else:
            return logits, None

    def predict(
            self,
            images: torch.Tensor,
            landmarks: Optional[torch.Tensor] = None,
            return_probs: bool = True
    ) -> torch.Tensor:
        """
        예측 (추론용)

        Args:
            images: (B, 3, 224, 224)
            landmarks: (B, 5, 2)
            return_probs: 확률 반환 (False면 logits)

        Returns:
            predictions: (B, num_classes) 확률 또는 logits
        """
        with torch.no_grad():
            logits, _ = self.forward(images, landmarks)

            if return_probs:
                if self.num_classes == 2:
                    # Binary classification
                    probs = torch.softmax(logits, dim=1)
                else:
                    probs = torch.softmax(logits, dim=1)
                return probs
            else:
                return logits


# 테스트 코드
if __name__ == "__main__":
    """
    특징 추출기 테스트
    실행: python src/feature_extraction/feature_extractor.py
    """
    print("특징 추출기 테스트\n")

    # 더미 데이터
    B = 4
    images = torch.randn(B, 3, 224, 224)
    landmarks = torch.rand(B, 5, 2) * 224

    # 1. 특징 추출기 테스트
    print("1. DeepfakeFeatureExtractor 테스트")
    extractor = DeepfakeFeatureExtractor(
        pretrained=False,  # 빠른 테스트
        use_attention=True
    )

    features, attention = extractor(images, landmarks, return_attention=True)
    print(f"   입력 이미지: {images.shape}")
    print(f"   출력 특징: {features.shape}")
    if attention is not None:
        print(f"   어텐션 맵: {attention.shape}")

    # 2. 다중 스케일 특징
    print("\n2. 다중 스케일 특징 추출")
    multi_features = extractor.extract_multi_scale_features(images, landmarks)
    for name, feat in multi_features.items():
        print(f"   {name}: {feat.shape}")

    # 3. 임베딩
    print("\n3. 정규화된 임베딩")
    embeddings = extractor.get_embedding(images, landmarks, normalize=True)
    print(f"   임베딩: {embeddings.shape}")
    print(f"   L2 norm: {torch.norm(embeddings[0]):.4f}")  # ~1.0이어야 함

    # 4. 전체 모델 테스트
    print("\n4. DeepfakeDetectionModel 테스트")
    model = DeepfakeDetectionModel(
        num_classes=2,
        pretrained=False
    )

    logits, feats = model(images, landmarks, return_features=True)
    print(f"   로짓: {logits.shape}")
    print(f"   특징: {feats.shape}")

    # 5. 예측
    print("\n5. 예측 테스트")
    probs = model.predict(images, landmarks, return_probs=True)
    print(f"   확률: {probs.shape}")
    print(f"   첫 샘플 확률: Real={probs[0, 0]:.3f}, Fake={probs[0, 1]:.3f}")

    # 6. 모델 정보
    print("\n6. 모델 정보")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")
    print(f"   모델 크기 (추정): {total_params * 4 / (1024 ** 2):.2f} MB")

    print("\n✅ 특징 추출기 테스트 완료!")