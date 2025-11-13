"""
분류기 모듈
위치: src/classification/classifier.py
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron 분류기

    특징 벡터 → 레이블 (Real/Fake)
    """

    def __init__(
            self,
            input_dim: int = 1792,
            hidden_dims: List[int] = [512, 128, 32],
            num_classes: int = 2,
            dropout_rate: float = 0.4,
            use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dims: 은닉층 차원 리스트
            num_classes: 출력 클래스 수
            dropout_rate: Dropout 비율
            use_batch_norm: BatchNorm 사용 여부
        """
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 은닉층
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        # 출력층
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            features: (B, input_dim) 특징 벡터

        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(features)


class AttentionClassifier(nn.Module):
    """
    Self-Attention 기반 분류기

    특징에 대한 중요도를 학습
    """

    def __init__(
            self,
            input_dim: int = 1792,
            hidden_dim: int = 512,
            num_classes: int = 2,
            dropout_rate: float = 0.4
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dim: 은닉 차원
            num_classes: 출력 클래스 수
            dropout_rate: Dropout 비율
        """
        super(AttentionClassifier, self).__init__()

        # Self-attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.num_classes = num_classes

    def forward(
            self,
            features: torch.Tensor,
            return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            features: (B, input_dim) 또는 (B, N, input_dim)
            return_attention: 어텐션 가중치 반환 여부

        Returns:
            logits: (B, num_classes)
            attention_weights: (B, N) - return_attention=True인 경우
        """
        # 배치 차원만 있는 경우
        if features.dim() == 2:
            features = features.unsqueeze(1)  # (B, 1, D)

        # Self-attention
        attention_weights = self.attention(features)  # (B, N, 1)

        # 가중합
        weighted_features = (features * attention_weights).sum(dim=1)  # (B, D)

        # 분류
        logits = self.classifier(weighted_features)

        if return_attention:
            return logits, attention_weights.squeeze(-1)
        else:
            return logits


class EnsembleClassifier(nn.Module):
    """
    여러 분류기의 앙상블
    """

    def __init__(
            self,
            input_dim: int = 1792,
            num_classes: int = 2,
            num_classifiers: int = 3,
            voting: str = 'soft'
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            num_classes: 출력 클래스 수
            num_classifiers: 분류기 개수
            voting: 'soft' (확률 평균) 또는 'hard' (다수결)
        """
        super(EnsembleClassifier, self).__init__()

        self.num_classifiers = num_classifiers
        self.num_classes = num_classes
        self.voting = voting

        # 여러 분류기 생성
        self.classifiers = nn.ModuleList([
            MLPClassifier(
                input_dim=input_dim,
                hidden_dims=[512, 128, 32] if i == 0
                else [256, 64] if i == 1
                else [1024, 256, 64],
                num_classes=num_classes,
                dropout_rate=0.3 + i * 0.1
            )
            for i in range(num_classifiers)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            features: (B, input_dim)

        Returns:
            logits: (B, num_classes)
        """
        # 각 분류기의 출력
        outputs = [classifier(features) for classifier in self.classifiers]

        if self.voting == 'soft':
            # 확률 평균
            probs = [torch.softmax(out, dim=1) for out in outputs]
            avg_probs = torch.stack(probs).mean(dim=0)
            # logits로 변환
            logits = torch.log(avg_probs + 1e-10)
        else:
            # hard voting (다수결)
            preds = [out.argmax(dim=1) for out in outputs]
            stacked_preds = torch.stack(preds)
            # 가장 많이 나온 클래스
            logits = torch.zeros(features.size(0), self.num_classes, device=features.device)
            for i in range(features.size(0)):
                votes = torch.bincount(stacked_preds[:, i], minlength=self.num_classes)
                logits[i, votes.argmax()] = 1.0

        return logits


# 테스트 코드
if __name__ == "__main__":
    """
    분류기 테스트
    실행: python src/classification/classifier.py
    """
    print("분류기 모듈 테스트\n")

    # 더미 데이터
    B = 8
    feature_dim = 1792
    features = torch.randn(B, feature_dim)

    # 1. MLP 분류기
    print("1. MLPClassifier 테스트")
    mlp = MLPClassifier(
        input_dim=feature_dim,
        hidden_dims=[512, 128, 32],
        num_classes=2
    )

    logits = mlp(features)
    print(f"   입력: {features.shape}")
    print(f"   출력: {logits.shape}")

    params = sum(p.numel() for p in mlp.parameters())
    print(f"   파라미터: {params:,}")

    # 2. Attention 분류기
    print("\n2. AttentionClassifier 테스트")
    attn_clf = AttentionClassifier(
        input_dim=feature_dim,
        hidden_dim=512,
        num_classes=2
    )

    logits, attn_weights = attn_clf(features, return_attention=True)
    print(f"   입력: {features.shape}")
    print(f"   출력: {logits.shape}")
    print(f"   어텐션: {attn_weights.shape}")

    params = sum(p.numel() for p in attn_clf.parameters())
    print(f"   파라미터: {params:,}")

    # 3. 앙상블 분류기
    print("\n3. EnsembleClassifier 테스트")
    ensemble = EnsembleClassifier(
        input_dim=feature_dim,
        num_classes=2,
        num_classifiers=3,
        voting='soft'
    )

    logits = ensemble(features)
    print(f"   입력: {features.shape}")
    print(f"   출력: {logits.shape}")

    params = sum(p.numel() for p in ensemble.parameters())
    print(f"   파라미터: {params:,}")

    print("\n✅ 분류기 테스트 완료!")