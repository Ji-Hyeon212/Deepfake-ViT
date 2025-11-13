"""
손실 함수 모듈
위치: src/training/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 클래스별 가중치 (C,)
            gamma: Focusing parameter
            reduction: 'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) 레이블

        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for feature learning

    실제/가짜 샘플 간 거리를 조정
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'euclidean'
    ):
        """
        Args:
            margin: 마진 값
            distance: 'euclidean' 또는 'cosine'
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: (B, D) 첫 번째 임베딩
            embeddings2: (B, D) 두 번째 임베딩
            labels: (B,) 0 (같은 클래스) 또는 1 (다른 클래스)

        Returns:
            loss: scalar
        """
        if self.distance == 'euclidean':
            distances = F.pairwise_distance(embeddings1, embeddings2)
        else:  # cosine
            cos_sim = F.cosine_similarity(embeddings1, embeddings2)
            distances = 1 - cos_sim

        # Contrastive loss
        loss_same = labels * distances.pow(2)
        loss_diff = (1 - labels) * F.relu(self.margin - distances).pow(2)

        loss = (loss_same + loss_diff).mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning

    anchor, positive, negative 샘플 간 관계 학습
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'euclidean'
    ):
        """
        Args:
            margin: 마진 값
            distance: 'euclidean' 또는 'cosine'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: (B, D) 앵커 임베딩
            positive: (B, D) 같은 클래스 임베딩
            negative: (B, D) 다른 클래스 임베딩

        Returns:
            loss: scalar
        """
        if self.distance == 'euclidean':
            dist_pos = F.pairwise_distance(anchor, positive)
            dist_neg = F.pairwise_distance(anchor, negative)
        else:  # cosine
            dist_pos = 1 - F.cosine_similarity(anchor, positive)
            dist_neg = 1 - F.cosine_similarity(anchor, negative)

        loss = F.relu(dist_pos - dist_neg + self.margin).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    여러 손실 함수의 가중 결합
    """

    def __init__(
        self,
        weights: dict,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            weights: {
                'ce': float,  # Cross-entropy 가중치
                'focal': float,  # Focal loss 가중치
                'contrastive': float  # Contrastive loss 가중치
            }
            class_weights: 클래스 불균형 가중치
        """
        super(CombinedLoss, self).__init__()

        self.weights = weights

        # 손실 함수들
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            logits: (B, C) 분류 로짓
            targets: (B,) 타겟 레이블
            features: (B, D) 특징 벡터 (contrastive용, 선택)

        Returns:
            losses: {
                'total': total_loss,
                'ce': ce_loss,
                'focal': focal_loss,
                'contrastive': contrastive_loss (if features provided)
            }
        """
        losses = {}
        total_loss = 0.0

        # Cross-entropy
        if 'ce' in self.weights and self.weights['ce'] > 0:
            ce = self.ce_loss(logits, targets)
            losses['ce'] = ce
            total_loss += self.weights['ce'] * ce

        # Focal loss
        if 'focal' in self.weights and self.weights['focal'] > 0:
            focal = self.focal_loss(logits, targets)
            losses['focal'] = focal
            total_loss += self.weights['focal'] * focal

        # Contrastive loss
        if features is not None and 'contrastive' in self.weights and self.weights['contrastive'] > 0:
            # 같은 배치 내에서 페어 생성
            B = features.size(0)
            if B >= 2:
                # 간단한 구현: 연속된 샘플 페어
                feat1 = features[:-1:2]
                feat2 = features[1::2]
                label1 = targets[:-1:2]
                label2 = targets[1::2]

                pair_labels = (label1 == label2).float()

                if len(feat1) > 0:
                    contrastive = self.contrastive_loss(feat1, feat2, pair_labels)
                    losses['contrastive'] = contrastive
                    total_loss += self.weights['contrastive'] * contrastive

        losses['total'] = total_loss

        return losses


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss

    과신(overconfidence) 방지
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1
    ):
        """
        Args:
            num_classes: 클래스 수
            smoothing: 스무딩 정도 [0, 1]
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) 레이블

        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(inputs, dim=1)

        # One-hot encoding with smoothing
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.fill_(self.smoothing / (self.num_classes - 1))
        targets_one_hot.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = (-targets_one_hot * log_probs).sum(dim=1).mean()

        return loss


# 테스트 코드
if __name__ == "__main__":
    """
    손실 함수 테스트
    실행: python src/training/losses.py
    """
    print("손실 함수 테스트\n")

    # 더미 데이터
    B, C, D = 8, 2, 1792
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    features = torch.randn(B, D)

    # 1. Focal Loss
    print("1. FocalLoss 테스트")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    # 2. Contrastive Loss
    print("\n2. ContrastiveLoss 테스트")
    contrastive_loss = ContrastiveLoss(margin=1.0)
    feat1, feat2 = features[:4], features[4:]
    labels = torch.tensor([0, 0, 1, 1])  # 0: same, 1: different
    loss = contrastive_loss(feat1, feat2, labels)
    print(f"   Loss: {loss.item():.4f}")

    # 3. Triplet Loss
    print("\n3. TripletLoss 테스트")
    triplet_loss = TripletLoss(margin=1.0)
    anchor = features[:3]
    positive = features[3:6]
    negative = features[6:9][:3]
    loss = triplet_loss(anchor, positive, negative)
    print(f"   Loss: {loss.item():.4f}")

    # 4. Combined Loss
    print("\n4. CombinedLoss 테스트")
    combined_loss = CombinedLoss(
        weights={'ce': 1.0, 'focal': 0.5, 'contrastive': 0.2}
    )
    losses = combined_loss(logits, targets, features)
    print(f"   Total: {losses['total'].item():.4f}")
    print(f"   CE: {losses['ce'].item():.4f}")
    print(f"   Focal: {losses['focal'].item():.4f}")
    if 'contrastive' in losses:
        print(f"   Contrastive: {losses['contrastive'].item():.4f}")

    # 5. Label Smoothing
    print("\n5. LabelSmoothingLoss 테스트")
    ls_loss = LabelSmoothingLoss(num_classes=C, smoothing=0.1)
    loss = ls_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    print("\n✅ 손실 함수 테스트 완료!")