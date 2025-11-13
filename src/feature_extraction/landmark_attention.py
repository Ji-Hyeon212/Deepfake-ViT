"""
랜드마크 기반 어텐션 모듈
위치: src/feature_extraction/landmark_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class LandmarkAttention(nn.Module):
    """
    얼굴 랜드마크 기반 공간 어텐션

    얼굴 주요 부위(눈, 코, 입)에 더 높은 가중치 부여
    """

    def __init__(
            self,
            feature_size: Tuple[int, int] = (7, 7),
            sigma: float = 1.5,
            learnable: bool = True
    ):
        """
        Args:
            feature_size: Feature map 크기 (H, W)
            sigma: 가우시안 표준편차
            learnable: 어텐션 가중치 학습 여부
        """
        super(LandmarkAttention, self).__init__()

        self.feature_size = feature_size
        self.sigma = sigma
        self.learnable = learnable

        # 학습 가능한 어텐션 가중치
        if learnable:
            self.attention_weights = nn.Parameter(
                torch.ones(5)  # 5개 랜드마크
            )
        else:
            self.register_buffer(
                'attention_weights',
                torch.ones(5)
            )

    def forward(
            self,
            feature_maps: torch.Tensor,
            landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            feature_maps: (B, C, H, W) 특징 맵
            landmarks: (B, 5, 2) 랜드마크 좌표 (224x224 기준)

        Returns:
            attended_features: (B, C, H, W) 어텐션 적용된 특징
        """
        B, C, H, W = feature_maps.shape

        # 어텐션 맵 생성
        attention_map = self._create_attention_map(
            landmarks, (H, W), feature_maps.device
        )  # (B, 1, H, W)

        # 어텐션 적용
        attended_features = feature_maps * attention_map

        return attended_features

    def _create_attention_map(
            self,
            landmarks: torch.Tensor,
            feature_size: Tuple[int, int],
            device: torch.device
    ) -> torch.Tensor:
        """
        랜드마크 기반 어텐션 맵 생성

        Args:
            landmarks: (B, 5, 2) 랜드마크 좌표
            feature_size: (H, W) 특징 맵 크기
            device: 디바이스

        Returns:
            attention_map: (B, 1, H, W)
        """
        B = landmarks.shape[0]
        H, W = feature_size

        # 224x224 좌표를 feature map 크기로 변환
        scale_x = W / 224.0
        scale_y = H / 224.0

        landmarks_scaled = landmarks.clone()
        landmarks_scaled[:, :, 0] *= scale_x
        landmarks_scaled[:, :, 1] *= scale_y

        # 좌표 그리드 생성
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

        # 각 랜드마크에 대한 가우시안 생성
        attention_map = torch.zeros(B, 1, H, W, device=device)

        for i in range(5):  # 5개 랜드마크
            lm_x = landmarks_scaled[:, i:i + 1, 0:1].view(B, 1, 1, 1)
            lm_y = landmarks_scaled[:, i:i + 1, 1:2].view(B, 1, 1, 1)

            # 가우시안 분포
            dist_sq = (x_coords - lm_x) ** 2 + (y_coords - lm_y) ** 2
            gaussian = torch.exp(-dist_sq / (2 * self.sigma ** 2))

            # 가중치 적용
            weighted_gaussian = gaussian * self.attention_weights[i]

            attention_map += weighted_gaussian

        # 정규화 [0, 1]
        attention_map = attention_map / (attention_map.max() + 1e-8)

        # 최소값 클리핑 (배경 영역도 약간의 가중치)
        attention_map = torch.clamp(attention_map, min=0.1, max=1.0)

        return attention_map

    def visualize_attention(
            self,
            landmarks: torch.Tensor,
            feature_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        어텐션 맵 시각화용 (고해상도)

        Args:
            landmarks: (B, 5, 2) 랜드마크
            feature_size: 출력 크기

        Returns:
            attention_map: (B, 1, H, W)
        """
        with torch.no_grad():
            return self._create_attention_map(
                landmarks, feature_size, landmarks.device
            )


class SpatialAttention(nn.Module):
    """
    학습 가능한 공간 어텐션 (CBAM 스타일)
    """

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Convolution 커널 크기
        """
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 입력 특징

        Returns:
            attended: (B, C, H, W) 어텐션 적용
        """
        # 채널 축 평균 및 최대값
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)

        # Convolution + Sigmoid
        attention = self.sigmoid(self.conv(concat))  # (B, 1, H, W)

        # 어텐션 적용
        return x * attention


class ChannelAttention(nn.Module):
    """
    채널 어텐션 (SE Block 스타일)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 입력 채널 수
            reduction: 차원 축소 비율
        """
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 입력 특징

        Returns:
            attended: (B, C, H, W) 어텐션 적용
        """
        B, C, _, _ = x.shape

        # Global pooling
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)

        # FC layers
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        # 합산 및 sigmoid
        out = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        # 어텐션 적용
        return x * out


class HybridAttention(nn.Module):
    """
    랜드마크 + 공간 + 채널 어텐션 결합
    """

    def __init__(
            self,
            channels: int,
            feature_size: Tuple[int, int] = (7, 7),
            use_landmark: bool = True,
            use_spatial: bool = True,
            use_channel: bool = True
    ):
        """
        Args:
            channels: 특징 채널 수
            feature_size: 특징 맵 크기
            use_landmark: 랜드마크 어텐션 사용
            use_spatial: 공간 어텐션 사용
            use_channel: 채널 어텐션 사용
        """
        super(HybridAttention, self).__init__()

        self.use_landmark = use_landmark
        self.use_spatial = use_spatial
        self.use_channel = use_channel

        if use_landmark:
            self.landmark_attn = LandmarkAttention(
                feature_size=feature_size,
                learnable=True
            )

        if use_spatial:
            self.spatial_attn = SpatialAttention()

        if use_channel:
            self.channel_attn = ChannelAttention(channels)

    def forward(
            self,
            feature_maps: torch.Tensor,
            landmarks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            feature_maps: (B, C, H, W)
            landmarks: (B, 5, 2) - 랜드마크 어텐션 사용 시 필요

        Returns:
            attended: (B, C, H, W)
        """
        x = feature_maps

        # 1. 랜드마크 어텐션
        if self.use_landmark and landmarks is not None:
            x = self.landmark_attn(x, landmarks)

        # 2. 채널 어텐션
        if self.use_channel:
            x = self.channel_attn(x)

        # 3. 공간 어텐션
        if self.use_spatial:
            x = self.spatial_attn(x)

        return x


# 테스트 코드
if __name__ == "__main__":
    """
    어텐션 모듈 테스트
    실행: python src/feature_extraction/landmark_attention.py
    """
    print("어텐션 모듈 테스트\n")

    # 더미 데이터
    B, C, H, W = 4, 1792, 7, 7
    feature_maps = torch.randn(B, C, H, W)
    landmarks = torch.rand(B, 5, 2) * 224  # 0~224 좌표

    # 1. 랜드마크 어텐션
    print("1. LandmarkAttention 테스트")
    lm_attn = LandmarkAttention(feature_size=(H, W))
    attended = lm_attn(feature_maps, landmarks)
    print(f"   입력: {feature_maps.shape}")
    print(f"   출력: {attended.shape}")
    print(f"   어텐션 가중치: {lm_attn.attention_weights.data}")

    # 2. 공간 어텐션
    print("\n2. SpatialAttention 테스트")
    sp_attn = SpatialAttention()
    attended = sp_attn(feature_maps)
    print(f"   입력: {feature_maps.shape}")
    print(f"   출력: {attended.shape}")

    # 3. 채널 어텐션
    print("\n3. ChannelAttention 테스트")
    ch_attn = ChannelAttention(channels=C)
    attended = ch_attn(feature_maps)
    print(f"   입력: {feature_maps.shape}")
    print(f"   출력: {attended.shape}")

    # 4. 하이브리드 어텐션
    print("\n4. HybridAttention 테스트")
    hybrid_attn = HybridAttention(
        channels=C,
        feature_size=(H, W),
        use_landmark=True,
        use_spatial=True,
        use_channel=True
    )
    attended = hybrid_attn(feature_maps, landmarks)
    print(f"   입력: {feature_maps.shape}")
    print(f"   출력: {attended.shape}")

    # 5. 파라미터 수
    print("\n5. 파라미터 수")
    for name, module in [
        ('Landmark', lm_attn),
        ('Spatial', sp_attn),
        ('Channel', ch_attn),
        ('Hybrid', hybrid_attn)
    ]:
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"   {name}: {params:,} (학습 가능: {trainable:,})")

    print("\n✅ 어텐션 모듈 테스트 완료!")