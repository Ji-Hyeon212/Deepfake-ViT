"""
EfficientNet-B4 백본 모델
위치: src/feature_extraction/efficientnet.py
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional, Tuple, Dict
from pathlib import Path

LOCAL_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "model" / "efficientnet-b4-6ed6700e.pth"
class EfficientNetB4Backbone(nn.Module):
    """
    EfficientNet-B4 백본

    특징:
    - ImageNet 사전학습 가중치
    - 다중 스케일 특징 추출
    - 1792-dim 특징 벡터 출력
    """

    def __init__(
            self,
            pretrained: bool = True,
            freeze_bn: bool = False,
            dropout_rate: float = 0.4,
            extract_features: bool = True
    ):
        """
        Args:
            pretrained: ImageNet 사전학습 가중치 사용
            freeze_bn: BatchNorm 레이어 고정
            dropout_rate: Dropout 비율
            extract_features: 특징만 추출 (분류 헤드 제거)
        """
        super(EfficientNetB4Backbone, self).__init__()

        # EfficientNet-B4 로드
        if pretrained:
            # EfficientNet.from_pretrained 대신 from_name을 사용하여 다운로드를 방지
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4',
                num_classes=1000  # ImageNet
            )

            # 로컬 경로에서 가중치를 로드
            if LOCAL_WEIGHTS_PATH.exists():
                state_dict = torch.load(str(LOCAL_WEIGHTS_PATH), map_location='cpu')
                self.backbone.load_state_dict(state_dict)
                print("✅ EfficientNet-B4 가중치 로컬 경로에서 로드 완료")
            else:
                print(f"⚠️ EfficientNet-B4 로컬 가중치를 찾을 수 없습니다: {LOCAL_WEIGHTS_PATH}")
                print("⚠️ EfficientNet-B4 랜덤 초기화")
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4',
                num_classes=1000
            )
            print("⚠️  EfficientNet-B4 랜덤 초기화")

        self.extract_features = extract_features
        self.freeze_bn = freeze_bn

        # 특징 추출 모드
        if extract_features:
            # 분류 헤드 제거
            self.backbone._fc = nn.Identity()

        # BatchNorm 고정
        if freeze_bn:
            self._freeze_bn_layers()

        # 특징 차원
        self.feature_dim = 1792  # EfficientNet-B4의 출력 차원

        # Dropout (추가)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 중간 레이어 특징 추출을 위한 hook
        self.intermediate_features = {}
        self._register_hooks()

    def _freeze_bn_layers(self):
        """BatchNorm 레이어 고정"""
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def _register_hooks(self):
        """중간 레이어 특징 추출을 위한 hook 등록"""

        def get_activation(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output

            return hook

        # EfficientNet-B4 주요 레이어
        # reduction_1: 112x112x24
        # reduction_2: 56x56x32
        # reduction_3: 28x28x56
        # reduction_4: 14x14x160
        # reduction_5: 7x7x448
        # reduction_6: 7x7x1792

        try:
            self.backbone._blocks[5].register_forward_hook(
                get_activation('reduction_2')
            )  # Low-level
            self.backbone._blocks[10].register_forward_hook(
                get_activation('reduction_4')
            )  # Mid-level
            self.backbone._blocks[21].register_forward_hook(
                get_activation('reduction_5')
            )  # High-level
        except:
            print("⚠️  Hook 등록 실패 - 중간 특징 사용 불가")

    def forward(
            self,
            x: torch.Tensor,
            return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass

        Args:
            x: 입력 이미지 (B, 3, 224, 224)
            return_intermediate: 중간 특징 반환 여부

        Returns:
            features: (B, 1792) 특징 벡터
            intermediate_features: 중간 레이어 특징 (선택)
        """
        # 특징 추출
        features = self.backbone.extract_features(x)  # (B, 1792, 7, 7)

        # Global Average Pooling
        features = self.backbone._avg_pooling(features)  # (B, 1792, 1, 1)
        features = features.flatten(1)  # (B, 1792)

        # Dropout
        features = self.dropout(features)

        if return_intermediate:
            return features, self.intermediate_features
        else:
            return features, None

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        7x7 특징 맵 반환 (어텐션 맵 적용 전)

        Args:
            x: 입력 이미지 (B, 3, 224, 224)

        Returns:
            feature_maps: (B, 1792, 7, 7)
        """
        return self.backbone.extract_features(x)

    def train(self, mode: bool = True):
        """학습 모드 설정 (BatchNorm 고정 유지)"""
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn_layers()
        return self


class EfficientNetB4WithFineTune(nn.Module):
    """
    Fine-tuning을 위한 EfficientNet-B4

    레이어별 학습률 조정 가능
    """

    def __init__(
            self,
            pretrained: bool = True,
            num_classes: int = 2,
            dropout_rate: float = 0.4,
            freeze_stages: int = 0
    ):
        """
        Args:
            pretrained: 사전학습 가중치 사용
            num_classes: 출력 클래스 수
            dropout_rate: Dropout 비율
            freeze_stages: 고정할 초기 스테이지 수 (0-7)
        """
        super(EfficientNetB4WithFineTune, self).__init__()

        # 백본 로드
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(
                'efficientnet-b4',
                num_classes=num_classes
            )
        else:
            self.backbone = EfficientNet.from_name(
                'efficientnet-b4',
                num_classes=num_classes
            )

        # 분류 헤드 교체
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        # 초기 스테이지 고정
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages: int):
        """
        초기 스테이지 고정

        Args:
            num_stages: 고정할 스테이지 수
        """
        # Stem 고정
        if num_stages >= 1:
            for param in self.backbone._conv_stem.parameters():
                param.requires_grad = False
            for param in self.backbone._bn0.parameters():
                param.requires_grad = False

        # Blocks 고정
        total_blocks = len(self.backbone._blocks)
        blocks_per_stage = total_blocks // 7  # EfficientNet-B4는 7 스테이지

        for i in range(min(num_stages - 1, 7) * blocks_per_stage):
            for param in self.backbone._blocks[i].parameters():
                param.requires_grad = False

        print(f"✅ 초기 {num_stages} 스테이지 고정")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 입력 이미지 (B, 3, 224, 224)

        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)

    def get_param_groups(
            self,
            base_lr: float = 1e-4,
            multiplier: float = 0.1
    ) -> list:
        """
        레이어별 학습률을 위한 파라미터 그룹

        Args:
            base_lr: 기본 학습률
            multiplier: 초기 레이어 학습률 감소 비율

        Returns:
            파라미터 그룹 리스트
        """
        param_groups = [
            # Stem (가장 낮은 학습률)
            {
                'params': list(self.backbone._conv_stem.parameters()) +
                          list(self.backbone._bn0.parameters()),
                'lr': base_lr * multiplier
            },
            # Blocks (점진적 증가)
            {
                'params': [p for block in self.backbone._blocks
                           for p in block.parameters()],
                'lr': base_lr * (multiplier + (1 - multiplier) * 0.5)
            },
            # Head (가장 높은 학습률)
            {
                'params': list(self.backbone._conv_head.parameters()) +
                          list(self.backbone._bn1.parameters()) +
                          list(self.backbone._fc.parameters()),
                'lr': base_lr
            }
        ]

        return param_groups


# 테스트 코드
if __name__ == "__main__":
    """
    EfficientNet-B4 테스트
    실행: python src/feature_extraction/efficientnet.py
    """
    print("EfficientNet-B4 테스트\n")

    # 1. 백본 테스트
    print("1. EfficientNetB4Backbone 테스트")
    model = EfficientNetB4Backbone(pretrained=False)  # 빠른 테스트

    dummy_input = torch.randn(2, 3, 224, 224)
    features, intermediate = model(dummy_input, return_intermediate=True)

    print(f"   입력 shape: {dummy_input.shape}")
    print(f"   출력 특징 shape: {features.shape}")
    print(f"   특징 차원: {model.feature_dim}")

    if intermediate:
        print("\n   중간 특징:")
        for name, feat in intermediate.items():
            print(f"     {name}: {feat.shape}")

    # 2. Feature maps 테스트
    print("\n2. Feature Maps 테스트")
    feature_maps = model.get_feature_maps(dummy_input)
    print(f"   Feature maps shape: {feature_maps.shape}")

    # 3. 파라미터 수
    print("\n3. 모델 정보")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")

    # 4. Fine-tune 모델 테스트
    print("\n4. Fine-tune 모델 테스트")
    finetune_model = EfficientNetB4WithFineTune(
        pretrained=False,
        num_classes=2,
        freeze_stages=2
    )

    logits = finetune_model(dummy_input)
    print(f"   출력 logits shape: {logits.shape}")

    # 5. 파라미터 그룹
    param_groups = finetune_model.get_param_groups(base_lr=1e-4)
    print(f"\n   파라미터 그룹 수: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"     그룹 {i}: {len(group['params'])} 파라미터, lr={group['lr']}")

    print("\n✅ EfficientNet-B4 테스트 완료!")