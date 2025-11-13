"""
Visualization utilities for deepfake detection
위치: src/utils/visualization.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, roc_curve, auc


# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_preprocessing_result(
    original_image: np.ndarray,
    aligned_face: np.ndarray,
    landmarks: np.ndarray,
    bbox: np.ndarray,
    quality_metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    전처리 결과 시각화

    Args:
        original_image: 원본 이미지 (H, W, 3) RGB
        aligned_face: 정렬된 얼굴 (224, 224, 3) RGB
        landmarks: 랜드마크 좌표 (5, 2)
        bbox: 바운딩 박스 [x1, y1, x2, y2]
        quality_metrics: 품질 메트릭 딕셔너리
        save_path: 저장 경로 (선택)

    Returns:
        시각화 이미지 (numpy array)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 원본 이미지 with bbox and landmarks
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image\nwith Detection', fontsize=12, fontweight='bold')

    # 바운딩 박스
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='lime',
        facecolor='none'
    )
    axes[0].add_patch(rect)

    # 랜드마크
    landmark_colors = ['red', 'blue', 'green', 'yellow', 'magenta']
    landmark_labels = ['L Eye', 'R Eye', 'Nose', 'L Mouth', 'R Mouth']
    for i, (x, y) in enumerate(landmarks):
        axes[0].scatter(x, y, c=landmark_colors[i], s=100,
                       marker='x', linewidths=3, label=landmark_labels[i])
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].axis('off')

    # 2. 정렬된 얼굴
    axes[1].imshow(aligned_face)
    axes[1].set_title('Aligned Face\n(224x224)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. 품질 메트릭
    axes[2].axis('off')
    axes[2].set_title('Quality Metrics', fontsize=12, fontweight='bold')

    # 메트릭 텍스트
    y_pos = 0.9
    for key, value in quality_metrics.items():
        if isinstance(value, (int, float)):
            text = f"{key}: {value:.3f}"
        else:
            text = f"{key}: {value}"

        # 색상 결정 (임계값 기반)
        color = 'green' if value > 0.7 else 'orange' if value > 0.5 else 'red'
        if not isinstance(value, (int, float)):
            color = 'black'

        axes[2].text(0.1, y_pos, text, fontsize=11,
                    transform=axes[2].transAxes, color=color)
        y_pos -= 0.08

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 저장: {save_path}")

    # Figure를 numpy array로 변환
    fig.canvas.draw()
    vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return vis_array


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    landmarks: Optional[torch.Tensor] = None,
    num_samples: int = 8,
    denormalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    배치 데이터 시각화

    Args:
        images: 이미지 텐서 (B, 3, 224, 224)
        labels: 레이블 텐서 (B,) - 0: real, 1: fake
        predictions: 예측 텐서 (B,) - 0: real, 1: fake (선택)
        landmarks: 랜드마크 텐서 (B, 5, 2) (선택)
        num_samples: 표시할 샘플 수
        denormalize: ImageNet 역정규화 적용 여부
        save_path: 저장 경로 (선택)
    """
    num_samples = min(num_samples, images.shape[0])
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() if num_samples > 1 else [axes]

    # ImageNet 역정규화 파라미터
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for i in range(num_samples):
        # 이미지 준비
        img = images[i].cpu()

        # 역정규화
        if denormalize:
            img = img * std.squeeze() + mean.squeeze()

        # CHW -> HWC
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # 표시
        axes[i].imshow(img)

        # 레이블 및 예측
        label_text = "Real" if labels[i] == 0 else "Fake"
        label_color = 'green' if labels[i] == 0 else 'red'

        if predictions is not None:
            pred_text = "Real" if predictions[i] == 0 else "Fake"
            correct = (predictions[i] == labels[i]).item()
            title = f"Label: {label_text}\nPred: {pred_text}"
            title_color = 'green' if correct else 'red'
        else:
            title = f"Label: {label_text}"
            title_color = label_color

        axes[i].set_title(title, color=title_color, fontweight='bold')

        # 랜드마크 표시
        if landmarks is not None:
            lm = landmarks[i].cpu().numpy()
            axes[i].scatter(lm[:, 0], lm[:, 1], c='cyan', s=30, marker='x')

        axes[i].axis('off')

    # 빈 subplot 제거
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"배치 시각화 저장: {save_path}")

    plt.show()
    plt.close(fig)


def visualize_feature_maps(
    feature_maps: torch.Tensor,
    num_features: int = 16,
    save_path: Optional[str] = None
) -> None:
    """
    특징 맵 시각화

    Args:
        feature_maps: 특징 맵 텐서 (B, C, H, W)
        num_features: 표시할 특징 맵 수
        save_path: 저장 경로 (선택)
    """
    # 첫 번째 샘플만 사용
    feature_maps = feature_maps[0].cpu().detach()

    num_features = min(num_features, feature_maps.shape[0])
    cols = 4
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_features > 1 else [axes]

    for i in range(num_features):
        fmap = feature_maps[i].numpy()
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].set_title(f'Feature {i}', fontsize=10)
        axes[i].axis('off')

    # 빈 subplot 제거
    for i in range(num_features, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"특징 맵 저장: {save_path}")

    plt.show()
    plt.close(fig)


def visualize_attention_maps(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    landmarks: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    denormalize: bool = True
) -> None:
    """
    어텐션 맵 시각화

    Args:
        image: 원본 이미지 (3, H, W) 또는 (H, W, 3)
        attention_map: 어텐션 맵 (1, H, W) 또는 (H, W)
        landmarks: 랜드마크 좌표 (5, 2) - 선택사항
        save_path: 저장 경로 (선택)
        denormalize: ImageNet 역정규화 적용
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 이미지 준비
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image.cpu().numpy()

        # 역정규화
        if denormalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
    else:
        img = image

    img = np.clip(img, 0, 1)

    # 어텐션 맵 준비
    if isinstance(attention_map, torch.Tensor):
        if attention_map.dim() == 3:
            attn = attention_map[0].cpu().numpy()
        else:
            attn = attention_map.cpu().numpy()
    else:
        attn = attention_map

    # 어텐션 맵을 이미지 크기로 리사이즈
    if attn.shape != img.shape[:2]:
        attn = cv2.resize(attn, (img.shape[1], img.shape[0]))

    # 1. 원본 이미지 (랜드마크 포함)
    axes[0].imshow(img)
    if landmarks is not None:
        if isinstance(landmarks, torch.Tensor):
            lm = landmarks.cpu().numpy()
        else:
            lm = landmarks

        # 랜드마크 표시
        landmark_colors = ['red', 'blue', 'green', 'yellow', 'magenta']
        for i, (x, y) in enumerate(lm):
            axes[0].scatter(x, y, c=landmark_colors[i], s=100, marker='x', linewidths=3)

    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. 어텐션 맵
    im = axes[1].imshow(attn, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 오버레이
    axes[2].imshow(img)
    axes[2].imshow(attn, cmap='hot', alpha=0.5, interpolation='bilinear')
    if landmarks is not None:
        for i, (x, y) in enumerate(lm):
            axes[2].scatter(x, y, c='cyan', s=80, marker='x', linewidths=2)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"어텐션 맵 저장: {save_path}")

    plt.show()
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    학습 곡선 시각화

    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        train_accs: 학습 정확도 리스트
        val_accs: 검증 정확도 리스트
        save_path: 저장 경로 (선택)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # 손실 그래프
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 정확도 그래프
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"학습 곡선 저장: {save_path}")

    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Real', 'Fake'],
    normalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    혼동 행렬 시각화

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        normalize: 정규화 여부
        save_path: 저장 경로 (선택)
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"혼동 행렬 저장: {save_path}")

    plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """
    ROC 곡선 시각화

    Args:
        y_true: 실제 레이블 (0 or 1)
        y_scores: 예측 확률 [0, 1]
        save_path: 저장 경로 (선택)

    Returns:
        AUC 점수
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC 곡선 저장: {save_path}")

    plt.show()
    plt.close()

    return roc_auc


def create_comparison_grid(
    images_dict: Dict[str, np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    여러 이미지 비교 그리드 생성

    Args:
        images_dict: {name: image} 딕셔너리
        titles: 제목 리스트 (선택)
        save_path: 저장 경로 (선택)
    """
    num_images = len(images_dict)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, (name, img) in enumerate(images_dict.items()):
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    # 빈 subplot 제거
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"비교 그리드 저장: {save_path}")

    plt.show()
    plt.close(fig)


def save_visualization(
    fig: plt.Figure,
    save_path: str,
    dpi: int = 150
) -> None:
    """
    Figure 저장 헬퍼 함수

    Args:
        fig: Matplotlib Figure 객체
        save_path: 저장 경로
        dpi: 해상도
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"시각화 저장: {save_path}")


# 테스트 코드
if __name__ == "__main__":
    """
    시각화 함수 테스트
    실행: python src/utils/visualization.py
    """
    print("시각화 유틸리티 테스트\n")

    # 더미 데이터 생성
    dummy_image = np.random.rand(224, 224, 3)
    dummy_aligned = np.random.rand(224, 224, 3)
    dummy_landmarks = np.array([[50, 60], [150, 60], [100, 120], [70, 170], [130, 170]])
    dummy_bbox = np.array([30, 40, 180, 190])
    dummy_quality = {
        'blur': 150.5,
        'brightness': 127.3,
        'contrast': 45.2,
        'overall': 0.85
    }

    # 1. 전처리 결과 시각화 테스트
    print("1. 전처리 결과 시각화...")
    visualize_preprocessing_result(
        dummy_image, dummy_aligned, dummy_landmarks,
        dummy_bbox, dummy_quality,
        save_path='test_preprocessing_vis.png'
    )

    # 2. 학습 곡선 테스트
    print("\n2. 학습 곡선 시각화...")
    dummy_train_loss = [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7]
    dummy_val_loss = [2.6, 2.1, 1.6, 1.3, 1.1, 0.9, 0.85]
    dummy_train_acc = [55, 65, 72, 78, 82, 85, 87]
    dummy_val_acc = [54, 63, 70, 76, 80, 83, 84]

    plot_training_curves(
        dummy_train_loss, dummy_val_loss,
        dummy_train_acc, dummy_val_acc,
        save_path='test_training_curves.png'
    )

    # 3. 혼동 행렬 테스트
    print("\n3. 혼동 행렬 시각화...")
    dummy_y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    dummy_y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])

    plot_confusion_matrix(
        dummy_y_true, dummy_y_pred,
        save_path='test_confusion_matrix.png'
    )

    # 4. ROC 곡선 테스트
    print("\n4. ROC 곡선 시각화...")
    dummy_y_scores = np.array([0.1, 0.2, 0.9, 0.4, 0.15, 0.85, 0.6, 0.95, 0.88, 0.3])

    auc_score = plot_roc_curve(
        dummy_y_true, dummy_y_scores,
        save_path='test_roc_curve.png'
    )
    print(f"   AUC: {auc_score:.3f}")

    print("\n✅ 모든 시각화 테스트 완료!")
    print("생성된 파일:")
    print("  - test_preprocessing_vis.png")
    print("  - test_training_curves.png")
    print("  - test_confusion_matrix.png")
    print("  - test_roc_curve.png")