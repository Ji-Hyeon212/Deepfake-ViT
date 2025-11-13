"""
모델 평가 모듈
위치: src/training/evaluator.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


class Evaluator:
    """
    모델 평가기
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_landmarks: bool = True
    ):
        """
        Args:
            model: 평가할 모델
            device: 디바이스
            use_landmarks: 랜드마크 사용 여부
        """
        self.model = model
        self.device = device
        self.use_landmarks = use_landmarks

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False
    ) -> Dict:
        """
        모델 평가

        Args:
            dataloader: 평가 데이터로더
            criterion: 손실 함수 (선택)
            return_predictions: 예측 결과 반환 여부

        Returns:
            metrics: {
                'loss': float,
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'auc': float,
                'ap': float (Average Precision),
                'confusion_matrix': np.ndarray,
                'predictions': np.ndarray (optional),
                'probabilities': np.ndarray (optional),
                'labels': np.ndarray (optional)
            }
        """
        self.model.eval()

        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 데이터 로드
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            landmarks = None
            if self.use_landmarks and 'landmarks' in batch:
                landmarks = batch['landmarks'].to(self.device)

            # Forward
            if self.use_landmarks:
                logits, _ = self.model(images, landmarks)
            else:
                logits, _ = self.model(images)

            # 손실 계산
            if criterion is not None:
                loss = criterion(logits, labels)

                # CombinedLoss는 딕셔너리 반환
                if isinstance(loss, dict):
                    loss_value = loss['total'].item()
                else:
                    loss_value = loss.item()

                total_loss += loss_value * images.size(0)

            # 예측 및 확률
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # 수집
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Concatenate
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 메트릭 계산
        metrics = self._compute_metrics(
            all_preds,
            all_probs,
            all_labels,
            total_loss / len(dataloader.dataset) if criterion else None
        )

        # 예측 결과 포함
        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['probabilities'] = all_probs
            metrics['labels'] = all_labels

        return metrics

    def _compute_metrics(
        self,
        preds: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        loss: Optional[float] = None
    ) -> Dict:
        """
        메트릭 계산

        Args:
            preds: 예측 레이블 (N,)
            probs: 예측 확률 (N, C)
            labels: 실제 레이블 (N,)
            loss: 평균 손실

        Returns:
            metrics: 메트릭 딕셔너리
        """
        metrics = {}

        # Loss
        if loss is not None:
            metrics['loss'] = loss

        # Accuracy
        acc = accuracy_score(labels, preds)
        metrics['accuracy'] = acc * 100  # 퍼센트

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        metrics['precision'] = precision * 100
        metrics['recall'] = recall * 100
        metrics['f1'] = f1 * 100

        # AUC (이진 분류)
        if probs.shape[1] == 2:
            try:
                auc = roc_auc_score(labels, probs[:, 1])
                metrics['auc'] = auc

                # Average Precision
                ap = average_precision_score(labels, probs[:, 1])
                metrics['ap'] = ap
            except:
                metrics['auc'] = 0.0
                metrics['ap'] = 0.0

        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm

        # 클래스별 정확도
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = tn
            metrics['false_positive'] = fp
            metrics['false_negative'] = fn
            metrics['true_positive'] = tp

            # Specificity (진짜를 진짜로)
            metrics['specificity'] = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

            # Sensitivity (가짜를 가짜로)
            metrics['sensitivity'] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

        return metrics

    def print_metrics(self, metrics: Dict, prefix: str = ""):
        """
        메트릭 출력

        Args:
            metrics: 메트릭 딕셔너리
            prefix: 접두사 (예: "Val")
        """
        print(f"\n{prefix} Metrics:")
        print(f"  Loss: {metrics.get('loss', 0):.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall: {metrics['recall']:.2f}%")
        print(f"  F1-Score: {metrics['f1']:.2f}%")

        if 'auc' in metrics:
            print(f"  AUC: {metrics['auc']:.4f}")
        if 'ap' in metrics:
            print(f"  AP: {metrics['ap']:.4f}")

        if 'specificity' in metrics:
            print(f"  Specificity: {metrics['specificity']:.2f}%")
        if 'sensitivity' in metrics:
            print(f"  Sensitivity: {metrics['sensitivity']:.2f}%")

        if 'confusion_matrix' in metrics:
            print(f"\n  Confusion Matrix:")
            print(f"    {metrics['confusion_matrix']}")


class MetricsTracker:
    """
    학습 중 메트릭 추적
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': [],
            'learning_rates': []
        }

        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_auc': 0.0,
            'best_val_f1': 0.0,
            'best_epoch': 0
        }

    def update(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float
    ):
        """
        에폭 메트릭 업데이트

        Args:
            epoch: 현재 에폭
            train_metrics: 학습 메트릭
            val_metrics: 검증 메트릭
            lr: 현재 학습률
        """
        # 학습 메트릭
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['train_acc'].append(train_metrics.get('accuracy', 0))

        # 검증 메트릭
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.history['val_auc'].append(val_metrics.get('auc', 0))
        self.history['val_f1'].append(val_metrics.get('f1', 0))

        # 학습률
        self.history['learning_rates'].append(lr)

        # 최고 성능 업데이트
        if val_metrics['accuracy'] > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = val_metrics['accuracy']
            self.best_metrics['best_epoch'] = epoch

        if val_metrics.get('auc', 0) > self.best_metrics['best_val_auc']:
            self.best_metrics['best_val_auc'] = val_metrics.get('auc', 0)

        if val_metrics.get('f1', 0) > self.best_metrics['best_val_f1']:
            self.best_metrics['best_val_f1'] = val_metrics.get('f1', 0)

    def get_history(self) -> Dict:
        """히스토리 반환"""
        return self.history

    def get_best_metrics(self) -> Dict:
        """최고 메트릭 반환"""
        return self.best_metrics

    def print_summary(self):
        """요약 출력"""
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Best Validation Accuracy: {self.best_metrics['best_val_acc']:.2f}% "
              f"(Epoch {self.best_metrics['best_epoch']})")
        print(f"Best Validation AUC: {self.best_metrics['best_val_auc']:.4f}")
        print(f"Best Validation F1: {self.best_metrics['best_val_f1']:.2f}%")
        print("="*60)


# 테스트 코드
if __name__ == "__main__":
    """
    평가 모듈 테스트
    실행: python src/training/evaluator.py
    """
    print("평가 모듈 테스트\n")

    # 더미 데이터
    n_samples = 100
    n_classes = 2

    preds = np.random.randint(0, n_classes, n_samples)
    probs = np.random.rand(n_samples, n_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, n_classes, n_samples)

    # Evaluator 인스턴스 (더미 모델)
    class DummyModel(nn.Module):
        def forward(self, x, lm=None):
            return torch.randn(x.size(0), 2), None

    dummy_model = DummyModel()
    evaluator = Evaluator(dummy_model, torch.device('cpu'))

    # 메트릭 계산
    print("1. 메트릭 계산 테스트")
    metrics = evaluator._compute_metrics(preds, probs, labels, loss=0.5)
    evaluator.print_metrics(metrics, prefix="Test")

    # MetricsTracker 테스트
    print("\n2. MetricsTracker 테스트")
    tracker = MetricsTracker()

    for epoch in range(5):
        train_metrics = {'loss': 0.5 - epoch * 0.05, 'accuracy': 70 + epoch * 2}
        val_metrics = {
            'loss': 0.6 - epoch * 0.05,
            'accuracy': 68 + epoch * 3,
            'auc': 0.7 + epoch * 0.03,
            'f1': 65 + epoch * 3
        }

        tracker.update(epoch, train_metrics, val_metrics, lr=1e-4 * (0.9 ** epoch))

    tracker.print_summary()

    print("\n✅ 평가 모듈 테스트 완료!")