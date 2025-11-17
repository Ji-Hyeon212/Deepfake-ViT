"""
모델 학습 트레이너
위치: src/training/trainer.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import time

from .evaluator import Evaluator, MetricsTracker
from .losses import CombinedLoss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import save_checkpoint, get_device


class Trainer:
    """
    모델 학습 트레이너
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: Optional[torch.device] = None,
            config: Optional[Dict] = None
    ):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            criterion: 손실 함수
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러 (선택)
            device: 디바이스
            config: 학습 설정
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else get_device()

        # 기본 설정
        default_config = {
            'num_epochs': 100,
            'save_dir': 'checkpoints',
            'log_dir': 'runs',
            'use_landmarks': True,
            'gradient_clip': 1.0,
            'print_freq': 10,
            'save_freq': 5,
            'early_stopping_patience': 15,
            'use_amp': True,  # Automatic Mixed Precision
            'accumulation_steps': 1
        }

        self.config = {**default_config, **(config or {})}

        # 모델을 디바이스로
        self.model.to(self.device)

        # 디렉토리 생성
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Evaluator
        self.evaluator = Evaluator(
            self.model,
            self.device,
            use_landmarks=self.config['use_landmarks']
        )

        # Metrics Tracker
        self.metrics_tracker = MetricsTracker()

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['use_amp'] else None

        # Early Stopping
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')

        print(f"✅ Trainer 초기화 완료")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Use AMP: {self.config['use_amp']}")

    def train_epoch(self, epoch: int) -> Dict:
        """
        1 에폭 학습

        Args:
            epoch: 현재 에폭

        Returns:
            metrics: 학습 메트릭
        """
        self.model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]",
            leave=True
        )

        for batch_idx, batch in enumerate(pbar):
            # 데이터 로드
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            landmarks = None
            if self.config['use_landmarks'] and 'landmarks' in batch:
                landmarks = batch['landmarks'].to(self.device)

            # Mixed Precision
            if self.config['use_amp']:
                with torch.cuda.amp.autocast():
                    # Forward
                    logits, features = self.model(images, landmarks, return_features=True)

                    # Loss
                    if isinstance(self.criterion, CombinedLoss):
                        losses = self.criterion(logits, labels, features)
                        loss = losses['total']
                    else:
                        loss = self.criterion(logits, labels)

                    # Gradient Accumulation
                    loss = loss / self.config['accumulation_steps']

                # Backward
                self.scaler.scale(loss).backward()

                # Optimizer step
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    # Gradient Clipping
                    if self.config['gradient_clip'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                logits, features = self.model(images, landmarks, return_features=True)

                if isinstance(self.criterion, CombinedLoss):
                    losses = self.criterion(logits, labels, features)
                    loss = losses['total']
                else:
                    loss = self.criterion(logits, labels)

                loss = loss / self.config['accumulation_steps']
                loss.backward()

                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    if self.config['gradient_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 통계
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0) * self.config['accumulation_steps']
            running_corrects += (preds == labels).sum().item()
            total_samples += images.size(0)

            # Progress bar 업데이트
            if batch_idx % self.config['print_freq'] == 0:
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': 100.0 * running_corrects / total_samples
                })

        # 에폭 메트릭
        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * running_corrects / total_samples

        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }

        return metrics

    def validate(self, epoch: int) -> Dict:
        """
        검증

        Args:
            epoch: 현재 에폭

        Returns:
            metrics: 검증 메트릭
        """
        metrics = self.evaluator.evaluate(
            self.val_loader,
            criterion=self.criterion
        )

        return metrics

    def train(self, start_epoch: int = 1):
        """
        전체 학습 루프
        """
        print("\n" + "=" * 70)
        print("학습 시작")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(start_epoch, self.config['num_epochs'] + 1):
            epoch_start = time.time()

            # 학습
            train_metrics = self.train_epoch(epoch)

            # 검증
            val_metrics = self.validate(epoch)

            # 학습률 스케줄러
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']

            # 체크포인트 저장
            is_best = val_metrics['accuracy'] > self.metrics_tracker.best_metrics['best_val_acc']

            if epoch % self.config['save_freq'] == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics, is_best)

            # Metrics Tracker 업데이트
            self.metrics_tracker.update(
                epoch, train_metrics, val_metrics, current_lr
            )

            # 출력
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.config['num_epochs']} - {epoch_time:.2f}s")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
                  f"AUC: {val_metrics.get('auc', 0):.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Early Stopping
            if self._early_stopping(val_metrics['loss']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break

        # 학습 완료
        total_time = time.time() - start_time
        print(f"\n학습 완료! 총 시간: {total_time / 60:.2f}분")

        # 요약
        self.metrics_tracker.print_summary()

    def _save_checkpoint(
            self,
            epoch: int,
            metrics: Dict,
            is_best: bool = False
    ):
        """체크포인트 저장"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': self.metrics_tracker.best_metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # 저장 경로
        save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        save_checkpoint(
            state,
            str(save_path),
            is_best=is_best,
            max_keep=5
        )

    def _early_stopping(self, val_loss: float) -> bool:
        """
        Early Stopping 체크

        Args:
            val_loss: 검증 손실

        Returns:
            stop: True면 학습 중단
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.config['early_stopping_patience']:
            return True

        return False

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        체크포인트에서 재개

        Args:
            checkpoint_path: 체크포인트 경로
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✅ 체크포인트 로드: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Best Val Acc: {checkpoint['best_metrics']['best_val_acc']:.2f}%")

        return checkpoint['epoch']


# 테스트 코드
if __name__ == "__main__":
    """
    Trainer 테스트
    실행: python src/training/trainer.py
    """
    print("Trainer 테스트\n")


    # 더미 모델 및 데이터
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, 2)

        def forward(self, x, lm=None, return_features=False):
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            if return_features:
                return out, x
            return out, None


    model = DummyModel()


    # 더미 데이터로더
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 224, 224),
                'label': torch.randint(0, 2, (1,)).item(),
                'landmarks': torch.rand(5, 2) * 224
            }


    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)

    # 옵티마이저 및 손실
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config={
            'num_epochs': 2,
            'use_amp': False,
            'print_freq': 2
        }
    )

    # 학습 (2 에폭만)
    trainer.train()

    print("\n✅ Trainer 테스트 완료!")