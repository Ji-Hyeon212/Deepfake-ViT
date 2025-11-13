"""
모델 학습 스크립트
위치: scripts/train.py

실행: python scripts/train.py --config config/training_config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.feature_extraction import DeepfakeDetectionModel
from src.training import Trainer, FocalLoss, CombinedLoss, LabelSmoothingLoss
from src.utils import setup_logger, get_device, load_config


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed 설정: {seed}")


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """옵티마이저 생성"""
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type']

    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=tuple(opt_config['betas'])
        )
    elif opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=tuple(opt_config['betas'])
        )
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config['momentum'],
            weight_decay=opt_config['weight_decay'],
            nesterov=opt_config['nesterov']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    print(f"✅ 옵티마이저: {opt_type}")
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """학습률 스케줄러 생성"""
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type']

    if sched_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max'],
            eta_min=sched_config['eta_min']
        )
    elif sched_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config['mode'],
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            min_lr=sched_config['min_lr']
        )
    elif sched_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_config['T_0'],
            T_mult=sched_config['T_mult'],
            eta_min=sched_config['eta_min_restart']
        )
    else:
        scheduler = None
        print("⚠️  스케줄러 없음")

    if scheduler:
        print(f"✅ 스케줄러: {sched_type}")

    return scheduler


def create_criterion(config: dict, class_weights=None) -> nn.Module:
    """손실 함수 생성"""
    loss_config = config['training']['loss']
    loss_type = loss_config['type']

    # 클래스 가중치
    if class_weights is not None and loss_config['class_weights'] is not None:
        class_weights = torch.tensor(loss_config['class_weights'], dtype=torch.float32)

    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'FocalLoss':
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=loss_config['focal_gamma']
        )
    elif loss_type == 'LabelSmoothing':
        criterion = LabelSmoothingLoss(
            num_classes=config['model']['classifier']['num_classes'],
            smoothing=loss_config['smoothing']
        )
    elif loss_type == 'CombinedLoss':
        criterion = CombinedLoss(
            weights=loss_config['weights'],
            class_weights=class_weights
        )
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

    print(f"✅ 손실 함수: {loss_type}")
    return criterion


def main(args):
    """메인 함수"""
    global start_epoch
    print("\n" + "=" * 70)
    print("Deepfake Detection 모델 학습")
    print("=" * 70)

    # 설정 로드
    config = load_config(args.config)
    print(f"✅ 설정 로드: {args.config}")

    # 시드 설정
    set_seed(config.get('seed', 42))

    # 디바이스
    device = get_device(config['hardware']['device'])

    # 로거
    logger = setup_logger(
        'training',
        log_file=Path(config['logging']['log_dir']) / 'training.log'
    )

    # 데이터로더
    print("\n데이터로더 생성 중...")
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_dir=config['data']['processed_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    # 클래스 가중치 계산
    class_weights = train_loader.dataset.get_class_weights()
    print(f"클래스 가중치: {class_weights}")

    # 모델 생성
    print("\n모델 생성 중...")
    model = DeepfakeDetectionModel(
        num_classes=config['model']['classifier']['num_classes'],
        pretrained=config['model']['feature_extractor']['pretrained'],
        feature_extractor_config=config['model']['feature_extractor'],
        classifier_hidden_dims=config['model']['classifier']['hidden_dims'],
        dropout_rate=config['model']['classifier']['dropout_rate']
    )

    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")

    # 옵티마이저
    optimizer = create_optimizer(model, config)

    # 스케줄러
    scheduler = create_scheduler(optimizer, config)

    # 손실 함수
    criterion = create_criterion(config, class_weights.to(device))

    # Trainer 생성
    trainer_config = {
        'num_epochs': config['training']['num_epochs'],
        'save_dir': config['checkpoint']['save_dir'],
        'log_dir': config['logging']['log_dir'],
        'use_landmarks': config['data']['use_landmarks'],
        'gradient_clip': config['training']['gradient_clip'],
        'print_freq': config['validation']['print_freq'],
        'save_freq': config['validation']['save_freq'],
        'early_stopping_patience': config['early_stopping']['patience'],
        'use_amp': config['training']['use_amp'],
        'accumulation_steps': config['training']['accumulation_steps']
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config
    )

    start_epoch = 1
    # 체크포인트에서 재개
    if args.resume:
        start_epoch = trainer.resume_from_checkpoint(args.resume)
        start_epoch += 1
        print(f"✅ 에폭 {start_epoch}부터 재개")

    # 학습 시작
    trainer.train(start_epoch=start_epoch)

    # 테스트 평가
    if test_loader is not None:
        print("\n" + "=" * 70)
        print("테스트 세트 평가")
        print("=" * 70)

        test_metrics = trainer.evaluator.evaluate(
            test_loader,
            criterion=criterion
        )

        trainer.evaluator.print_metrics(test_metrics, prefix="Test")

    print("\n✅ 학습 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepfake Detection 모델 학습')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='학습 설정 파일 경로'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='재개할 체크포인트 경로'
    )

    args = parser.parse_args()

    main(args)