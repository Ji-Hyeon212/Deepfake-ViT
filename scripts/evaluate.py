"""
모델 평가 스크립트
위치: scripts/evaluate.py

실행: python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config config/training_config.yaml
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.feature_extraction import DeepfakeDetectionModel
from src.training import Evaluator
from src.utils import load_config, get_device, load_checkpoint

import numpy as np


def main(args):
    """메인 함수"""
    print("\n" + "=" * 70)
    print("모델 평가")
    print("=" * 70)

    # 설정 로드
    config = load_config(args.config)
    device = get_device()

    # 데이터로더
    print("\n데이터로더 생성 중...")
    _, _, test_loader = create_dataloaders(
        processed_dir=config['data']['processed_dir'],
        batch_size=args.batch_size,
        num_workers=config['data']['num_workers']
    )

    # 모델 생성
    print("\n모델 생성 중...")
    model = DeepfakeDetectionModel(
        num_classes=config['model']['classifier']['num_classes'],
        pretrained=False,  # 체크포인트에서 로드
        feature_extractor_config=config['model']['feature_extractor'],
        classifier_hidden_dims=config['model']['classifier']['hidden_dims'],
        dropout_rate=config['model']['classifier']['dropout_rate']
    )

    # 체크포인트 로드
    print(f"\n체크포인트 로드: {args.checkpoint}")
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        device=device
    )

    model.to(device)
    model.eval()

    # Evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        use_landmarks=config['data']['use_landmarks']
    )

    # 평가
    print("\n평가 시작...")
    metrics = evaluator.evaluate(
        test_loader,
        return_predictions=True
    )

    # 결과 출력
    evaluator.print_metrics(metrics, prefix="Test")

    # 상세 분석
    if args.detailed:
        print("\n" + "=" * 70)
        print("상세 분석")
        print("=" * 70)

        # 클래스별 정확도
        for class_idx in range(2):
            class_name = 'Real' if class_idx == 0 else 'Fake'
            mask = metrics['labels'] == class_idx
            class_acc = (metrics['predictions'][mask] == class_idx).mean() * 100
            print(f"{class_name} 정확도: {class_acc:.2f}%")

        # 신뢰도별 정확도
        if 'probabilities' in metrics:
            probs = metrics['probabilities']
            max_probs = probs.max(axis=1)

            for threshold in [0.5, 0.7, 0.9]:
                high_conf_mask = max_probs >= threshold
                if high_conf_mask.sum() > 0:
                    acc = (metrics['predictions'][high_conf_mask] ==
                           metrics['labels'][high_conf_mask]).mean() * 100
                    coverage = high_conf_mask.mean() * 100
                    print(f"신뢰도 >= {threshold}: 정확도 {acc:.2f}%, 커버리지 {coverage:.2f}%")

    print("\n✅ 평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 평가')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='평가할 체크포인트 경로'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='배치 크기'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='시각화 생성'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='상세 분석'
    )

    args = parser.parse_args()
    main(args)