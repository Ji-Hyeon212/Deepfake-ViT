"""
전체 모듈 통합 테스트
위치: scripts/test_all_modules.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """모든 모듈 임포트 테스트"""
    print("=" * 70)
    print("모듈 임포트 테스트")
    print("=" * 70 + "\n")

    tests = []

    # 1. Preprocessing 모듈
    print("1. Preprocessing 모듈")
    try:
        from src.preprocessing import (
            FaceDetector,
            FaceAligner,
            QualityChecker,
            PreprocessingPipeline,
            PreprocessingOutput
        )
        print("   ✅ 모든 preprocessing 모듈 임포트 성공")
        tests.append(True)
    except ImportError as e:
        print(f"   ❌ 실패: {e}")
        tests.append(False)

    # 2. Data 모듈
    print("\n2. Data 모듈")
    try:
        from src.data import (
            PreprocessedFaceDataset,
            create_dataloaders,
            PreprocessingToFeatureInterface,
            FeatureExtractionInput
        )
        print("   ✅ 모든 data 모듈 임포트 성공")
        tests.append(True)
    except ImportError as e:
        print(f"   ❌ 실패: {e}")
        tests.append(False)

    # 3. Utils 모듈
    print("\n3. Utils 모듈")
    try:
        from src.utils import (
            setup_logger,
            visualize_detection_result,
            plot_training_curves,
            load_json,
            save_checkpoint
        )
        print("   ✅ 모든 utils 모듈 임포트 성공")
        tests.append(True)
    except ImportError as e:
        print(f"   ❌ 실패: {e}")
        tests.append(False)

    return all(tests)


def test_preprocessing_module():
    """Preprocessing 모듈 기능 테스트"""
    print("\n" + "=" * 70)
    print("Preprocessing 모듈 기능 테스트")
    print("=" * 70 + "\n")

    try:
        import yaml
        from src.preprocessing import create_pipeline_from_config

        # 설정 파일 확인
        config_path = Path("config/preprocessing_config.yaml")
        if not config_path.exists():
            print("⚠️  설정 파일 없음 - 스킵")
            return True

        # 파이프라인 생성
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        from src.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline(config)

        print("✅ Preprocessing 파이프라인 생성 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_data_module():
    """Data 모듈 기능 테스트"""
    print("\n" + "=" * 70)
    print("Data 모듈 기능 테스트")
    print("=" * 70 + "\n")

    try:
        from src.data import PreprocessedFaceDataset

        # CSV 파일 확인
        train_csv = Path("data/processed/splits/train.csv")
        if not train_csv.exists():
            print("⚠️  전처리 데이터 없음 - 스킵")
            return True

        # 데이터셋 생성
        dataset = PreprocessedFaceDataset(
            csv_file=str(train_csv),
            processed_dir="data/processed",
            load_landmarks=True
        )

        print(f"✅ Dataset 생성 성공 ({len(dataset)} 샘플)")

        # 샘플 로드 테스트
        sample = dataset[0]
        print(f"   샘플 keys: {sample.keys()}")
        print(f"   Image shape: {sample['image'].shape}")

        return True

    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils_module():
    """Utils 모듈 기능 테스트"""
    print("\n" + "=" * 70)
    print("Utils 모듈 기능 테스트")
    print("=" * 70 + "\n")

    try:
        from src.utils import setup_logger, save_json, load_json
        import tempfile

        # Logger 테스트
        logger = setup_logger("test", level="INFO")
        logger.info("테스트 로그 메시지")
        print("✅ Logger 생성 성공")

        # I/O 테스트
        test_data = {'test': 'data', 'value': 123}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        save_json(test_data, temp_path)
        loaded = load_json(temp_path)

        assert loaded == test_data
        print("✅ I/O 기능 테스트 성공")

        # 임시 파일 삭제
        Path(temp_path).unlink()

        return True

    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_visualization_module():
    """Visualization 모듈 테스트"""
    print("\n" + "=" * 70)
    print("Visualization 모듈 기능 테스트")
    print("=" * 70 + "\n")

    try:
        import numpy as np
        from src.utils import plot_training_curves

        # 더미 데이터
        train_losses = [0.5, 0.4, 0.3, 0.2]
        val_losses = [0.6, 0.5, 0.4, 0.35]

        # 시각화 (저장만, 표시 안함)
        plot_training_curves(
            train_losses, val_losses,
            show=False,
            save_path="test_viz_output.png"
        )

        if Path("test_viz_output.png").exists():
            print("✅ Visualization 테스트 성공")
            Path("test_viz_output.png").unlink()  # 삭제
            return True
        else:
            print("❌ 시각화 파일 생성 실패")
            return False

    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 22 + "전체 모듈 테스트" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝\n")

    results = []

    # 1. 임포트 테스트
    results.append(("임포트", test_imports()))

    # 2. Preprocessing 테스트
    results.append(("Preprocessing", test_preprocessing_module()))

    # 3. Data 테스트
    results.append(("Data", test_data_module()))

    # 4. Utils 테스트
    results.append(("Utils", test_utils_module()))

    # 5. Visualization 테스트
    # results.append(("Visualization", test_visualization_module()))

    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)

    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name:20s}: {status}")

    print("=" * 70)

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 모든 테스트 통과!")
        print("\n다음 단계:")
        print("1. EfficientNet-B4 특징 추출 모델 구현")
        print("2. 학습 스크립트 작성")
        print("3. 모델 학습 시작")
    else:
        print("\n⚠️  일부 테스트 실패")
        print("실패한 모듈을 확인하고 수정하세요.")

    print()


if __name__ == "__main__":
    main()