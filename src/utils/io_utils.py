"""
I/O utilities for model checkpointing and configuration management
위치: src/utils/io_utils.py
"""

import torch
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 반환

    Returns:
        프로젝트 루트 Path 객체
    """
    # 현재 파일에서 3단계 상위 (utils -> src -> project_root)
    return Path(__file__).parent.parent.parent


def ensure_dir(path: str) -> Path:
    """
    디렉토리 생성 (존재하지 않으면)

    Args:
        path: 디렉토리 경로

    Returns:
        Path 객체
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: str) -> Dict[str, Any]:
    """
    JSON 파일 로드

    Args:
        filepath: JSON 파일 경로

    Returns:
        딕셔너리
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    JSON 파일 저장

    Args:
        data: 저장할 데이터
        filepath: 저장 경로
        indent: 들여쓰기 수준
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"JSON 저장: {filepath}")


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    YAML 파일 로드

    Args:
        filepath: YAML 파일 경로

    Returns:
        딕셔너리
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """
    YAML 파일 저장

    Args:
        data: 저장할 데이터
        filepath: 저장 경로
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML 저장: {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Pickle 파일 로드

    Args:
        filepath: Pickle 파일 경로

    Returns:
        저장된 객체
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: str):
    """
    Pickle 파일 저장

    Args:
        data: 저장할 객체
        filepath: 저장 경로
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Pickle 저장: {filepath}")


def save_checkpoint(
    state: Dict[str, Any],
    save_path: str,
    is_best: bool = False,
    max_keep: int = 5
) -> None:
    """
    모델 체크포인트 저장

    Args:
        state: 저장할 상태 딕셔너리
            {
                'epoch': int,
                'model_state_dict': OrderedDict,
                'optimizer_state_dict': OrderedDict,
                'scheduler_state_dict': OrderedDict (optional),
                'best_acc': float,
                'best_loss': float,
                ...
            }
        save_path: 저장 경로
        is_best: 최고 성능 모델 여부
        max_keep: 유지할 최대 체크포인트 수
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 체크포인트 저장
    torch.save(state, save_path)
    print(f"체크포인트 저장: {save_path}")

    # 최고 성능 모델 별도 저장
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        shutil.copyfile(save_path, best_path)
        print(f"최고 성능 모델 저장: {best_path}")

    # 오래된 체크포인트 삭제 (최신 max_keep개만 유지)
    if max_keep > 0:
        checkpoints = sorted(
            save_path.parent.glob('checkpoint_epoch_*.pth'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for old_ckpt in checkpoints[max_keep:]:
            old_ckpt.unlink()
            print(f"오래된 체크포인트 삭제: {old_ckpt}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    체크포인트 로드

    Args:
        checkpoint_path: 체크포인트 경로
        model: 모델 객체
        optimizer: 옵티마이저 (선택)
        scheduler: 스케줄러 (선택)
        device: 디바이스

    Returns:
        체크포인트 딕셔너리
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    print(f"체크포인트 로드: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])

    # 옵티마이저 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 스케줄러 로드
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Accuracy: {checkpoint.get('best_acc', 'N/A')}")
    print(f"  Best Loss: {checkpoint.get('best_loss', 'N/A')}")

    return checkpoint


# Alias functions for compatibility
def save_config(config: Dict[str, Any], save_path: str) -> None:
    """설정 파일 저장 (YAML or JSON)"""
    save_path = Path(save_path)

    if save_path.suffix in ['.yaml', '.yml']:
        save_yaml(config, str(save_path))
    elif save_path.suffix == '.json':
        save_json(config, str(save_path))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {save_path.suffix}")


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드 (YAML or JSON)"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    if config_path.suffix in ['.yaml', '.yml']:
        return load_yaml(str(config_path))
    elif config_path.suffix == '.json':
        return load_json(str(config_path))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {config_path.suffix}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    모델 파라미터 개수 계산

    Args:
        model: PyTorch 모델
        trainable_only: 학습 가능한 파라미터만 카운트

    Returns:
        파라미터 개수
    """
    if trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


def get_device(device: Optional[str] = None) -> torch.device:
    """
    사용 가능한 디바이스 반환

    Args:
        device: 원하는 디바이스 ('cuda', 'cpu', 또는 None)

    Returns:
        torch.device 객체
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            device = torch.device('cpu')
        else:
            print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 버전: {torch.version.cuda}")

    print(f"디바이스: {device}")

    return device


def print_model_info(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """
    모델 정보 출력

    Args:
        model: PyTorch 모델
        input_size: 입력 크기
    """
    print("\n" + "="*60)
    print("모델 정보")
    print("="*60)

    # 파라미터 개수
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"학습 불가 파라미터: {total_params - trainable_params:,}")

    # 모델 크기 추정 (MB)
    param_size = total_params * 4 / (1024 ** 2)  # float32 기준
    print(f"모델 크기 (추정): {param_size:.2f} MB")

    # 입력 크기
    print(f"입력 크기: {input_size}")

    print("="*60 + "\n")


def save_metrics(
    metrics: Dict[str, Any],
    save_path: str,
    append: bool = False
) -> None:
    """
    메트릭 저장 (JSON)

    Args:
        metrics: 메트릭 딕셔너리
        save_path: 저장 경로
        append: 기존 파일에 추가할지 여부
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if append and save_path.exists():
        # 기존 데이터 로드
        existing_data = load_json(str(save_path))

        # 리스트면 append, 딕셔너리면 update
        if isinstance(existing_data, list):
            existing_data.append(metrics)
            data_to_save = existing_data
        elif isinstance(existing_data, dict):
            existing_data.update(metrics)
            data_to_save = existing_data
        else:
            data_to_save = [existing_data, metrics]
    else:
        data_to_save = metrics

    # 저장
    save_json(data_to_save, str(save_path))


# 테스트 코드
if __name__ == "__main__":
    """
    I/O 유틸리티 테스트
    실행: python src/utils/io_utils.py
    """
    print("I/O 유틸리티 테스트\n")

    # 1. 프로젝트 루트
    print("1. 프로젝트 루트")
    root = get_project_root()
    print(f"   {root}\n")

    # 2. 디렉토리 생성
    print("2. 디렉토리 생성 테스트")
    test_dir = ensure_dir("test_outputs/checkpoints")
    print(f"   생성됨: {test_dir}\n")

    # 3. JSON 저장/로드
    print("3. JSON 저장/로드 테스트")
    test_data = {'model': 'efficientnet-b4', 'batch_size': 32}
    save_json(test_data, "test_outputs/config.json")
    loaded_data = load_json("test_outputs/config.json")
    print(f"   로드된 데이터: {loaded_data}\n")

    # 4. YAML 저장/로드
    print("4. YAML 저장/로드 테스트")
    save_yaml(test_data, "test_outputs/config.yaml")
    loaded_yaml = load_yaml("test_outputs/config.yaml")
    print(f"   로드된 데이터: {loaded_yaml}\n")

    # 5. 더미 모델로 체크포인트 테스트
    print("5. 체크포인트 저장 테스트")
    dummy_model = torch.nn.Linear(10, 2)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

    state = {
        'epoch': 10,
        'model_state_dict': dummy_model.state_dict(),
        'optimizer_state_dict': dummy_optimizer.state_dict(),
        'best_acc': 95.5,
        'best_loss': 0.123
    }

    save_checkpoint(state, "test_outputs/checkpoints/checkpoint_epoch_10.pth", is_best=True)
    print()

    # 6. 파라미터 개수
    print("6. 모델 파라미터 계산")
    total = count_parameters(dummy_model)
    trainable = count_parameters(dummy_model, trainable_only=True)
    print(f"   총 파라미터: {total}")
    print(f"   학습 가능: {trainable}\n")

    # 7. 디바이스 확인
    print("7. 디바이스 확인")
    device = get_device()
    print()

    # 8. 모델 정보 출력
    print("8. 모델 정보")
    print_model_info(dummy_model, input_size=(1, 10))

    print("\n✅ 모든 I/O 유틸리티 테스트 완료!")

    # 정리
    print("\n정리 중...")
    import shutil
    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
        print("테스트 파일 삭제 완료")