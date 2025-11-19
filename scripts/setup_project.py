"""
프로젝트 폴더 구조 생성 스크립트
위치: scripts/setup_project.py

실행: python scripts/setup_project.py
"""

from pathlib import Path


def create_project_structure():
    """프로젝트 폴더 구조 생성"""

    # 생성할 디렉토리 목록
    directories = [
        # 소스 코드
        "src/preprocessing",
        "src/data",
        "src/feature_extraction",
        "src/classification",
        "src/training",
        "src/utils",

        # 설정
        "config",

        # 스크립트
        "scripts",

        # 데이터
        "data/raw/LFW-FER/images",
        "data/raw/DeeperForensics/real",
        "data/raw/DeeperForensics/fake",
        "data/processed/faces",
        "data/processed/landmarks",
        "data/processed/metadata",
        "data/processed/splits",

        # 체크포인트
        "checkpoints",

        # 로그
        "runs",
        "outputs/logs",
        "outputs/evaluation",
        "outputs/visualizations",

        # 테스트
        "tests"
    ]

    print("프로젝트 폴더 구조 생성 중...\n")

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

        # .gitkeep 생성 (빈 폴더 추적용)
        if not any(path.iterdir()):
            gitkeep = path / ".gitkeep"
            gitkeep.touch()

    print("\n✅ 폴더 구조 생성 완료!")

    # README 파일들 생성
    create_readme_files()


def create_readme_files():
    """각 디렉토리에 README 생성"""

    readme_contents = {
        "data/raw/README.md": """# Raw Data

이 디렉토리는 원본 데이터를 저장합니다.

## 데이터셋 구조

- `LFW-FER/images/`: LFW-FER 데이터셋 이미지
- `DeeperForensics/real/`: DeeperForensics 실제 얼굴 이미지
- `DeeperForensics/fake/`: DeeperForensics 가짜 얼굴 이미지

## 데이터 다운로드

1. LFW-FER: [다운로드 링크]
2. DeeperForensics-1.0: [다운로드 링크]
""",

        "data/processed/README.md": """# Processed Data

이 디렉토리는 전처리된 데이터를 저장합니다.

## 구조

- `faces/`: 정렬된 얼굴 이미지 (224x224)
- `landmarks/`: 얼굴 랜드마크 좌표
- `metadata/`: 품질 메트릭 및 메타데이터
- `splits/`: train/val/test 분할 정보

## 생성 방법

```bash
python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets all
```
""",

        "checkpoints/README.md": """# Model Checkpoints

이 디렉토리는 학습된 모델 체크포인트를 저장합니다.

## 파일 형식

- `checkpoint_epoch_X.pth`: X 에폭의 체크포인트
- `best_model.pth`: 최고 성능 모델

## 로드 방법

```python
from src.utils import load_checkpoint

checkpoint = load_checkpoint('checkpoints/best_model.pth', model)
```
""",

        "runs/README.md": """# TensorBoard Logs

이 디렉토리는 TensorBoard 로그를 저장합니다.

## 사용 방법

```bash
tensorboard --logdir runs
```

브라우저에서 http://localhost:6006 접속
""",

        "outputs/README.md": """# Outputs

이 디렉토리는 학습 및 평가 결과물을 저장합니다.

## 구조

- `logs/`: 학습 로그
- `evaluation/`: 평가 결과 (confusion matrix, ROC curve 등)
- `visualizations/`: 시각화 결과
"""
    }

    print("\nREADME 파일 생성 중...\n")

    for filepath, content in readme_contents.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ {filepath}")

    print("\n✅ README 파일 생성 완료!")


def create_gitignore():
    """
    .gitignore 파일 생성
    """
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/*
!data/raw/.gitkeep
!data/raw/README.md
data/processed/*
!data/processed/.gitkeep
!data/processed/README.md

# Models
checkpoints/*.pth
!checkpoints/.gitkeep
!checkpoints/README.md

# Logs
runs/*
!runs/.gitkeep
!runs/README.md
outputs/logs/*.log
*.log

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
temp/
"""

    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)

    print("\n✅ .gitignore 파일 생성 완료!")


def print_project_tree():
    """프로젝트 트리 출력"""
    print("\n" + "=" * 70)
    print("프로젝트 구조")
    print("=" * 70)
    print("""
deepfake_detection/
├── src/
│   ├── preprocessing/          # 1단계: 전처리
│   ├── data/                   # 데이터 로더
│   ├── feature_extraction/     # 2단계: 특징 추출
│   ├── classification/         # 3단계: 분류
│   ├── training/               # 학습 관련
│   └── utils/                  # 유틸리티
│
├── config/
│   ├── preprocessing_config.yaml
│   └── model_config.yaml
│
├── scripts/
│   ├── preprocess_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── test_feature_extraction.py
│
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리된 데이터
│
├── checkpoints/                # 모델 체크포인트
├── runs/                       # TensorBoard 로그
└── outputs/                    # 결과물
""")
    print("=" * 70)


def main():
    """메인 함수"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "프로젝트 초기화" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝\n")

    # 폴더 구조 생성
    create_project_structure()

    # .gitignore 생성
    create_gitignore()

    # 프로젝트 트리 출력
    print_project_tree()

    print("\n다음 단계:")
    print("1. 데이터 다운로드 및 배치")
    print("2. 전처리 실행: python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets all")
    print("3. 학습 시작: python scripts/train.py --config config/model_config.yaml")
    print()


if __name__ == "__main__":
    main()