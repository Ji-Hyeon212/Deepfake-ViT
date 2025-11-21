Deepfake Detection Project (EfficientNet-B4 + Hybrid Attention)
본 프로젝트는 EfficientNet-B4를 백본으로 사용하고 랜드마크 기반 하이브리드 어텐션을 결합하여 이미지 및 동영상 속 딥페이크 여부를 판별하는 이진 분류 모델 학습 파이프라인을 구축합니다.

✨ 주요 특징 (Features)
하이브리드 특징 추출: EfficientNet-B4와 랜드마크 기반의 공간/채널 어텐션(HybridAttention)을 결합하여 얼굴의 미세한 아티팩트를 탐지합니다.

견고한 전처리: RetinaFaceDetector를 이용한 얼굴 검출 및 정렬, 그리고 블러/노이즈를 걸러내는 품질 검사(QualityChecker)가 파이프라인에 포함되어 있습니다.

고급 손실 함수: CrossEntropy, FocalLoss, ContrastiveLoss를 가중 결합한 CombinedLoss를 사용하여 불균형 데이터셋에서 학습 성능을 최적화합니다.

📁 프로젝트 폴더 구조
프로젝트는 모듈성과 확장성을 위해 계층적으로 구성되어 있습니다.
```
deepfake-detection/
├── 📁 config/
│   ├── preprocessing_config.yaml  # 전처리 설정 (RetinaFace, alignment)
│   └── model_config.yaml       # 학습 하이퍼파라미터 설정
│
├── 📁 data/
│   ├── 📁 raw/                 # 원본 데이터셋 (LFW-FER, DeeperForensics, GenAI, FaceForensics++ 등)
│   └── 📁 processed/           # 전처리된 얼굴 이미지, 랜드마크, 분할 정보 (train/val/test)
│
├── 📁 scripts/                 # 실행 가능한 스크립트 모음
│   ├── preprocess_dataset.py  # 데이터 전처리 및 분할 실행
│   ├── train.py               # 모델 학습 실행                 
│   └── evaluate.py            # 모델 테스트
│
├── 📁 src/                     # 핵심 소스 코드 (모듈)
│   ├── 📁 preprocessing/      # 얼굴 검출/정렬/품질 검사 엔진
│   ├── 📁 feature_extraction/ # EfficientNet 백본, Attention 모듈
│   ├── 📁 training/           # Trainer, Evaluator, Loss 함수
│   └── 📁 data/               # DataLoader, Dataset, 인터페이스
│
├── 📁 checkpoints/             # 학습된 모델 가중치 저장소 (best_model.pth)
├── 📄 requirements.txt         # 파이썬 라이브러리 목록
└── 📄 task.ipynb               # 대회 제출용 노트북 파일
```

🚀 실행 가이드 (Quick Start)
## 1. 개발 환경 설정
Python 3.9 환경 및 GPU/CUDA 환경을 준비합니다.

### 1. 프로젝트 폴더 초기화 (필수 아님, 폴더 구조만 생성)
```commandline
python scripts/setup_project.py
```

### 2. 파이썬 라이브러리 설치
```commandline
pip install -r requirements.txt
```
## 2. 데이터 준비
모든 원본 데이터셋(LFW-FER, DeeperForensics, GenAI 등)은 data/raw/ 폴더 내에 config/preprocessing_config.yaml 파일에 정의된 경로에 위치해야 합니다.

## 3. 데이터 전처리 및 분할
원본 데이터를 읽어 얼굴 검출, 정렬, 품질 검사를 수행하고, 최종 학습에 사용될 data/processed/ 폴더의 CSV 파일들을 생성합니다.

### 모든 데이터셋을 전처리하고 train/val/test 분할 파일을 생성
```commandline
python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets all
```

# (옵션) 특정 데이터셋(예: GenAI)만 전처리
```commandline
python scripts/preprocess_dataset.py --config config/preprocessing_config.yaml --datasets gen_ai
```

4. 모델 학습 시작
전처리된 데이터를 기반으로 모델 학습을 시작합니다.

# 학습을 시작하고, 최고 성능 모델은 checkpoints/best_model.pth에 저장됩니다.
```commandline
python scripts/train.py --config config/training_config.yaml
```
