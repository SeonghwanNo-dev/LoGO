# gcsudo docker build --network host -t my-experiment .
# gcsudo docker run --gpus all -it --network host -v $(pwd):/workspace my-experiment /bin/bash
# exit


FROM python:3.12.3

WORKDIR /workspace

# 1. 최상위 핵심 AI/딥러닝 프레임워크 설치
RUN pip install --no-cache-dir \
    torch \
    accelerate

# 2. 허깅페이스 생태계 및 대형 모델 학습/평가 패키지 설치
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    peft \
    trl \
    safetensors \
    lm_eval

# 3. 데이터 분석, 시각화 및 유틸리티 도구 설치
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    wandb

COPY . .