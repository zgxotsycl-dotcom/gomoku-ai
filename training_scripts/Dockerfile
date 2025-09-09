# 1. 기본 환경 설정: NVIDIA가 제공하는 CUDA 11.8 및 cuDNN 8이 설치된 Ubuntu 22.04 환경을 사용합니다.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 환경 변수 설정 (설치 중 대화형 프롬프트 방지)
ENV DEBIAN_FRONTEND=noninteractive
# TensorFlow oneDNN 최적화 기능 끄기 (cuDNN 버그 우회)
ENV TF_ENABLE_ONEDNN_OPTS=0
# cuDNN 자동 튜닝 기능 끄기 (버그 우회)
ENV TF_CUDNN_USE_AUTOTUNE=0
# GPU 메모리 점진적 할당 (메모리 충돌 방지)
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# cuDNN 버그를 우회하기 위해 안정적인 알고리즘을 사용하도록 설정
ENV TF_CUDNN_DETERMINISTIC=1

# 2. 필수 패키지 및 Node.js 설치
RUN apt-get update && \
    apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 프로젝트 파일 복사 및 라이브러리 설치
# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 먼저 package.json과 package-lock.json을 복사합니다.
# (이렇게 하면, 코드만 변경되었을 때 npm install을 다시 실행하지 않아 빌드 속도가 향상됩니다.)
COPY package.json package-lock.json* ./

# npm install을 실행하여 모든 라이브러리를 설치합니다.
RUN npm install

# 나머지 모든 프로젝트 파일을 복사합니다.
COPY . .

# 4. TypeScript 컴파일
# tsc 명령어를 실행하여 .ts 파일을 .js 파일로 변환합니다.
RUN npm run build

# 서버가 8080 포트를 사용함을 명시합니다.
EXPOSE 8080

# 5. 컨테이너 실행 시 기본 명령어 설정
# 이 컨테이너가 실행될 때 기본적으로 전체 파이프라인 관리자 스크립트가 실행되도록 설정합니다.
CMD ["npm", "run", "start:pipeline"]
