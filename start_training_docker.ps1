# Stop on any error
$ErrorActionPreference = "Stop"

# 1. 프로젝트 루트 디렉토리로 이동
Set-Location -Path $PSScriptRoot

# 2. 최신 코드로 Docker 이미지 빌드
echo "Building training pipeline image..."
docker build -t final-training-pipeline -f Dockerfile .

# 3. 기존 컨테이너가 있다면 중지 및 삭제
echo "Stopping and removing old container if it exists..."
docker stop gomoku-training-pipeline-container -t 0 2>$null | Out-Null
docker rm gomoku-training-pipeline-container 2>$null | Out-Null

# 4. 새 컨테이너 실행 (--rm 옵션 제거)
echo "Starting training pipeline in new Docker container..."
docker run --gpus all --name gomoku-training-pipeline-container final-training-pipeline
