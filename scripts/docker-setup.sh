#!/bin/bash

# Docker 설정 스크립트
echo "🚀 Docker 환경 설정을 시작합니다..."

# .env 파일이 없으면 생성
if [ ! -f .env ]; then
    echo "📝 .env 파일을 생성합니다..."
    cp env.example .env
    echo "✅ .env 파일이 생성되었습니다. API 키를 설정해주세요."
else
    echo "✅ .env 파일이 이미 존재합니다."
fi

# 필요한 디렉토리 생성
echo "📁 필요한 디렉토리를 생성합니다..."
mkdir -p data
mkdir -p lightgcn_data
mkdir -p logs
mkdir -p ssl

# 권한 설정
echo "🔐 파일 권한을 설정합니다..."
chmod 755 scripts/*.sh 2>/dev/null || true

echo "✅ Docker 환경 설정이 완료되었습니다!"
echo ""
echo "다음 단계:"
echo "1. .env 파일에서 OPENAI_API_KEY를 설정하세요"
echo "2. docker-compose up -d 명령으로 서비스를 시작하세요" 