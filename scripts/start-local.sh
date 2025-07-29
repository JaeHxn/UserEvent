#!/bin/bash

# 로컬 개발 환경 실행 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 로컬 개발 환경을 시작합니다...${NC}"

# 로컬 환경 설정 파일 복사
if [ -f env.local ]; then
    cp env.local .env
    echo -e "${GREEN}✅ 로컬 환경 설정이 적용되었습니다.${NC}"
else
    echo -e "${BLUE}⚠️ env.local 파일이 없습니다. 기본 설정을 사용합니다.${NC}"
fi

# 필요한 디렉토리 생성
mkdir -p data lightgcn_data logs

# Python 가상환경 확인
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Python 가상환경을 생성합니다...${NC}"
    python -m venv venv
fi

# 가상환경 활성화
echo -e "${BLUE}🔧 가상환경을 활성화합니다...${NC}"
source venv/bin/activate

# 의존성 설치
echo -e "${BLUE}📦 Python 의존성을 설치합니다...${NC}"
pip install -r requirements.txt

# 서버 시작
echo -e "${BLUE}🚀 서버를 시작합니다...${NC}"
echo -e "${GREEN}📍 메인 애플리케이션: http://localhost:7070${NC}"
echo -e "${GREEN}📍 관리자 대시보드: http://localhost:7071/admin${NC}"
echo -e "${BLUE}⏹️  중지하려면 Ctrl+C를 누르세요.${NC}"

# 메인 애플리케이션 실행
python app.py 