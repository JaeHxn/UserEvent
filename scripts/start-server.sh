#!/bin/bash

# 서버 프로덕션 환경 실행 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVER_IP=114.110.135.96

echo -e "${BLUE}🚀 서버 프로덕션 환경을 시작합니다...${NC}"

# 서버 환경 설정 파일 복사
if [ -f env.production ]; then
    cp env.production .env
    echo -e "${GREEN}✅ 서버 환경 설정이 적용되었습니다.${NC}"
else
    echo -e "${YELLOW}⚠️ env.production 파일이 없습니다. 기본 설정을 사용합니다.${NC}"
fi

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}❌ Docker가 설치되지 않았습니다. Docker를 먼저 설치해주세요.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}❌ Docker Compose가 설치되지 않았습니다. Docker Compose를 먼저 설치해주세요.${NC}"
    exit 1
fi

# 필요한 디렉토리 생성
mkdir -p data lightgcn_data logs ssl

# Docker 이미지 빌드
echo -e "${BLUE}🔨 Docker 이미지를 빌드합니다...${NC}"
docker-compose build --no-cache

# 서비스 시작
echo -e "${BLUE}🚀 Docker 서비스를 시작합니다...${NC}"
docker-compose up -d

# 헬스체크
echo -e "${BLUE}🔍 서비스 상태를 확인합니다...${NC}"
sleep 10

SERVER_HOST=0.0.0.0
SERVER_PORT=${SERVER_PORT:-7070}
ADMIN_PORT=${ADMIN_PORT:-7071}

if curl -f http://${SERVER_HOST}:${SERVER_PORT}/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 메인 애플리케이션이 정상 동작 중입니다.${NC}"
    echo -e "${GREEN}📍 외부 접속 주소: http://${SERVER_IP}:${SERVER_PORT}${NC}"
else
    echo -e "${YELLOW}⚠️ 메인 애플리케이션에 접속할 수 없습니다.${NC}"
fi

if curl -f http://${SERVER_HOST}:${ADMIN_PORT}/admin > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 관리자 대시보드가 정상 동작 중입니다.${NC}"
    echo -e "${GREEN}📍 외부 접속 주소: http://${SERVER_IP}:${ADMIN_PORT}/admin${NC}"
else
    echo -e "${YELLOW}⚠️ 관리자 대시보드에 접속할 수 없습니다.${NC}"
fi

echo -e "${BLUE}📊 로그 확인: docker-compose logs -f${NC}"
echo -e "${BLUE}⏹️  서비스 중지: docker-compose down${NC}" 