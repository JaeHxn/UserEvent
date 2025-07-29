#!/bin/bash

# AI 상품 추천 시스템 Docker 실행 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Docker 설치 확인
check_docker() {
    log_info "Docker 설치 상태를 확인합니다..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되지 않았습니다. 먼저 Docker를 설치해주세요."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되지 않았습니다. 먼저 Docker Compose를 설치해주세요."
        exit 1
    fi
    
    log_success "Docker 및 Docker Compose가 설치되어 있습니다."
}

# 환경 설정 확인
check_environment() {
    log_info "환경 설정을 확인합니다..."
    
    if [ ! -f .env ]; then
        log_warning ".env 파일이 없습니다. env.example에서 복사합니다..."
        if [ -f env.example ]; then
            cp env.example .env
            log_warning ".env 파일이 생성되었습니다. API 키를 설정해주세요."
        else
            log_error "env.example 파일을 찾을 수 없습니다."
            exit 1
        fi
    fi
    
    # OpenAI API 키 확인
    if grep -q "your_openai_api_key_here" .env; then
        log_warning "OpenAI API 키가 설정되지 않았습니다. .env 파일을 편집해주세요."
    fi
    
    log_success "환경 설정 확인 완료"
}

# 디렉토리 생성
create_directories() {
    log_info "필요한 디렉토리를 생성합니다..."
    
    mkdir -p data
    mkdir -p lightgcn_data
    mkdir -p logs
    mkdir -p ssl
    
    log_success "디렉토리 생성 완료"
}

# Docker 이미지 빌드
build_images() {
    log_info "Docker 이미지를 빌드합니다..."
    
    docker-compose build --no-cache
    
    log_success "Docker 이미지 빌드 완료"
}

# 서비스 시작
start_services() {
    log_info "서비스를 시작합니다..."
    
    docker-compose up -d
    
    log_success "서비스 시작 완료"
}

# 헬스체크
health_check() {
    log_info "서비스 상태를 확인합니다..."
    
    sleep 10
    
    # 환경에 따른 서버 주소 설정
    SERVER_HOST=${SERVER_HOST:-localhost}
    SERVER_PORT=${SERVER_PORT:-7070}
    ADMIN_PORT=${ADMIN_PORT:-7071}
    
    # 메인 애플리케이션 헬스체크
    if curl -f http://${SERVER_HOST}:${SERVER_PORT}/ > /dev/null 2>&1; then
        log_success "메인 애플리케이션이 정상 동작 중입니다. (http://${SERVER_HOST}:${SERVER_PORT})"
    else
        log_warning "메인 애플리케이션에 접속할 수 없습니다. 로그를 확인해주세요."
    fi
    
    # 관리자 대시보드 헬스체크
    if curl -f http://${SERVER_HOST}:${ADMIN_PORT}/admin > /dev/null 2>&1; then
        log_success "관리자 대시보드가 정상 동작 중입니다. (http://${SERVER_HOST}:${ADMIN_PORT}/admin)"
    else
        log_warning "관리자 대시보드에 접속할 수 없습니다. 로그를 확인해주세요."
    fi
}

# 메인 함수
main() {
    case "${1:-start}" in
        "start")
            check_docker
            check_environment
            create_directories
            build_images
            start_services
            health_check
            ;;
        "stop")
            log_info "서비스를 중지합니다..."
            docker-compose down
            log_success "서비스 중지 완료"
            ;;
        "restart")
            log_info "서비스를 재시작합니다..."
            docker-compose down
            sleep 2
            docker-compose up -d
            health_check
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "status")
            docker-compose ps
            ;;
        "clean")
            log_info "Docker 리소스를 정리합니다..."
            docker-compose down -v
            docker system prune -f
            log_success "정리 완료"
            ;;
        "help"|"-h"|"--help")
            echo "사용법: $0 [명령어]"
            echo ""
            echo "명령어:"
            echo "  start   - 서비스 시작 (기본값)"
            echo "  stop    - 서비스 중지"
            echo "  restart - 서비스 재시작"
            echo "  logs    - 로그 확인"
            echo "  status  - 서비스 상태 확인"
            echo "  clean   - Docker 리소스 정리"
            echo "  help    - 도움말 표시"
            ;;
        *)
            log_error "알 수 없는 명령어: $1"
            echo "도움말을 보려면: $0 help"
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@" 