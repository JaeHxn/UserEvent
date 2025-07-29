# AI 상품 추천 시스템

AI 기반 상품 추천 시스템으로, 사용자 검색어와 조건에 맞는 상품을 똑똑하게 추천해주는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **AI 기반 상품 추천**: OpenAI GPT와 Milvus 벡터 데이터베이스를 활용한 지능형 추천
- **실시간 검색**: 사용자 검색어와 가격 조건에 따른 실시간 상품 검색
- **사용자 이벤트 추적**: 상품 조회, 검색 이력 등 사용자 행동 분석
- **관리자 대시보드**: 사용자 통계, 인기 상품, 검색 이력 관리
- **새로고침 방지**: 페이지 새로고침 시 검색 결과 초기화

## 🛠️ 기술 스택

- **Backend**: Flask, Python 3.9
- **AI/ML**: OpenAI GPT-4, Milvus 벡터 데이터베이스
- **Database**: SQLite (사용자 이벤트)
- **Frontend**: HTML5, CSS3, JavaScript
- **Container**: Docker, Docker Compose
- **Proxy**: Nginx (선택사항)

## 📋 사전 요구사항

- Docker 및 Docker Compose 설치
- OpenAI API 키
- Milvus 서버 접근 권한

## 🐳 Docker로 실행하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd MCP

# Docker 환경 설정 스크립트 실행
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh
```

### 2. 환경별 실행

#### 로컬 개발 환경
```bash
# 로컬 환경 실행 스크립트
chmod +x scripts/start-local.sh
./scripts/start-local.sh
```

#### 서버 프로덕션 환경
```bash
# 서버 환경 실행 스크립트
chmod +x scripts/start-server.sh
./scripts/start-server.sh
```

### 3. 수동 환경 변수 설정

#### 로컬 환경 (.env 파일)
```bash
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# 서버 설정
SERVER_HOST=localhost
SERVER_PORT=7070
ADMIN_PORT=7071
ENVIRONMENT=development
```

#### 서버 환경 (.env 파일)
```bash
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# 서버 설정
SERVER_HOST=114.110.135.96
SERVER_PORT=7070
ADMIN_PORT=7071
ENVIRONMENT=production
```

### 4. Docker 컨테이너 실행

```bash
# 모든 서비스 시작 (백그라운드)
docker-compose up -d

# 로그 확인
docker-compose logs -f main-app
docker-compose logs -f admin-dashboard
```

## 🐳 서버 운영 원칙

- **내부 바인딩(host)**: 0.0.0.0 (Flask, FastAPI 등 모든 서버)
- **외부 접속 URL**: http://114.110.135.96:7070 (또는 80/443)

---

### 4. 접속 확인

#### 로컬 환경
- **메인 애플리케이션**: http://localhost:7070
- **관리자 대시보드**: http://localhost:7071/admin

#### 서버 환경 (외부에서 접속)
- **메인 애플리케이션**: http://114.110.135.96:7070
- **관리자 대시보드**: http://114.110.135.96:7071/admin

> 서버 내부 바인딩은 항상 0.0.0.0, 외부에서는 서버IP로 접속해야 합니다.

## 🔧 개발 환경에서 실행하기

### 1. Python 환경 설정

```bash
# Python 3.9 이상 설치 필요
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp env.example .env
# .env 파일 편집하여 API 키 설정
```

### 3. 애플리케이션 실행

```bash
# 메인 애플리케이션 실행
python app.py

# 관리자 대시보드 실행 (별도 터미널)
python admin_dashboard.py
```

## 📊 사용법

### 1. 상품 검색 및 추천

#### 로컬 환경
1. 웹 브라우저에서 http://localhost:7070 접속
2. 검색어 입력 (예: "방수 등산화 8사이즈")
3. 최소/최대 가격 설정 (선택사항)
4. "추천 검색" 버튼 클릭
5. AI가 추천한 상품 목록 확인

#### 서버 환경
1. 웹 브라우저에서 http://114.110.135.96:7070 접속
2. 검색어 입력 (예: "방수 등산화 8사이즈")
3. 최소/최대 가격 설정 (선택사항)
4. "추천 검색" 버튼 클릭
5. AI가 추천한 상품 목록 확인

### 2. 상품 상세보기

- 각 상품 카드의 "상세보기" 버튼 클릭
- 새 탭에서 상품 상세 정보 확인
- 사용자 조회 이벤트가 자동으로 기록됨

### 3. 관리자 대시보드

#### 로컬 환경
- http://localhost:7071/admin 접속
- 사용자 통계, 인기 상품, 검색 이력 확인
- 기간별 데이터 분석

#### 서버 환경
- http://114.110.135.96:7071/admin 접속
- 사용자 통계, 인기 상품, 검색 이력 확인
- 기간별 데이터 분석

## 🐳 Docker 명령어

```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down

# 서비스 재시작
docker-compose restart

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f main-app

# 컨테이너 상태 확인
docker-compose ps

# 이미지 재빌드
docker-compose build --no-cache

# 볼륨 정리
docker-compose down -v
```

## 📁 프로젝트 구조

```
MCP/
├── app.py                 # 메인 Flask 애플리케이션
├── admin_dashboard.py     # 관리자 대시보드
├── user_events.py         # 사용자 이벤트 관리
├── requirements.txt       # Python 의존성
├── Dockerfile            # Docker 이미지 설정
├── docker-compose.yml    # Docker Compose 설정
├── nginx.conf            # Nginx 설정
├── env.example           # 환경 변수 예시
├── templates/
│   ├── index.html        # 메인 페이지
│   └── admin_dashboard.html
├── scripts/
│   └── docker-setup.sh   # Docker 설정 스크립트
├── data/                 # 데이터 저장소
├── lightgcn_data/        # LightGCN 모델 데이터
└── logs/                 # 로그 파일
```

## 🔍 문제 해결

### 일반적인 문제들

1. **포트 충돌**
   ```bash
   # 사용 중인 포트 확인
   netstat -tulpn | grep :7070
   
   # 다른 포트 사용
   docker-compose up -d -p 8080:7070
   ```

2. **API 키 오류**
   ```bash
   # .env 파일 확인
   cat .env | grep OPENAI_API_KEY
   
   # 환경 변수 직접 설정
   export OPENAI_API_KEY=your_key_here
   ```

3. **Milvus 연결 오류**
   ```bash
   # Milvus 서버 상태 확인
   telnet 114.110.135.96 19530
   
   # 환경 변수 확인
   echo $MILVUS_HOST $MILVUS_PORT
   ```

### 로그 확인

```bash
# 실시간 로그 확인
docker-compose logs -f main-app

# 특정 시간 이후 로그
docker-compose logs --since="2024-01-01T00:00:00" main-app

# 에러 로그만 확인
docker-compose logs main-app | grep ERROR
```

## 🔒 보안 고려사항

- `.env` 파일에 민감한 정보가 포함되므로 `.gitignore`에 추가
- 프로덕션 환경에서는 HTTPS 사용 권장
- API 키는 안전하게 관리하고 정기적으로 갱신
- 방화벽 설정으로 불필요한 포트 차단

## 📈 성능 최적화

- Milvus 벡터 검색 최적화
- Redis 캐싱 도입 (선택사항)
- Nginx 리버스 프록시로 로드 밸런싱
- 데이터베이스 인덱싱 최적화

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요. 