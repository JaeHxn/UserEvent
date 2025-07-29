# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p /app/data
RUN mkdir -p /app/lightgcn_data
RUN mkdir -p /app/logs

# 권한 설정
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# 포트 노출
EXPOSE 7070 7071

# 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7070/ || exit 1

# 실행 명령
CMD ["python", "app.py"] 