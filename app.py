import numpy as np
from langchain_openai import OpenAI, OpenAIEmbeddings
from pymilvus import Collection, connections
import os
from openai import OpenAI as OpenAIClient      # 공식 OpenAI 클라이언트
import json
import base64
import urllib
from concurrent.futures import ThreadPoolExecutor
import redis
import uvicorn

from langdetect import detect
from collections import deque
# langchain 의존성을 선택적으로 import
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

import time
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import secrets
from user_events import event_manager
from datetime import datetime
from dotenv import load_dotenv

import sqlite3
from lightgcn_data_prep import LightGCNDataPreprocessor


import hmac
import hashlib

from fastapi import  Cookie, Depends, status
from fastapi.responses import RedirectResponse,PlainTextResponse, Response,FileResponse
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    RedisChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
import re, unicodedata, difflib
from typing import Optional, Union, List, Dict, Any, Tuple, Iterable

#가격인식 임포트
from decimal import Decimal, InvalidOperation
from rapidfuzz import fuzz

import gc
import sys
from rank_categories import get_top_categories

import asyncio
import functools
import logging
import uuid


executor = ThreadPoolExecutor()

#시간 출력 관련
import builtins


# 환경변수 로드
load_dotenv()

###시간 출력 관련 start#####
##############################

original_print = builtins.print

def timed_print(*args, **kwargs):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    original_print(f"[{current_time}]", *args, **kwargs)

# built-in print 교체
builtins.print = timed_print

##############################
###시간 출력 관련 END#####



# ─── ENV & 유틸 (위쪽 공용 영역에 추가/교체) ─────────────────────────────

def _s(x):  # None 가드
    return x if isinstance(x, str) else ""

def _eq_cs(a, b):  # 비밀번호용(대소문자 구분 + 상수시간)
    a, b = _s(a), _s(b)
    return secrets.compare_digest(a, b)

def _eq_ci(a, b):  # 아이디용(대소문자 무시하고 싶을 때)
    return _s(a).casefold() == _s(b).casefold()


# ── 설정 ─────────────────────────────────────────────────────────
API_KEY    = os.getenv('OPENAI_API_KEY', '')

REDIS_URL = "redis://localhost:6379/0"                    # ← 환경변수에서 로드
COLLECTION = "ownerclan"            # Milvus 컬렉션 이름
MILVUS_HOST = os.getenv('MILVUS_HOST', '114.110.135.96')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
LLM_MODEL  = "gpt-4.1-mini-2025-04-14"
# LLM_MODEL  = "gpt-5-mini-2025-08-07"
EMB_MODEL  = "text-embedding-3-small"
EMB_MODEL_LARGE  = "text-embedding-3-large"

# 클라이언트 및 래퍼
client    = OpenAIClient(api_key=API_KEY)
llm       = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)    # ← embedder 정의 추가
embedder_large  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL_LARGE)    # ← embedder 정의 추가
API_URL = "https://fb-narosu.duckdns.org"  # 예: http://114.110.135.96:8011

#사용자 이벤트 관리자 대시보드
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE_SECONDS", "10800")) # 3시간
ADMIN_COOKIE_NAME = os.getenv("ADMIN_COOKIE_NAME", "admin_session")


# 🆕 세션 자동 만료 설정
SESSION_TIMEOUT_MINUTES = 2 # 2분 후 자동 초기화시간 (테스트용)
SESSION_WARNING_MINUTES = 1  # 1분 후 경고 메시지

TIMEOUT_SECONDS = int(SESSION_TIMEOUT_MINUTES * 60)  # 정수로 변환
WARNING_SECONDS = SESSION_WARNING_MINUTES * 60  # 경고를 쓸 경우

HOMEPAGE_URL = "https://www.chatmall.kr/"  # 홈페이지 이동 응답용
DELIVERY_INQUIRY_URL = "https://www.chatmall.kr/shop/orderinquiry.php"  # 배송/주문 조회 안내









# 1) Milvus 서버에 먼저 연결
connections.connect(
    alias="default",
    host=MILVUS_HOST,    # 예: "114.110.135.96"
    port=MILVUS_PORT     # 예: "19530"
)
print("✅ Milvus에 연결되었습니다.")

# cat_col = Collection("category_embed_onerclean_l2_ivf")
cat_col = Collection("ownerclan_category_Large")
results = cat_col.query(
    expr="category_full != ''",
    output_fields=["category_full"]
)

# ── 중복 제거하며 순서 보존해서 리스트 만들기 ─────────
seen = set()
categories = []
for row in results:
    cat = row["category_full"]
    if cat and cat not in seen:
        seen.add(cat)
        categories.append(cat)

print(f"✅ Milvus에서 불러온 카테고리 개수: {len(categories)}")


app = FastAPI(title="AI 상품 추천 시스템", version="1.0.0")
templates = Environment(loader=FileSystemLoader("templates"))


# 설정: 2분(120초)에 최대 20회 (원하면 환경변수로 바꿔도 됨)
RATE_WINDOW_SECONDS = 120
RATE_MAX_REQUESTS   = 20

SOFT_BAN_SECONDS    = 5 * 60       # 5분
HARD_BAN_SECONDS    = 2 * 60 * 60  # 2시간
BAN_HARASS_THRESHOLD = 10           # 밴 중 n회 이상 두드리면 하드 밴


SAFE_PREFIXES = (
    "/robots.txt", "/favicon.ico",
    "/docs", "/openapi.json", "/redoc",
    "/static", "/assets", "/health", "/status"
)
WHITELIST_IPS = {'127.0.0.1', '::1', 'localhost'}  # 로컬호스트 등 예외 IP 목록

# ===== 상태 저장 =====
_ip_hits       = {}       # dict[str, deque[float]]: 평시 슬라이딩 창
_ip_ban_until  = {}       # dict[str, float]        : 밴 해제 시각(epoch)
_ip_ban_tier   = {}       # dict[str, str]          : "soft" | "hard"
_ip_ban_hits   = {}       # dict[str, deque[float]] : 밴 중 반복 타격 카운트

# ===== 메시지 템플릿 (상황별 여러 개 → 랜덤 선택) =====
MESSAGES = {
    "soft_ban_start": "You're sending requests too quickly. Please try again in {minutes} minute(s).",
    "soft_ban_still": "Temporary limit is active. Please try again in {remain} second(s).",
    "hard_ban_start": "Repeated requests detected. Access is restricted for {hours} hour(s).",
    "hard_ban_still": "Protection mode is active. Please try again in {remain} second(s).",
    "rate_limit_hit": "Rate limit reached. Please try again shortly."
}


async def run_sync_llm(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        functools.partial(func, *args, **kwargs)
    )

def _client_ip(req: Request) -> str:
    # 프록시 없는 구조: X-Forwarded-For 신뢰 금지
    return req.client.host or "unknown"

def _starts_with_any(path: str, prefixes: tuple) -> bool:
    return any(path.startswith(p) for p in prefixes)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path

    # 예외 경로/화이트리스트 우선 통과
    if _starts_with_any(path, SAFE_PREFIXES):
        return await call_next(request)

    ip = _client_ip(request)
    if ip in WHITELIST_IPS:
        return await call_next(request)

    now = time.time()

    # ===== 1) 밴 집행 & 에스컬레이션 =====
    ban_until = _ip_ban_until.get(ip, 0.0)
    if ban_until and now < ban_until:
        # 밴 중 추가 요청 기록
        dq = _ip_ban_hits.setdefault(ip, deque())
        dq.append(now)

        # 최근 SOFT_BAN_SECONDS 내 기록만 유지
        cutoff_ban = now - SOFT_BAN_SECONDS
        while dq and dq[0] < cutoff_ban:
            dq.popleft()

        # 소프트 밴 상태에서 반복 타격 → 하드 밴 승격
        if _ip_ban_tier.get(ip) == "soft" and len(dq) >= BAN_HARASS_THRESHOLD:
            _ip_ban_until[ip] = now + HARD_BAN_SECONDS
            _ip_ban_tier[ip]  = "hard"
            _ip_ban_hits[ip].clear()
            return PlainTextResponse(
                MESSAGES["hard_ban_start"].format(hours=int(HARD_BAN_SECONDS / 3600)),
                status_code=429,
                headers={"Retry-After": str(HARD_BAN_SECONDS)}
            )

        # 여전히 밴 중(소프트/하드 공통): 남은 시간 안내
        remain = max(1, int(ban_until - now))
        key = "hard_ban_still" if _ip_ban_tier.get(ip) == "hard" else "soft_ban_still"
        return PlainTextResponse(
            MESSAGES[key].format(remain=remain),
            status_code=429,
            headers={"Retry-After": str(remain)}
        )

    # 만료된 밴 정리
    if ban_until and now >= ban_until:
        _ip_ban_until.pop(ip, None)
        _ip_ban_tier.pop(ip, None)
        _ip_ban_hits.pop(ip, None)

    # ===== 2) 평시 카운트 =====
    q = _ip_hits.setdefault(ip, deque())
    cutoff = now - RATE_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

    # ===== 3) 임계 초과 → 소프트 밴 발동 =====
    if len(q) >= RATE_MAX_REQUESTS:
        _ip_ban_until[ip] = now + SOFT_BAN_SECONDS
        _ip_ban_tier[ip]  = "soft"
        _ip_ban_hits[ip]  = deque([now])
        return PlainTextResponse(
            MESSAGES["soft_ban_start"].format(minutes=int(SOFT_BAN_SECONDS / 60)),
            status_code=429,
            headers={"Retry-After": str(SOFT_BAN_SECONDS)}
        )

    # 허용 → 기록 + 통과
    q.append(now)
    resp = await call_next(request)

    # 429 표준화(다른 경로에서 429가 나와도)
    if resp.status_code == 429 and "Retry-After" not in resp.headers:
        resp.headers["Retry-After"] = "60"
    return resp












class ProductViewData(BaseModel):
    product_code: str
    product_data: dict

class SearchData(BaseModel):
    query: str

# 누적 쿼리에서 [clarify], 선택지(A/B/C/D), 숫자, 공백 등 불필요한 토큰 제거
def clean_accumulated_query(parts: list) -> str:
    """
    누적 쿼리에서 [clarify], 선택지(A/B/C/D/1/2/3/4), 공백 등 불필요한 토큰 제거
    """
    cleaned = []
    for p in parts:
        s = p.strip()
        # [clarify] 태그 제거
        s = re.sub(r'^\[clarify\]\s*', '', s)
        # 단일 선택지(A/B/C/D/1/2/3/4)만 남은 경우 제거
        if s in {'A', 'B', 'C', 'D', '1', '2', '3', '4','a','b','c','d'}:
            continue
        # 완전히 비어있으면 제외
        if not s:
            continue
        cleaned.append(s)
    return " ".join(cleaned)

# 메인 질의 + 재질문 답변 분리 헬퍼
def extract_main_and_clarifies(session_history, current_query: str):
    """
    세션에서 '메인 질의' 1개와 '재질문에 대한 짧은 답변들'만 분리해서 가져온다.
    - main_query: 처음에 사용자가 길게 적은 메인 검색 질의
    - clarify_parts: 이후 재질문에 대한 짧은 답변들(A/B/C/D 단답 제외)
    """
    messages = getattr(session_history, "messages", []) or []
    user_messages = [
        m for m in messages
        if getattr(m, "type", "") == "human"
    ]

    main_query = None
    clarify_parts = []

    # 1) 첫 번째 사용자 메시지를 메인 질의로 고정
    if user_messages:
        first = (getattr(user_messages[0], "content", "") or "").strip()
        if first:
            main_query = first

    # 2) 두 번째 이후 human 메시지는 대부분 재질문 답변이므로 짧은 것만 clarify로 모은다
    for m in user_messages[1:]:
        text = (getattr(m, "content", "") or "").strip()
        if not text:
            continue

        upper = text.upper()

        # A/B/C/D만 있는 단답은 굳이 붙일 필요 없음 → 스킵
        if upper in ("A", "B", "C", "D"):
            continue

        # 너무 긴 문장은 새 메인 질의일 가능성이 높으니 여기선 제외
        if len(text) > 40:
            continue

        clarify_parts.append(text)

    # 3) 세션이 거의 없으면, 이번 질의를 메인 질의로 사용
    if not main_query:
        main_query = current_query

    return main_query, clarify_parts




#사용자 문장에서 부정적인 단어 처리를 위한 코드
def strip_minus_terms(q: str) -> str:
    # -뒤에 따옴표 구/일반 단어 제거, - 뒤에 띄어쓰기 있어도 처리
    cleaned = re.sub(r'(?<!\S)-\s*(?:"[^"]+"|“[^”]+”|\'[^\']+\'|[^\s,;|]+)', '', q)
    return re.sub(r'\s{2,}', ' ', cleaned).strip()

#이벤트 관련 코드
def _sign(msg: str) -> str:
    return hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256).hexdigest()

def make_session_token(username: str) -> str:
    ts = str(int(time.time()))
    msg = f"{username}:{ts}"
    sig = _sign(msg)
    return f"{username}:{ts}:{sig}"

def verify_session_token(token: str) -> bool:
    try:
        username, ts, sig = token.split(":", 2)
        msg = f"{username}:{ts}"
        if not hmac.compare_digest(_sign(msg), sig):
            return False
        if time.time() - int(ts) > SESSION_MAX_AGE:
            return False
        return username == ADMIN_USERNAME
    except Exception:
        return False

# ─────────────────────
# 2) 관리자 보호 의존성
# ─────────────────────

def require_admin(admin_session: Optional[str] = Cookie(default=None, alias=ADMIN_COOKIE_NAME)):
    if not admin_session or not verify_session_token(admin_session):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="관리자 인증 필요")

# ─────────────────────
# 3) FastAPI 앱/예외 핸들러
# ─────────────────────

@app.exception_handler(HTTPException)
async def http_exc_redirect_login(request: Request, exc: HTTPException):
    # 401이면 로그인 화면으로 보내 사용자 경험 개선
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return RedirectResponse(url="/login")
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

# ─────────────────────
# 4) 공개 페이지
# ─────────────────────

@app.get('/', response_class=HTMLResponse)
async def index_get(request: Request):
    """메인 페이지 GET 요청 - 챗봇 시스템"""
    # 매번 새로운 세션 생성 (개발용)
    session_id = secrets.token_hex(16)
    
    template = templates.get_template("index.html")
    
    # 템플릿 렌더링 후 응답 객체 생성
    response = HTMLResponse(
        template.render(
            request=request, 
            error=None, 
            results=None
        )
    )
    
    # 새 세션 쿠키 설정
    response.set_cookie(
        key='session_id',
        value=session_id,
        httponly=True  # 개발용이므로 만료시간 설정 안함
    )
    
    return response

# ─────────────────────
# 5) 로그인/로그아웃
# ─────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_get():
    return HTMLResponse(
        """
        <html><body>
          <h2>관리자 로그인</h2>
          <form method="post">
            <label>Username</label><br/>
            <input name="username"/><br/>
            <label>Password</label><br/>
            <input name="password" type="password"/><br/><br/>
            <button type="submit">로그인</button>
          </form>
        </body></html>
        """
    )

@app.post("/login")
async def login_post(username: str = Form(...), password: str = Form(...)):
    # 혹시 위의 부팅 검증이 비활성화된 환경 대비
    if not isinstance(ADMIN_USERNAME, str) or not isinstance(ADMIN_PASSWORD, str):
        return HTMLResponse("<h3>서버 설정 오류: 관리자 자격 미설정</h3>", status_code=500)

    # 아이디: 대소문자 무시를 원하면 _eq_ci, 정확히 일치 원하면 _eq_cs로 바꾸세요.
    ok_user = _eq_ci(username, ADMIN_USERNAME)   # ← 필요시 _eq_cs로 교체
    # 비밀번호: 절대 변형하지 말고 상수시간 비교
    ok_pass = _eq_cs(password, ADMIN_PASSWORD)

    if ok_user and ok_pass:
        token = make_session_token(username)
        resp = RedirectResponse(url="/admin", status_code=302)
        resp.set_cookie(
            key=ADMIN_COOKIE_NAME or "admin_session",
            value=token,
            httponly=True,
            secure=True,   # HTTPS면 True 유지
            samesite="lax",
            max_age=SESSION_MAX_AGE,
        )
        return resp

    return HTMLResponse("<h3>로그인 실패</h3><a href='/login'>다시 시도</a>", status_code=401)

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(ADMIN_COOKIE_NAME)
    return resp


##보안측면 추가 
###################
@app.get("/_whoami")
async def whoami(req: Request):
    return {
        "client_host": req.client.host,
        "x_forwarded_for": req.headers.get("x-forwarded-for"),
        "user_agent": req.headers.get("user-agent"),
    }











# 요청 모델
class QueryRequest(BaseModel):
    query: str

class Ranker_DirectSearch:
    _SEP   = re.compile(r'[ \t\u00A0\-\_/\\\|\+\·•,.:;()\[\]{}<>~"“”‘’]+')
    _CHUNK = re.compile(r'[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?')
    _NUM   = re.compile(r'^\d+(?:\.\d+)?$')

    def __init__(self, near_threshold: float = 0.90) -> None:
        self.near_threshold = float(near_threshold)
        self.W_EXACT, self.W_NEAR, self.W_PART, self.W_MISS = 1.0, 0.8, 0.5, 0.0

    # ---------- 핵심: 정제된 쿼리 꺼내기 ----------
    def prepare_query(self, user_text: str, min_token_len: int = 1) -> Dict[str, Any]:
        """
        사용자 원문을 정규화/토큰화해서 '정제 쿼리'를 생성.
        반환: {
          "original": 원문,
          "normalized": 정규화 문자열,
          "tokens": 토큰 리스트,
          "canonical_query": "토큰을 공백으로 합친 문자열"
        }
        """
        norm = self._normalize(user_text)
        toks = [t for t in self._tokenize(user_text) if len(t) >= min_token_len]
        canonical = " ".join(toks)
        return {
            "original": user_text,
            "normalized": norm,
            "tokens": toks,
            "canonical_query": canonical
        }

    # ---------- 점수 API ----------
    def score_text(self, user_text: str, item_text: str) -> int:
        u_tokens = self._tokenize(user_text)  # 매칭용(중복 제거)
        i_tokens = self._tokenize(item_text)
        if not u_tokens:
            return 0
        
        # [수정] 단일 단어 검색일 경우 정확히 구분된 단어만 인정
        if len(u_tokens) == 1:
            search_word = u_tokens[0]
            # 단어 앞뒤로 공백이나 문장 경계가 있는지 확인
            normalized_text = self._normalize(item_text)
            words = normalized_text.split()
            if search_word in words:  # 정확히 독립된 단어로 존재하는지 확인
                return 1000
            return 0
            
        # 여러 단어 검색일 경우 기존 연속 가중치 매핑 적용
        base_frac = sum(self._best_weight(kw, i_tokens) for kw in u_tokens) / float(len(u_tokens))
        score = 1000.0 * min(1.0, base_frac)

        # 길이 페널티 적용
        u_raw = self._tokenize_raw(user_text)
        i_raw = self._tokenize_raw(item_text)
        len_factor = self._relative_length_factor(u_raw, i_raw)
        score *= len_factor

        return int(round(max(0.0, min(1000.0, score))))

    def score_items(
        self,
        user_text: str,
        items: List[Union[str, Dict[str, Any]]],
        fields: Optional[Iterable[str]] = None,
        return_query_meta: bool = False
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        qmeta = self.prepare_query(user_text)
        for it in items:
            text = self._build_item_text(it, fields)
            s = self.score_text(user_text, text)
            row = {"item": it, "direct_text_score": s, "text_used": text}
            if return_query_meta:
                row["query_meta"] = qmeta  # 원문/정규화/토큰/정제쿼리 함께 반환
            out.append(row)
        out.sort(key=lambda x: x["direct_text_score"], reverse=True)
        return out

    # ---------- 내부: 정규화/토큰화 ----------
    def _normalize(self, txt: str) -> str:
        t = unicodedata.normalize('NFKC', txt)
        return ''.join(ch.lower() if 'A' <= ch <= 'Z' else ch for ch in t)

    def _is_number(self, s: str) -> bool:
        return bool(self._NUM.match(s))

    def _split_mixed(self, tok: str) -> List[str]:
        return self._CHUNK.findall(tok)

    def _num_alpha_variants(self, pieces: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(pieces):
            cur = pieces[i]; out.append(cur)
            if i + 1 < len(pieces):
                nxt = pieces[i + 1]
                if (self._is_number(cur) and not self._is_number(nxt)) or \
                   (self._is_number(nxt) and not self._is_number(cur)):
                    out.append(cur + nxt)  # "10","g" -> "10g"
            i += 1
        return out

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        raw = [p for p in self._SEP.split(text) if p]
        pieces: List[str] = []
        for p in raw:
            pieces.extend(self._num_alpha_variants(self._split_mixed(p)))
        seen = set(); ordered: List[str] = []
        for t in pieces:
            if t and t not in seen:
                seen.add(t); ordered.append(t)
        return ordered

    # ---------- 내부: 매칭 ----------
    def _similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    ######### [변경] 계단식 → 연속 가중치 매핑
    def _best_weight(self, kw: str, item_tokens: List[str]) -> float:
        if not item_tokens:
            return self.W_MISS
        if any(kw == t for t in item_tokens):
            return self.W_EXACT

        has_partial = False
        best_sim = 0.0
        for t in item_tokens:
            if (kw in t) or (t in kw):
                has_partial = True
            s = self._similarity(kw, t)
            if s > best_sim:
                best_sim = s

        s = best_sim
        s_low = 0.30  # 부분 매칭 하한
        if s >= self.near_threshold:
            # near(0.8)에서 시작해 s=1.0에선 약 0.95까지 부드럽게
            return self.W_NEAR + 0.15 * (s - self.near_threshold) / (1.0 - self.near_threshold)
        if s <= s_low:
            return 0.10 if has_partial else self.W_MISS
        # s_low ~ near_threshold 구간: 곡선 증가(감마=1.2)
        a = ((s - s_low) / (self.near_threshold - s_low)) ** 1.2
        base = 0.20 if has_partial else 0.10
        return base + (self.W_PART - base) * a

    # ---------- 내부: 아이템 텍스트 구성(스키마 자유) ----------
    def _build_item_text(
        self,
        item: Union[str, Dict[str, Any]],
        fields: Optional[Iterable[str]]
    ) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if fields is not None:
                parts = [str(item[k]) for k in fields if (k in item and item[k])]
                return " ".join(parts)
            for k in ("제목", "title"):
                if k in item and item[k]:
                    return str(item[k])
            parts = [v for v in item.values() if isinstance(v, str) and v]
            return " ".join(parts)
        return str(item)
    
    # --- (Ranker_DirectSearch 내부) 상대 길이 페널티 ---
    def _relative_length_factor(
        self,
        u_tokens: list,
        i_tokens: list,
        r0: float = 2.0,          # 기준 비율(이하이면 페널티 없음)  아이템 길이 ÷ 쿼리 길이  쿼리길이보다 2.0배 이상이면 패널티 있다는거지.
        gamma: float = 0.9,       # 페널티 기울기(1.0이면 비례, 0.7~1.2에서 튜닝)
        min_factor: float = 0.85,  # 최저 배수(과도 감점 방지)
        query_floor: int = 3      # 너무 짧은 쿼리 보호용 바닥
    ) -> float:
        U = max(len(u_tokens), query_floor)
        I = max(len(i_tokens), 1)
        r = I / float(U)
        if r <= r0:
            return 1.0
        # (r0 / r)^gamma, 단 min_factor 미만으로는 깎지 않음
        factor = (r0 / r) ** gamma
        return max(min_factor, factor)

    def _tokenize_raw(self, text: str) -> List[str]:
        """중복 제거 없이 토큰을 그대로 반환 (길이/반복 페널티 계산용)"""
        text = self._normalize(text)
        raw = [p for p in self._SEP.split(text) if p]
        pieces: List[str] = []
        for p in raw:
            pieces.extend(self._num_alpha_variants(self._split_mixed(p)))
        return [t for t in pieces if t]



def external_search_and_generate_response(request: Union[QueryRequest, str], session_id: str = None) -> dict:
    # 🔧 재질문 임계값 통일 설정
    THRESHOLD = 0.59             # 평균 점수 임계값 (이상이면 검색 진행)
    DIRECT_MATCH_HIGH = 0.80      # 직접 매칭 높은 신뢰도 (단독 통과 가능)
    FACET_COVERAGE_MIN = 0.50     # 최소 속성 커버리지 (미달 시 재질문)
    ATTRIBUTE_MIN = 0.50         # 속성 매칭 최소값
    FACET_SUFFICIENT = 0.50       # 충분한 속성 커버리지
    gc.collect()


    def check_session_timeout(session_history, session_id: str) -> dict:
        """
        세션의 마지막 활동 시간을 체크하여 자동 만료 처리.
        만료(=TTL==0)일 때만 초기화하고, 그 외에는 TTL만 설정/연장.
        만료 시에는 즉시 상위로 반환하여 이후 로직(검색 등)을 진행하지 않도록 한다.
        """
        try:
            r = redis.from_url(REDIS_URL)
            session_key = f"message_store:{session_id}"
            ttl = r.ttl(session_key)  # 초 단위
            print(f"[세션체크] 세션 {session_id} TTL={ttl}초 (기준={TIMEOUT_SECONDS}초={SESSION_TIMEOUT_MINUTES}분)")

            # 1) 키 없음(-2) → 새 세션: TTL만 설정하고 진행
            if ttl == -2:   
                print("[세션체크] 새 세션 - TTL 설정")
                r.expire(session_key, TIMEOUT_SECONDS)
                return None

            # 2) 만료시간 미설정(-1) → TTL 설정
            if ttl == -1:
                print("[세션체크] TTL 미설정 - TTL 설정")
                r.expire(session_key, TIMEOUT_SECONDS)
                return None

            # 3) 만료(0) → 초기화 후 안내 반환 (이 요청에서는 즉시 종료)
            #    경계 흔들림이 걱정되면 ttl <= 1 로 완화 가능
            if ttl == 0:
                print(f"[세션만료 감지] TTL=0, 세션 {session_id} 자동 초기화 시작")

                # 기존 메시지 삭제
                clear_message_history(session_id)

                # 안내 메시지
                msg = (
                    f"💫 It was automatically reset after there was no response for a certain period of time since the last conversation. Start searching for a new product! Enter a search term 😊\n\n"
                )

                # 새 세션에 안내 메시지 저장 + TTL 설정
                try:
                    new_session_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
                    new_session_history.add_ai_message(msg)
                    r.expire(session_key, TIMEOUT_SECONDS)
                    print("[세션만료] 새 세션 생성 및 초기화 메시지 저장 완료")
                except Exception as e:
                    print(f"[세션만료 안내 기록 오류] {e}")

                print("[세션만료] 응답 반환: session_expired=True")
                return {
                    "query": "auto_reset",
                    "assistant_message": msg,
                    "UserMessage": msg,
                    "RawContext": [],
                    "results": [],
                    "combined_message_text": msg,
                    "needs_clarification": False,
                    "auto_reset": True,
                    "session_expired": True,
                }

            # 4) 정상 세션(ttl > 0) → TTL 연장
            print("[세션체크] 정상 세션 - TTL 연장")
            r.expire(session_key, TIMEOUT_SECONDS)
            return None

        except Exception as e:
            print(f"[세션체크 오류] {e}")
            return None




    

    def calculate_completion_rate(avg_score: float, threshold: float = THRESHOLD) -> tuple:
        """
        평균 점수를 기준으로 질문 완성률을 계산
        THRESHOLD(0.80)을 100%로 기준하여 완성률 산출
        
        Args:
            avg_score: LLM이 준 평균 점수 (0.0~1.0)
            threshold: 완성 기준 점수 (기본값: 0.80)
            
        Returns:
            tuple: (완성률 비율 0.0~1.0, 완성률 퍼센트 0~100)
            
        예시:
            - avg_score=0.80 → 100% 완성
            - avg_score=0.64 → 80% 완성  
            - avg_score=0.40 → 50% 완성
            - avg_score=0.00 → 0% 완성
        """
        if avg_score >= threshold:
            return 1.0, 100  # 100% 완성 (검색 진행)ㅌ
        
        # 0.80을 100%로 기준하여 비례 계산
        completion_ratio = min(1.0, max(0.0, avg_score / threshold))
        completion_percent = int(round(completion_ratio * 100))
        
        return completion_ratio, completion_percent
    
    def is_delivery_intent(text: str) -> bool:
        """
        배송/주문 조회 의도를 LLM 없이 패턴 기반으로 감지
        - 의미론적 패턴: "어디", "언제", "상태", "확인" 등 + 주문/배송 관련
        - 영어 표현 대폭 확장
        """
        if not text:
            return False
            
        t = text.lower()
        
        # 🎯 핵심 개념 기반 패턴 (영어 대폭 확장)
        delivery_concepts = {
            # 추적/조회 개념 (15개 → 30개)
            "tracking": [
                # 한국어
                "track", "추적", "조회", "확인",
                # 영어 - 추적 관련
                "track", "tracking", "trace", "tracing", "follow", "following",
                "locate", "locating", "find", "finding", "check", "checking",
                "monitor", "monitoring", "view", "viewing", "see", "seeing",
                "watch", "watching", "observe", "observing", "inspect", "inspecting",
                "verify", "verifying", "confirm", "confirming", "review", "reviewing"
            ],
            
            # 위치/장소 개념 (8개 → 20개)
            "location": [
                # 한국어
                "어디", "위치", "장소",
                # 영어 - 위치 질문
                "where", "where is", "where's", "location", "place", "position",
                "whereabouts", "spot", "site", "area", "zone",
                "current location", "present location", "at what place",
                "in what location", "what place", "which place"
            ],
            
            # 상태 확인 개념 (12개 → 35개)
            "status": [
                # 한국어
                "상태", "확인", "현황", "진행", "진행상황",
                # 영어 - 상태 관련
                "status", "state", "condition", "situation", "standing",
                "update", "updates", "progress", "stage", "phase",
                "check on", "look up", "look into", "verify", "confirm",
                "see", "view", "check status", "get status", "know status",
                "information", "info", "details", "data", "record",
                "current status", "latest status", "order status"
            ],
            
            # 도착/배송 시간 개념 (15개 → 40개)
            "arrival": [
                # 한국어
                "when", "언제", "arrive", "도착", "배송일", "받을", "도착예정",
                # 영어 - 시간 관련
                "when", "when will", "when does", "when can", "what time",
                "arrive", "arrival", "arriving", "reached", "delivered",
                "get here", "get to me", "come", "coming", "receive", "receiving",
                "expect", "expected", "expecting", "anticipate", "anticipated",
                "delivery date", "delivery time", "arrival date", "arrival time",
                "eta", "estimated", "estimate", "how long", "how soon", "how many days",
                "ship date", "shipping date", "dispatch date", "sent date"
            ],
            
            # 주문 개념 (8개 → 25개) - 🔥 order 추가!
            "order": [
                # 한국어
                "order", "주문", "구매", "산거", "주문한", "구매한",
                # 영어 - 주문 관련 (문맥 필수!)
                "my order", "the order", "this order", "that order", "order status",
                "order number", "order id", "order code", "order reference",
                "purchase", "purchased", "bought", "ordered", "placed order",
                "order history", "recent order", "last order", "latest order",
                "order info", "order information", "order details"
            ],
            
            # 배송/택배 개념 (15개 → 45개)
            "package": [
                # 한국어
                "package", "배송", "delivery", "shipping", "택배", "소포", "화물",
                # 영어 - 배송 관련
                "delivery", "deliveries", "deliver", "delivering",
                "shipping", "shipment", "shipped", "ship", "shipper",
                "package", "packages", "parcel", "parcels", "item", "items",
                "mail", "post", "postal", "courier", "carrier", "freight",
                "express", "express delivery", "fast delivery", "quick delivery",
                "dispatch", "dispatched", "dispatching", "sent", "send", "sending",
                "consignment", "goods", "cargo", "product delivery","Order Inquiry","OrderInquiry"
            ]
        }
        
        # 패턴 매칭: 각 개념별로 키워드 체크
        matched_concepts = []
        for concept, keywords in delivery_concepts.items():
            if any(kw in t for kw in keywords):
                matched_concepts.append(concept)
        
        print(f"[배송의도감지] 입력: '{text}' → 매칭된 개념: {matched_concepts}")
        
        # 🔥 의도 판별 로직 (개념 조합으로 판단)
        # 1) "추적/조회" + "주문/배송" → 배송조회
        if "tracking" in matched_concepts and ("order" in matched_concepts or "package" in matched_concepts):
            print(f"[배송의도감지] ✅ 패턴1: 추적+주문/배송")
            return True
            
        # 2) "위치/상태" + "배송/주문" → 배송조회  
        if ("location" in matched_concepts or "status" in matched_concepts) and \
           ("package" in matched_concepts or "order" in matched_concepts):
            print(f"[배송의도감지] ✅ 패턴2: 위치/상태+배송/주문")
            return True
            
        # 3) "언제 도착" + "주문/배송" → 배송조회
        if "arrival" in matched_concepts and ("package" in matched_concepts or "order" in matched_concepts):
            print(f"[배송의도감지] ✅ 패턴3: 도착시간+주문/배송")
            return True
        
        # 4) "추적/조회" + "위치/상태" → 배송조회 (배송 단어 없어도)
        if "tracking" in matched_concepts and ("location" in matched_concepts or "status" in matched_concepts):
            print(f"[배송의도감지] ✅ 패턴4: 추적+위치/상태")
            return True
        
        print(f"[배송의도감지] ❌ 패턴 미일치")
        return False


    def is_homepage_intent(text: str) -> bool:
        """
        사용자가 홈페이지로 이동/주소 요청 의도인지 감지.
        공백 제거 패턴도 함께 체크.
        """
        t = (text or "").lower()
        t_nospace = re.sub(r"\s+", "", t)
        keywords = [
            "홈페이지", "홈페이지주소", "홈페이지링크", "홈페이지로이동", "홈페이지접속",
            "사이트주소", "사이트링크", "사이트로이동", "사이트접속",
            "website", "home page", "homepage","web URL","website address","go to website","visit website"
        ]
        for kw in keywords:
            if kw in t or kw in t_nospace:
                return True
        return False

    def is_greeting_or_intro_intent(text: str) -> bool:
        """
        인사 또는 챗봇 소개 요청 의도를 감지한다.
        - 한국어/영어 인사말, "뭐 하는지", "누구세요", "기능" 등
        """
        if not text:
            return False
            
        t = text.lower().strip()
        t_nospace = re.sub(r"\s+", "", t)
        
        # 인사/소개 키워드
        keywords = [
            # 한국어 인사
            "안녕", "안녕하세요", "안녕하십니까", "반갑습니다", "반가워요",
            "하이", "헬로","good morning", "good afternoon",
            
            # 챗봇 소개 요청
            "뭐하는", "뭐 하는", "무엇을하는", "무슨기능", "기능이뭐", "뭘도와주는",
            "누구", "누구세요", "누구야", "정체가뭐", "어떤챗봇",
            "what do you do", "what can you do", "who are you", "what are you",
            "introduce yourself", "tell me about yourself", "your function",
            "how can you help", "what is your purpose","What can I do?"
        ]
        
        for kw in keywords:
            kw_clean = re.sub(r"\s+", "", kw.lower())
            if kw_clean in t_nospace or kw.lower() in t:
                return True
        
        # 추가 패턴: 짧은 인사말 (3글자 이하 한글)
        if len(t) <= 5 and re.match(r'^[ㄱ-ㅎ가-힣]+$', t):
            if any(word in t for word in ["안녕", "하이", "헬로", "hi"]):
                return True
        
        return False








    
    collection = Collection(COLLECTION)   #다시 상품 DB 컬렉션으로 연결
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return obj.item()
        return obj

    PRODUCT_CACHE = {}

    def clean_html_content(html_raw: str) -> str:
        try:
            html_cleaned = html_raw.replace('\n', '').replace('\r', '')
            html_cleaned = html_cleaned.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
            if html_cleaned.count("<center>") > html_cleaned.count("</center>"):
                html_cleaned += "</center>"
            if html_cleaned.count("<p") > html_cleaned.count("</p>"):
                html_cleaned += "</p>"
            return html_cleaned
        except Exception as e:
            print(f"❌ HTML 정제 오류: {e}")
            return html_raw
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Redis에서 세션 기록을 가져옵니다."""
        try:
            history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
            # 세션이 존재하는지 확인
            messages = history.messages
            return history
        except Exception as e:
            print(f"❌ 대화 기록 가져오기 오류: {e}")
            # 에러 발생 시 새로운 히스토리 객체 생성
            return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    
    def clear_message_history(session_id: str):
        """
        Redis에 저장된 특정 세션의 대화 기록을 초기화합니다.
        """
        try:
            history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
            history.clear()
            print(f"✅ 세션 {session_id}의 대화 기록이 초기화되었습니다.")
        except Exception as e:
            print(f"❌ Redis 초기화 오류: {e}")
            raise HTTPException(status_code=500, detail="대화 기록 초기화 중 오류가 발생했습니다.")
        



    ##############################################################################원본 사용자 쿼리
    # ✅ 세션 초기화 명령 처리
    total_start_time = time.time()  # 전체 시작 시간 기록
    # ✅ 입력 쿼리 추출 및 타입 확인
    query = request if isinstance(request, str) else request.query
    print(f"🔍 사용자 검색어: {query}")

    if not isinstance(query, str):
        raise TypeError(f"❌ [ERROR] 잘못된 query 타입: {type(query)}")
    
    if query.lower() == "reset":
        if session_id:
            clear_message_history(session_id)
        return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}
    

    print("[Debug] Raw query:", query)            # ← 여기에!
    # lang_code = detect(query)

    # ✅ Redis 세션 기록 불러오기 및 최신 입력 저장
    session_history = get_session_history(session_id)



    # � **핵심 수정**: 리셋 직후에는 세션 타임아웃 체크 건너뛰기
    # 리셋 직후 첫 메시지인지 확인 (history가 비어있거나 1개 이하면 리셋 직후)
    messages = session_history.messages if hasattr(session_history, 'messages') else []
    is_after_reset = len(messages) <= 1

    # 리셋 직후 첫 메시지는 timeout 체크 스킵
    if not is_after_reset:
        timeout_result = check_session_timeout(session_history, session_id)
        print(f"[세션체크] timeout_result={timeout_result}")
        if timeout_result and timeout_result.get("session_expired"):
            return timeout_result
    else:
        print("[세션체크] 리셋 직후 첫 메시지 → timeout 체크 스킵")
    
    # # 🕒 세션 자동 만료 체크 (경고 없이 바로 처리)
    # timeout_result = check_session_timeout(session_history, session_id)
    # print(f"[세션체크] timeout_result={timeout_result}")
    # if timeout_result and timeout_result.get("session_expired"):
    #     # 세션이 만료된 경우 자동 초기화 메시지를 즉시 반환 (이미 Redis에 저장됨)
    #     print(f"[세션만료] 자동 초기화 메시지 반환: {timeout_result['assistant_message'][:50]}...")
    #     return timeout_result





    if not locals().get('skip_add_user_message'):
        session_history.add_user_message(query)

    # 🔁 마지막 활동 기준으로 TTL 갱신(슬라이딩)
    try:
        r = redis.from_url(REDIS_URL)
        r.expire(f"message_store:{session_id}", TIMEOUT_SECONDS)  # 활동할 때마다 연장
    except Exception as e:
        print(f"[세션 TTL 갱신 오류] {e}")



    previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
    if query in previous_queries:
        previous_queries.remove(query)
    
    # ✅ 전체 중복 제거 (최신 입력을 제외한 나머지에서)
    previous_queries = list(dict.fromkeys(previous_queries))


    raw = detect(query)
    lang_code = raw.lower().split("-")[0]
    print("[Debug] lang_code →", lang_code)   # ← 이 줄 추가!

    # 가격 조건 처리 함수
    def extract_price_condition(text: str) -> Optional[str]:
        # 🔧 크기/규격 단위 목록 (가격 인식에서 제외)
        SIZE_UNITS = [
            # 길이/크기 단위
            '인치', 'inch', 'cm', '센티', '미터', 'm', 'mm', '밀리미터',
            # 무게 단위  
            'kg', '킬로', '그램', 'g', 'ton', '톤',
            # 용량 단위
            '리터', 'l', 'ml', '밀리리터', '갤런',
            # 전자기기 단위
            '기가', 'gb', 'mb', 'tb', '테라', 'hz', '헤르츠',
            # 기타 규격
            '사이즈', 'size', '호', '번', '단', '개', '매', '장', '권', '병', '포'
        ]

        # 🔧 크기 조건 패턴 먼저 체크 (가격보다 우선 처리)
        size_condition_patterns = [
            (r'(\d+(?:\.\d+)?)\s*인치\s*(이하|미만|이상|초과)', "인치"),
            (r'(\d+(?:\.\d+)?)\s*inch\s*(under|below|over|above)', "inch"),
            (r'(\d+(?:\.\d+)?)\s*cm\s*(이하|미만|이상|초과)', "cm"),
            (r'(\d+(?:\.\d+)?)\s*센티\s*(이하|미만|이상|초과)', "cm"),
            (r'(\d+(?:\.\d+)?)\s*kg\s*(이하|미만|이상|초과)', "kg"),
            (r'(\d+(?:\.\d+)?)\s*킬로\s*(이하|미만|이상|초과)', "kg"),
            (r'(\d+(?:\.\d+)?)\s*리터\s*(이하|미만|이상|초과)', "리터"),
            (r'(\d+(?:\.\d+)?)\s*l\s*(이하|미만|이상|초과)', "리터"),
        ]
        
        op_map = {
            "이하": "<=", "미만": "<", "이상": ">=", "초과": ">",
            "under": "<", "below": "<", "over": ">", "above": ">"
        }
        
        query_lower = text.lower()
        
        # 크기 조건 체크
        for pattern, unit in size_condition_patterns:
            m = re.search(pattern, query_lower)
            if m:
                value = float(m.group(1))
                op_text = m.group(2).strip()
                operator = op_map.get(op_text)
                
                if operator:
                    print(f"[Debug] 크기 조건 감지: {value}{unit} {op_text} (가격 인식 건너뛰기)")
                    return f"SIZE_CONDITION_{unit}_{operator}_{value}"  # 특별한 크기 조건 반환
        
        # 🔧 가격 조건도 함께 있는지 먼저 체크
        price_keywords = ['원', '만원', '천원', '억원', '달러', 'dollar', 'usd', '이상', '이하', '미만', '초과', '에서', '사이', '부터', '까지', '~', '-']
        has_price_keywords = any(keyword in query_lower for keyword in price_keywords)
        
        if has_price_keywords:
            print(f"[Debug] 크기+가격 조건 동시 감지 → 가격 인식 계속 진행")
            # 가격 인식을 계속 진행
        else:
            # 🔧 기타 크기/규격 단위만 있고 가격 키워드가 없는 경우만 가격 인식 건너뛰기
            size_patterns = [rf'\d+\s*{re.escape(unit)}\b' for unit in SIZE_UNITS]
            for pattern in size_patterns:
                if re.search(pattern, query_lower):
                    print(f"[Debug] 크기/규격 패턴만 감지로 가격 인식 제외: '{pattern}' 매치 in '{text}'")
                    return None
        


        # 숫자 단위 정규화
        def normalize_price_units(text: str) -> str:

            # 소문자 통일
            t = text.lower()
            
            # 0) "10,000" 같은 콤마 포함 숫자 → 콤마 제거
            def _strip_commas(m):
                return m.group(1).replace(",", "")
            t = re.sub(r'(\d{1,3}(?:,\d{3})+)', _strip_commas, t)

            # 한글 단위와 기본값 매핑
            kr_unit_map = {
                "천": ("000", "1000"),
                "만": ("0000", "10000"),
                "십만": ("00000", "100000"),
                "백만": ("000000", "1000000"),
                "천만": ("0000000", "10000000"),
                "억": ("00000000", "100000000")
            }
            # 영어 단위와 기본값 매핑
            en_unit_map = {
                "k": ("000", "1000"),
                "thousand": ("000", "1000"),
                "m": ("000000", "1000000"),
                "million": ("000000", "1000000")
            }
            
            
            # ── 한글 숫자+단위: "3만", "2.5억", "3만 원" → 정수로 치환 ──
            pat_kr_num_unit = re.compile(r'(\d+(?:\.\d+)?)\s*(십만|백만|천만|천|만|억)(?:\s*원)?')
            def repl_kr_num_unit(m):
                num_txt = m.group(1)
                unit = m.group(2)
                zeros, _default = kr_unit_map[unit]
                try:
                    num = Decimal(num_txt)
                    multiplier = Decimal(10) ** len(zeros)
                    val = int(num * multiplier)
                    return str(val)
                except (InvalidOperation, ValueError):
                    return m.group(0)  # 실패 시 원문 유지
            t = pat_kr_num_unit.sub(repl_kr_num_unit, t)

            # ── 한글 단위만 있는 경우: "십만원", "억 원" → 기본값 숫자 ──
            pat_kr_unit_only = re.compile(r'\b(십만|백만|천만|천|만|억)\s*원\b')
            def repl_kr_only(m):
                unit = m.group(1)
                _zeros, default_val = kr_unit_map[unit]
                return f"{default_val}원"
            t = pat_kr_unit_only.sub(repl_kr_only, t)

            # ── 영어 숫자+단위(Compact): "10k", "2.5m" → 정수 ──
            pat_en_compact = re.compile(r'(\d+(?:\.\d+)?)\s*(k|m)\b')
            def repl_en_compact(m):
                num_txt = m.group(1)
                unit = m.group(2)
                zeros, _default = en_unit_map[unit]
                try:
                    num = Decimal(num_txt)
                    multiplier = Decimal(10) ** len(zeros)
                    val = int(num * multiplier)
                    return str(val)
                except (InvalidOperation, ValueError):
                    return m.group(0)
            t = pat_en_compact.sub(repl_en_compact, t)

            # ── 영어 숫자+단위(Word): "10 thousand", "3 million" → 정수 ──
            pat_en_word = re.compile(r'(\d+(?:\.\d+)?)\s*(thousand|million)\b')
            def repl_en_word(m):
                num_txt = m.group(1)
                unit = m.group(2)
                zeros, _default = en_unit_map[unit]
                try:
                    num = Decimal(num_txt)
                    multiplier = Decimal(10) ** len(zeros)
                    val = int(num * multiplier)
                    return str(val)
                except (InvalidOperation, ValueError):
                    return m.group(0)
            t = pat_en_word.sub(repl_en_word, t)

            # ⚠️ 주의: 단독 unit(예: "k", "m") 치환은 하지 않음 → women’s/summer 오염 방지

            # 3) 통화 단위 통일: 숫자 + (won|dollars|usd|원) → '원'
            t = re.sub(r'(\d+)\s*(won|dollars|usd|원)\b', r'\1원', t)

            print(f"[Debug] normalize_price_units 입력: {t}")
            return t

        # 쿼리 정규화 및 디버깅
        query = normalize_price_units(text.lower())
        print(f"[Debug] 정규화된 쿼리: {query}")

        # 가격 범위 패턴 (한글 + 영어)
        range_patterns = [
            # 한글 복합 범위 패턴 (이상/이하)
            r'(\d+)[^\d]*원?\s*이상\s*(\d+)[^\d]*원?\s*이하',   # "20000 이상 30000 이하"
            r'(\d+)[^\d]*이상\s*(\d+)[^\d]*이하',              # "20000이상 30000이하" (원 없는 버전)
            r'(\d+)[^\d]*원\s*이상\s*(\d+)[^\d]*원\s*이하',    # "20000원 이상 30000원 이하"
            
            # 한글 복합 범위 패턴 (초과/미만)
            r'(\d+)[^\d]*원?\s*초과\s*(\d+)[^\d]*원?\s*미만',   # "20000 초과 30000 미만"
            r'(\d+)[^\d]*초과\s*(\d+)[^\d]*미만',              # "20000초과 30000미만"
            r'(\d+)[^\d]*원\s*초과\s*(\d+)[^\d]*원\s*미만',    # "20000원 초과 30000원 미만"
            
            # 한글 범위 구분자 패턴
            r'(\d+)[^\d]*원?\s*(?:~|에서|부터)\s*(\d+)[^\d]*원?',    # "20000~30000원"
            r'(\d+)[^\d]*원?\s*부터\s*(\d+)[^\d]*원?\s*까지',        # "20000원부터 30000원까지"
            r'(\d+)[^\d]*에서\s*(\d+)[^\d]*원?\s*사이',             # "20000에서 30000원 사이"
            
            # 영어 범위 패턴 (기본)
            r'between\s*(\d+)\s*and\s*(\d+)(?:\s*원?)',             # "between 20000 and 30000"
            r'from\s*(\d+)\s*to\s*(\d+)(?:\s*원?)',                # "from 20000 to 30000"
            r'(\d+)\s*(?:to|-|~)\s*(\d+)(?:\s*원?)',               # "20000 to 30000", "20000-30000"
            
            # 영어 범위 패턴 (상세)
            r'(\d+)\s*(?:or\s+more)\s+(?:but|and)\s+(?:less\s+than|under|below)\s*(\d+)',  # "20000 or more but less than 30000"
            r'(?:more\s+than|over|above)\s*(\d+)\s*(?:but|and)\s*(?:less\s+than|under|below)\s*(\d+)',  # "more than 20000 but less than 30000"
            r'(\d+)\s*(?:or\s+more)\s+(?:but|and)\s+(?:not\s+more\s+than|no\s+more\s+than)\s*(\d+)',   # "20000 or more but not more than 30000"
            
            # 통화 단위 포함 패턴
            r'(?:USD|USD\$|\$)\s*(\d+)\s*(?:to|-|~)\s*(?:USD|USD\$|\$)\s*(\d+)',  # "$20000 to $30000"
            r'(?:KRW|₩)\s*(\d+)\s*(?:to|-|~)\s*(?:KRW|₩)\s*(\d+)',               # "₩20000 to ₩30000"
        ]

        # 단일 가격 패턴 (한글 + 영어)
        single_patterns = [
            # 한글 패턴
            r'(\d+)[^\d]*원?(?:\s*)(이하|미만|이상|초과)',
            # 영어 패턴
            r'(?:under|below|less than|up to)\s*(\d+)(?:\s*원?)',
            r'(?:over|above|more than|at least)\s*(\d+)(?:\s*원?)',
            r'(\d+)(?:\s*원?)\s*(?:or less|or more)',
            # 단순 숫자+원 패턴 (이상으로 해석)
            r'(\d+)[^\d]*원\s*'
        ]

        # 연산자 매핑
        op_map_kr = {"이하": "<=", "미만": "<", "이상": ">=", "초과": ">"}
        op_map_en = {
            "under": "<", "below": "<", "less than": "<", "up to": "<=",
            "over": ">", "above": ">", "more than": ">", "at least": ">=",
            "or less": "<=", "or more": ">="
        }

        try:
            # 1. 범위 검색 시도
            for pattern in range_patterns:
                m = re.search(pattern, query)
                if m:
                    # "이상-이하" 또는 기타 범위 패턴 모두 일관되게 처리
                    min_price = int(m.group(1))  # 첫 번째 숫자
                    max_price = int(m.group(2))  # 두 번째 숫자
                    
                    # 가격 순서가 뒤바뀐 경우 교정
                    if min_price > max_price:
                        min_price, max_price = max_price, min_price
                        
                    print(f"[Debug] 가격 범위 감지: {min_price}원 ~ {max_price}원")
                    return f"market_price >= {min_price} && market_price <= {max_price}"

            # 2. 단일 가격 검색 시도
            for pattern in single_patterns:
                m = re.search(pattern, query)
                if not m:
                    continue

                amount = int(m.group(1))
                comp = m.group(2) if len(m.groups()) > 1 else None

                # 단일 숫자가 1개(1, 2 등)일 때는 가격으로 인식하지 않음 (예: '1개', '2개' 등)
                if amount in [1, 2]:
                    print(f"[Debug] 단일 숫자({amount})는 가격으로 인식하지 않음")
                    continue

                # (A) 한글 연산자 우선 처리
                if comp in op_map_kr:
                    price_op = op_map_kr[comp]
                    print(f"[Debug] 한글 가격 조건 감지: {amount}원 {comp}")
                    return f"market_price {price_op} {amount}"

                # (B) 영어 연산자 우선 처리: 매치된 구간 안에서만 찾아서 안전하게 판정
                matched_span = query[m.start():m.end()]
                for op_text, op_symbol in op_map_en.items():
                    if op_text in matched_span:
                        print(f"[Debug] 영어 가격 조건 감지: {op_text} {amount}") 
                        return f"market_price {op_symbol} {amount}"

                # (C) 여기까지도 못 정하면 **정말 마지막** 폴백
                #     오직 '(\d+)[^\d]*원\s*' 패턴에만 적용해 '이상'으로 처리
                if pattern == r'(\d+)[^\d]*원\s*':
                    print(f"[Debug] 단순 가격 감지(원 폴백): {amount}원 (이상으로 처리)")
                    return f"market_price >= {amount}"

        except Exception as e:
            print(f"[Warning] 가격 조건 처리 중 오류 발생: {str(e)}")
            return None

        return None

    # 가격 조건 추출
    price_cond = extract_price_condition(query)
    print(f"[Debug] 최종 가격 조건: {price_cond if price_cond else '제한 없음'}")


    # 2) 언어 코드 → 사람말 매핑
    lang_map = {
        "ko": "한국어",
        "en": "English",
        "zh": "中文",
        "ja": "日本語",
        "vi": "Tiếng Việt",  # 베트남어
        "th": "ไทย",        # 태국어


        # "fr": "Français",
        # "de": "Deutsch",
        # "es": "Español",
        # "it": "Italiano",
        # "pt": "Português",
        # "ar": "العربية",
        # "fa": "فارسی",
        # "he": "עברית",
        # "sw": "Kiswahili",
    }

    target_lang = lang_map.get(lang_code, "English")
    print("[Debug] Detected language →", target_lang)

    # 시즌 판단
    # 시즌 관련 상수 정의
    SPRING_KEYWORDS = [
        # 한국어
        "봄","봄맞이","봄소풍","봄나들이","벚꽃","꽃샘추위","춘분","간절기","환절기",
        "트렌치코트","스프링코트","바람막이","가디건","플로럴","꽃무늬","파스텔",
        # English
        "spring","vernal","cherry blossom","blossom","cold snap","windbreaker",
        "trench coat","spring coat","cardigan","floral","floral print","pastel","layering"
    ]

    SUMMER_KEYWORDS = [
        # 한국어
        "여름","썸머","자외선차단","uv차단","썬캡","선바이저","비치","바캉스",
        "래시가드","수영복","비키니","아쿠아슈즈","워터레깅스","비치샌들",
        "라피아","스트로","스트로우","햇빛가리개","쿨링","냉감","쿨터치",
        "흡한속건","속건","통풍","메쉬","린넨","리넨","시어서커","UPF",
        # English
        "summer","uv","uv protection","uv-cut","upf","sun protection","sun cap","sun visor",
        "beach","beachwear","rashguard","rash guard","swimsuit","bikini",
        "aqua shoes","water shoes","water leggings","flip-flops","sandals",
        "raffia","straw","straw hat","sun shade","neck shade",
        "cooling","cool-touch","quick-dry","moisture wicking","breathable","ventilated","mesh","linen","seersucker"
    ]

    AUTUMN_KEYWORDS = [
        # 한국어
        "가을","오텀","단풍","추석","한가위","가을맞이","수확제","간절기","환절기",
        "플란넬","코듀로이","트위드","체크","체커드","셔켓","오버셔츠","울혼방","니트베스트",
        # English
        "autumn","fall","fall foliage","foliage","chuseok","harvest festival",
        "flannel","corduroy","tweed","plaid","checkered","shacket","overshirt","wool blend","layering"
    ]

    WINTER_KEYWORDS = [
        # 한국어
        "겨울","윈터","방한","보온","발열","기모","기모안감","플리스","보아","보아털","쉐르파",
        "다운","패딩","롱패딩","구스다운","덕다운","웰론","충전재",
        "스노우부츠","핫팩","핫팩손난로","바라클라바","넥워머","귀마개",
        "목도리","머플러","장갑","내복","롱존","이너웨어",
        "방풍","방수","발수",
        "비니","니트","울","캐시미어","퍼","두꺼운옷","두툼한","보온성",
        # English
        "winter","thermal","heat-retaining","fleece","boa","sherpa",
        "down","goose down","duck down","synthetic down","insulation","puffer","parka",
        "snow boots","hot pack","hand warmer","balaclava","neck warmer","earmuffs",
        "scarf","muffler","gloves","mittens","thermal underwear","long johns","base layer",
        "windproof","waterproof","water-repellent",
        "beanie","wool","cashmere","fur","thick","lined"
    ]


    def detect_season(query: str) -> str:
        """쿼리에서 시즌 정보를 감지하는 함수"""
        query_lower = query.lower()
        
        for season, keywords in [ 
            ("봄", SPRING_KEYWORDS),
            ("여름", SUMMER_KEYWORDS),
            ("가을", AUTUMN_KEYWORDS),
            ("겨울", WINTER_KEYWORDS)
        ]:
            if any(keyword in query_lower for keyword in keywords):
                return season
        return "미정"

    # 시즌 설정
    season = detect_season(query)
    print(f"🌞 감지된 시즌: {season}")

    # --- (옵션) 시즌 보정만 유지: 여름/겨울이면 살짝 정렬 보정 ---
    # 시즌 보정을 위한 긍정/부정 세트 정의
    _SEASON_POS = {
        "봄": tuple(SPRING_KEYWORDS),
        "여름": tuple(SUMMER_KEYWORDS),
        "가을": tuple(AUTUMN_KEYWORDS),
        "겨울": tuple(WINTER_KEYWORDS),
    }

    # 봄/가을엔 극여름·극겨울 단어 감점, 여름/겨울엔 서로 반대 시즌만 감점
    _SEASON_NEG = {
        "봄": tuple(SUMMER_KEYWORDS + WINTER_KEYWORDS),
        "가을": tuple(SUMMER_KEYWORDS + WINTER_KEYWORDS),
        "여름": tuple(WINTER_KEYWORDS),
        "겨울": tuple(SUMMER_KEYWORDS),
    }

    def _season_adjust(name: str, season_hint: str) -> float:
        """
        카테고리명(name)에 시즌 강 키워드가 있으면 L2 거리를 미세 조정.
        - 보너스(거리↓): 해당 시즌의 강 키워드 포함
        - 페널티(거리↑): 반대(혹은 극단) 시즌 강 키워드 포함
        """
        if season_hint not in _SEASON_POS:
            return 0.0

        n = name.lower()
        pos_hit = any(w.lower() in n for w in _SEASON_POS[season_hint])
        neg_hit = any(w.lower() in n for w in _SEASON_NEG[season_hint])

        POS_BONUS   = 0.04   # 거리 감소 → 순위 소폭 상승
        NEG_PENALTY = 0.12   # 거리 증가 → 순위 소폭 하향

        adj = 0.0
        if pos_hit: adj -= POS_BONUS
        if neg_hit: adj += NEG_PENALTY
        return adj

    def season_filter_items(items: list, season_hint: str):
        """시즌에 맞는 상품만 필터링하여 반환"""
        if not season_hint or season_hint == "미정" or not items:
            return items
            
        def matches_season(item_name: str, season: str) -> bool:
            name_lower = item_name.lower()
            
            # 해당 시즌의 키워드가 있으면 True
            if season == "봄" and any(kw.lower() in name_lower for kw in SPRING_KEYWORDS):
                return True
            elif season == "여름" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                return True
            elif season == "가을" and any(kw.lower() in name_lower for kw in AUTUMN_KEYWORDS):
                return True
            elif season == "겨울" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                return True
                
            # 반대 시즌 키워드가 있으면 False
            if season == "여름" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                return False
            elif season == "겨울" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                return False
                
            # 아무 키워드도 없으면 True (중립)
            return True
        
        # 필터링 적용
        filtered = [
            item for item in items 
            if matches_season(item.get("제목", ""), season_hint) and 
            matches_season(item.get("카테고리", ""), season_hint)
        ]
        
        return filtered if filtered else items  # 필터링 결과가 없으면 원본 반환

    # season 기본값 보장 (이미 위에서 세팅되어 있으면 이 부분은 그대로 둬도 됨)
    try:
        season
    except NameError:
        season = "미정"


    # 대화 이력 가져오기
    history_messages = [msg.content for msg in session_history.messages]
    conversation_context = "\n".join([f"이전 대화: {msg}" for msg in history_messages[-10:]]) if history_messages else "이전 대화 없음"



















    # ===== 사용자 의도 파악  LLM 재질문 Start =====
    def _extract_json_block(text: str) -> dict:
        # 1) 코드펜스 제거
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("` \n")
            # 첫 줄에 json 같은 언어 힌트 제거
            if t.lower().startswith("json"):
                t = t[len("json"):].lstrip()
        # 2) 중괄호 블록만 추출
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1 and e > s:
            t = t[s:e+1]
        return json.loads(t)

    def _clamp01(x):
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    def _build_recent_context(session_history, k: int = 7) -> str:
        lines = []
        if not session_history:
            return ""
        try:
            for m in session_history.messages[-20:]:
                txt = getattr(m, "content", "") or ""
                if not txt:
                    continue
                s = txt.strip()
                if "[INTENT_GATE]" in s:
                    continue
                if s.startswith(("ERROR:", "Traceback", "```", "#", "INFO:", "DEBUG:")):
                    continue
                s = s.replace("\n", " ")[:160]
                if s:
                    lines.append("- " + s)
        except Exception:
            pass
        return "\n".join(lines[-k:])



    # 점수 가중치 (반드시 합이 1.0이 되도록 설계)
    W_DIRECT  = 0.65  # 상품이 뚜렷한지
    W_ATTR    = 0.15  # 필터/속성 얼마나 채워졌는지
    W_CONTEXT = 0.10  # 대화 맥락 일관성
    W_BRAND   = 0.10  # 브랜드 명확도


    def _recompute_route(scores: Dict[str, Any]) -> Tuple[float, str]:
        """
        디테일한 점수 계산 및 라우팅 결정
        정의된 임계값들을 활용한 정교한 판정 시스템
        """
        d = _clamp01(scores.get("direct_match", 0.0))
        a = _clamp01(scores.get("attribute_match", 0.0))
        c = _clamp01(scores.get("context_match", 0.0))
        b = _clamp01(scores.get("brand_match", 0.0))
        
        # 🎯 가중평균 계산 (direct_match 중심)
        avg = W_DIRECT*d + W_ATTR*a + W_CONTEXT*c + W_BRAND*b
        
        # 🎯 우선: 컨텍스트 + 직접매칭 조합 (특별 케이스)
        if 0.3 <= c <= 0.4 and d >= DIRECT_MATCH_HIGH:
            route = "proceed"
            print(f"[Route] 컨텍스트+직접매칭 조합({c:.3f}≥0.3 + {d:.3f}≥0.7) → 특별 통과")
        # 🚀 매우 높은 직접 매칭
            
        # 🚀 1차: 매우 높은 직접 매칭만 단독 통과
        elif d >= DIRECT_MATCH_HIGH:
            route = "proceed"
            print(f"[Route] 직접매칭 높음({d:.3f}≥{DIRECT_MATCH_HIGH}) → 단독 통과")
            
        # 🔍 2차: 직접매칭 + 속성매칭 둘 다 높아야 통과 (조건 강화)
        elif d >= 0.80 and a >= ATTRIBUTE_MIN:  # 둘 다 높아야 함
            route = "proceed"
            print(f"[Route] 직접매칭+속성매칭 높음({d:.3f}≥0.80 + {a:.3f}≥{ATTRIBUTE_MIN}) → 통과")

        # 📊 3차: 종합 점수 + 최소 속성 커버리지 (둘 다 더 까다롭게)
        elif avg >= THRESHOLD and a >= FACET_COVERAGE_MIN:  
            route = "proceed" 
            print(f"[Route] 종합점수 높음({avg:.3f}≥{THRESHOLD}) + 속성커버 충족({a:.3f}≥{FACET_COVERAGE_MIN}) → 통과")
            
        # 🎲 4차: 매우 높은 속성 커버리지 + 괜찮은 점수 (조건 강화)
        elif a >= FACET_SUFFICIENT and avg >= 0.70: 
            route = "proceed"
            print(f"[Route] 속성커버 충분({a:.3f}≥{FACET_SUFFICIENT}) + 기본 점수({avg:.3f}≥0.70) → 완화 통과")
            
        # ❌ 5차: 그 외 모든 경우 재질문 (더 엄격)
        else:
            route = "clarify"
            print(f"[Route] 점수 부족(avg={avg:.3f}<{THRESHOLD}, d={d:.3f}, a={a:.3f}) → 재질문")
        
        return round(avg, 4), route


    direct_search_patterns = [
        '나로수','narosu'
    ]

    def check_direct_search_command(user_query: str) -> bool:
        """
        바로검색 명령어가 포함되어 있는지 확인
        """
        
        query_lower = user_query.lower().replace(' ', '')
        for pattern in direct_search_patterns:
            if pattern.replace(' ', '') in query_lower:
                return True
        return False
    
    def remove_direct_search_keywords(text: str) -> str:
        """
        바로검색 명령어 패턴을 모두 제거
        """
        t = text
        for kw in direct_search_patterns:
            t = t.replace(kw, "")
            t = t.replace(kw.replace(" ", ""), "")
        return t.strip()





    def analyze_conversation_context(session_history) -> Dict[str, Any]:
        """
        이전 대화를 분석하여 종합적인 맥락 정보 추출
        """
        context_info = {
            "main_category": None,
            "mentioned_attributes": [],
            "rejected_items": [],
            "price_mentions": [],
            "brand_mentions": [],
            "repeated_queries": [],
            "clarification_history": []
        }
        
        if not session_history:
            return context_info
            
        try:
            user_messages = []
            ai_messages = []
            
            for m in session_history.messages[-15:]:  # 최근 15개 메시지 분석
                content = getattr(m, "content", "") or ""
                if not content.strip():
                    continue
                    
                if hasattr(m, 'type'):
                    if m.type == 'human':
                        user_messages.append(content)
                    elif m.type == 'ai':
                        ai_messages.append(content)
                        
            # 주요 카테고리 추출
            categories = ['옷', '의류', '상의', '하의', '신발', '운동화', '구두', '가방', '핸드백', '화장품', '음식', '전자제품']
            for msg in user_messages:
                for cat in categories:
                    if cat in msg:
                        context_info["main_category"] = cat
                        break
                        
            # 속성 언급 추출
            attributes = ['색깔', '색상', '크기', '사이즈', '재질', '브랜드', '가격', '저렴', '비싸', '고급', '예쁜', '멋진']
            for msg in user_messages:
                for attr in attributes:
                    if attr in msg and attr not in context_info["mentioned_attributes"]:
                        context_info["mentioned_attributes"].append(attr)
                        
            # 반복 질문 패턴 감지
            if len(user_messages) >= 2:
                recent_queries = user_messages[-3:]  # 최근 3개 질문
                for i, query in enumerate(recent_queries):
                    if len([q for q in recent_queries if q.strip()[:10] == query.strip()[:10]]) > 1:
                        context_info["repeated_queries"].append(query[:50])
                        
        except Exception as e:
            print(f"[Context Analysis Error] {e}")
            
        return context_info


    
    # --- LLM 프롬프트: 의도 파악 + 스마트 재질문 통합 (ui_message) ---

    INTENT_GATE_PROMPT = lambda user_query, recent_context: f"""
    **Professional Product Consultation AI - Specific Product Tailored Questioning**
    Output must be a single JSON object only. No other text or comments.
    All text values MUST be plain text without any emojis or emoticons (no 😊, 😂 etc.).

    The user searched for "{user_query}".
    **CRITICAL: Previous Conversation Analysis (MUST CHECK FIRST!)**
    [Previous conversation summary] 
    {recent_context or 'none'}

    Read the query and conversation summary, then:
    - Understand what product or category the user is trying to find.
    - Select exactly one axis that will most effectively narrow down the product candidates.
    - Ask one follow-up question with 2–4 choices, labeled in order as:
      - 2 options: A), B)
      - 3 options: A), B), C)
      - 4 options: A), B), C), D)
    - Decide the number of options (2~4) based on how many **meaningful and distinct** values exist on that axis. Do NOT fabricate options just to reach 4.
    - Never repeat axes or values that are already fixed in the conversation.

    --------------------------------------------------
     Critical Axis Lock Non-Violation Principle
    --------------------------------------------------

    Concept / Axis Lock (must be strictly enforced):
    - Treat every axis and value already mentioned by the user as locked.
    - Treat every axis that was already used in a previous clarify_question as locked.
    - Do not ask again about any locked axis (e.g., spiciness already chosen, “for bedroom use” already fixed, “soup vs dry” already decided).

    --------------------------------------------------
    Product / Category detection & “which-type” axis
    --------------------------------------------------


    1) Detect the main product or category keyword from the query and context.
    - Examples: clothes, coat, dress, shoes, bag, backpack, wallet, AirPods, wireless earphones, laptop, smartphone, ramen, cup noodles, snacks, coffee, bed, chair, desk, light, etc.
    - Treat this as the anchor product/category.

    2) When the anchor product/category is clear but the specific type/model is not fixed yet:
    - The first clarification axis MUST be a “which-type” axis:
        - “Which [product_word] do you want?” / “Which [product_word] are you looking for?”
    - The clarify_question MUST literally include the anchor product word.
        - For example:
        - If the user_query implies “clothes” → the question must include “clothes”.
        - If the user_query implies “AirPods” → the question must include “AirPods”.
        - If the user_query implies “ramen” → the question must include “ramen”.

    3) Choices for the “which-type” axis (2~4 options):
    - Options MUST be concrete, real-world variants or representative types within that product.
    - Examples of the pattern (do not hard-code these; adapt to the product):
        - Clothes: shirt / jacket / coat / dress
        - Earphones: entry model / noise-cancelling model / sports-fit model / premium model
        - Ramen: mild ramen / medium ramen / spicy ramen / extra spicy ramen
    - Each option must be a distinct type/model/variant that a user could realistically pick as a starting point for search.
    - If there are only 2 or 3 truly meaningful variants, output only 2 or 3 options (do NOT force 4).

    The “which-type” axis has top priority whenever the main product/category is clear but the specific variant is not.


    --------------------------------------------------
    Generic axis library (used after “which-type” is resolved)
    --------------------------------------------------

    When the “which-type” axis is already clarified or not applicable, consider other axes.
    From the remaining (unlocked) axes, prioritize roughly in this order:

    1) Usage purpose / location / situation (who, where, for what)
    2) Size / volume / capacity / scale
    3) Core function / performance level
    4) Style / design / material / feel
    5) Budget / price range / brand preference

    Never reuse any locked axis.

    --------------------------------------------------

    Each clarify_question must:

    - Length:
    - ≤ 200 characters if written in Korean
    - ≤ 250 characters if written in English

    - Structure:
    1) One sentence of background:
        - Explain briefly why this single axis is important to narrow down products now.
    2) One~two sentences to understand the situation:
        - Reference who/where/how often/how it will be used, but remain aligned with the same axis.
    3) One sentence for an additional consideration:
        - Add one overlooked but relevant factor for this axis (e.g., long-time comfort, storage space, sensitivity to spicy food).

    Then immediately present 2–4 options (A)~B) or A)~C) or A)~D)) as short choices (1~3 words each) for that single axis.

    --------------------------------------------------

    Choice (A,B,C,D) rules (2~4 options)
    --------------------------------------------------

    - Always generate **between 2 and 4 options**:
        - If there are only 2 meaningful values on that axis → use A), B)
        - If there are 3 meaningful values → use A), B), C)
        - If there are 4 meaningful values → use A), B), C), D)
    - Never invent unnatural or meaningless options just to reach 4.
    - The count of options must match the number of **realistic** and **distinct** values on that axis (min 2, max 4).

    - Each option MUST:
    - Represent a single, clear product attribute, type, or variant **along the chosen axis only**.
    - Be 1~3 words long (short phrase-level, not full sentences).
    - Be directly usable as a filter or a product group in a shopping/search system.
    - Be a pure value on that axis, not a vague description of the whole product.

    - If the chosen axis is **flavor / taste** (for foods, snacks, drinks, etc.):
    - Good patterns (allowed):
        - "plain / original", "lightly salted", "spicy", "mild", "barbecue", "cheese", "onion", "wasabi", "sour cream", "lemon", "chocolate".
    - Bad patterns (forbidden as option labels):
        - Vague or marketing-style phrases such as:
        - "classic chips", "classic taste", "premium chips", "special chips", "best flavor", "signature taste".
    - If the intended meaning is "basic salted/original flavor", normalize the label to:
        - "plain/original" or "lightly salted".
        - You MUST NOT use "classic" as a flavor or type label.

    - If the chosen axis is **form / type / style**:
    - Examples of GOOD options:
        - Bags: "backpack", "tote bag", "crossbody", "shoulder bag"
        - Shoes: "sneakers", "loafers", "sandals", "boots"
        - Lighting: "ceiling light", "desk lamp", "floor lamp", "wall lamp"
    - Do NOT mix marketing adjectives into the option labels:
        - Forbidden: "premium sneakers", "classic bag", "best desk lamp" as option texts.

    - If the chosen axis is **size / volume / capacity**:
    - Use clean size values only:
        - "small", "medium", "large", "XL", "single pack", "multi pack", "500 ml", "1 L"
    - Do NOT mix unrelated info in one option:
        - Forbidden: "large cheap set" (mixes size + price)
        - Keep each option focused on size/volume only.

    - Generic rules for ALL products and axes:
    - Do NOT:
        - Use pure situations such as "commuting", "home use", "office" as options themselves.
        (These may appear in the question text, but not as option labels.)
        - Mix unrelated axes inside one option
        (e.g., "cheap AND premium brand AND waterproof" combines price + brand + function).
        - Include "Other", "Etc", "Type it yourself", or any equivalent wording as A, B, C, or D.
        - Use vague marketing or emotional words as option labels, such as:
        - "classic", "premium", "special", "best", "signature",
            "high quality", "deluxe", "basic type", "standard type",
            "trendy", "stylish", "hot item", "popular".
    - Prefer:
        - Axis-pure values such as:
        - flavor: "spicy", "barbecue", "cheese"
        - size: "small", "large"
        - fit: "slim fit", "regular fit", "oversized"
        - material: "cotton", "leather", "stainless steel"
        - Option texts that can be used as-is as catalog filter tags.

    - If allowing free-text input is useful, keep the 2~4 options as concrete values and optionally add a separate sentence after the list, such as:
    - "If none of these match, you can answer in your own words."
    This must not be treated as an extra option.



    --------------------------------------------------
    Route rules
    --------------------------------------------------

    - Default: "route" = "clarify".
    - When:
    - the category/product is fixed, AND
    - at least three key attributes (including, if relevant, the which-type axis) are fixed,
    then:
    - set "route" = "proceed"
    - and set "clarify_question" = "" (empty string).

    When "route" = "clarify":
    - "clarify_question" must NEVER be an empty string.
    - If "clarify_question" is empty while "route" = "clarify", the system will malfunction.

    Core principle:
    - If the query is vague, first clarify which product type the user actually wants (using the which-type axis).
    - If the product type is already clear, focus on detailed selection criteria (size, spec level, design, pack count, etc.).


    Scoring scheme (0.00~1.00, step 0.01)
    Scoring mindset:
    - Treat scoring like an audit: use the entire dialogue (current query + history + locked axes).
    - Base every score on explicit evidence, never on vague intuition.
    - Use the full 0.00–1.00 range in 0.01 steps; do NOT cluster everything around the same value.

    direct_match
    - How specific and unambiguous the product/category is.
    - Low: vague intent, no clear product.
    - High: clear category + brand/model/line fixed.

    attribute_match
    - How many concrete, catalog-usable attributes are fixed (size, flavor, fit, usage, material, pack size, etc.).
    - More and clearer attributes ⇒ higher score.

    context_match
    - How well the current interpretation respects all previous constraints and locked axes.
    - Penalize any conflict or ignored lock; reward strict consistency.

    brand_match
    - How clearly the intent is tied to a brand or product line across the whole conversation.
    - No or weak hint ⇒ low; explicit and repeated ⇒ high.

    avg_score
    - Compute numerically:
    - avg_score = (direct_match + attribute_match + context_match + brand_match) / 4
    - Round to two decimals.

    expanded_terms
    - From the user’s answer (choice or free text), extract 3~6 **Korean retail search keywords** (remove stopwords/emotional words, keep noun-like tokens).
    Route rules
    - Default is route="clarify". If the query is sufficiently specific (category + 3 or more key attributes), switch to route="proceed" (only then set clarify_question="").

    The length of the follow-up question must be within 200 characters in Korean or 250 characters in English.
    
    **Important output rules:**
    - When route="clarify", clarify_question must **never** be an empty string.
    - If clarify_question is empty, the system will malfunction. You must always generate a valid question.
    - Only when route="proceed" may clarify_question be set to the empty string ("").
    

    - All string fields in the JSON (clarify_question, expanded_terms, notes, etc.)
        MUST NOT contain any emojis or emoticons.
        Use only plain text words. (No emojis like 😊, 😂, ❤️ and no emoticons.)


    Language rules
    - Detect the user’s language → then output **only in {target_lang}**. Do not generate text in any other language. No mixing of languages.
    - Do not use emojis or emoticons in any field. Plain text only.



    JSON schema:
    {{
    "direct_match": 0.0,
    "context_match": 0.0,
    "attribute_match": 0.0,
    "brand_match": 0.0,
    "avg_score": 0.0,
    "route": "clarify",
    "clarify_question": "Natural follow-up question like a sales expert + 2~4 choices (A,B[,C[,D]])",
    "expanded_terms": ["3~6 Korean keywords useful for search"],
    "notes": ["Rationale for your judgment"]
    }}
    """


    

    def run_intent_gate(user_query: str, session_history, client, model=LLM_MODEL):
        # 🚀 바로검색 명령어 확인 - 재질문 패스
        if check_direct_search_command(user_query):
            print(f"[바로검색] 명령어 감지: '{user_query}' → 재질문 패스하고 바로 검색")
            return {
                "direct_match": 0.9, "context_match": 0.8,
                "attribute_match": 0.7, "brand_match": 0.6,
                "avg_score": 0.85, "route": "proceed",
                "clarify_question": "",
                "notes": ["direct_search_command"],
                "completion_ratio": 1.0,
                "completion_percent": 100,
                "completion_message": "🚀 바로검색 모드"
            }
        

        # 🏠 홈페이지 이동 의도 감지 체크 추가
        if is_homepage_intent(user_query):
            print(f"[홈페이지 이동] 의도 감지: '{user_query}' → 재질문 패스")
            homepage_message = f"🏠 홈페이지로 이동합니다: {HOMEPAGE_URL}" if target_lang == "한국어" else f"🏠 Redirecting to homepage: {HOMEPAGE_URL}"
            
            return {
                "direct_match": 1.0,
                "context_match": 1.0,
                "attribute_match": 1.0,
                "brand_match": 1.0,
                "avg_score": 1.0,
                "route": "homepage",
                "clarify_question": "",
                "homepage_url": HOMEPAGE_URL,
                "homepage_message": homepage_message,
                "is_homepage_intent": True,
                "expanded_terms": [],
                "notes": ["Homepage navigation intent detected"],
                "completion_ratio": 1.0,
                "completion_percent": 100
            }

        # 🚚 배송/주문 조회 의도 감지 체크 추가
        if is_delivery_intent(user_query):
            print(f"[배송조회] 의도 감지: '{user_query}' → 재질문 패스")
            delivery_message = (
                f"배송/주문 조회는 {DELIVERY_INQUIRY_URL} 에서 바로 확인할 수 있어요. " if target_lang == "한국어" else f"Delivery/Order status: {DELIVERY_INQUIRY_URL}"
            )
            
            return {
                "direct_match": 1.0,
                "context_match": 1.0,
                "attribute_match": 1.0,
                "brand_match": 1.0,
                "avg_score": 1.0,
                "route": "delivery",
                "clarify_question": "",
                "delivery_url": DELIVERY_INQUIRY_URL,
                "delivery_message": delivery_message,
                "is_delivery_intent": True,
                "expanded_terms": [],
                "notes": ["Delivery inquiry intent detected"],
                "completion_ratio": 1.0,
                "completion_percent": 100
            }

        # 👋 인사/소개 의도 감지 (새로 추가)
        if is_greeting_or_intro_intent(user_query):
            print(f"[인사/소개] 의도 감지: '{user_query}' → 챗봇 소개 응답")
            
            # 언어별 소개 메시지
            greeting_messages = {
                "한국어": (
                    "안녕하세요! 👋\n\n"
                    "저는 상품 검색을 도와드리는 AI 챗봇입니다. 🤖\n"
                    "찾으시는 상품을 말씀해 주시면 최적의 결과를 추천해 드리겠습니다!\n\n"
                    "예: '여름용 가방', '겨울 따뜻한 장갑', '운동화 추천해줘' 등\n\n"
                    "채팅으로 지원하는 1.홈페이지URL 2.주문조회 3.인사/소개 등을 입력해 보세요."
                ),
                "English": (
                    "Hello! 👋\n\n"
                    "I'm an AI chatbot that helps you search for products. 🤖\n"
                    "Tell me what you're looking for, and I'll recommend the best results!\n\n"
                    "Examples: 'summer bag', 'warm winter gloves', 'recommend sneakers'\n\n"
                    "You can also chat with me using 1. Homepage URL 2. Order Inquiry 3. Greeting/Introduction."
                )
            }
            
            greeting_message = greeting_messages.get(target_lang, greeting_messages["English"])
            
            return {
                "direct_match": 1.0,
                "context_match": 1.0,
                "attribute_match": 1.0,
                "brand_match": 1.0,
                "avg_score": 1.0,
                "route": "greeting",
                "clarify_question": "",
                "greeting_message": greeting_message,
                "is_greeting_intent": True,
                "expanded_terms": [],
                "notes": ["Greeting or introduction intent detected"],
                "completion_ratio": 1.0,
                "completion_percent": 100
            }


        # 1) 최근 문맥 안정 구성
        recent_context = _build_recent_context(session_history, k=7)
        
        # 🧠 대화 맥락 종합 분석
        context_analysis = analyze_conversation_context(session_history)
        print(f"[맥락분석] 주요카테고리={context_analysis['main_category']}, "
              f"언급속성={context_analysis['mentioned_attributes']}, "
              f"반복질문={len(context_analysis['repeated_queries'])}")

        # 2) 호출 (JSON 강제)
        raw = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """JSON ONLY. No prose. CRITICAL: if route='clarify', clarify_question MUST NOT be empty string.

                
                                IMPORTANT LINE FORMATTING:
                                - In clarify_question, put each choice A), B), (C), (D) on separate lines.
                                - If there are only 2 options, output only A) and B).
                                - If there are 3 options, output only A), B), C).
                                - Use actual line breaks in the JSON string value.
                                - Add ONE EXTRA line break between question and choices for better readability.

                                Format examples:

                                "clarify_question": "It's a question about a binary choice.

                                A) Option 1
                                B) Option 2"

                                or

                                "clarify_question": "It's a question with three choices.

                                A) Option 1
                                B) Option 2
                                C) Option 3"

                                DO NOT write choices in one line like "A) Select1 B) Select2 C) Select3 D) Select4".
                                PUT EACH CHOICE ON NEW LINE with real line breaks inside JSON string.
                                ALWAYS add an empty line between question text and the first choice A)."""},
                {"role": "user", "content": INTENT_GATE_PROMPT(user_query, recent_context)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1000
        ).choices[0].message.content


        # 3) 안전 파싱 (+한 번 더 보정 시도)
        try:
            intent_eval = json.loads(raw)
        except Exception:
            try:
                intent_eval = _extract_json_block(raw)
            except Exception:
                print(f"[JSON 파싱 실패] LLM 응답 파싱 불가 → 기본 재질문 생성")
                intent_eval = {
                    "direct_match": 0.0, "context_match": 0.0,
                    "attribute_match": 0.0, "brand_match": 0.0,
                    "avg_score": 0.0, "route": "clarify",
                    "clarify_question": "",
                }

        # expanded_terms 프린트 출력
        if "expanded_terms" in intent_eval:
            print(f"[LLM] expanded_terms: {intent_eval['expanded_terms']}")

        # 안내 문구를 언어에 따라 다르게 설정
        notice_ko = "※ 선택지에 없는 경우 자유롭게 글로 서술해서 입력하셔도 됩니다."
        notice_en = "※ If none of the choices fit, feel free to write your answer in your own words."
        notice_zh = "※ 如果选项中没有合适的内容，可以自由用文字描述后输入。"
        notice_ja = "※ 選択肢にない場合は、自由に文章で入力していただいて構いません。"
        notice_vi = "※ Nếu không có lựa chọn nào phù hợp, bạn có thể tự do nhập câu trả lời bằng lời văn của mình."
        notice_th = "※ หากไม่มีตัวเลือกใดที่ตรงกับคุณ คุณสามารถพิมพ์คำตอบเป็นข้อความได้อย่างอิสระ"
        notice_ru = "※ Если ни один из вариантов не подходит, вы можете свободно ввести ответ в произвольной форме."

        # target_lang은 이미 위에서 결정됨 target_lang 예시: "한국어", "English", "中文", "日本語", "Tiếng Việt", "ไทย", "Русский" 등 
        if target_lang == "한국어":
            notice = notice_ko
        elif target_lang in ("English", "영어"):
            notice = notice_en
        elif target_lang in ("中文", "중국어"):
            notice = notice_zh
        elif target_lang in ("日本語", "일본어"):
            notice = notice_ja
        elif target_lang in ("Tiếng Việt", "베트남어"):
            notice = notice_vi
        elif target_lang in ("ไทย", "태국어"):
            notice = notice_th
        elif target_lang in ("Русский", "러시아어"):
            notice = notice_ru
        else:
            # 혹시 매핑 안 된 언어면 기본 영어로 폴백
            notice = notice_en


        if intent_eval.get("clarify_question"):
            intent_eval["clarify_question"] = intent_eval["clarify_question"].strip()
            if notice not in intent_eval["clarify_question"]:
                intent_eval["clarify_question"] += f"\n \n{notice}"






        # 4) 서버에서 재계산(단일 진실원칙)
        avg_score, route = _recompute_route(intent_eval)
        intent_eval["avg_score"] = avg_score
        intent_eval["route"] = route

        # 🎯 완성률 계산 및 추가
        completion_ratio, completion_percent = calculate_completion_rate(avg_score, THRESHOLD)
        
        
        intent_eval.update({
            "completion_ratio": completion_ratio,
            "completion_percent": completion_percent,
            "context_analysis": context_analysis,
            "_raw_llm": raw,  # 👈 원문 JSON 그대로 보관
        })


        # 5) 디버그(원하면 남기되, recent_context엔 안 넣도록 태그 유지)
        session_history.add_ai_message(f"[INTENT_GATE]{json.dumps(intent_eval, ensure_ascii=False)}")
    
         # 🔁 AI 응답 시에도 TTL 슬라이딩

        try:
            r = redis.from_url(REDIS_URL)
            r.expire(f"message_store:{session_id}", TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[세션 TTL 갱신 오류] {e}")


        return intent_eval

















    # --- 선택지 해석 유틸 ---
    def _parse_abcd_answer(text: str):
        raw = re.sub(r'[\s\)\].,:;~…!\?-]+', '', text).lower()
        # 한글 표기/서수어/숫자까지 매핑
        mapping = {
            'a':'A','1':'A','에이':'A','첫번째':'A','첫째':'A','첫번':'A',
            'b':'B','2':'B','비':'B','두번째':'B','둘째':'B','두번':'B',
            'c':'C','3':'C','씨':'C','세번째':'C','셋째':'C','세번':'C',
            'd':'D','4':'D','디':'D','네번째':'D','넷째':'D','네번':'D',
        }
        return mapping.get(raw)

    def _extract_last_choices(history) -> dict:
        """이전 대화에서 선택지 A), B), C), D) 추출 - 개선된 버전"""

        
        # AI 메시지에서 clarify_question 내용 추출 (최신 메시지부터 역순으로)
        for msg in reversed(history.messages):
            if not hasattr(msg, 'type') or msg.type != 'ai':
                continue
                
            content = getattr(msg, 'content', '') or ''
            
            # 1️⃣ [INTENT_GATE] 태그가 있는 메시지 처리
            if content.startswith('[INTENT_GATE]'):
                try:
                    json_str = content[len('[INTENT_GATE]'):]
                    data = json.loads(json_str)
                    clarify_question = data.get('clarify_question', '')
                    
                    if clarify_question and re.search(r'\b[A-D]\s*\)', clarify_question, re.I):
                        choices = _parse_choices_from_text_enhanced(clarify_question)
                        if choices:  # 🔥 선택지가 있으면 즉시 반환 (최신 것 우선)
                            print(f"[Debug] 최신 INTENT_GATE 선택지 사용: {choices}")
                            return choices
                except Exception as e:
                    print(f"[Debug] INTENT_GATE JSON 파싱 오류: {e}")
                    continue
            
            # 2️⃣ 일반 메시지에서 선택지 패턴 감지
            elif re.search(r'\b[A-D]\s*\)', content, re.I):
                choices = _parse_choices_from_text_enhanced(content)
                if choices:  # 🔥 선택지가 있으면 즉시 반환 (최신 것 우선)
                    print(f"[Debug] 최신 일반 메시지 선택지 사용: {choices}")
                    return choices
        
        print(f"[Debug] 선택지를 찾을 수 없음")
        return {}
    
    def _parse_choices_from_text_enhanced(text: str) -> dict:
        """개선된 선택지 파싱 - \n\nA) 패턴까지 완벽 지원"""
        pairs = {}
        
        # 🎯 강화된 정규식 패턴들
        patterns = [
            # 패턴 1: \n\nA) 형태 (줄바꿈 2개 + 선택지)
            r'\n\n([A-D])\s*\)\s*([^\n\r]+?)(?=\s*\n\n[A-D]\s*\)|※|$)',
            
            # 패턴 2: \nA) 형태 (줄바꿈 1개 + 선택지) 
            r'\n([A-D])\s*\)\s*([^\n\r]+?)(?=\s*\n[A-D]\s*\)|※|$)',
            
            # 패턴 3: 일반적인 A) 형태
            r'\b([A-D])\s*\)\s*([^\n\r]+?)(?=\s*[A-D]\s*\)|※|$)',
            
            # 패턴 4: 공백 다음에 오는 A) 형태
            r'\s+([A-D])\s*\)\s*([^\n\r]+?)(?=\s*[A-D]\s*\)|※|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                label = match.group(1).upper()
                choice_text = match.group(2).strip()
                
                # 정제: 불필요한 문자 제거
                choice_text = re.sub(r'[※].*$', '', choice_text, flags=re.MULTILINE).strip()
                choice_text = choice_text.replace('\\n', '').replace('\n', '').replace('\r', '')
                choice_text = re.sub(r'\s+', ' ', choice_text).strip()
                
                if choice_text and label not in pairs:  # 중복 방지
                    pairs[label] = choice_text
                    print(f"[Debug] 선택지 추출 성공: {label}) {choice_text}")
        
        # 🔍 대안 방법: 간단한 split 방식도 시도
        if not pairs:
            print(f"[Debug] 정규식 실패, 대안 파싱 시도...")
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                # A) 로 시작하는 라인 찾기
                if re.match(r'^[A-D]\s*\)', line):
                    match = re.match(r'^([A-D])\s*\)\s*(.+)$', line)
                    if match:
                        label = match.group(1).upper()
                        choice_text = match.group(2).strip()
                        
                        # 정제
                        choice_text = re.sub(r'[※].*$', '', choice_text).strip()
                        
                        if choice_text:
                            pairs[label] = choice_text
                            print(f"[Debug] 대안 파싱 성공: {label}) {choice_text}")
        
        print(f"[Debug] 최종 추출된 선택지: {pairs}")
        return pairs
    





    

    # --- 여기부터 삽입 (run_intent_gate 호출 전에) ---
    choice_letter = _parse_abcd_answer(query)
    if choice_letter:
        options = _extract_last_choices(session_history)
        if options.get(choice_letter):
            resolved = options[choice_letter]
            
            # 🧠 스마트한 기존 질문 추출
            def extract_original_context(history) -> str:
                """이전 대화에서 의미있는 원본 질문들을 추출"""
                context_parts = []
                
                # 최근 10개 메시지에서 사용자 질문들 수집
                recent_messages = history.messages[-10:]  # 최근 10개 메시지 확인
                
                for msg in recent_messages:
                    if hasattr(msg, 'type') and msg.type == 'human':
                        content = getattr(msg, 'content', '') or ''
                        
                        # 필터링 조건
                        if ("[clarify]" in content or  # clarify 태그 제외
                            _parse_abcd_answer(content.strip()) or  # A/B/C/D 답변 제외
                            content.strip().lower() in ['네', '아니오', 'yes', 'no']):  # 단순 답변 제외
                            continue
                        
                        # 의미있는 질문이면 추가
                        clean_content = content.strip()
                        if clean_content and clean_content not in context_parts:
                            context_parts.append(clean_content)
                
                return " ".join(context_parts[-2:])  # 최근 2개 질문만 결합
            
            original_context = extract_original_context(session_history)
            
            # 🎯 지능적 쿼리 결합
            if original_context:
                # 중복 단어 제거하면서 결합
                context_words = set(original_context.lower().split())
                resolved_words = set(resolved.lower().split())
                
                # 완전히 다른 내용이면 결합, 비슷한 내용이면 선택지만 사용
                overlap_ratio = len(context_words & resolved_words) / len(context_words | resolved_words)
                
                if overlap_ratio < 0.3:  # 30% 미만 겹치면 결합
                    combined_query = f"{original_context} {resolved}"
                    print(f"[스마트 결합] 기존맥락 + 선택지: '{original_context}' + '{resolved}' = '{combined_query}'")
                else:  # 많이 겹치면 선택지만 사용 (중복 방지)
                    combined_query = resolved
                    print(f"[중복방지] 선택지만 사용: '{resolved}' (기존맥락과 {overlap_ratio:.1%} 유사)")
            else:
                combined_query = resolved
                print(f"[선택지매핑] 기존맥락 없음: '{query}' → '{resolved}'")
            
            query = combined_query
            session_history.add_user_message(f"[clarify]{resolved}")
            skip_add_user_message = True
            skip_intent_gate = True             # ✅ 이번 턴에는 Intent Gate를 다시 돌리지 않겠다
            resolved_choice_text = resolved     # ✅ 나중에 expanded_terms 등에 쓸 수 있게 저장



    # ===== Intent Gate 실행 및 게이트 판정 =====
    intent_eval = run_intent_gate(query, session_history, client, model=LLM_MODEL)




    # ✅ 홈페이지 이동 의도 체크 (재질문 카운팅 제외 + 검색 리셋)
    if intent_eval.get("is_homepage_intent"):
        homepage_message = intent_eval.get("homepage_message", "🏠 홈페이지로 이동합니다.")
        homepage_url = intent_eval.get("homepage_url", HOMEPAGE_URL)

        # 🔄 검색/대화 리셋
        try:
            clear_message_history(session_id)              # 전체 히스토리 삭제
            session_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
            session_history.add_ai_message(homepage_message)
            PRODUCT_CACHE.clear()                          # (선택) 상품 캐시도 초기화
            print("[홈페이지 이동] 검색/세션 리셋 완료")
        except Exception as e:
            print(f"[홈페이지 리셋 오류] {e}")

        return {
            "query": "",                     # 검색어 비움
            "assistant_message": homepage_message,
            "UserMessage": homepage_message,
            "RawContext": [],                # 리셋 이후 빈 컨텍스트
            "results": [],                   # 결과 없음
            "combined_message_text": homepage_message,
            "homepage_url": homepage_url,
            "is_homepage_intent": True,
            "needs_clarification": False,
            "clarification_count": 0,
            "reset": True,                   # ✅ 리셋 플래그
        }
    
    # ✅ 배송/주문 조회 의도 체크 (재질문 카운팅 제외 + 검색 리셋)
    if intent_eval.get("is_delivery_intent"):
        delivery_message = intent_eval.get("delivery_message", "배송 조회 페이지로 이동합니다.")
        delivery_url = intent_eval.get("delivery_url", DELIVERY_INQUIRY_URL)

        # 🔄 검색/대화 리셋
        try:
            clear_message_history(session_id)
            session_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
            session_history.add_ai_message(delivery_message)
            PRODUCT_CACHE.clear()
            print("[배송조회 이동] 검색/세션 리셋 완료")
        except Exception as e:
            print(f"[배송 리셋 오류] {e}")

        return {
            "query": "",
            "assistant_message": delivery_message,
            "UserMessage": delivery_message,
            "RawContext": [],
            "results": [],
            "combined_message_text": delivery_message,
            "delivery_url": delivery_url,
            "is_delivery_intent": True,
            "needs_clarification": False,
            "clarification_count": 0,
            "reset": True,
        }

    # ✅ 인사/소개 의도 체크 (재질문 카운팅 제외)
    if intent_eval.get("is_greeting_intent"):
        greeting_message = intent_eval.get("greeting_message", "안녕하세요!")
        
        # 세션에 AI 메시지 저장
        try:
            session_history.add_ai_message(f"[GREETING_INTENT]{greeting_message}")
        except Exception as e:
            print(f"[인사 메시지 저장 오류] {e}")
        
        return {
            "query": query,
            "assistant_message": greeting_message,
            "UserMessage": greeting_message,
            "RawContext": [m.content for m in session_history.messages],
            "results": [],
            "combined_message_text": greeting_message,
            "is_greeting_intent": True,
            "needs_clarification": False,
            "clarification_count": 0,
        }




    avg_score = float(intent_eval.get("avg_score", 0.0))
    route = (intent_eval.get("route") or "").lower()
    completion_percent = intent_eval.get("completion_percent", 0)
    print(f"[IntentGate] avg={avg_score:.2f} route={route}"
        f"D={intent_eval.get('direct_match')} C={intent_eval.get('context_match')} "
        f"A={intent_eval.get('attribute_match')} B={intent_eval.get('brand_match')}")
    

    # 🔄 재질문 횟수 추적 및 누적 쿼리 관리 - 완전 개선
    clarification_count = 0
    user_query_parts = []

    # def _is_short_answer(text: str) -> bool:
    #     """A/B/숫자/네/아무거나 같은 단답형 필터"""
    #     t = (text or "").strip()
    #     if not t:
    #         return False

    #     short_tokens = {
    #         "a", "b", "c", "d",
    #         "A", "B", "C", "D",
    #         "네", "예", "아니오", "아니요",
    #         "응", "웅", "ㅇㅇ", "ㅇㅋ", "ㅇㅋㅇㅋ",
    #         "상관없어요", "상관 없어", "아무거나", "다 좋아",
    #     }
    #     if t in short_tokens:
    #         return True

    #     # 숫자 하나/두 개 같은 선택지
    #     if len(t) <= 2 and t.isdigit():
    #         return True

    #     # 너무 짧은 건 검색어로 안 씀
    #     if len(t) < 4:
    #         return True

    #     return False

    def sanitize_user_fragment(text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r'(찾아\s*줘|찾아줘|찾아|알려줘|추천해줘)\b', '', t)
        t = re.sub(r'\s{2,}', ' ', t)
        return t.strip()

    # 세션 히스토리에서 재질문 횟수와 사용자 질문들 수집
    try:
        print(f"[디버그] 전체 메시지 수: {len(session_history.messages)}")
        for i, msg in enumerate(session_history.messages[-20:]):  # 최근 20개 메시지 확인
            content = getattr(msg, "content", "") or ""
            if not content.strip():
                continue

        
            mtype = getattr(msg, 'type', 'unknown')
            # 🏠 홈페이지 안내 메시지는 재질문 카운트에서 제외
            # ✅ 안내 메시지 그대로 유지 (continue 제거)
            if content.startswith("[HOMEPAGE_INTENT]"):
                print(f"[Context 보존] 홈페이지 안내 유지: '{content[:50]}'")
            elif content.startswith("[DELIVERY_INTENT]"):
                print(f"[Context 보존] 배송조회 안내 유지: '{content[:50]}'")
            elif content.startswith("[GREETING_INTENT]"):
                print(f"[Context 보존] 인사 안내 유지: '{content[:50]}'")



            print(f"[디버그] 메시지 {i}: type={getattr(msg, 'type', 'unknown')}, content='{content[:100]}'")

            # 🎯 clarify 재질문 카운트 (AI INTENT_GATE)
            if mtype == 'ai' and content.startswith("[INTENT_GATE]"):
                try:
                    j = json.loads(content[len("[INTENT_GATE]"):])
                    if (j.get("route") or "").lower() == "clarify":
                        clarification_count += 1
                        print(f"[재질문 감지] {clarification_count}번째 (route=clarify)")
                except Exception:
                    pass
                continue  # INTENT_GATE 전체는 누적쿼리 구성 제외

            # ✅ 사용자 메시지 처리
            if mtype == 'human':
                low = content.lower()

                # 홈페이지/배송/인사/URL 의도 사용자 입력은 누적 제외 (요구사항 유지)
                if (is_homepage_intent(content) or
                    is_delivery_intent(content) or
                    is_greeting_or_intro_intent(content) or
                    "chatmall.kr" in low or "http://" in low or "https://" in low):
                    print(f"[누적쿼리 제외] 네비/배송/인사/URL 사용자 입력: '{content}'")
                    continue

                if "[INTENT_GATE]" in content:
                    continue

                clean_content = sanitize_user_fragment(content)
                if not clean_content:
                    continue

                # if _is_short_answer(clean_content):
                #     print(f"[단답형 제외] '{clean_content}'")
                #     continue

                if len(clean_content) < 50:
                    user_query_parts.append(clean_content)
                    print(f"[사용자 질문 수집] '{clean_content}'")



    except Exception as e:
        print(f"[재질문 추적 오류] {e}")

    # 사용자 질문들을 의미있게 조합 (최신 것을 우선으로)
    if user_query_parts:
        # 중복 제거하면서 순서 유지 (전체)
        unique_parts = []
        for part in user_query_parts:
            if part not in unique_parts:
                unique_parts.append(part)

        # 리스트 그대로 넣기
        combined_query = clean_accumulated_query(unique_parts)
    else:
        combined_query = query

    print(f"[재질문 추적 최종] 감지횟수={clarification_count}, "
          f"사용자질문={user_query_parts}, 누적쿼리='{combined_query}'")





    # 🚀 **핵심 수정**: 바로검색 명령어 우선 확인 (Intent Gate 이전)
    clarify_answer = None
    
    if check_direct_search_command(combined_query if len(user_query_parts) > 1 else query):
        search_query = combined_query if len(user_query_parts) > 1 else query
        print(f"[바로검색] 명령어 감지: '{search_query}' → Intent Gate 건너뛰고 바로 검색")
        query = search_query
        pass  # 바로 검색으로 진행
    else:
        # clarify 답변 수집
        try:
            if isinstance(request, dict):
                clarify_answer = request.get("clarify_answer")
            else:
                clarify_answer = getattr(request, "clarify_answer", None)
        except Exception:
            clarify_answer = None

        # ★ route 우선 적용 (수정)
        if route == "proceed":
            pass  # 다음 단계 진행
        elif clarification_count >= 4:  # 재질문 횟수 4번 초과 시 강제 진행
            print(f"[재질문 제한] {clarification_count}번 재질문 완료 → 누적쿼리로 강제 진행: '{combined_query}'")
            query = combined_query
        # 🔥 수정: route가 clarify면 무조건 재질문 (avg_score 조건 제거)
        elif route == "clarify" and not clarify_answer:
            # 🧠 누적 쿼리로 재평가 (쿼리가 실제로 달라졌을 때만)
            if combined_query != query and len(user_query_parts) > 1:
                print(f"[누적 재평가] 기존 쿼리='{query}' → 누적 쿼리='{combined_query}'")
                # 누적된 쿼리로 다시 intent gate 실행
                intent_eval_combined = run_intent_gate(combined_query, session_history, client, model=LLM_MODEL)
                avg_score_combined = float(intent_eval_combined.get("avg_score", 0.0))
                route_combined = (intent_eval_combined.get("route") or "").lower()
                
                if route_combined == "proceed":
                    print(f"[누적 재평가] clarify→proceed 전환 → 검색 진행")
                    query = combined_query
                    # 재질문 건너뛰고 검색으로 진행
                else:
                    print(f"[누적 재평가] 여전히 부족 avg={avg_score_combined:.3f} → 재질문")
                    # 누적 평가 결과를 사용
                    intent_eval = intent_eval_combined
                    
                    # 재질문 생성
                    followup = intent_eval.get("clarify_question") or "조금만 더 구체화해주실래요?"
                    enhanced = followup

                    return {
                        "query": combined_query,
                        "assistant_message": enhanced,      
                        "UserMessage": enhanced,            
                        "RawContext": [m.content for m in session_history.messages],
                        "results": [],
                        "combined_message_text": enhanced,  
                        "intent_gate": intent_eval,
                        "needs_clarification": True,
                        "completion_percent": intent_eval.get("completion_percent", 0),
                        "clarification_count": clarification_count + 1,
                        "accumulated_query": combined_query
                    }
            else:
                # 다른 재질문 조건들
                followup = intent_eval.get("clarify_question") or "조금만 더 구체화해주실래요?"
                enhanced = followup  # 완성률 메시지 없이 질문만

                return {
                    "query": combined_query,  # 누적 쿼리로 저장
                    "assistant_message": enhanced,      # 👈 프론트가 우선 사용
                    "UserMessage": enhanced,            # 기존 호환
                    "RawContext": [m.content for m in session_history.messages],
                    "results": [],
                    "combined_message_text": enhanced,  # 기존 호환
                    "intent_gate": intent_eval,
                    "needs_clarification": True,
                    "completion_percent": intent_eval.get("completion_percent", 0),
                    "clarification_count": clarification_count + 1,
                    "accumulated_query": combined_query
                }

    # 보강답변이 있으면 재평가
    if clarify_answer:
        query = f"{query} {clarify_answer}".strip()
        try:
            session_history.add_user_message(f"[clarify] {clarify_answer}")
        except Exception:
            pass
        intent_eval = run_intent_gate(query, session_history, client, model=LLM_MODEL)
        avg_score = float(intent_eval.get("avg_score", 0.0))
        route = (intent_eval.get("route") or "").lower()
        print(f"[IntentGate-RE] avg={avg_score:.2f} route={route} "
            f"D={intent_eval.get('direct_match')} C={intent_eval.get('context_match')} "
            f"A={intent_eval.get('attribute_match')} B={intent_eval.get('brand_match')}")
        if route != "proceed" and avg_score < THRESHOLD:
            # 🎯 완성률 정보 포함된 재질문 생성
            followup = intent_eval.get("clarify_question") or "조금만 더 구체화해주실래요?"

            enhanced = followup  # 완성률 메시지 없이 질문만
            return {
                "query": query,
                "UserMessage": enhanced,
                "RawContext": [m.content for m in session_history.messages],
                "results": [],
                "combined_message_text": enhanced,
                "intent_gate": intent_eval,
                "needs_clarification": True,
                "completion_percent": intent_eval.get("completion_percent", 0),
            }









    # ===== 여기까지 내려오면 통과 → 아래 전처리/카테고리/임베딩 검색 계속 =====
    # ===== 통과 시 아래 원래 system_prompt 로직 계속 진행    재질문 END =====

    system_prompt = (
        f"""System:
            당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 물건을 추천해주는 전문가입니다.
            입력 언어가 무엇이든 먼저 한국어로 의미 보존 번역을 수행합니다.
            
            
            [🔥 대화 맥락 강제 반영 원칙 - 절대 준수!]
            1. 이전 대화 이력:
            {conversation_context}
             
            2. 🚨 맥락 반영 강제 규칙 (반드시 실행):
            **STEP 1: 맥락 분석**
            - 이전 대화에서 상품명 추출: 예) "겨울장갑", "여름가방", "운동화" 등
            - 이전 대화에서 조건 추출: 예) "국내산", "저렴한", "브랜드", "작은" 등
            
            **STEP 2: 현재 입력 분석**
            - 현재 입력에 상품명이 있는가? YES/NO
            - 현재 입력이 조건/속성 추가인가? YES/NO
            
            **STEP 3: 강제 결합 (반드시 실행!)**
             - 현재 입력에 상품명 없고 + 이전에 상품명 있음 → **[이전 상품명] + [현재 조건]**
             - 현재 입력이 조건 추가임 → **[이전 상품명] + [이전 조건들] + [현재 조건]**
             - 예시:
               - 이전: "겨울장갑" → 현재: "국내산" → **결과: "겨울장갑 국내산"**
               - 이전: "여름가방" → 현재: "작은거" → **결과: "여름가방 작은"**
               - 이전: "겨울장갑 국내산" → 현재: "저렴한" → **결과: "겨울장갑 국내산 저렴한"**
            
            **🚨 STEP 4: 바로검색 키워드 완전 제거 (절대 필수!)**
             - **금지 키워드**: "나로수", "narosu" 등 바로검색 명령어는 절대 포함 금지!
             - 입력에 바로검색 키워드가 있어도 최종 결과에서는 완전히 제거할 것!
             - 예시: "여성 구두 나로수" → **결과: "여성 구두"** (나로수 완전 삭제)
            
            3. 🎯 최종 검색어 생성 강제 공식:
            - **절대 규칙**: 이전 맥락 무시 금지! 반드시 누적하여 검색어 생성할 것!
            - **절대 규칙**: 바로검색 키워드는 절대 포함 금지!

 
            [전처리 원칙]
            *사용자의 문장을 이해해서 추측해서 문장 생성한다. 질문에 대해서 답을 같이 생성해서 제일 앞에 단어를 붙인다.*

            - 브랜드/스펙/색/규격/수량/가격 등 명시 제약이 있으면 보존.
            2) 한국 쇼핑 맥락의 일반명은 단수 표면형을 우선(복수/어미/조사 제거).
            3) '용' 같은 불용미사/꼬리표는 제거.
            4) 불필요한 구두점은 제거. OR, |, 콤마 대신 **공백 나열**만 사용.
            5) **부정/제외 처리(아주 중요)**:
            - 다음 패턴을 부정 신호로 인식한다: "싫어/싫다/말고/빼고/제외(하고/한)/아닌/제외해줘/빼줘/미포함/제외 부탁" 등.
            - "A 말고 B", "A는 싫고 B", "B 찾는데 A 제외" 형태에서는 **A는 제외(-A), B는 유지**한다.
            - 제외 토큰은 단어·구를 **표준형으로 정규화**하고 **하이픈(-토큰)** 으로 표기한다(예: -호박맛, -딸기, -화이트).
            - 다단 제외가 있으면 공백으로 나열한다: 예) "사과 주스 -호박맛 -배맛".
            - 핵심 품목이 불명확하고 제외만 존재할 경우, **추정하지 말고 원문 유지**(의미 보존 원칙).

            [doc2query: “사람이 검색할 법한 질의” — 형제(같은 레벨) 품목만]
            - 목적: 메인 품목의 동의어/속성 변형/액세서리 제외, **같은 상위 카테고리**를 공유하는 **형제 하위 품목**만 6개 생성
            - 언어/길이: 한국어 우선, 4~28자(공백 포함), 실제 검색창 스타일
            - 생성 대상(허용)
            · 형제 카테고리의 **대체** 품목(상위 카테고리 동일)
            · (선택) 동일 코너에서 함께 **비교되는** 하위 유형
            - 금지(엄격)
            · 메인 품목 명칭 포함 금지
            · 메인 품목의 동의어/옵션/속성 변형/복수형/철자 변형 금지
            · 액세서리/소모품/부품/설치 키트/세트 구성품 금지
            · 브랜드/모델/규격 날조 금지
            · 조사·어미만 다른 중복 변형 금지
            - 품질 기준
            · 실제 쇼핑 맥락에서 **함께 비교·대체**되는 형제를 우선
            - 셀프 체크(생성 직후 자체 필터)
            [ ] 각 질의에 메인 품목 단어가 들어가 있지 않은가?
            [ ] 동의어/옵션/액세서리/소모품/부품이 아닌가?
            [ ] 의미 중복(조사·어미만 차이) 제거했는가?


            [Category Search Text:카테고리 검색용 상품 요약 문장 생성]
                사용자가 찾고자 하는 상품을 이해하고, 카테고리 검색에 최적화된 자연스러운 한국어 문장으로 요약하세요.
                
                **요구사항:**
                1) 길이: 8~16자(공백 포함, 가능하면 12자 내외)
                2) 관형어 1개 이상 포함: 예) 먹는/쓰는/입는/바르는/메는/신는/쓰는(착용) 등등 문장과 자연스럽게.
                3) 브랜드/트렌드/마케팅 단어 금지: “다양한, 트렌드, 제품, 합리적, 프리미엄, 스타일리시, 최적”
                4) 불필요한 종결어미 생략
                5) 핵심 품목 불명확 시: **추정 금지, 원문 유지**
                6) **부정/제외 표현(제외/빼고/말고/without 등) 금지.** 결과 문장은 제외 대상 자체를 언급하지 말 것.
                7) 문장 제일 앞에 너가 추측해서 1등 일것 같은 카테고리 추가. 예시) 화과자, 베이컨, 밀가루

                **올바른 예시:**
                - "여름용 작은 가방" → "여름 작은 가방 입니다"
                - "운동할 때 신는 신발" → "운동할때 신는 신발 입니다"  
                - "겨울 따뜻한 외투" → "겨울 따뜻한 외투 입니다"
                
                **잘못된 예시 (키워드 나열):**
                - "여름 작은 가방 쿨 소재 여성" (X)
                - "운동 신발 편안한" (X)
                - "겨울 외투 따뜻한" (X)

                
            [Category Top3: 쇼핑몰 카테고리 Top3 후보 생성]
                사용자가 찾고자 하는 상품을 이해하고, 국내 쇼핑몰에서 실제로 사용할 법한 카테고리 명칭 Top3를 생성하세요.

                **역할:**
                - 검색 엔진이 카테고리 필터를 걸 수 있도록, "가장 가능성 높은 카테고리 3개"를 점수 순으로 나열합니다.
                - 각 카테고리는 쇼핑몰에서 흔히 쓰는 "대분류>중분류>소분류" 형식을 우선합니다.
                  예) "식품>과자/스낵>감자칩", "패션잡화>가방>크로스백"

                **요구사항:**
                1) 정확히 3개 생성: Top1, Top2, Top3
                2) 서로 다른 카테고리일 것(완전히 동일한 문자열 금지)
                3) 브랜드/트렌드/마케팅 단어 금지: “다양한, 트렌드, 제품, 합리적, 프리미엄, 스타일리시, 최적” 등
                4) 문장형 설명 금지: 카테고리 이름만 사용(“입니다/추천/찾기 좋은/검색어” 등 문장형 금지)
                5) 핵심 품목이 다소 애매할 경우,
                   - 과도하게 세분화된 소분류로 날조하지 말고
                   - 비교적 상위 레벨(대분류/중분류 중심)에서 보수적으로 추론할 것.
                6) 부정/제외 표현(제외/빼고/말고/without 등)은 **카테고리 명칭에는 반영하지 말 것.**

                **예시:**
                - 입력: "여름용 작은 가방"
                  → Category Top3:
                    "패션잡화>가방>크로스백 | 패션잡화>가방>미니백 | 패션잡화>가방>숄더백"


            [출력 규칙(반드시 정확히 준수)]
            오직 세 줄만 출력, 따옴표 포함. 추가 설명/불릿/번호/코드블록 절대 금지.
            Raw Query: "<query>"
            Preprocessed Query: "<전처리된_쿼리(핵심 품목 + 유의미 속성만, ‘용’ 제거 후 표준형)>"
            Category Search Text: "<예상1순위카테고리> <요약 문장>"
            Doc2Query: "<q1> | <q2> | <q3> | <q4> | <q5> | <q6>"
            Category Top3: "<Top1_카테고리> | <Top2_카테고리> | <Top3_카테고리>"

        """    
    )



    # 🔥 맥락 반영 강제 사용자 메시지 구성
    user_message = f"""
        🚨 현재 사용자 입력: "{query}"

        **필수 실행 단계:**
        1. 위 이전 대화 이력에서 상품명/조건 추출
        2. 현재 입력과 이전 맥락을 강제로 결합
        3. 결합된 검색어로 전처리 수행
        4. 🚨 **바로검색 키워드 완전 제거**: "나로수", "narosu" 등은 절대 포함하지 말 것!

        반드시 이전 맥락을 반영한 검색어를 생성하되, 바로검색 키워드는 완전히 제거하세요!
        """

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    llm_response = resp.choices[0].message.content.strip()
    print("[Debug] LLM full response:\n", llm_response)  # ← 여기에!

    def extract_preprocessed(llm_text: str, fallback: str) -> str:
        # 0) 혹시 JSON으로 온 경우 대비
        try:
            obj = json.loads(llm_text)
            v = (obj.get("Preprocessed Query") or
                obj.get("preprocessed_query") or
                obj.get("query_preprocessed"))
            if v and isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass

        # 1) 영문 라벨 (권장 형식)
        m = re.search(r'(?i)^\s*preprocessed\s*query\s*:\s*["“]?(.+?)["”]?\s*$', llm_text, re.M)
        if m:
            return m.group(1).strip()

        # 2) 한글 라벨(모델이 한국어로 라벨을 바꿔도 커버)
        for pat in [
            r'^\s*최종\s*검색어\s*[:=]\s*["“]?(.+?)["”]?\s*$',
            r'^\s*전처리된_?쿼리\s*[:=]\s*["“]?(.+?)["”]?\s*$',
        ]:
            m = re.search(pat, llm_text, re.M)
            if m:
                return m.group(1).strip()

        # 3) 마지막 폴백: 따옴표 속 가장 그럴듯한 문자열
        quotes = re.findall(r'["“]([^"\n”]{2,64})["”]', llm_text)
        if quotes:
            # 공백 포함/길이 기준 간단 휴리스틱
            quotes.sort(key=lambda s: (-(' ' in s), -len(s)))
            return quotes[0].strip()

        # 4) 최종 폴백: 원문
        return fallback



    # Category Search Text 추출 함수 사용
    def extract_category_text_new(llm_text: str):
        """LLM 응답에서 Category Search Text를 추출"""
        try:
            # 1) JSON 응답 체크
            if llm_text.strip().startswith('{'):
                obj = json.loads(llm_text)
                text = obj.get("Category Search Text", "")
                if text and isinstance(text, str):
                    return text.strip()
            
            # 2) 라벨 기반 추출
            patterns = [
                r'(?i)^\s*category\s*search\s*text\s*:\s*[""]?(.+?)[""]?\s*$',
                r'^\s*카테고리\s*검색\s*문장\s*[:=]\s*[""]?(.+?)[""]?\s*$'
            ]
            
            for pattern in patterns:
                m = re.search(pattern, llm_text, re.MULTILINE)
                if m:
                    result = m.group(1).strip()
                    print(f"[Debug] Category Search Text 추출 성공: '{result}'")
                    return result
            
            # 3) 세 번째 라인 폴백
            lines = llm_text.strip().split('\n')
            if len(lines) >= 3:
                third_line = lines[2].strip()
                quote_match = re.search(r'[""]([^"\n"]+)[""]', third_line)
                if quote_match:
                    result = quote_match.group(1).strip()
                    print(f"[Debug] Category Search Text 폴백 추출: '{result}'")
                    return result
            
            return ""
            
        except Exception as e:
            print(f"[Error] Category Search Text 추출 오류: {e}")
            return ""
    



    def extract_category_top3(response_text: str):
        """
        LLM 응답 전체 문자열에서
        Category Top3: "<c1> | <c2> | <c3>"
        이 한 줄을 찾아서 [c1, c2, c3] 리스트로 반환
        """
        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        
        cat_line = None
        for line in lines:
            if line.startswith("Category Top3:"):
                cat_line = line
                break

        if cat_line is None:
            return []

        # Category Top3: " ... "
        m = re.search(r'Category Top3:\s*"(.*)"', cat_line)
        if not m:
            return []

        raw = m.group(1)  # "<Top1> | <Top2> | <Top3>" 안쪽 내용
        cats = [c.strip() for c in raw.split("|") if c.strip()]

        # 길이 보정 (3개 미만이면 None 채워넣을 수도 있음)
        while len(cats) < 3:
            cats.append(None)

        top1, top2, top3 = cats[:3]
        return top1, top2, top3



# #####증강 부분#####
#     def extract_doc2query(llm_text: str, min_len: int = 4, max_len: int = 28):


#         def _split_pipes(s: str):
#             return [x.strip() for x in s.split("|") if isinstance(x, str) and x.strip()]

#         def _norm_key(s: str):
#             return re.sub(r"\s+", " ", s.lower()).strip()

#         # 0) 원본 보존 + 선처리
#         text = (llm_text or "").strip()

#         # 1) JSON 우선 파싱 (코드펜스/앞뒤 잡음 제거 + 중괄호 블록만 추출)
#         try:
#             _t = text
#             if _t.startswith("```"):
#                 _t = _t.strip("` \n")
#                 if _t.lower().startswith("json"):
#                     _t = _t[len("json"):].lstrip()
#             i, j = _t.find("{"), _t.rfind("}")
#             if i != -1 and j != -1 and j > i:
#                 _t = _t[i:j+1]
#             obj = json.loads(_t)
#             if "Doc2Query" in obj:
#                 dq = obj["Doc2Query"]
#                 if isinstance(dq, list):
#                     raw = [str(x).strip() for x in dq if str(x).strip()]
#                 elif isinstance(dq, str):
#                     raw = _split_pipes(dq)
#                 else:
#                     raw = []
#                 # 정제
#                 seen, out = set(), []
#                 for q in raw:
#                     if len(q) < min_len or len(q) > max_len:
#                         continue
#                     k = _norm_key(q)
#                     if k in seen:
#                         continue
#                     seen.add(k)
#                     out.append(q)
#                 return out
#         except Exception:
#             pass

#         # 2) 라벨 기반 추출: Doc2Query: "q1 | q2 | q3"
#         m = re.search(r'^\s*Doc2Query\s*:\s*["“]?(.+?)["”]?\s*$', text, re.M | re.I)
#         if m:
#             raw = _split_pipes(m.group(1))
#             seen, out = set(), []
#             for q in raw:
#                 if len(q) < min_len or len(q) > max_len:
#                     continue
#                 k = _norm_key(q)
#                 if k in seen:
#                     continue
#                 seen.add(k)
#                 out.append(q)
#             return out

#         # 3) 폴백: 파이프 구분 라인 자동 감지 (Doc2Query 라벨이 없어도)

#         lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#         candidate = ""
#         for ln in lines:
#             if "|" in ln:
#                 parts = _split_pipes(ln)
#                 if len(parts) >= 2 and len(ln) <= 400:
#                     candidate = ln
#                     break
#         if candidate:
#             raw = _split_pipes(candidate)
#             seen, out = set(), []
#             for q in raw:
#                 if len(q) < min_len or len(q) > max_len:
#                     continue
#                 k = _norm_key(q)
#                 if k in seen:
#                     continue
#                 seen.add(k)
#                 out.append(q)
#             return out

#         # 4) 최종 빈 리스트
#         return []
#     # ===================== Helper: Doc2Query를 Preprocessed에 병합 검색 단어 증강 =====================
#     def merge_doc2_into_preprocessed(pre_q: str, doc2_list, k: int = 2, max_len: int = 120) -> str:
#         """
#         Preprocessed Query 뒤에 Doc2Query 상위 k개를 공백으로 이어붙여 합친다.
#         - 부정(-토큰) 충돌 질의 제외
#         - 중복/공백 정리
#         - 최종 길이 제한
#         """
#         if not isinstance(pre_q, str):
#             pre_q = str(pre_q or "")
#         doc2_list = doc2_list or []

#         # 부정 토큰 수집 (-화이트, -XL 등)
#         minus_terms = {t[1:].strip() for t in pre_q.split() if t.startswith("-") and len(t) > 1}

#         def _clean(s: str) -> str:
#             s = (s or "").strip()
#             s = s.replace("|", " ").replace("\"", " ").replace("“", " ").replace("”", " ")
#             s = s.replace("’", "'").replace("‘", "'")
#             s = " ".join(s.split())
#             return s

#         picked = []
#         for q in doc2_list:
#             if len(picked) >= k:
#                 break
#             cq = _clean(q)
#             if not cq:
#                 continue
#             if len(cq) < 2 or len(cq) > 28:
#                 continue
#             # 부정 토큰 충돌 제외
#             if any(mt and (mt in cq) for mt in minus_terms):
#                 continue
#             # 중복 제거(공백 무시)
#             if any(cq == p or cq.replace(" ", "") == p.replace(" ", "") for p in picked):
#                 continue
#             picked.append(cq)

#         tail = " ".join(picked).strip()
#         merged = (pre_q + " " + tail).strip() if tail else pre_q

#         if len(merged) > max_len:
#             parts = merged.split()
#             cut, cur = [], 0
#             for w in parts:
#                 add = len(w) + (1 if cur > 0 else 0)
#                 if cur + add > max_len:
#                     break
#                 cut.append(w)
#                 cur += add
#             merged = " ".join(cut).strip()
#         return merged







#     # --- Doc2Query 추출 및 정리 (preprocessed_query와 동일한 위치에서 이어서) start---
#     doc2query_list = extract_doc2query(llm_response)            # 리스트 형태
#     doc2query_str  = " | ".join(doc2query_list) if doc2query_list else ""  # 파이프 병합 문자열
#     # 디버그 로그
#     print(f"[Debug] Doc2Query 리스트 -> {doc2query_list if doc2query_list else '[]'}")
#     print(f"[Debug] Doc2Query 문자열 -> '{doc2query_str}'")
#     # --- Doc2Query 추출 및 정리 (preprocessed_query와 동일한 위치에서 이어서) end---


    top1, top2, top3 = extract_category_top3(llm_response)
    category_search_text = extract_category_text_new(llm_response)
    terms = extract_preprocessed(llm_response, query)
    terms = remove_direct_search_keywords(terms)
    preprocessed_query = strip_minus_terms(terms)

    # 🔥 여기에 추가!
    preprocessed_query = remove_direct_search_keywords(preprocessed_query)
    category_search_text = remove_direct_search_keywords(category_search_text)
    print(f"[Debug] 바로검색 키워드 최종 제거: '{preprocessed_query}'")

    # # ===== Doc2Query 상위 2~3개를 Preprocessed Query에 합치기 =====
    # preprocessed_query_after_merge = merge_doc2_into_preprocessed(preprocessed_query, doc2query_list, k=2, max_len=120)

    # print(f"[Debug] Preprocessed+Doc2 합성 -> '{preprocessed_query_after_merge}'  (before='{preprocessed_query}')")


    
    # LLM 전처리된 결과에서 가격 조건 재추출 (SIZE_CONDITION은 무시)
    temp_price = extract_price_condition(preprocessed_query)
    if temp_price and not temp_price.startswith("SIZE_CONDITION_"):
        price_cond = temp_price
        print(f"[Debug] 유효한 가격 조건 재추출: {price_cond}")
    elif temp_price and temp_price.startswith("SIZE_CONDITION_"):
        print(f"[Debug] 크기 조건 감지됨, 기존 가격 조건 유지: {price_cond if price_cond else '제한 없음'}")
    else:
        print(f"[Debug] LLM 전처리된 쿼리에서 가격 조건 없음, 기존 조건 유지: {price_cond if price_cond else '제한 없음'}")

    print(f"[맥락반영검증] 원본 입력: '{query}'")
    print(f"[맥락반영검증] LLM 처리 결과: '{preprocessed_query}'")
    print(f"[맥락반영검증] 맥락 반영 여부: {'✅ 반영됨' if query != preprocessed_query else '❌ 미반영'}")

    if not category_search_text or not isinstance(category_search_text, str) or not category_search_text.strip():
        print(f"[Fallback] Category Search Text가 비어있음, preprocessed_query 사용: '{preprocessed_query}'")
        category_search_text = preprocessed_query

    print("[Debug] Preprocessed Query_Before ->", terms)
    print("[Debug] Preprocessed Query ->", preprocessed_query)
    print("[Debug] Category Search Text ->", category_search_text)


 

    # ================== 벡터검색후 100개 리스트에서 Nano 카테고리 Top3추출 부분 이였는데==================
    # ================== LLM에서 Top1~3 추출후 각각 벡터검색후 30개씩 추출후 합계 90개 리스트에서 Nano 카테고리 Top3 추출 부분 ==================

    cat_match_results = []
    
    # 검색어 유효성 체크
    if not category_search_text or len(category_search_text.strip()) < 2:
        print(f"[Warning] 유효하지 않은 검색어, query로 폴백: '{query}'")
        category_search_text = query
    
    # 요약 문장으로 카테고리 벡터 검색 수행 (get_top_categories 사용)
    try:
        category_names = get_top_categories(top1,top2,top3,category_search_text)  # 상위 3개 카테고리 가져오기
        print(f"[Debug] get_top_categories 결과: {len(category_names)}개")
        
        # 기존 형식에 맞춰 변환
        matches = []
        for i, name in enumerate(category_names):
            matches.append({"name": name, "distance": 0.0})  # distance는 0.0으로 설정
        
        if matches:
            print(f"[Debug] 상위 매치:")
            for i, m in enumerate(matches, 1):
                print(f"  {i}. {m.get('name', 'N/A')}")
        else:
            print(f"[Warning] 매치 결과가 없음")
            
    except Exception as e:
        print(f"[Error] get_top_categories 오류: {e}")
        matches = []
    
    seen_names = set()
    
    # 상위 3개 카테고리 선택 (중복 제거)
    for m in matches[:3]:
        name = m.get("name", "")
        if name and name not in seen_names:
            cat_match_results.append({"input": category_search_text, "matches": [m]})
            seen_names.add(name)
            print(f"\n[CatMatch] '{category_search_text}' → '{name}' 매칭 완료")
            
            if len(cat_match_results) >= 3:  # Top3까지만
                break
    
    # 매칭 결과 확인
    print(f"\n[CatMatch] 총 {len(cat_match_results)}개 카테고리 매칭 완료")

    # 전체 힌트 통합 Global Top3 (중복 제거 완료된 상태)
    global_top3 = []
    for r in cat_match_results:
        if not isinstance(r, dict):
            continue
        for m in (r.get("matches") or []):
            global_top3.append({
                "input": r.get("input", ""),
                "name": m.get("name", ""),
                "distance": float(m.get("distance", 0.0))
            })

    print("\n[CatMatch] 중복 제거된 Global Top3:")
    for idx, r in enumerate(global_top3, 1):
        print(f"  {idx}. {r['input']} → {r['name']}")





    

    # # --- 쿼리 임베딩 (L2 정규화) 카테고리 임베딩---
    q_vec = np.array(embedder.embed_query(preprocessed_query), dtype=np.float32)
    n = np.linalg.norm(q_vec)
    if np.isfinite(n) and n != 0.0:
        q_vec = q_vec / n


    # #증강된 쿼리 입력 (방법1만 적용)
    # q_vec_plus = np.array(embedder.embed_query(preprocessed_query_after_merge), dtype=np.float32)
    # n = np.linalg.norm(q_vec_plus)
    # if np.isfinite(n) and n != 0.0:
    #     q_vec_plus = q_vec_plus / n
    # print(f"[Debug] q_vec dim: {q_vec_plus.shape}, norm: {np.linalg.norm(q_vec_plus):.4f}")




    def _build_info_from_hit(hit):
        e = hit.entity
        
        # 미리 기본값 설정
        preview_url = "https://naver.com"
        option_raw = ""
        option_display = "없음"
        
        # preview_url 생성
        try:
            html_raw = e.get("description", "") or ""
            html_cleaned = clean_html_content(html_raw)
            if isinstance(html_raw, bytes):
                html_raw = html_raw.decode("cp949")
            encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
            preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
        except Exception:
            preview_url = "https://naver.com"

        # 옵션 파싱
        try:
            option_raw = str(e.get("composite_options", "")).strip()
            if option_raw.lower() not in ["", "nan", "none", "없음"]:
                parsed = []
                for line in option_raw.splitlines():
                    try:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            extra = int(float(parts[1]))
                            parsed.append(f"{name}{f' (＋{extra:,}원)' if extra>0 else ''}")
                    except Exception:
                        parsed.append(line.strip())
                option_display = "\n".join(parsed) if parsed else "없음"
        except Exception as err:
            print(f"⚠️ 옵션 파싱 오류: {err}")

        return {
            "상품코드":     str(e.get("product_code", "없음")),
            "제목":        e.get("market_product_name", "제목 없음"),
            "가격":        convert_to_serializable(e.get("market_price", 0)),
            "배송비":      convert_to_serializable(e.get("shipping_fee", 0)),
            "이미지":      e.get("image_url", "이미지 없음"),
            "원산지":      e.get("origin", "정보 없음"),
            "상품링크":    preview_url,
            "옵션":        option_display,
            "조합형옵션":  option_raw,
            "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
            "카테고리":    e.get("category_name", "카테고리 없음"),
            "검색방식":    "벡터검색",
        }
   

    # #################################################################
    # #                         NEW 방법 1 시작                                #
    # #################################################################
    ####메타 데이터의  카테고리와 임베딩 된 카테고리를 서로 매칭 시키기 위한 로직.




    # # 1) 전 카테고리 1000개 벡터 검색 (global_top3 제외)
    # print("\n[방법1/RRF] 1000개 벡터 검색 시작 (방법2 카테고리 제외)")

    # # global_top3에서 제외할 카테고리명 추출
    # excluded_categories = [item.get("name", "") for item in global_top3 if item.get("name")]
    # print(f"[Debug] 제외할 카테고리: {excluded_categories}")

    # 검색 조건 구성
    search_conditions = []
    fuzz_number = 65  # 제목 중복제거용 유사도 기준값


    # 가격/크기 조건 처리
    if price_cond and not price_cond.startswith("SIZE_CONDITION_"):
        search_conditions.append(price_cond)
        print(f"[Debug] Milvus 검색에 가격 조건 적용: {price_cond}")
    elif price_cond and price_cond.startswith("SIZE_CONDITION_"):
        try:
            parts = price_cond.split("_")
            if len(parts) >= 4:
                unit = parts[2]  # "인치"
                operator = parts[3]  # ">="
                value = parts[4]  # "30.0"
                
                # 크기 조건 처리 (이하/이상 조건은 복잡하므로 일단 무시)
                if unit == "인치":
                    size_value = int(float(value))
                    if operator in ["==", "="]:
                        # 정확한 크기만 검색 가능
                        size_condition = f"market_product_name LIKE '%{size_value}인치%'"
                        search_conditions.append(size_condition)
                        print(f"[Debug] 정확한 크기 조건 적용: {size_condition}")
                    else:
                        # 이하/이상/초과/미만 조건은 현재 지원 안함
                        print(f"[Debug] 크기 범위 조건 '{operator} {size_value}인치'는 현재 지원하지 않음 - 무시")
                else:
                    print(f"[Debug] 지원하지 않는 크기 단위: {unit}")
        except Exception as e:
            print(f"[Debug] SIZE_CONDITION 파싱 오류: {e}")
            print(f"[Debug] SIZE_CONDITION 무시: {price_cond}")

    # # 카테고리 제외 조건 추가 (Milvus 호환 문법 사용)
    # if excluded_categories:
    #     # 빈 문자열 제거 및 정제
    #     valid_categories = [cat.strip() for cat in excluded_categories if cat.strip()]
        
    #     if valid_categories:
    #         # Milvus에서 지원하는 not in 연산자 사용
    #         if len(valid_categories) == 1:
    #             category_exclude_expr = f"category_name != '{valid_categories[0]}'"
    #         else:
    #             category_list = "', '".join(valid_categories)
    #             category_exclude_expr = f"category_name not in ['{category_list}']"
            
    #         search_conditions.append(category_exclude_expr)    # 최종 검색 조건 생성
    final_search_expr = " && ".join(search_conditions) if search_conditions else None

    print(f"[Debug] 방법1 검색 조건: {final_search_expr}")




    vector_hits_1000 = collection.search(
        data=[q_vec],
        anns_field="emb",
        param={"metric_type":"L2","params":{"nprobe":128}},
        limit=1000,
        expr=final_search_expr,  # 가격 조건 추가
        output_fields=[
            "product_code","category_code","category_name","market_product_name",
            "market_price","shipping_fee","shipping_type","max_quantity",
            "composite_options","image_url","manufacturer","model_name",
            "origin","keywords","description","return_shipping_fee",
        ]
    )










    # 검색 결과 -> dict 리스트
    # 상품 제목이 fuzz_number 같으면 상품제목 중복제거 추가1
    vector_items = []
    for hits in vector_hits_1000:
        for idx, hit in enumerate(hits):
            item = _build_info_from_hit(hit)
            item["vector_match_score"] = 1000 - idx
            title = item.get("제목", "")
            # difflib → RapidFuzz로 변경
            if any(fuzz.ratio(title, v.get("제목", "")) >= fuzz_number for v in vector_items):
                continue

            vector_items.append(item)

    print(f"[방법1/RRF] 벡터 검색 1000개 완료: {len(vector_items)}개")

    # ---------- 시즌 필터링 적용 ----------
    if season != "미정":
        original_count = len(vector_items)
        vector_items = season_filter_items(vector_items, season)
        print(f"[시즌필터] {season} 시즌에 맞는 상품으로 필터링: {original_count}개 → {len(vector_items)}개")

    # ---------- 직접매칭(문자열) 점수: '제목'만 ----------
    ds = Ranker_DirectSearch()
    for it in vector_items:
        it["direct_title_score"] = ds.score_text(preprocessed_query, it.get("제목", ""))

    # ---------- 순위를 만들어야 RRF를 적용할 수 있음 ----------
    def make_rank(items, score_key):
        # 점수>0 인 것만 순위화 (1등=1)
        ranked = sorted(items, key=lambda x: x.get(score_key, 0) or 0, reverse=True)
        rank = {}
        r = 1
        for it in ranked:
            sc = it.get(score_key, 0) or 0
            if sc <= 0:
                continue
            code = it.get("상품코드")
            if code and code not in rank:
                rank[code] = r                                                                                                                                                                                                                                                                                                                                                                                                                                  
                r += 1
        return rank

    # 벡터도 "순위"만 사용 (값 자체 가중치는 안 씀)
    vector_rank = {
        it.get("상품코드"): i + 1
        for i, it in enumerate(sorted(vector_items, key=lambda x: x.get("vector_match_score", 0), reverse=True))
    }
    title_rank = make_rank(vector_items, "direct_title_score")

    BASE_K = 60

    def _safe_rrf_from_rank(rank: Optional[int], k: int = BASE_K) -> float:
        """정수 순위 → RRF 점수(없으면 0)"""
        return 0.0 if rank is None else (1.0 / (k + rank))



    def _rrf_to_1000(rrf_val: float, k: int = BASE_K) -> float:
        """RRF 값을 1등 기준으로 1000점 환산(로그용)"""
        if rrf_val <= 0:
            return 0.0
        rrf_best = 1.0 / (k + 1)  # 1등일 때의 RRF
        return 1000.0 * (rrf_val / rrf_best)

    # 프리-RRF 로깅
    print("\n[방법1/RRF] RRF 점수 계산 전 결과:")
    for idx, it in enumerate(vector_items[:10], 1):  # 상위 10개 출력
        print(
            f"{idx}. {it['제목']} | {it['카테고리']} | "
            f"벡터 점수: {it.get('vector_match_score', 0)} | "
            f"제목 직접 점수: {it.get('direct_title_score', 0)}"
        )

    # 최종 RRF 계산: vec / title 두 축만 사용
    for it in vector_items:
        # 벡터/제목 각 축의 RRF 점수
        code = it.get("상품코드")
        rv = vector_rank.get(code)         # 벡터 순위(정수)
        rt = title_rank.get(code)          # 제목 직접 순위(정수)
        rrf_vec   = _safe_rrf_from_rank(rv, BASE_K)
        rrf_title = _safe_rrf_from_rank(rt, BASE_K)

        # 저장 + 최종 평균 (벡터 + 직접매칭만)
        it["rrf_vec"]   = rrf_vec
        it["rrf_title"] = rrf_title
        it["rrf_all"]   = (rrf_vec + rrf_title) / 2

        # (로그용) 1000점 환산도 함께 저장
        it["vecScore1000"]   = _rrf_to_1000(rrf_vec, BASE_K)
        it["titleScore1000"] = _rrf_to_1000(rrf_title, BASE_K)

    # 3) RRF 점수 계산 후 결과 출력
    # print("\n[방법1/RRF] RRF 점수 계산 후 결과(샘플):")
    for idx, item in enumerate(vector_items[:10], 1):
        code = item.get("상품코드")
        rv = vector_rank.get(code); rt = title_rank.get(code)


    # ---------- 최종 정렬 ----------
    final_sorted = sorted(vector_items, key=lambda x: x["rrf_all"], reverse=True)

    final_results, seen_codes = [], set()
    for it in final_sorted:
        if len(final_results) >= 40:
            break
        code = it.get("상품코드")
        if code and code not in seen_codes:
            seen_codes.add(code)
            final_results.append(it)

    method1_all_sets = final_results  # 리스트 그대로 (40개 아이템)
    print(f"\n[방법1/RRF] 상위 {len(method1_all_sets)}개 선택 완료")

    if method1_all_sets:
        print("\n[방법1/RRF] 세트1 미리보기 (RRF최종 점수로 Sort완료)")
        for idx, it in enumerate(method1_all_sets[:10], 1):
            print(
                f"{idx}. {it['제목']} | {it['카테고리']} | {it['가격']:,}원 | "
                f"vecRRF={it['rrf_vec']:.6f} | titleRRF={it['rrf_title']:.6f} | "
                f"vec≈{it['vecScore1000']:.1f} / title≈{it['titleScore1000']:.1f}| "
                f"RRF(mean-2)={it['rrf_all']:.6f}"
            )










    # #################################################################
    # #                         NEW 방법 1 끝                             #
    # #################################################################

    # #################################################################
    # #                      NEW 방법 2 Start                          #
    # #################################################################




    print("\n[방법2] 카테고리별 검색 시작")

    def search_by_category_method2(category: str, size: int = 50) -> List[Dict]:
        """지정된 카테고리에 대해 벡터+직접 검색 수행 (방법2용)"""
        try:
            # 검색 조건 구성
            search_conditions = []
            
            # 1. 카테고리 조건 (정확 매칭)
            search_conditions.append(f"category_name == '{category}'")
            
            # 2. 가격/크기 조건 처리 
            if price_cond and not price_cond.startswith("SIZE_CONDITION_"):
                search_conditions.append(price_cond)
                print(f"[Debug] 카테고리별 검색에 가격 조건 적용: {price_cond}")
            elif price_cond and price_cond.startswith("SIZE_CONDITION_"):
                # SIZE_CONDITION을 제목 검색 조건으로 변환
                try:
                    parts = price_cond.split("_")
                    if len(parts) >= 4:
                        unit = parts[2]  # "인치"
                        operator = parts[3]  # "<=", ">=", 등
                        value = parts[4]  # "30.0"
                        
                        if unit == "인치":
                            size_value = int(float(value))
                            if operator in ["==", "="]:
                                # 정확한 크기만 검색 가능
                                size_condition = f"market_product_name LIKE '%{size_value}인치%'"
                                search_conditions.append(size_condition)
                                print(f"[Debug] 카테고리별 정확한 크기 조건 적용: {size_condition}")
                            else:
                                # 이하/이상 조건은 현재 지원 안함
                                print(f"[Debug] 카테고리별 크기 범위 조건 '{operator} {size_value}인치'는 지원하지 않음 - 무시")
                except Exception as e:
                    print(f"[Debug] 카테고리별 SIZE_CONDITION 파싱 오류: {e}")

            # 최종 검색 조건 생성
            final_expr = " && ".join(f"({cond})" for cond in search_conditions)

            # 벡터 검색 실행
            vector_hits = collection.search(
                data=[q_vec],
                anns_field="emb",
                param={"metric_type":"L2","params":{"nprobe":128}},
                limit=size,
                expr=final_expr,
                output_fields=[
                    "product_code", "category_name", "market_product_name",
                    "market_price", "shipping_fee", "shipping_type", "max_quantity",
                    "composite_options", "image_url", "manufacturer", "model_name",
                    "origin", "keywords", "description", "return_shipping_fee",
                ]
            )

            # 검색 결과 -> dict 리스트
            # 상품 제목이 fuzz_number 같으면 상품제목 중복제거 추가2
            items = []
            for hits in vector_hits:
                for idx, hit in enumerate(hits):
                    item = _build_info_from_hit(hit)
                    item["vector_match_score"] = size - idx


                    title = item.get("제목", "")
                    # difflib → RapidFuzz로 변경
                    if any(fuzz.ratio(title, v.get("제목", "")) >= fuzz_number for v in items):
                        continue

                    items.append(item)

                   

            # 직접 매칭 점수 계산
            ds = Ranker_DirectSearch()
            for it in items:
                it["direct_title_score"] = ds.score_text(preprocessed_query, it.get("제목", ""))

            # RRF 계산 (벡터 + 직접매칭만)
            vector_rank = {it.get("상품코드"): i + 1 for i, it in enumerate(
                sorted(items, key=lambda x: x.get("vector_match_score", 0), reverse=True)
            )}
            title_rank = make_rank(items, "direct_title_score")

            # 최종 RRF 점수 계산
            for it in items:
                code = it.get("상품코드")
                rv = vector_rank.get(code)
                rt = title_rank.get(code)
                
                rrf_vec = _safe_rrf_from_rank(rv, BASE_K)
                rrf_title = _safe_rrf_from_rank(rt, BASE_K)
                
                it["rrf_vec"] = rrf_vec
                it["rrf_title"] = rrf_title
                it["rrf_all"] = (rrf_vec*3 + rrf_title) / 4
                it["검색방식"] = f"카테고리검색_{category}"

            # 정렬 및 중복 제거
            results = []
            seen = set()
            for it in sorted(items, key=lambda x: x["rrf_all"], reverse=True):
                code = it.get("상품코드")
                if code not in seen:
                    seen.add(code)
                    results.append(it)

            # 시즌 필터링 적용
            if season != "미정":
                original_count = len(results)
                results = season_filter_items(results, season)
                print(f"[방법2/{category}] {season} 시즌 필터링: {original_count}개 → {len(results)}개")

            return results

        except Exception as e:
            print(f"⚠️ 카테고리 {category} 검색 중 오류: {str(e)}")
            return []

    # global_top3에서 바로 카테고리명 사용
    top1_category = global_top3[0]["name"] if len(global_top3) > 0 else None
    top2_category = global_top3[1]["name"] if len(global_top3) > 1 else None  
    top3_category = global_top3[2]["name"] if len(global_top3) > 2 else None

    print(f"[방법2] Top1 카테고리: '{top1_category}' ")
    print(f"[방법2] Top2 카테고리: '{top2_category}' ")
    print(f"[방법2] Top3 카테고리: '{top3_category}' ")

    # 각 카테고리별로 미리 40개씩 뽑아서 리스트 생성
    top1_list = []
    top2_list = []
    top3_list = []

    # Top1 카테고리에서 40개
    if top1_category:
        top1_results = search_by_category_method2(top1_category, size=40)
        top1_list = top1_results[:40]  # 상위 40개만
        print(f"[방법2] Top1 ({top1_category}): {len(top1_results)}개 중 40개 선택")

    # Top2 카테고리에서 40개
    if top2_category:
        top2_results = search_by_category_method2(top2_category, size=40)
        top2_list = top2_results[:40]  # 상위 40개만
        print(f"[방법2] Top2 ({top2_category}): {len(top2_results)}개 중 40개 선택")

    # Top3 카테고리에서 40개
    if top3_category:
        top3_results = search_by_category_method2(top3_category, size=40)
        top3_list = top3_results[:40]  # 상위 40개만
        print(f"[방법2] Top3 ({top3_category}): {len(top3_results)}개 중 40개 선택")





    # ===== 방법2 구성 설정 =====
    METHOD2_TOP1_COUNT = 4  # Top1 카테고리에서 가져올 개수 (3 → 4개)
    METHOD2_TOP2_COUNT = 4  # Top2 카테고리에서 가져올 개수 (3 → 4개)  
    METHOD2_TOP3_COUNT = 3  # Top3 카테고리에서 가져올 개수 (2 → 3개)
    METHOD2_ITEMS_PER_SET = METHOD2_TOP1_COUNT + METHOD2_TOP2_COUNT + METHOD2_TOP3_COUNT  # 세트당 총 11개
    METHOD2_TOTAL_FETCH = 40  # 각 카테고리에서 미리 가져올 총 개수

    print(f"[방법2 설정] Top1:{METHOD2_TOP1_COUNT}개 + Top2:{METHOD2_TOP2_COUNT}개 + Top3:{METHOD2_TOP3_COUNT}개 = 총 {METHOD2_ITEMS_PER_SET}개/세트")


    # 방법2 결과를 세트로 구성 (3:3:2 비율)
    method2_all_sets = []

    # 사이클 수 계산
    max_cycles = min(
        len(top1_list) // METHOD2_TOP1_COUNT if METHOD2_TOP1_COUNT > 0 else 0,
        len(top2_list) // METHOD2_TOP2_COUNT if METHOD2_TOP2_COUNT > 0 else 0,
        len(top3_list) // METHOD2_TOP3_COUNT if METHOD2_TOP3_COUNT > 0 else 0,
        10  # 최대 10사이클
    )


    
    print(f"[방법2] 최대 {max_cycles}개 사이클 생성 가능")

    for cycle in range(max_cycles):
        cycle_items = []
        
        # Top1에서 3개 (cycle*3 인덱스부터)
        start_idx_top1 = cycle * METHOD2_TOP1_COUNT
        if start_idx_top1 + METHOD2_TOP1_COUNT <= len(top1_list):
            cycle_items.extend(top1_list[start_idx_top1:start_idx_top1 + METHOD2_TOP1_COUNT])
        elif start_idx_top1 < len(top1_list):
            cycle_items.extend(top1_list[start_idx_top1:])
        
        # Top2에서 3개 (cycle*3 인덱스부터)
        start_idx_top2 = cycle * METHOD2_TOP2_COUNT
        if start_idx_top2 + METHOD2_TOP2_COUNT <= len(top2_list):
            cycle_items.extend(top2_list[start_idx_top2:start_idx_top2 + METHOD2_TOP2_COUNT])
        elif start_idx_top2 < len(top2_list):
            cycle_items.extend(top2_list[start_idx_top2:])
        
        # Top3에서 2개 (cycle*2 인덱스부터)
        start_idx_top3 = cycle * METHOD2_TOP3_COUNT
        if start_idx_top3 + METHOD2_TOP3_COUNT <= len(top3_list):
            cycle_items.extend(top3_list[start_idx_top3:start_idx_top3 + METHOD2_TOP3_COUNT])
        elif start_idx_top3 < len(top3_list):
            cycle_items.extend(top3_list[start_idx_top3:])
        
        # 세트 추가 조건
        if len(cycle_items) == METHOD2_ITEMS_PER_SET:  # 정확히 8개인 경우
            method2_all_sets.append(cycle_items)
            print(f"[방법2] 사이클 {cycle+1}: 완전한 세트 ({METHOD2_ITEMS_PER_SET}개) 추가")
        elif len(cycle_items) >= METHOD2_ITEMS_PER_SET - 2:  # 최소 6개 이상이면 추가
            method2_all_sets.append(cycle_items)
            print(f"[방법2] 사이클 {cycle+1}: 부분 세트 ({len(cycle_items)}개) 추가")

    print(f"\n[방법2] 총 {len(method2_all_sets)}개 세트 생성됨")

    

    # 첫 번째 세트 미리보기
    if method2_all_sets:
        print(f"\n[방법2] 첫 번째 세트 미리보기 (총 {len(method2_all_sets[0])}개):")
        for idx, item in enumerate(method2_all_sets[0], 1):
            source = ""
            if idx <= METHOD2_TOP1_COUNT:
                source = f"Top1-{idx}"
            elif idx <= METHOD2_TOP1_COUNT + METHOD2_TOP2_COUNT:
                source = f"Top2-{idx - METHOD2_TOP1_COUNT}"
            else:
                source = f"Top3-{idx - METHOD2_TOP1_COUNT - METHOD2_TOP2_COUNT}"
            
            print(f"  {idx}. [{source}] {item['제목']} | {item['카테고리']} | {item['가격']:,}원")
    



    
    # #################################################################
    # #                         NEW 방법 2 end                          #
    # #################################################################


    # ===== 15개 후보 생성: 방법2(8) + 방법1(7) =====
    METHOD1_ITEMS_COUNT = 4  # 방법1에서 가져올 개수 (7 → 4개)

    # 방법2 첫 세트 11개
    method2_top11 = method2_all_sets[0] if method2_all_sets else []
    if len(method2_top11) > METHOD2_ITEMS_PER_SET:
        method2_top11 = method2_top11[:METHOD2_ITEMS_PER_SET]
    print(f"[15개 후보] 방법2 첫 세트: {len(method2_top11)}개")

    # 방법1 상위 4개 (안전하게 처리)
    if isinstance(method1_all_sets, list):
        method1_top4 = method1_all_sets[:METHOD1_ITEMS_COUNT]
    else:
        method1_top4 = []
    print(f"[15개 후보] 방법1 RRF 상위: {len(method1_top4)}개")

    # 11+4=15 합치고 중복 제거(상품코드랑 상품제목을 같이 봄)
    combined_15 = method2_top11 + method1_top4
    seen_codes_15, unique_15 = set(), []
    for item in combined_15:
        c = (item.get("상품코드") or "").strip().upper()
        t = (item.get("제목") or "").strip()
        if not c or not t:
            continue     

        # 제목 유사도 중복 체크(기존 fuzz_number 임계값 사용)
        is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_15)
        if c in seen_codes_15 or is_title_dup:
            continue

        seen_codes_15.add(c)
        unique_15.append(item)
    print(f"[15개 후보] 중복 제거 후: {len(unique_15)}개 (목표 15개)")





    # 🔥 백필: 15개 미달 시 방법2 Top1 카테고리 다음 순위로 채우기
    method1_next_idx = METHOD1_ITEMS_COUNT  # 4 (다음 세트에서 사용할 시작 인덱스 추적)
    top1_next_idx = METHOD2_TOP1_COUNT      # 4 (Top1에서 다음 사용할 인덱스)

    if len(unique_15) < 15 and len(top1_list) > top1_next_idx:
        backfill_needed = 15 - len(unique_15)
        print(f"[15개 후보] {backfill_needed}개 부족 → 방법2 Top1 다음 순위로 백필")
        
        backfilled_count = 0
        
        for item in top1_list[top1_next_idx:]:
            if backfilled_count >= backfill_needed:
                break
            c = (item.get("상품코드") or "").strip().upper()
            t = (item.get("제목") or "").strip()
            if not c or not t:
                continue

            # 제목 유사도 중복 체크(기존 fuzz_number 임계값 사용)
            is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_15)
            if c in seen_codes_15 or is_title_dup:
                continue

            seen_codes_15.add(c)
            unique_15.append(item)
            backfilled_count += 1
            top1_next_idx += 1
        
        print(f"[15개 후보] 백필 완료: {backfilled_count}개 추가 (Top1 인덱스: {METHOD2_TOP1_COUNT} → {top1_next_idx})")

    print(f"[15개 후보] 최종: {len(unique_15)}개")


    # ===== LLM 리랭킹: 15개 → 상위 10개 =====
    def rerank_15_to_10(products_15, ranking_query: str, top1: str = None):
        if len(products_15) <= 10:
            return products_15[:10]

        # 제목만 추출해서 LLM에 전달 (인덱스 유지)
        products_text = "\n".join([
        f"{i}. {p.get('제목', '제목없음')}"

            for i, p in enumerate(products_15)
        ]) 
        
        ranking_prompt = f"""사용자 검색: "{ranking_query}"

        상품 목록:
        {products_text}

        지시사항: 상품제목인 위 {len(products_15)}개 중 사용자 검색과 추측한 카테고리인 {top1} 에 맞는 최적의 상위 10개 인덱스를 콤마로만 출력하세요.
        예: 2,0,5,1,8,3,7,4,9,6
        답변:"""
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role":"system","content":"Respond with ONLY 10 numbers (0-based), comma-separated. No other text."},
                    {"role":"user","content":ranking_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            txt = resp.choices[0].message.content.strip()
            nums = re.findall(r'\d+', txt)
            idxs = [int(x) for x in nums[:10] if 0 <= int(x) < len(products_15)]
            picked, used = [], set()
            for i in idxs:
                if i not in used:
                    picked.append(products_15[i]); used.add(i)
            # 부족하면 앞에서 채우기
            for i, p in enumerate(products_15):
                if len(picked) >= 10:
                    break
                if i not in used:
                    picked.append(p); used.add(i)
            return picked[:10]
        except Exception as e:
            print(f"[LLM 리랭킹 오류] {e} → 원본 상위 10개 사용")
            return products_15[:10]

    # 리랭킹 쿼리 선택
    if preprocessed_query and len(preprocessed_query.strip()) > 2:
        ranking_query = preprocessed_query
    elif 'combined_query' in locals() and combined_query and combined_query != query:
        ranking_query = combined_query
    else:
        ranking_query = query

    # 🔥 리랭킹 전 15개 상품 목록 출력 (세트 1)
    print(f"\n{'='*60}")
    print(f"📋 [세트 1 - 리랭킹 전] 15개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(unique_15, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")


    # ===== 세트2: 방법2(4~6등 × 3카테고리) + 방법1(8~14등) = 15개 =====
    set2_items = []
    
    # 방법2: Top1/Top2/Top3에서 두 번째 구간 가져오기
    # Top1: (1*3)~(2*3) = 3~6 인덱스 → 4~6등
    set2_top1_start = METHOD2_TOP1_COUNT * 1
    set2_top1_end = METHOD2_TOP1_COUNT * 2
    if len(top1_list) >= set2_top1_end:
        set2_items.extend(top1_list[set2_top1_start:set2_top1_end])
    
    # Top2: (1*3)~(2*3) = 3~6 인덱스 → 4~6등
    set2_top2_start = METHOD2_TOP2_COUNT * 1
    set2_top2_end = METHOD2_TOP2_COUNT * 2
    if len(top2_list) >= set2_top2_end:
        set2_items.extend(top2_list[set2_top2_start:set2_top2_end])
    
    # Top3: (1*2)~(2*2) = 2~4 인덱스 → 3~4등
    set2_top3_start = METHOD2_TOP3_COUNT * 1
    set2_top3_end = METHOD2_TOP3_COUNT * 2
    if len(top3_list) >= set2_top3_end:
        set2_items.extend(top3_list[set2_top3_start:set2_top3_end])
    
    # 🔥 방법1: 세트1에서 사용한 다음부터 7개 가져오기
    set2_m1_start = method1_next_idx  # 세트1이 사용한 다음 인덱스
    set2_m1_end = set2_m1_start + METHOD1_ITEMS_COUNT
    if len(method1_all_sets) >= set2_m1_end:
        set2_items.extend(method1_all_sets[set2_m1_start:set2_m1_end])
        method1_next_idx = set2_m1_end  # 다음 세트를 위해 업데이트
    
    print(f"[세트 2] 방법1 사용 구간: [{set2_m1_start}:{set2_m1_end}] (인덱스)")
    
    # 중복 제거
    seen_codes_set2, unique_set2 = set(), []
    for item in set2_items:
        c = (item.get("상품코드") or "").strip().upper()
        t = (item.get("제목") or "").strip()
        if not c or not t:
            continue
        is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_set2)
        if c in seen_codes_set2 or is_title_dup:
            continue
        seen_codes_set2.add(c)
        unique_set2.append(item)
    
    print(f"[세트 2] 방법2({METHOD2_ITEMS_PER_SET}개) + 방법1({METHOD1_ITEMS_COUNT}개) = {len(unique_set2)}개 후보 생성")
    
    # 🔥 백필: 15개 미달 시 방법2 Top1 카테고리 다음 순위로 채우기
    if len(unique_set2) < 15 and len(top1_list) > top1_next_idx:
        backfill_needed = 15 - len(unique_set2)
        print(f"[세트 2] {backfill_needed}개 부족 → 방법2 Top1 다음 순위로 백필")
        
        backfilled_count = 0
        
        for item in top1_list[top1_next_idx:]:
            if backfilled_count >= backfill_needed:
                break
            c = (item.get("상품코드") or "").strip().upper()
            t = (item.get("제목") or "").strip()
            if not c or not t:
                continue
            is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_set2)
            if c in seen_codes_set2 or is_title_dup:
                continue
            seen_codes_set2.add(c)
            unique_set2.append(item)
            backfilled_count += 1
            top1_next_idx += 1
        
        print(f"[세트 2] 백필 완료: {backfilled_count}개 추가 → 총 {len(unique_set2)}개")
        print(f"[세트 2] Top1 다음 시작 인덱스: {top1_next_idx}")
    
    # 🔥 리랭킹 전 15개 상품 목록 출력 (세트 2)
    print(f"\n{'='*60}")
    print(f"📋 [세트 2 - 리랭킹 전] 15개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(unique_set2, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")


    # ===== 세트3: 방법2(7~9등 × 3카테고리) + 방법1(15~21등) = 15개 =====
    set3_items = []
    
    # 방법2: Top1/Top2/Top3에서 세 번째 구간 가져오기
    # Top1: (2*3)~(3*3) = 6~9 인덱스 → 7~9등
    set3_top1_start = METHOD2_TOP1_COUNT * 2
    set3_top1_end = METHOD2_TOP1_COUNT * 3
    if len(top1_list) >= set3_top1_end:
        set3_items.extend(top1_list[set3_top1_start:set3_top1_end])
    
    # Top2: (2*3)~(3*3) = 6~9 인덱스 → 7~9등
    set3_top2_start = METHOD2_TOP2_COUNT * 2
    set3_top2_end = METHOD2_TOP2_COUNT * 3
    if len(top2_list) >= set3_top2_end:
        set3_items.extend(top2_list[set3_top2_start:set3_top2_end])
    
    # Top3: (2*2)~(3*2) = 4~6 인덱스 → 5~6등
    set3_top3_start = METHOD2_TOP3_COUNT * 2
    set3_top3_end = METHOD2_TOP3_COUNT * 3
    if len(top3_list) >= set3_top3_end:
        set3_items.extend(top3_list[set3_top3_start:set3_top3_end])
    
    # 🔥 방법1: 세트2에서 사용한 다음부터 7개 가져오기
    set3_m1_start = method1_next_idx  # 세트2가 사용한 다음 인덱스
    set3_m1_end = set3_m1_start + METHOD1_ITEMS_COUNT
    if len(method1_all_sets) >= set3_m1_end:
        set3_items.extend(method1_all_sets[set3_m1_start:set3_m1_end])
        method1_next_idx = set3_m1_end  # 업데이트
    
    print(f"[세트 3] 방법1 사용 구간: [{set3_m1_start}:{set3_m1_end}] (인덱스)")
    
    # 중복 제거 (상품코드 + 제목 유사도)
    seen_codes_set3, unique_set3 = set(), []
    for item in set3_items:
        c = (item.get("상품코드") or "").strip().upper()
        t = (item.get("제목") or "").strip()
        if not c or not t:
            continue
        is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_set3)
        if c in seen_codes_set3 or is_title_dup:
            continue
        seen_codes_set3.add(c)
        unique_set3.append(item)
    
    print(f"[세트 3] 방법2({METHOD2_ITEMS_PER_SET}개) + 방법1({METHOD1_ITEMS_COUNT}개) = {len(unique_set3)}개 후보 생성")
    
    # 🔥 백필: 15개 미달 시 방법2 Top1 카테고리 다음 순위로 채우기
    if len(unique_set3) < 15 and len(top1_list) > top1_next_idx:
        backfill_needed = 15 - len(unique_set3)
        print(f"[세트 3] {backfill_needed}개 부족 → Top1에서만 백필")
        backfilled_count = 0
        for item in top1_list[top1_next_idx:]:
            if backfilled_count >= backfill_needed:
                break
            c = (item.get("상품코드") or "").strip().upper()
            t = (item.get("제목") or "").strip()
            if not c or not t:
                continue
            is_title_dup = any(fuzz.ratio(t, v.get("제목", "")) >= fuzz_number for v in unique_set3)
            if c in seen_codes_set3 or is_title_dup:
                continue
            seen_codes_set3.add(c)
            unique_set3.append(item)
            backfilled_count += 1
            top1_next_idx += 1
        
        print(f"[세트 3] 백필 완료: {backfilled_count}개 추가 → 총 {len(unique_set3)}개")
        print(f"[세트 3] Top1 최종 사용 인덱스: {top1_next_idx}")
    
    # 🔥 리랭킹 전 15개 상품 목록 출력 (세트 3)
    print(f"\n{'='*60}")
    print(f"📋 [세트 3 - 리랭킹 전] 15개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(unique_set3, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")


    # =====🎯 세트1/2/3 LLM 리랭킹을 스레드풀로 동시에 실행 =====
    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(rerank_15_to_10, unique_15,   ranking_query, top1)
        f2 = ex.submit(rerank_15_to_10, unique_set2, ranking_query, top1)
        f3 = ex.submit(rerank_15_to_10, unique_set3, ranking_query, top1)

        final_10   = f1.result()
        set2_final = f2.result()
        set3_final = f3.result()

    print(f"[LLM 리랭킹] 세트1/2/3 병렬 리랭킹 완료")

    # 🔥 리랭킹 후 10개 상품 목록 출력 (세트1)
    print(f"\n{'='*60}")
    print(f"🎯 [세트 1 - 리랭킹 후] 10개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(final_10, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")

    # 세트2
    print(f"\n{'='*60}")
    print(f"🎯 [세트 2 - 리랭킹 후] 10개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(set2_final, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")

    # 세트3
    print(f"\n{'='*60}")
    print(f"🎯 [세트 3 - 리랭킹 후] 10개 상품 목록:")
    print(f"{'='*60}")
    for idx, item in enumerate(set3_final, 1):
        print(f"{idx}. {item.get('제목', '제목없음')} | {item.get('카테고리', '')} | {item.get('가격', 0):,}원")
    print(f"{'='*60}\n")

    # ===== 세트1/2/3 결과를 최종 리스트에 합치기 =====
    final_results = []
    final_results.extend(final_10)
    print(f"[세트 1] 리랭킹된 10개 사용 (누적 {len(final_results)}개)")

    final_results.extend(set2_final)
    print(f"[세트 2] 리랭킹 후 10개 추가 (누적 {len(final_results)}개)")

    final_results.extend(set3_final)
    print(f"[세트 3] 리랭킹 후 10개 추가 (누적 {len(final_results)}개)")

    print(f"\n[최종] 총 {len(final_results)}개 완성")



    # 검색 결과가 없으면 메시지 반환
    if len(final_results) == 0:
        nores_msg = "No search results. Please specify your search terms more or search for other conditions."
        try:
            session_history.add_ai_message(nores_msg)
        except Exception:
            pass
        return {
            "query": query,
            "UserMessage": nores_msg,
            "RawContext": [m.content for m in session_history.messages],
            "results": [],
            "combined_message_text": nores_msg,
            "message_history": [
                {"type": type(msg).__name__, "content": getattr(msg, "content", "")}
                for msg in session_history.messages
            ]
        }

















    # LLM 메시지 생성용 텍스트 (방법2 우선 표시)
    products_text = "\n".join([
        f"- 코드: {p['상품코드']} | 제목: {p['제목']} | 가격: {p['가격']:,}원 | 카테고리: {p['카테고리']}"
        for p in final_results[:10]  # 방법2(5개) + 방법1(5개) 순서
    ])

    # 1) 템플릿을 바로 f-string으로
    prompt = f"""
    **무조건 어떤 언어가 들어와도 {target_lang}로만 답변하세요!반드시**

    답변 언어는 무조건 {target_lang}로만 작성해주세요. 다른 언어, 혼합 표현 절대 금지.
    당신은 고객의 니즈를 정확히 파악하는 프리미엄 온라인 쇼핑몰의 VIP 상품 추천 전문가입니다.
    주어진 상품 목록을 바탕으로, 고객이 더 나은 선택을 할 수 있도록 도와주세요.

    [매우 중요]

    - 고객이 원하는 상품을 정확히 찾을 수 있도록, 후보 상품의 특징을 활용해 고객의 선호를 파악하는 질문을 작성하세요.

    후보 상품 목록:
    {products_text}

    - 만약 고객이 "{preprocessed_query}"(을)를 찾고 있다면, 같은 상황에서 함께 고려할 수 있는 다른 상품(예: 비옷을 찾는다고 하면 비옷과 같은 레벨의 제품 우산,장화 등등 이런 제품은 어떤지 물어본다.)도 자연스럽게 “이런 상품도 있으니 어떠세요?” 식으로 한 문장으로 제안해 주세요. (단, 상품 리스트에는 포함하지 마세요.)

    요청사항:
    1) {preprocessed_query}의 문장을 이해하고 분석해서 고객 의도 확인과 선호 파악을 위한 '확인형 질문'을 반드시! 물어보는 답변이면 그 문장에 반드시 답변하세요!


    2) 상품 코드나 구체 모델명/스펙 나열은 금지합니다. 후보에서 추출한 '특징 키워드'만 요약해 언급하세요. preprocessed_query과 관련없는 상품이 있다면 없다고 설명해주는 문장을 생성하세요.
    3) 친근하면서도 전문적인 대화체를 유지하세요.


    **무조건 어떤 언어가 들어와도 {target_lang}로만 답변하세요**
    **무조건 답변은 반드시 한글은 150자 이내 ,영어는 200자 이내로 자세하게 작성하되 요약본으로 답변을 작성하세요.

    """

    print("target_lang->", target_lang)

    # 2) 호출부 동일 (system에 실어 보내는 현재 구조 유지)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )

    txt1 = response.choices[0].message.content or ""
    clean = txt1.strip()
    print(f"[Top2-Stage] 전체 카테고리 추가 질문:\n{clean}\n")

    # 결과 출력
    print(f"\n총 {len(final_results)}개의 상품이 최종 리스트에 저장되었습니다.")







    # ========== 예전 로직 (사용 안 함) ==========
    # def rerank_block_with_llm(block_products, ranking_query: str, block_name: str = ""):
    #     """
    #     block_products: final_results의 일부 (길이 보통 10개)
    #     ranking_query: combined_query / query 그대로 사용
    #     """
    #     print(f"\n{'='*60}")
    #     print(f"🎯 [LLM 리랭킹 블록] {block_name} 상품 수: {len(block_products)}개")
    #     print(f"{'='*60}")
    #
    #     # 상품이 10개 미만이면 그냥 스킵 (원본 유지)
    #     if len(block_products) < 10:
    #         print(f"[LLM 리랭킹] {block_name} 상품 수 부족({len(block_products)}개) → 리랭킹 스킵")
    #         return block_products
    #
    #     # LLM에 넘길용 텍스트 구성
    #     products_for_ranking = [
    #         f"{idx}. {item['제목']}"
    #         for idx, item in enumerate(block_products)
    #     ]
    #
    #     ranking_prompt = f"""사용자 검색: "{ranking_query}"
    #
    #     상품 목록:
    #     {chr(10).join(products_for_ranking)}
    #
    #     지시사항: 위 10개 상품을 사용자 검색 의도에 맞게 재정렬하세요.
    #     응답은 반드시 다음 형식으로만 답변: 0,1,2,3,4,5,6,7,8,9
    #
    #     예시: 2,0,5,1,8,3,7,4,9,6
    #
    #     답변:
    #     """
    #
    #     try:
    #         rerank_response = client.chat.completions.create(
    #             model=LLM_MODEL,
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You must respond with ONLY numbers and commas. No other text allowed."
    #                 },
    #                 {"role": "user", "content": ranking_prompt}
    #             ],
    #             temperature=0.1,
    #             max_tokens=50
    #         )
    #
    #         rerank_order = rerank_response.choices[0].message.content.strip()
    #         print(f"[LLM 리랭킹] {block_name} LLM 응답: '{rerank_order}'")
    #
    #         # '없음' 계열 응답 방어
    #         if "없음" in rerank_order or "해당" in rerank_order or "적절" in rerank_order:
    #             print(f"[LLM 리랭킹] {block_name} '없음' 패턴 감지 → 원본 순서 유지")
    #             raise ValueError("LLM이 '없음' 응답 반환")
    #
    #         numbers_only = re.findall(r'\d+', rerank_order)
    #         print(f"[LLM 리랭킹] {block_name} 추출된 숫자들: {numbers_only}")
    #
    #         # 정확히 10개 숫자인지 확인
    #         if len(numbers_only) != 10:
    #             print(f"[LLM 리랭킹] {block_name} 숫자 개수 오류 ({len(numbers_only)}개) → 원본 순서 유지")
    #             raise ValueError(f"숫자 개수가 10개가 아님: {len(numbers_only)}개")
    #
    #         order_indices = [int(x) for x in numbers_only]
    #
    #         # 0~9 범위 체크
    #         if not all(0 <= x <= 9 for x in order_indices):
    #             print(f"[LLM 리랭킹] {block_name} 숫자 범위 오류 → 원본 순서 유지")
    #             raise ValueError("숫자가 0-9 범위를 벗어남")
    #
    #         # 중복 체크
    #         if len(set(order_indices)) != 10:
    #             print(f"[LLM 리랭킹] {block_name} 중복 숫자 감지 → 원본 순서 유지")
    #             raise ValueError("중복된 숫자 존재")
    #
    #         print(f"[LLM 리랭킹] {block_name} 파싱된 순서: {order_indices}")
    #
    #         reranked_block = [block_products[i] for i in order_indices]
    #         print(f"[LLM 리랭킹] {block_name} 성공적으로 재정렬됨")
    #
    #         # 디버그 출력
    #         print(f"\n{'='*60}")
    #         print(f"🎯 [리랭킹 후] {block_name} 재정렬된 상품 제목:")
    #         print(f"{'='*60}")
    #         for idx, item in enumerate(reranked_block):
    #             print(f"{idx}. {item['제목']}")
    #         print(f"{'='*60}")
    #
    #         return reranked_block
    #
    #     except Exception as e:
    #         print(f"[LLM 리랭킹] {block_name} 오류: {e} → 이 블록은 원본 순서 유지")
    #         return block_products





    # ========== 예전 로직 (사용 안 함) ==========
    # # 🎯 LLM 리랭킹: 상위 30개를 10개씩 3세트로 재정렬
    # # 1) 리랭킹용 쿼리 선택
    # 
    # # ✅ 우선순위: preprocessed_query > combined_query > query
    # if preprocessed_query and len(preprocessed_query.strip()) > 2:
    #     ranking_query = preprocessed_query  # LLM이 정제한 쿼리 (최우선)
    #     query_type = "전처리쿼리"
    # elif len(user_query_parts) > 1 and combined_query != query:
    #     ranking_query = combined_query      # 누적된 대화 맥락
    #     query_type = "누적쿼리"
    # else:
    #     ranking_query = query               # 원본 쿼리 (폴백)
    #     query_type = "원본쿼리"
    #
    # print(f"[LLM 리랭킹] {query_type} 사용: '{ranking_query}'")
    # print(f"[LLM 리랭킹] 비교 - 원본: '{query}' / 누적: '{combined_query}' / 전처리: '{preprocessed_query}'")
    #
    # # 2) 디버깅용: 리랭킹 전 상위 10개만 일단 출력 (원래 하던 것 유지)
    # top_10_products = final_results[:10]
    # print(f"\n{'='*60}")
    # print(f"📋 [리랭킹 전] 상위 10개 상품 제목:")
    # print(f"{'='*60}")
    # for idx, item in enumerate(top_10_products):
    #     print(f"{idx}. {item['제목']}")
    # print(f"{'='*60}")
    #
    # # 3) 세트 나누기
    # block1 = final_results[0:10]   # 1~10위
    # block2 = final_results[10:20]  # 11~20위
    # block3 = final_results[20:30]  # 21~30위
    #
    # # 4) 세트별 리랭킹 (LLM 3번 호출)
    # block1_r = rerank_block_with_llm(block1, ranking_query, "1세트 (1~10위)")
    # block2_r = rerank_block_with_llm(block2, ranking_query, "2세트 (11~20위)")
    # block3_r = rerank_block_with_llm(block3, ranking_query, "3세트 (21~30위)")
    #
    #
    # # 5) 리랭킹 결과를 final_results에 다시 반영
    # final_results[0:0+len(block1_r)] = block1_r
    # final_results[10:10+len(block2_r)] = block2_r
    # final_results[20:20+len(block3_r)] = block3_r
















    # 상품 캐시에 저장 (final_results 기준)
    for info in final_results:
        PRODUCT_CACHE[info["상품코드"]] = info

    print("\n상품의 상세 정보:")
    for idx, info in enumerate(final_results, start=1):
        PRODUCT_CACHE[info["상품코드"]] = info
        
        if idx % 5 == 0:  # 5개마다 한 번씩만 출력
            print(f"\n처리 중... {idx}/{len(final_results)}개 완료")
        
    print(f"================================")
    print(f"보여주는 상품의 개수: {len(final_results)}")

    # ✅ 상품 정보에 옵션 관련 필드 확인 및 추가
    for product in final_results:
        if "옵션" not in product:
            product["옵션"] = "없음"
        if "조합형옵션" not in product:
            product["조합형옵션"] = ""
        if "최대구매수량" not in product:
            product["최대구매수량"] = 0       


    result_payload = {
        "query": query,  # 사용자가 입력한 원본 쿼리
        "UserMessage": llm_response,  # 정제된 쿼리
        "RawContext": previous_queries + [query],  # 전체 대화 맥락
        "results": final_results,  # 검색 결과 리스트 (방법2 5개 + 방법1 5개 = 10개)
        "combined_message_text": clean,  # 🎯 세션 만료 메시지 포함 가능
        "message_history": [
            {"type": type(msg).__name__, "content": getattr(msg, "content", "")}
            for msg in session_history.messages
        ]  # 전체 메시지 기록 (디버깅용)
    }
    total_time = time.time() - total_start_time  # 전체 처리 시간 계산
    print(f"\n⏱ 전체 검색 및 응답 생성 소요시간: {total_time:.2f}초")
    
    return result_payload






























@app.post('/track_view')
async def track_product_view(data: ProductViewData, request: Request):
    """상품 상세보기 클릭 이벤트 추적"""
    try:
        # 기존 세션 확인
        session_id = request.cookies.get('session_id')
        
        # 세션이 없는 경우에만 새로 생성
        is_new_session = False
        if not session_id:
            session_id = secrets.token_hex(16)
            is_new_session = True

        ip_address = request.client.host
        user_agent = request.headers.get('user-agent', '')
        
        # 사용자 ID 생성 또는 가져오기
        user_id = event_manager.get_or_create_user(session_id, ip_address, user_agent)
        
        # 상품 조회 이벤트 기록
        view_id = event_manager.record_product_view(
            user_id=user_id,
            session_id=session_id,
            product_data=data.product_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # 응답 생성
        response = JSONResponse({
            'status': 'success', 
            'view_id': view_id, 
            'user_id': user_id
        })
        
        # 새 세션인 경우에만 쿠키 설정
        if is_new_session:
            response.set_cookie(
                key='session_id',
                value=session_id,
                max_age=86400 * 3,  # 3일
                httponly=True
            )
            
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/user_stats')
async def user_stats(request: Request):
    """사용자 통계 정보"""
    try:
        session_id = request.cookies.get('session_id')
        if not session_id:
            raise HTTPException(status_code=400, detail='세션이 없습니다.')
        
        user_id = event_manager.get_or_create_user(session_id)
        
        # 사용자별 통계
        user_views = event_manager.get_user_product_views(user_id, limit=10)
        user_searches = event_manager.get_user_search_history(user_id, limit=10)
        
        return JSONResponse({
            'status': 'success',
            'user_id': user_id,
            'recent_views': user_views,
            'recent_searches': user_searches
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/popular_products')
async def popular_products(days: int = 7, limit: int = 20):
    """인기 상품 조회"""
    try:
        popular = event_manager.get_popular_products(days=days, limit=limit)
        return JSONResponse({'status': 'success', 'products': popular})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 디버깅용 요청 모델
class DebugRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    clarify_answer: Optional[str] = None 

@app.post('/debug-search')
async def debug_search(data: DebugRequest, request: Request):
    """
    external_search_and_generate_response를 호출하고
    검색 이벤트도 함께 기록합니다.
    """
    try: 
        # 검색어 검증
        if not data.query.strip():
            return JSONResponse({'status': 'error', 'message': '검색어가 필요합니다.'})
        
        # 세션 ID 확인 및 생성
        session_id = data.session_id or secrets.token_hex(16)
        ip_address = request.client.host
        user_agent = request.headers.get('user-agent', '')
        
        # 사용자 ID 생성 또는 가져오기
        user_id = event_manager.get_or_create_user(session_id, ip_address, user_agent)
        
        # 검색 실행
        result = external_search_and_generate_response(data.query, session_id)
        
        # 검색 이벤트 기록 추가
        results_count = len(result.get('results', []))  # 검색 결과 수 계산
        try:
            event_manager.record_search_event(
                user_id=user_id,
                session_id=session_id,
                query=data.query.strip(),
                results_count=results_count,
                ip_address=ip_address,
                user_agent=user_agent
            )
            print(f"검색 이벤트 기록 성공: query={data.query.strip()}, results_count={results_count}")
        except Exception as e:
            print(f"검색 이벤트 기록 실패: {str(e)}")
        
        # 응답 생성
        response = JSONResponse(content=result)
        
        # 세션 ID 쿠키 설정 (새로운 세션인 경우)
        if not data.session_id:
            response.set_cookie(
                key='session_id',
                value=session_id,
                max_age=86400 * 7,  # 7일
                httponly=True
            )
        
        return response
        
    except Exception as e:
        print(f"DEBUG - Search error: {str(e)}")  # 디버그용 로그
        return JSONResponse(
            status_code=500,
            content={"error": f"검색 중 오류가 발생했습니다: {str(e)}"}
        )

#####관리자 페이지#####
# Pydantic 모델 정의
class ResetStatisticsData(BaseModel):
    confirm: bool = False
    
@app.get('/admin', response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """관리자 대시보드 페이지"""
    # 세션 쿠키 확인
    admin_session = request.cookies.get(ADMIN_COOKIE_NAME)
    
    if not admin_session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # 세션 유효성 및 만료 검사
    if not verify_session_token(admin_session):
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        # 만료된 세션 쿠키 삭제
        response.delete_cookie(ADMIN_COOKIE_NAME)
        return response
        
    template = templates.get_template("admin_dashboard.html")
    return HTMLResponse(template.render(request=request))

@app.get('/api/stats')
async def get_statistics():
    """전체 통계 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 전체 사용자 수
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # 전체 상품 조회 수
        cursor.execute('SELECT COUNT(*) FROM product_views')
        total_views = cursor.fetchone()[0]
        
        # 전체 검색 수
        cursor.execute('SELECT COUNT(*) FROM search_events')
        total_searches = cursor.fetchone()[0]

        # 오늘 날짜 구하기
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 오늘 상품 조회 수
        cursor.execute('''
            SELECT COUNT(*) FROM product_views 
            WHERE DATE(timestamp) = ?
        ''', (current_date,))
        today_views = cursor.fetchone()[0]
        
        # 오늘 검색 수
        cursor.execute('''
            SELECT COUNT(*) FROM search_events 
            WHERE DATE(timestamp) = ?
        ''', (current_date,))
        today_searches = cursor.fetchone()[0]

        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'total_users': total_users,
                'total_views': total_views,
                'total_searches': total_searches,
                'today_views': today_views,
                'today_searches': today_searches
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/popular_products')
async def get_popular_products(days: int = 7):
    """인기 상품 API"""
    try:
        popular_products = event_manager.get_popular_products(days=days, limit=20)
        return JSONResponse({'status': 'success', 'data': popular_products})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/user_activity')
async def get_user_activity(days: int = 7):
    """사용자 활동 추이 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 일별 상품 조회 수
        cursor.execute('''
            WITH RECURSIVE dates(date) AS (
                SELECT date(datetime('now', '-{} days', '+9 hours'))
                UNION ALL
                SELECT date(datetime(date, '+1 day'))
                FROM dates
                WHERE date < date('now', '+9 hours')
            )
            SELECT 
                dates.date,
                COUNT(product_views.timestamp) as view_count
            FROM dates
            LEFT JOIN product_views ON 
                date(datetime(product_views.timestamp, '+9 hours')) = dates.date
            GROUP BY dates.date
            ORDER BY dates.date
        '''.format(days))
        
        daily_views = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 일별 검색 수
        cursor.execute('''
            WITH RECURSIVE dates(date) AS (
                SELECT date(datetime('now', '-{} days', '+9 hours'))
                UNION ALL
                SELECT date(datetime(date, '+1 day'))
                FROM dates
                WHERE date < date('now', '+9 hours')
            )
            SELECT 
                dates.date,
                COUNT(search_events.timestamp) as search_count
            FROM dates
            LEFT JOIN search_events ON 
                date(datetime(search_events.timestamp, '+9 hours')) = dates.date
            GROUP BY dates.date
            ORDER BY dates.date
        '''.format(days))
        
        daily_searches = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 시간별 활동 (오늘)
        cursor.execute('''
            SELECT strftime('%H', datetime(timestamp, '+9 hours')) as hour, COUNT(*) as count
            FROM product_views 
            WHERE DATE(datetime(timestamp, '+9 hours')) = DATE('now', '+9 hours')
            GROUP BY strftime('%H', datetime(timestamp, '+9 hours'))
            ORDER BY hour
        ''')
        
        hourly_activity = [{'hour': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'daily_views': daily_views,
                'daily_searches': daily_searches,
                'hourly_activity': hourly_activity
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/period_stats')
async def get_period_statistics(days: int = 7):
    """기간별 통계 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 기간별 사용자 수 (한국 시간 기준)
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as user_count
            FROM users 
            WHERE datetime(created_at, '+9 hours') >= datetime('now', '+9 hours', ? || ' days')
        ''', (-days,))
        period_users = cursor.fetchone()[0]
        
        # 기간별 상품 조회 수 (한국 시간 기준)
        cursor.execute('''
            SELECT COUNT(*) as view_count
            FROM product_views 
            WHERE datetime(timestamp, '+9 hours') >= datetime('now', '+9 hours', ? || ' days')
        ''', (-days,))
        period_views = cursor.fetchone()[0]
        
        # 기간별 검색 수 (한국 시간 기준)
        cursor.execute('''
            SELECT COUNT(*) as search_count
            FROM search_events 
            WHERE datetime(timestamp, '+9 hours') >= datetime('now', '+9 hours', ? || ' days')
        ''', (-days,))
        period_searches = cursor.fetchone()[0]
        
        # 기간별 일일 평균
        daily_avg_views = period_views / days if days > 0 else 0
        daily_avg_searches = period_searches / days if days > 0 else 0
        
        # 기간별 인기 상품 (한국 시간 기준)
        cursor.execute('''
            SELECT product_code, 
                   product_name, 
                   COUNT(*) as view_count
            FROM product_views 
            WHERE datetime(timestamp, '+9 hours') >= datetime('now', '+9 hours', ? || ' days')
            GROUP BY product_code, product_name
            ORDER BY view_count DESC
            LIMIT 10
        ''', (-days,))
        period_popular_products = [
            {'product_code': row[0], 'product_name': row[1], 'view_count': row[2]} 
            for row in cursor.fetchall()
        ]
        
        # 기간별 인기 카테고리 (한국 시간 기준)
        cursor.execute('''
            SELECT category, 
                   COUNT(*) as view_count
            FROM product_views 
            WHERE datetime(timestamp, '+9 hours') >= datetime('now', '+9 hours', ? || ' days')
            AND category IS NOT NULL
            AND category != ''
            GROUP BY category
            ORDER BY view_count DESC
            LIMIT 10
        ''', (-days,))
        period_popular_categories = [
            {'category': row[0], 'view_count': row[1]} 
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'period_days': days,
                'period_users': period_users,
                'period_views': period_views,
                'period_searches': period_searches,
                'daily_avg_views': round(daily_avg_views, 2),
                'daily_avg_searches': round(daily_avg_searches, 2),
                'popular_products': period_popular_products,
                'popular_categories': period_popular_categories
            }
        })
    except Exception as e:
        print(f"기간별 통계 로드 중 오류: {str(e)}")  # 디버깅용 로그 추가
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/category_stats')
async def get_category_stats(days: int = None):
    """카테고리별 통계 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 기간 필터 조건
        date_filter = ""
        if days:
            date_filter = f"WHERE timestamp >= datetime('now', '-{days} days') AND category != '' AND category IS NOT NULL"
        else:
            date_filter = "WHERE category != '' AND category IS NOT NULL"
        
        # 카테고리별 상품 조회 수
        cursor.execute(f'''
            SELECT category, COUNT(*) as view_count
            FROM product_views 
            {date_filter}
            GROUP BY category
            ORDER BY view_count DESC
            LIMIT 20
        ''')
        
        category_stats = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return JSONResponse({'status': 'success', 'data': category_stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/reset_statistics')
async def reset_statistics(data: ResetStatisticsData):
    """통계 데이터 초기화 API"""
    try:
        if not data.confirm:
            raise HTTPException(status_code=400, detail='확인 파라미터가 필요합니다.')
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 모든 이벤트 데이터 삭제
        cursor.execute('DELETE FROM user_events')
        cursor.execute('DELETE FROM product_views')
        cursor.execute('DELETE FROM search_events')
        cursor.execute('DELETE FROM users')
        
        # AUTOINCREMENT를 사용하지 않으므로 sqlite_sequence 테이블은 존재하지 않음
        # 따라서 시퀀스 리셋은 불필요
        
        conn.commit()
        conn.close()
        
        return JSONResponse({
            'status': 'success', 
            'message': '모든 통계 데이터가 성공적으로 초기화되었습니다.'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/lightgcn_data')
async def get_lightgcn_data():
    """LightGCN 데이터 상태 확인 API"""
    try:
        preprocessor = LightGCNDataPreprocessor()
        data = preprocessor.load_lightgcn_data()
        
        if not data:
            raise HTTPException(status_code=404, detail='LightGCN 데이터가 없습니다.')
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'n_users': data.get('n_users', 0),
                'n_products': data.get('n_products', 0),
                'n_interactions': data.get('n_interactions', 0),
                'sparsity': data.get('sparsity', 0)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/regenerate_lightgcn_data')
async def regenerate_lightgcn_data(min_interactions: int = 1):
    """LightGCN 데이터 재생성 API"""
    try:
        preprocessor = LightGCNDataPreprocessor()
        print(f"최소 상호작용 횟수: {min_interactions}")
        data = preprocessor.prepare_lightgcn_data(min_interactions=min_interactions)
        
        if data:
            return JSONResponse({
                'status': 'success',
                'message': 'LightGCN 데이터가 성공적으로 재생성되었습니다.',
                'data': {
                    'n_users': data['n_users'],
                    'n_products': data['n_products'],
                    'n_interactions': data['n_interactions']
                }
            })
        else:
            raise HTTPException(status_code=400, detail='데이터가 충분하지 않습니다.')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/user_details/{user_id}')
async def get_user_details(user_id: str):
    """사용자 상세 정보 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 사용자 기본 정보
        cursor.execute('''
            SELECT user_id, session_id, ip_address, created_at, last_activity
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        user_info = cursor.fetchone()
        if not user_info:
            raise HTTPException(status_code=404, detail='사용자를 찾을 수 없습니다.')
        
        # 사용자 상품 조회 기록
        cursor.execute('''
            SELECT product_code, product_name, category, price, timestamp
            FROM product_views 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (user_id,))
        
        product_views = [
            {
                'product_code': row[0],
                'product_name': row[1],
                'category': row[2],
                'price': row[3],
                'timestamp': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        # 사용자 검색 기록
        cursor.execute('''
            SELECT query, price_min, price_max, results_count, timestamp
            FROM search_events 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (user_id,))
        
        search_history = [
            {
                'query': row[0],
                'price_min': row[1],
                'price_max': row[2],
                'results_count': row[3],
                'timestamp': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'user_info': {
                    'user_id': user_info[0],
                    'session_id': user_info[1],
                    'ip_address': user_info[2],
                    'created_at': user_info[3],
                    'last_activity': user_info[4]
                },
                'product_views': product_views,
                'search_history': search_history
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ───── 모든 라우트 등록이 끝난 뒤, 마지막에 ─────

if __name__ == '__main__':
    
    # 환경 변수에서 설정 가져오기
    host = '0.0.0.0'
    port = 8011
    debug = True
    TRUSTED = os.getenv("FORWARDED_ALLOW_IPS", "")

    print(f"🚀 FastAPI 서버 시작: {host}:{port} (debug={debug})")
    uvicorn.run("app:app", host=host, port=port, reload=debug, proxy_headers=True,forwarded_allow_ips=TRUSTED)
