# 데이터의 구조가 바뀐 시초의 ownerclan
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

from langdetect import detect
from collections import defaultdict, Counter
import math
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
from fastapi.responses import RedirectResponse
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





executor = ThreadPoolExecutor()

# 환경변수 로드
load_dotenv()

# ─── ENV & 유틸 (위쪽 공용 영역에 추가/교체) ─────────────────────────────

def _s(x):  # None 가드
    return x if isinstance(x, str) else ""

def _eq_cs(a, b):  # 비밀번호용(대소문자 구분 + 상수시간)
    a, b = _s(a), _s(b)
    return secrets.compare_digest(a, b)

def _eq_ci(a, b):  # 아이디용(대소문자 무시하고 싶을 때)
    return _s(a).casefold() == _s(b).casefold()


# ── 설정 ─────────────────────────────────────────────────────────
API_KEY    = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')

REDIS_URL = "redis://localhost:6379/0"                    # ← 환경변수에서 로드
COLLECTION = "ownerclan"            # Milvus 컬렉션 이름
MILVUS_HOST = os.getenv('MILVUS_HOST', '114.110.135.96')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
LLM_MODEL  = "gpt-4.1-mini-2025-04-14"
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


# 1) Milvus 서버에 먼저 연결
connections.connect(
    alias="default",
    host=MILVUS_HOST,    # 예: "114.110.135.96"
    port=MILVUS_PORT     # 예: "19530"
)
print("✅ Milvus에 연결되었습니다.")

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

# Pydantic 모델 정의
class ProductViewData(BaseModel):
    product_code: str
    product_data: dict

class SearchData(BaseModel):
    query: str

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
    THRESHOLD = 0.6             # 평균 점수 임계값 (이상이면 검색 진행)
    DIRECT_MATCH_HIGH = 0.6      # 직접 매칭 높은 신뢰도 (단독 통과 가능)
    FACET_COVERAGE_MIN = 0.2     # 최소 속성 커버리지 (미달 시 재질문)
    ATTRIBUTE_MIN = 0.3          # 속성 매칭 최소값
    FACET_SUFFICIENT = 0.35       # 충분한 속성 커버리지


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
                    f"💫 The last conversation was not present for 0.5 minutes and was automatically initialized. Start Search for New Products! Enter Your Search Word 😊\n\n"
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

    def format_completion_message(completion_percent: int, avg_score: float, threshold: float = THRESHOLD) -> str:
        """
        완성률에 따른 친절한 메시지 생성
        
        Args:
            completion_percent: 완성률 퍼센트 (0~100)
            avg_score: 평균 점수
            threshold: 완성 기준 점수
            
        Returns:
            str: 완성률 메시지
        """
        if completion_percent >= 100:
            return f"💯{completion_percent}%"
        elif completion_percent >= 80:
            return f"🎯{completion_percent}%"
        elif completion_percent >= 60:
            return f"📈{completion_percent}%"
        elif completion_percent >= 40:
            return f"📊{completion_percent}%"
        elif completion_percent >= 20:
            return f"📉{completion_percent}%"
        else:
            return f"❓{completion_percent}%"
    
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
        

    def set_session_ttl(session_id: str, ttl_seconds: int = TIMEOUT_SECONDS):
        """
        세션의 TTL을 설정하는 공통 함수
        """
        try:
            r = redis.from_url(REDIS_URL)
            session_key = f"message_store:{session_id}"
            r.expire(session_key, ttl_seconds)
            print(f"[TTL 설정] 세션 {session_id} TTL={ttl_seconds}초 설정 완료")
            return True
        except Exception as e:
            print(f"[TTL 설정 오류] 세션 {session_id}: {e}")
            return False

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
    lang_code = detect(query)

    # ✅ Redis 세션 기록 불러오기 및 최신 입력 저장
    session_history = get_session_history(session_id)



    # � **핵심 수정**: 리셋 직후에는 세션 타임아웃 체크 건너뛰기
    # 리셋 직후 첫 메시지인지 확인 (history가 비어있거나 1개 이하면 리셋 직후)
    messages = session_history.messages if hasattr(session_history, 'messages') else []
    is_after_reset = len(messages) <= 1
    
    # 🕒 세션 자동 만료 체크 (경고 없이 바로 처리)
    timeout_result = check_session_timeout(session_history, session_id)
    print(f"[세션체크] timeout_result={timeout_result}")
    if timeout_result and timeout_result.get("session_expired"):
        # 세션이 만료된 경우 자동 초기화 메시지를 즉시 반환 (이미 Redis에 저장됨)
        print(f"[세션만료] 자동 초기화 메시지 반환: {timeout_result['assistant_message'][:50]}...")
        return timeout_result





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


        "fr": "Français",
        "de": "Deutsch",
        "es": "Español",
        "it": "Italiano",
        "pt": "Português",
        "ar": "العربية",
        "fa": "فارسی",
        "he": "עברית",
        "sw": "Kiswahili",
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

    def _build_recent_context(session_history, k: int = 5) -> str:
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
        avg = 0.6*d + 0.2*a + 0.1*c + 0.1*b
        
        # 🎯 우선: 컨텍스트 + 직접매칭 조합 (특별 케이스)
        if c >= 0.3 and c <= 0.4 and d >= DIRECT_MATCH_HIGH:
            route = "proceed"
            print(f"[Route] 컨텍스트+직접매칭 조합({c:.3f}≥0.3 + {d:.3f}≥0.7) → 특별 통과")
            
        # 🚀 1차: 높은 직접 매칭 - 단독 통과 가능
        elif d >= DIRECT_MATCH_HIGH:
            route = "proceed"
            print(f"[Route] 직접매칭 높음({d:.3f}≥{DIRECT_MATCH_HIGH}) → 단독 통과")
            
        # 🔍 2차: 속성 매칭 검증
        elif a >= ATTRIBUTE_MIN and d >= 0.5:
            # 직접매칭이 중간 이상이고 속성매칭이 충분한 경우
            route = "proceed"
            print(f"[Route] 속성매칭 충분({a:.3f}≥{ATTRIBUTE_MIN}) + 직접매칭 중간({d:.3f}≥0.5) → 통과")

        # 📊 3차: 종합 점수 + 최소 속성 커버리지 검증
        elif avg >= THRESHOLD * 0.6:  # THRESHOLD의 60% 이상
            if a >= FACET_COVERAGE_MIN:
                route = "proceed" 
                print(f"[Route] 종합점수 높음({avg:.3f}≥{THRESHOLD*0.6:.3f}) + 속성커버 충족({a:.3f}≥{FACET_COVERAGE_MIN}) → 통과")
            else:
                route = "clarify"
                print(f"[Route] 종합점수 높지만 속성커버 부족({a:.3f}<{FACET_COVERAGE_MIN}) → 재질문")
                
        # 🎲 4차: 충분한 속성 커버리지가 있는 경우 점수 완화
        elif a >= FACET_SUFFICIENT and avg >= 0.5:
            route = "proceed"
            print(f"[Route] 속성커버 충분({a:.3f}≥{FACET_SUFFICIENT}) + 기본 점수({avg:.3f}≥0.5) → 완화 통과")
            
        # ❌ 5차: 그 외 모든 경우 재질문
        else:
            route = "clarify"
            print(f"[Route] 점수 부족(avg={avg:.3f}, d={d:.3f}, a={a:.3f}) → 재질문")
        
        return round(avg, 4), route
    
    def _build_recent_context(session_history, k: int = 5) -> str:
        """
        사용자/판매 관련 메시지 위주로 최근 5줄 요약.
        디버그/코드/로그/[INTENT_GATE]는 제외.
        """
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
                # 너무 긴 건 자르기
                s = s.replace("\n", " ")[:160]
                if s:
                    lines.append("- " + s)
        except Exception:
            pass
        return "\n".join(lines[-k:])

    def check_direct_search_command(user_query: str) -> bool:
        """
        바로검색 명령어가 포함되어 있는지 확인
        """
        direct_search_patterns = [
            '바로검색', '바로 검색', '즉시검색', '즉시 검색',
            '그냥검색', '그냥 검색', '바로찾아', '바로 찾아',
            '즉시찾아', '즉시 찾아', '그냥찾아', '그냥 찾아',
            '바로해', '바로 해',

                # 영어 패턴
            'direct search', 'immediate search', 'instant search',
            'quick search', 'fast search', 'just search',
            'search now', 'find now', 'direct find',
            'immediate find', 'instant find', 'quick find',
            'fast find', 'just find', 'search directly',
            'find directly', 'search immediately', 'find immediately',
            'go search', 'go find', 'do search', 'do find',
            
            # 축약형/간단한 명령어
            'direct', 'immediate', 'instant', 'now',
            'go', 'just do it', 'skip', 'proceed'
        ]
        
        query_lower = user_query.lower().replace(' ', '')
        for pattern in direct_search_patterns:
            if pattern.replace(' ', '') in query_lower:
                return True
        return False

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
    **AI 통합 분석 & 재질문 시스템**
    다음 사용자 질의를 읽고 '의도 파악 신뢰도'를 평가하고 맞춤 재질문을 생성하세요.
    출력은 반드시 JSON 한 덩어리만. 다른 문장/주석 금지.

    [사용자 질의] {user_query}
    [이전 대화 요약] {recent_context or '없음'}

    ⚡ **0단계: 엄격한 자동 진행 규칙 (Strict Auto-Proceed)**
    다음에 **정확히** 해당하는 경우만 재질문 없이 바로 진행하세요.

    **🎯 절대 확실한 경우만 자동 진행:**
    1. **완전한 브랜드+모델명**: "아이폰 15 프로", "갤럭시 S24", "RTX 4070"
    2. **구체적 규격+제품**: "30인치 모니터", "256GB SSD", "1TB 외장하드"
    3. **명확한 단일 상품 + 구매 의사**: "에어팟 구매", "맥북 최저가", "아이패드 찾아줘"

    **🚨 재질문이 필요한 경우들 (점수 낮게 부여):**
    - **모호한 카테고리**: "운동용품", "화장품", "전자제품" → direct_match=0.2
    - **형용사만 있는 경우**: "좋은", "저렴한", "예쁜" → direct_match=0.1
    - **일반 상품명만**: "가방", "신발", "옷" → direct_match=0.4 (재질문 필요!)
    - **추상적 요청**: "뭔가", "괜찮은 거", "적당한" → direct_match=0.1
    - **불완전한 정보**: "겨울용", "여행갈 때" → attribute_match=0.3

    **엄격한 점수 부여 원칙:**
    - **direct_match**: 정확한 상품명이 아니면 0.5 미만으로 부여
    - **attribute_match**: 용도/재질/크기/브랜드 중 2개 이상 명시되어야 0.5 이상
    - **context_match**: 명확한 구매 의사가 있어야 0.6 이상
    - **총 avg_score ≥ 0.7**: 매우 엄격하게 적용

    🔒 **현재 탐색 카테고리(잠금)**
    - 질의에서 드러난 1차 카테고리를 잠급니다.
    - 이 카테고리 **외부로 전환 금지**. 하위 유형/속성 안에서만 질문·선택지를 구성하세요.

    📋 **미션: 핵심 정보만 수집하여 검색 품질 극대화**

    🔍 **1단계: 이미 확정된 정보 파악**
    이전 대화에서 사용자가 이미 선택하거나 명시한 정보들을 정확히 파악하세요.
    ⚠️ **절대 금지**: 이미 확정된 정보를 다시 묻는 것!

    🌟 **2단계: 용어/동의어 확장**
    외국어/브랜드/생소어 존재 시 한국 리테일 검색어로 확장.

    확장 예시:
    - "프릭남쁠라" → "매운 칠리소스, 태국 핫소스, 스리라차, 고추소스, 매콤한 소스"
    - "타바스코" → "핫소스, 매운소스, 칠리소스, 고추소스"  
    - "삼발올렉" → "인도네시아 칠리소스, 매운 디핑소스, 아시아 핫소스"
    - "우산" → "우산, 양산, 접이식우산, 장우산, 자동우산, 방수우산"
    
    🏷️ **브랜드 확장 역할 (중요!):**
    사용자가 "브랜드 운동화", "브랜드 신발", "유명한 브랜드", "명품 브랜드" 등을 언급하면:
    
    **당신의 역할**: 해당 카테고리에서 가장 잘 알려진 대표 브랜드 3-5개를 자동으로 선정하여 expanded_terms에 포함시키세요.
    
    **브랜드 선정 기준**:
    - 한국 시장에서 인지도가 높은 브랜드 우선
    - 해당 카테고리의 대표적인 글로벌 브랜드
    - 온라인 쇼핑몰에서 쉽게 찾을 수 있는 브랜드
    - 다양한 가격대를 아우르는 브랜드 (고급/중급/보급)
    
    **자동 추론 예시**:
    - "브랜드 운동화" → 운동화 분야 대표 브랜드들을 스스로 선정
    - "명품 가방" → 럭셔리 가방 브랜드들을 스스로 선정  
    - "브랜드 화장품" → 유명 화장품 브랜드들을 스스로 선정
    - "브랜드 시계" → 시계 브랜드들을 스스로 선정
    
    **주의사항**: 리스트를 암기하지 말고, 상황에 맞는 브랜드를 유연하게 추론하세요.
    
    🔍 **일반 확장 역할**:
    사용자가 추상적이거나 모호한 표현을 사용할 때도 구체적인 검색어로 확장하세요:
    
    **예시 상황별 확장 방법**:
    - "겨울 아이템" → 겨울에 필요한 구체적인 상품들로 확장
    - "운동용품" → 다양한 운동 관련 구체적 용품들로 확장  
    - "건강식품" → 실제 건강식품 카테고리들로 확장
    - "생활용품" → 일상생활 필수 아이템들로 확장
    - "패션 아이템" → 구체적인 의류/악세서리 종류들로 확장
    
    **확장 원칙**: 
    1. 사용자 의도를 정확히 파악하고
    2. 해당 카테고리의 대표적인 하위 개념들을 3-6개 선정
    3. 검색에 실제로 도움이 되는 구체적인 키워드로 변환


    **3단계: 점수 평가**
    direct_match (60%), attribute_match (20%), brand_match (10%), context_match (10%)

    **특별 점수 처리:**
    - "우산/모자/가방" 등등 명확한 일상 상품명의 문맥 파악 문장 → direct_match=0.8
    - "있나?/찾아줘/알려줘" 같은 직접 요청같은 문맥의 문장 → context_match=0.7

    **4단계: 점수상승 지향 재질문 설계**
    재질문은 아래 4점수를 올리기 위한 목적이어야 함:
    - direct_match: 구체 명사/규격(타입, 치수, 형태) 확정
    - attribute_match: 용도/무게/재질/기능/가격대 등 이런 타입 2~3개 선택 유도
    - brand_match: 브랜드 지정 vs 무관 토글
    - context_match: 직전/누적쿼리 자세히 요약 후 확인

    **재질문 품질 규칙 - 정보 풍부화 원칙**
    - **상품별 핵심 정보 제공**: 각 선택지마다 주요 특징, 용도, 가격대, 적합한 상황 등 구매 결정에 필요한 구체적 정보 포함
    - **비교 기준 명시**: 크기, 재질, 기능, 브랜드, 가격대 등 실제 비교할 수 있는 기준 제시
    - **사용 시나리오**: 언제, 어디서, 누가 사용하는지에 간결히
    - **성능/품질 차이**: 고급형 vs 보급형, 전문용 vs 일반용 등의 차이 설명
    - 한 문단+선택지 이모지 1️⃣~4️⃣ (의미 중복/모호어 금지)
    - 마지막에 자유기입 한 줄

    **역할 중재(Explain-or-Search, 암묵 추론)**
    - 문자열 규칙이 아니라 **지배적 의도**로 판단.

    - 지배적 의도가 **설명/정의/차이/의미 질문**(예: "~가 뭐야", "~차이", "~뜻?")이면:
    1) {target_lang}로 **한 줄 정의(최대 170자)**를 먼저 제시하고,

    2) 즉시 이어서 **짧은 선택지형 브릿지 질문( 1️⃣~4️⃣)**로 구매 탐색으로 연결한다.

    3) 이 두 줄을 **clarify_question**에 넣고, 기본적으로 **route="clarify"**로 둔다.
        (단, 질의 안에 **명시적 구매 의사**와 **구체 항목**이 함께 있으면 route="proceed" 가능)

    4) **expanded_terms**에는 실제 검색에 유용한 3~6개 키워드를 넣는다(동의어/유사 스타일/연관 카테고리).
    - 예시(형식 참고용):
    clarify_question:



    **정보성 질의에도 적용되는 재질문 가이드(모든 카테고리 공통):**
    - 제공되는 문장을 잘 읽어보고 판단해서 문맥에 어울리는 답변을 제대로 답변.
    - 과장/마켓팅 문구 금지. 구체 키워드만.
        
    🌍 **언어 매칭 규칙:**
    **사용자가 사용한 언어를 감지하고, 그 나라 언어로 답변하세요!!**

    **무조건 어떤 언어가 들어와도 {target_lang}로만 답변하세요!반드시**

    답변 언어는 무조건 {target_lang}로만 작성해주세요. 다른 언어, 혼합 표현 절대 금지.
    

    JSON 스키마:
    {{
    "direct_match": 0.0,
    "context_match": 0.0,
    "attribute_match": 0.0,
    "brand_match": 0.0,
    "avg_score": 0.0,
    "route": "clarify",
    "clarify_question": "한 줄 정의 + 한 줄 선택지 질문(또는 순수 선택지 질문)",
    "expanded_terms": ["검색에 유용한 한국어 키워드 3~6개"],
    "refine_terms": ["보강 키워드/사이즈확장 결과"],
    "notes": ["판단 근거(간단히)"]

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
                "refine_terms": ["direct_search"],
                "notes": ["direct_search_command"],
                "completion_ratio": 1.0,
                "completion_percent": 100,
                "completion_message": "🚀 바로검색 모드"
            }
        
        # 1) 최근 문맥 안정 구성
        recent_context = _build_recent_context(session_history, k=5)
        
        # 🧠 대화 맥락 종합 분석
        context_analysis = analyze_conversation_context(session_history)
        print(f"[맥락분석] 주요카테고리={context_analysis['main_category']}, "
              f"언급속성={context_analysis['mentioned_attributes']}, "
              f"반복질문={len(context_analysis['repeated_queries'])}")

        # 2) 호출 (JSON 강제)
        raw = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "JSON ONLY. No prose."},
                {"role": "user", "content": INTENT_GATE_PROMPT(user_query, recent_context)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=600
        ).choices[0].message.content

        # 3) 안전 파싱 (+한 번 더 보정 시도)
        try:
            intent_eval = json.loads(raw)
        except Exception:
            try:
                intent_eval = _extract_json_block(raw)
            except Exception:
                intent_eval = {
                    "direct_match": 0.0, "context_match": 0.0,
                    "attribute_match": 0.0, "brand_match": 0.0,
                    "avg_score": 0.0, "route": "clarify",
                    "clarify_question": "",
                    "refine_terms": [], "notes": ["parse_fail"]
                }

        # 4) 서버에서 재계산(단일 진실원칙)
        avg_score, route = _recompute_route(intent_eval)
        intent_eval["avg_score"] = avg_score
        intent_eval["route"] = route

        # 🎯 완성률 계산 및 추가
        completion_ratio, completion_percent = calculate_completion_rate(avg_score, THRESHOLD)
        completion_message = format_completion_message(completion_percent, avg_score, THRESHOLD)
        
        # 🧠 **통합 완료** - INTENT_GATE_PROMPT에서 이미 재질문이 생성됨
        # intent_eval["clarify_question"]에 AI가 생성한 스마트 재질문이 이미 포함되어 있음
        
        intent_eval.update({
            "completion_ratio": completion_ratio,
            "completion_percent": completion_percent,
            "completion_message": completion_message,
            "context_analysis": context_analysis,
            "_raw_llm": raw,  # 👈 원문 JSON 그대로 보관
        })

        print(f"[완성률] avg_score={avg_score:.3f} → {completion_message}")

        # 5) 디버그(원하면 남기되, recent_context엔 안 넣도록 태그 유지)
        session_history.add_ai_message(f"[INTENT_GATE]{json.dumps(intent_eval, ensure_ascii=False)}")
    
         # 🔁 AI 응답 시에도 TTL 슬라이딩

        try:
            r = redis.from_url(REDIS_URL)
            r.expire(f"message_store:{session_id}", TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[세션 TTL 갱신 오류] {e}")


        return intent_eval

    # ===== Intent Gate 실행 및 게이트 판정 =====
    intent_eval = run_intent_gate(query, session_history, client, model=LLM_MODEL)
    avg_score = float(intent_eval.get("avg_score", 0.0))
    route = (intent_eval.get("route") or "").lower()
    completion_percent = intent_eval.get("completion_percent", 0)
    print(f"[IntentGate] avg={avg_score:.2f} route={route} 완성률={completion_percent}% "
        f"D={intent_eval.get('direct_match')} C={intent_eval.get('context_match')} "
        f"A={intent_eval.get('attribute_match')} B={intent_eval.get('brand_match')}")

    

    # 🔄 재질문 횟수 추적 및 누적 쿼리 관리 - 완전 개선
    clarification_count = 0
    user_query_parts = []
    
    # 세션 히스토리에서 재질문 횟수와 사용자 질문들 수집
    try:
        print(f"[디버그] 전체 메시지 수: {len(session_history.messages)}")
        
        for i, msg in enumerate(session_history.messages[-20:]):  # 최근 20개 메시지 확인
            content = getattr(msg, "content", "") or ""
            if not content.strip():
                continue
                
            print(f"[디버그] 메시지 {i}: type={getattr(msg, 'type', 'unknown')}, content='{content[:100]}'")
            
            # 🎯 AI 메시지에서 실제 재질문 패턴 감지
            if hasattr(msg, 'type') and msg.type == 'ai':
                # 실제 재질문에서 나타나는 패턴들
                clarification_patterns = [
                    "어떤 종류를 원하시나요",
                    "어떤 스타일을",
                    "어떤 종류를 찾",
                    "1️⃣", "2️⃣", "3️⃣",  # 선택지 이모지
                    "더 자세히 알려주세요",
                    "더 구체적으로",
                    "추가로",
                    "알려주시면 더 정확한"
                ]
                
                # 패턴 매칭 확인
                for pattern in clarification_patterns:
                    if pattern in content:
                        clarification_count += 1
                        print(f"[재질문 감지] {clarification_count}번째 재질문 패턴 '{pattern}' 발견")
                        break  # 한 메시지에서 여러 패턴 매칭되어도 1회만 카운트
                
            # 🔍 사용자 메시지 수집 (INTENT_GATE 태그 제외)
            elif hasattr(msg, 'type') and msg.type == 'human':
                if "[INTENT_GATE]" not in content and content.strip():
                    clean_content = content.strip().replace("찾아줘", "").replace("찾아", "").replace("을", "").replace("를", "").strip()
                    if clean_content and len(clean_content) < 50:  # 길이 제한 완화
                        user_query_parts.append(clean_content)
                        print(f"[사용자 질문 수집] '{clean_content}'")
                    
    except Exception as e:
        print(f"[재질문 추적 오류] {e}")
    
    # 사용자 질문들을 의미있게 조합 (최신 것을 우선으로)
    if user_query_parts:
        # 중복 제거하면서 순서 유지
        unique_parts = []
        for part in reversed(user_query_parts[-3:]):  # 최근 3개만
            if part not in unique_parts:
                unique_parts.insert(0, part)
        combined_query = " ".join(unique_parts)
    else:
        combined_query = query
    
    print(f"[재질문 추적 최종] 감지횟수={clarification_count}, 사용자질문={user_query_parts}, 누적쿼리='{combined_query}'")

    # 🚀 **핵심 수정**: 바로검색 명령어 우선 확인 (Intent Gate 이전)
    if check_direct_search_command(combined_query if len(user_query_parts) > 1 else query):
        search_query = combined_query if len(user_query_parts) > 1 else query
        print(f"[바로검색] 명령어 감지: '{search_query}' → Intent Gate 건너뛰고 바로 검색")
        query = search_query
        pass  # 바로 검색으로 진행
    else:
        # clarify 답변 수집
        clarify_answer = None
        try:
            if isinstance(request, dict):
                clarify_answer = request.get("clarify_answer")
            else:
                clarify_answer = getattr(request, "clarify_answer", None)
        except Exception:
            clarify_answer = None

        # ★ route 우선 적용 (direct_match≥0.8 포함)
        if route == "proceed":
            pass  # 다음 단계 진행
        elif clarification_count >= 3:
            # 🚨 재질문 횟수 제한 (3번 초과 시 강제 진행)   #재질문 횟수 3번만에 대화 완료 목표 4번째에 검색 돌입.
            print(f"[재질문 제한] {clarification_count}번 재질문 완료 → 누적쿼리로 강제 진행: '{combined_query}'")
            query = combined_query  # 누적된 쿼리로 검색 진행
            # 강제로 검색 진행 (재질문 건너뛰기)
        elif avg_score < THRESHOLD and not clarify_answer:
            # 평균 점수 미달 + 보강답변 없음 → 재질문 (3번 미만일 때만)
            # 🧠 누적 쿼리로 재평가 (쿼리가 실제로 달라졌을 때만)
            if combined_query != query and len(user_query_parts) > 1:
                print(f"[누적 재평가] 기존 쿼리='{query}' → 누적 쿼리='{combined_query}'")
                # 누적된 쿼리로 다시 intent gate 실행
                intent_eval_combined = run_intent_gate(combined_query, session_history, client, model=LLM_MODEL)
                avg_score_combined = float(intent_eval_combined.get("avg_score", 0.0))
                route_combined = (intent_eval_combined.get("route") or "").lower()
                
                if route_combined == "proceed" or avg_score_combined >= THRESHOLD:
                    print(f"[누적 재평가] 통과! avg={avg_score_combined:.3f} → 검색 진행")
                    query = combined_query
                    # 재질문 건너뛰고 검색으로 진행
                else:
                    print(f"[누적 재평가] 여전히 부족 avg={avg_score_combined:.3f} → 재질문")
                    # 누적 평가 결과를 사용
                    intent_eval = intent_eval_combined
                    
                    # 재질문 생성
                    followup = intent_eval.get("clarify_question")
                    completion_message = intent_eval.get("completion_message", f"💡{intent_eval.get('completion_percent', 0)}%")
                    enhanced = f"{followup}\n\n{completion_message}"
                    
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
                        "completion_message": completion_message,
                        "clarification_count": clarification_count + 1,
                        "accumulated_query": combined_query
                    }
            else:
                # 다른 재질문 조건들
                followup = intent_eval.get("clarify_question")
                completion_message = intent_eval.get("completion_message", f"💡{intent_eval.get('completion_percent', 0)}%")
                enhanced = f"{followup}\n\n{completion_message}"
                
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
                    "completion_message": completion_message,
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
                followup = intent_eval.get("clarify_question") or "조금만 더 구체화해주실래요? (용도/예산/브랜드 중 하나)"
                completion_message = intent_eval.get("completion_message", f"💡 질문 완성률: {intent_eval.get('completion_percent', 0)}%")
                
                # 재질문에 완성률 정보 추가
                enhanced = f"{followup}\n\n{completion_message}"
                
                return {
                    "query": query,
                    "UserMessage": enhanced,
                    "RawContext": [m.content for m in session_history.messages],
                    "results": [],
                    "combined_message_text": enhanced,
                    "intent_gate": intent_eval,
                    "needs_clarification": True,
                    "completion_percent": intent_eval.get("completion_percent", 0),
                    "completion_message": completion_message
                }


    # ===== 여기까지 내려오면 통과 → 아래 전처리/카테고리/임베딩 검색 계속 =====
    # ===== 통과 시 아래 원래 system_prompt 로직 계속 진행    재질문 END =====
















    system_prompt = (
        f"""System:
            당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
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
            ✅ 현재 입력에 상품명 없고 + 이전에 상품명 있음 → **[이전 상품명] + [현재 조건]**
            ✅ 현재 입력이 조건 추가임 → **[이전 상품명] + [이전 조건들] + [현재 조건]**
            ✅ 예시:
               - 이전: "겨울장갑" → 현재: "국내산" → **결과: "겨울장갑 국내산"**
               - 이전: "여름가방" → 현재: "작은거" → **결과: "여름가방 작은"**
               - 이전: "겨울장갑 국내산" → 현재: "저렴한" → **결과: "겨울장갑 국내산 저렴한"**
            
            3. 🎯 최종 검색어 생성 강제 공식:
            - **절대 규칙**: 이전 맥락 무시 금지! 반드시 누적하여 검색어 생성할 것!

 
            [전처리 원칙]
            1) 문장을 이해해서 분석하여 사용자가 찾고자 하는 상품의 모호한 상위개념으로 묻는 경우(예: "한국식 과자", "여름 원피스", "세차용품") → 
            카테고리/유형/대표 상품명으로 확장된 표면어 다발을 만든다.
            *사용자의 문장을 이해해서 추측도 반드시 해본다! 질문에 대해서 답을 생성도 같이해서 제일 앞에 단어를 붙인다.*

            - 확장은 3~7개 사이의 "핵심 하위유형/동의어"로 한정(오버확장 금지).
            - 브랜드/스펙/색/규격/수량/가격 등 명시 제약이 있으면 보존.
            - "추천/최고/인기" 같은 수식어는 제거하고, 실제 검색에 유의미한 토큰만 남긴다.
            2) 한국 쇼핑 맥락의 일반명은 단수 표면형을 우선(복수/어미/조사 제거).
            3) '용' 같은 불용미사/꼬리표는 제거.
            4) 불필요한 구두점은 제거. OR, |, 콤마 대신 **공백 나열**만 사용.
            5) **부정/제외 처리(아주 중요)**:
            - 다음 패턴을 부정 신호로 인식한다: "싫어/싫다/말고/빼고/제외(하고/한)/아닌/제외해줘/빼줘/미포함/제외 부탁" 등.
            - "A 말고 B", "A는 싫고 B", "B 찾는데 A 제외" 형태에서는 **A는 제외(-A), B는 유지**한다.
            - 제외 토큰은 단어·구를 **표준형으로 정규화**하고 **하이픈(-토큰)** 으로 표기한다(예: -호박맛, -딸기, -화이트).
            - 다단 제외가 있으면 공백으로 나열한다: 예) "사과 주스 -호박맛 -배맛".
            - 핵심 품목이 불명확하고 제외만 존재할 경우, **추정하지 말고 원문 유지**(의미 보존 원칙).

            [중요: 의미 보존]
            - 핵심 품목이 명확하지 않으면 원문 유지(축약·추정 금지)

            [Category Search Text:카테고리 검색용 상품 요약 문장 생성]
                사용자가 찾고자 하는 상품을 이해하고, 카테고리 검색에 최적화된 자연스러운 한국어 문장으로 요약하세요.
                
                **요구사항:**
                1) 길이: 8~16자(공백 제외, 가능하면 12자 내외)
                2) 관형어 1개 이상 포함: 예) 먹는/쓰는/입는/바르는/메는/신는/쓰는(착용)
                3) 브랜드/트렌드/마케팅 단어 금지: “다양한, 트렌드, 제품, 합리적, 프리미엄, 스타일리시, 최적”
                4) 불필요한 종결어미, 존댓말 생략: “~합니다/입니다” → 생략
                5) 핵심 품목 불명확 시: **추정 금지, 원문 유지**
                6) **부정/제외 표현(제외/빼고/말고/without 등) 금지.** 결과 문장은 제외 대상 자체를 언급하지 말 것.

                **올바른 예시:**
                - "여름용 작은 가방" → "여름용 작은 가방을 찾고 있습니다"
                - "운동할 때 신는 신발" → "운동용 신는 신발을 찾고 있습니다"  
                - "겨울 따뜻한 외투" → "겨울철 따뜻한 외투을 찾고 있습니다."
                
                **잘못된 예시 (키워드 나열):**
                - "여름 작은 가방 쿨 소재 여성" (X)
                - "운동 신발 편안한" (X)
                - "겨울 외투 따뜻한" (X)


            [출력 규칙(반드시 정확히 준수)]
            오직 세 줄만 출력, 따옴표 포함. 추가 설명/불릿/번호/코드블록 절대 금지.
            Raw Query: "<query>"
            Preprocessed Query: "<전처리된_쿼리(핵심 품목 + 유의미 속성만, ‘용’ 제거 후 표준형)>"
            Category Search Text: "<상품을_완전한_문장으로_설명하는_자연스러운_한국어_문장>"
        """    
    )



    # 🔥 맥락 반영 강제 사용자 메시지 구성
    user_message = f"""
        🚨 현재 사용자 입력: "{query}"

        **필수 실행 단계:**
        1. 위 이전 대화 이력에서 상품명/조건 추출
        2. 현재 입력과 이전 맥락을 강제로 결합
        3. 결합된 검색어로 전처리 수행

        반드시 이전 맥락을 반영한 검색어를 생성하세요!
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
                obj.get("final_query") or
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
    
    category_search_text = extract_category_text_new(llm_response)
    terms = extract_preprocessed(llm_response, query)
    preprocessed_query = strip_minus_terms(terms)
    
    # LLM 전처리된 결과에서 가격 조건 재추출 (SIZE_CONDITION은 무시)
    temp_price = extract_price_condition(preprocessed_query)
    if temp_price and not temp_price.startswith("SIZE_CONDITION_"):
        price_cond = temp_price
        print(f"[Debug] 유효한 가격 조건 재추출: {price_cond}")
    elif temp_price and temp_price.startswith("SIZE_CONDITION_"):
        print(f"[Debug] 크기 조건 감지됨, 기존 가격 조건 유지: {price_cond if price_cond else '제한 없음'}")
    else:
        print(f"[Debug] LLM 전처리된 쿼리에서 가격 조건 없음, 기존 조건 유지: {price_cond if price_cond else '제한 없음'}")
    
    # �🔥 맥락 반영 검증
    print(f"[맥락반영검증] 원본 입력: '{query}'")
    print(f"[맥락반영검증] LLM 처리 결과: '{preprocessed_query}'")
    print(f"[맥락반영검증] 맥락 반영 여부: {'✅ 반영됨' if query != preprocessed_query else '❌ 미반영'}")

    # Category Search Text가 비어있으면 preprocessed_query를 폴백으로 사용
    if not category_search_text or not isinstance(category_search_text, str) or not category_search_text.strip():
        print(f"[Fallback] Category Search Text가 비어있음, preprocessed_query 사용: '{preprocessed_query}'")
        category_search_text = preprocessed_query

    print("[Debug] Preprocessed Query_Before ->", terms)
    print("[Debug] Preprocessed Query ->", preprocessed_query)   #마이너스 같은걸 빼줌.
    print("[Debug] Category Search Text ->", category_search_text)

    

    # ================== LLM 카테고리 힌트 → 카테고리 임베딩 Top3 매칭 ==================
    try:
        cat_col
    except NameError:
        cat_col = Collection("ownerclan_category_Large")  # fields: id, category_full, embedding

    ###large 으로 변경
    def _embed_text_unit(text: str) -> np.ndarray:
        v = np.array(embedder_large.embed_query(text), dtype=np.float32)
        n = np.linalg.norm(v)
        if np.isfinite(n) and n != 0.0:
            v /= n
        return v

    def find_topk_category_names(cat_hint: str, k: int = 3) -> list:
        if not (isinstance(cat_hint, str) and cat_hint.strip()):
            return []
        v = _embed_text_unit(cat_hint)
        try:
            res = cat_col.search(
                data=[v],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 32}},
                limit=k,
                output_fields=["category_full"]
            )
        except Exception as e:
            print(f"[CatMatch] Milvus 검색 오류: {e}")
            return []

        out = []
        if res and res[0]:
            for hit in res[0]:
                name = hit.entity.get("category_full") or getattr(hit, "category_full", None)
                dist = float(getattr(hit, "distance", 0.0))
                if name:
                    out.append({"name": name, "distance": dist})
        out.sort(key=lambda x: x["distance"])   # 가까운 순
        return out



    # category_search_text를 사용해서 Top3 카테고리 매칭
    print(f"[Debug] 카테고리 검색 시작 - 검색어: '{category_search_text}'")
    print(f"[Debug] 검색어 타입: {type(category_search_text)}, 길이: {len(str(category_search_text))}")
    
    cat_match_results = []
    
    # 검색어 유효성 체크
    if not category_search_text or len(category_search_text.strip()) < 2:
        print(f"[Warning] 유효하지 않은 검색어, query로 폴백: '{query}'")
        category_search_text = query
    
    # 요약 문장으로 카테고리 벡터 검색 수행
    try:
        matches = find_topk_category_names(category_search_text, k=15)  # 15개 후보 가져오기
        print(f"[Debug] find_topk_category_names 결과: {len(matches)}개")
        
        if matches:
            print(f"[Debug] 상위 5개 매치:")
            for i, m in enumerate(matches[:5], 1):
                print(f"  {i}. {m.get('name', 'N/A')} (L2={m.get('distance', 0.0):.6f})")
        else:
            print(f"[Warning] 매치 결과가 없음")
            
    except Exception as e:
        print(f"[Error] find_topk_category_names 오류: {e}")
        matches = []
    
    seen_names = set()
    
    # 상위 3개 카테고리 선택 (중복 제거)
    for m in matches[:3]:
        name = m.get("name", "")
        if name and name not in seen_names:
            cat_match_results.append({"input": category_search_text, "matches": [m]})
            seen_names.add(name)
            print(f"\n[CatMatch] '{category_search_text}' → '{name}' (L2={float(m.get('distance',0.0)):.6f})")
            
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
        print(f"  {idx}. {r['input']} → {r['name']} (L2={r['distance']:.6f})")





    

    # # --- 쿼리 임베딩 (L2 정규화) 카테고리 임베딩---
    q_vec = np.array(embedder.embed_query(preprocessed_query), dtype=np.float32)
    n = np.linalg.norm(q_vec)
    if np.isfinite(n) and n != 0.0:
        q_vec = q_vec / n
    print(f"[Debug] q_vec dim: {q_vec.shape}, norm: {np.linalg.norm(q_vec):.4f}")

    # --- ownerclan_category에서 L2로 Top5 카테고리 검색 (방법2에 해당하는 카테고리를 임베딩 벡터검색)---

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
    # cat_col = Collection("ownerclan_category_Large")  # 스키마: id, category_full, embedding ...




    # 1) 전 카테고리 1000개 벡터 검색 (global_top3 제외)
    print("\n[방법1/RRF] 1000개 벡터 검색 시작 (방법2 카테고리 제외)")

    # global_top3에서 제외할 카테고리명 추출
    excluded_categories = [item.get("name", "") for item in global_top3 if item.get("name")]
    print(f"[Debug] 제외할 카테고리: {excluded_categories}")

    # 검색 조건 구성
    search_conditions = []

    # 가격/크기 조건 처리
    if price_cond and not price_cond.startswith("SIZE_CONDITION_"):
        search_conditions.append(price_cond)
        print(f"[Debug] Milvus 검색에 가격 조건 적용: {price_cond}")
    elif price_cond and price_cond.startswith("SIZE_CONDITION_"):
        # SIZE_CONDITION을 실제 검색 가능한 조건으로 변환
        # SIZE_CONDITION_인치_>=_30.0 → title LIKE '%30인치%' (제목에서 크기 검색)
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

    # 카테고리 제외 조건 추가 (Milvus 호환 문법 사용)
    if excluded_categories:
        # 빈 문자열 제거 및 정제
        valid_categories = [cat.strip() for cat in excluded_categories if cat.strip()]
        
        if valid_categories:
            # Milvus에서 지원하는 not in 연산자 사용
            if len(valid_categories) == 1:
                category_exclude_expr = f"category_name != '{valid_categories[0]}'"
            else:
                category_list = "', '".join(valid_categories)
                category_exclude_expr = f"category_name not in ['{category_list}']"
            
            search_conditions.append(category_exclude_expr)    # 최종 검색 조건 생성
    final_search_expr = " && ".join(search_conditions) if search_conditions else None

    print(f"[Debug] 방법1 검색 조건: {final_search_expr}")


    vector_hits_1000 = collection.search(
        data=[q_vec],
        anns_field="emb",
        param={"metric_type":"L2","params":{"nprobe":64}},
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
    vector_items = []
    for hits in vector_hits_1000:
        for idx, hit in enumerate(hits):
            item = _build_info_from_hit(hit)            # 제목/카테고리만 담긴 (본문 X)
            item["vector_match_score"] = 1000 - idx     # 벡터 1등=1000, 1000등=1 (순위 신호로만 사용)
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
        it["rrf_all"]   = (rrf_vec + rrf_title) / 2.0

        # (로그용) 1000점 환산도 함께 저장
        it["vecScore1000"]   = _rrf_to_1000(rrf_vec, BASE_K)
        it["titleScore1000"] = _rrf_to_1000(rrf_title, BASE_K)

    # 3) RRF 점수 계산 후 결과 출력
    # print("\n[방법1/RRF] RRF 점수 계산 후 결과(샘플):")
    for idx, item in enumerate(vector_items[:10], 1):
        code = item.get("상품코드")
        rv = vector_rank.get(code); rt = title_rank.get(code)

        # print(
        #     f"{idx}. {item['제목']} | {item['카테고리']} | "
        #     f"벡터RRF={item['rrf_vec']:.6f} (등수={rv}, score≈{item['vecScore1000']:.1f}) | "
        #     f"제목직접RRF={item['rrf_title']:.6f} (등수={rt}, score≈{item['titleScore1000']:.1f}) | "
        #     f"최종RRF={item['rrf_all']:.6f}"
        # )

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

    method1_all_sets = []
    for i in range(0, len(final_results), 10):
        block = final_results[i:i+10]
        if len(block) == 10:
            method1_all_sets.append(block)

    print(f"\n[방법1/RRF] 세트 수: {len(method1_all_sets)}  (총 {sum(len(s) for s in method1_all_sets)}개)")
    if method1_all_sets:
        print("\n[방법1/RRF] 세트1 미리보기 (RRF최종 점수로 Sort완료)")
        for idx, it in enumerate(method1_all_sets[0], 1):
            print(
                f"{idx}. {it['제목']} | {it['카테고리']} | {it['가격']:,}원 | "
                f"vecRRF={it['rrf_vec']:.6f} | titleRRF={it['rrf_title']:.6f} | "
                f"vec≈{it['vecScore1000']:.1f} / title≈{it['titleScore1000']:.1f}| "
                f"RRF(mean-2)={it['rrf_all']:.6f}"
            )
        print("\n[방법1/RRF] 세트2 미리보기 (RRF최종 점수로 Sort완료)")
        for idx, it in enumerate(method1_all_sets[1], 1):
            print(
                f"{idx}. {it['제목']} | {it['카테고리']} | {it['가격']:,}원 | "
                f"vecRRF={it['rrf_vec']:.6f} | titleRRF={it['rrf_title']:.6f} |  "
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
                param={"metric_type":"L2","params":{"nprobe":64}},
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
            items = []
            for hits in vector_hits:
                for idx, hit in enumerate(hits):
                    item = _build_info_from_hit(hit)
                    item["vector_match_score"] = size - idx
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
                it["rrf_all"] = (rrf_vec + rrf_title) / 2.0
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

    # 각 카테고리별로 미리 10개씩 뽑아서 리스트 생성
    top1_list = []
    top2_list = []
    top3_list = []

    # Top1 카테고리에서 10개
    if top1_category:
        top1_results = search_by_category_method2(top1_category, size=100)
        top1_list = top1_results[:10]  # 상위 10개만
        print(f"[방법2] Top1 ({top1_category}): {len(top1_results)}개 중 10개 선택")

    # Top2 카테고리에서 10개  
    if top2_category:
        top2_results = search_by_category_method2(top2_category, size=100)
        top2_list = top2_results[:10]  # 상위 10개만
        print(f"[방법2] Top2 ({top2_category}): {len(top2_results)}개 중 10개 선택")

    # Top3 카테고리에서 10개
    if top3_category:
        top3_results = search_by_category_method2(top3_category, size=100)
        top3_list = top3_results[:10]  # 상위 10개만
        print(f"[방법2] Top3 ({top3_category}): {len(top3_results)}개 중 10개 선택")

    # 방법2 결과를 10개씩 세트로 구성 (2:2:1 비율 유지)
    method2_all_sets = []
    
    # 5싸이클 생성 (각 싸이클마다 2:2:1 비율로 5개씩)
    for cycle in range(5):
        cycle_items = []
        
        # Top1에서 2개 (cycle*2 인덱스부터)
        start_idx = cycle * 2
        if start_idx < len(top1_list) and start_idx + 1 < len(top1_list):
            cycle_items.extend(top1_list[start_idx:start_idx+2])
        elif start_idx < len(top1_list):
            cycle_items.extend(top1_list[start_idx:start_idx+1])
        
        # Top2에서 2개 (cycle*2 인덱스부터)
        if start_idx < len(top2_list) and start_idx + 1 < len(top2_list):
            cycle_items.extend(top2_list[start_idx:start_idx+2])
        elif start_idx < len(top2_list):
            cycle_items.extend(top2_list[start_idx:start_idx+1])
        
        # Top3에서 1개 (cycle 인덱스)
        if cycle < len(top3_list):
            cycle_items.extend(top3_list[cycle:cycle+1])
        
        # 5개 세트가 완성되면 추가 (10개로 채우지 않음!)
        if len(cycle_items) == 5:  # 정확히 5개인 경우만
            method2_all_sets.append(cycle_items)
        elif len(cycle_items) >= 3:  # 최소 3개 이상이면 그대로 추가
            method2_all_sets.append(cycle_items)

    print(f"\n[방법2] 총 {len(method2_all_sets)}개 세트 생성됨")
    
    # 첫 번째 세트 미리보기
    if method2_all_sets:
        print("\n[방법2] 첫 번째 세트 미리보기:")
        for idx, item in enumerate(method2_all_sets[0], 1):
            print(f"  {idx}. {item['제목']} | {item['카테고리']} | {item['가격']:,}원 | {item['검색방식']}")
    



    
    # #################################################################
    # #                         NEW 방법 2 end                          #
    # #################################################################

    
    
    # final_results 초기화 - 5세트 반복으로 50개 생성
    final_results = []
    used_codes = set()  # 전체 중복 방지용

    # 5세트 반복 생성 (총 50개: 10개씩 5세트)
    for set_num in range(5):
        print(f"\n[세트 {set_num + 1}] 생성 시작")
        
        # 각 세트마다 10개씩 생성
        set_items = []
        
        # 1. 방법2 결과 5개 먼저 추가
        if method2_all_sets and set_num < len(method2_all_sets):
            method2_items = method2_all_sets[set_num][:5]  # 해당 세트에서 5개
            method2_added = 0
            for item in method2_items:
                if item['상품코드'] not in used_codes:
                    set_items.append(item)
                    used_codes.add(item['상품코드'])
                    method2_added += 1
            print(f"[세트 {set_num + 1}] 방법2 결과 {method2_added}개 추가")
        
        # 2. 방법1 결과로 부족한 만큼 채우기 (최대 5개)
        if method1_all_sets and set_num < len(method1_all_sets):
            needed_count = 10 - len(set_items)  # 10개가 되도록 부족한 개수
            method1_added = 0
            
            for product in method1_all_sets[set_num]:  # 해당 세트에서
                if method1_added >= min(needed_count, 5):  # 최대 5개까지만
                    break
                if product['상품코드'] not in used_codes:
                    set_items.append(product)
                    used_codes.add(product['상품코드'])
                    method1_added += 1
            
            print(f"[세트 {set_num + 1}] 방법1 결과 {method1_added}개 추가 (중복 제거 후)")
        
        # 3. 현재 세트를 final_results에 추가
        final_results.extend(set_items)
        print(f"[세트 {set_num + 1}] 완성: {len(set_items)}개 (누적 총 {len(final_results)}개)")

    print(f"\n[최종표시] 총 {len(final_results)}개 상품이 {len(final_results)//10}개 세트로 구성됨")
    
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
    - 특정 품목/모델로 단정하지 말고, 후보 리스트의 특징을 근거로 질문하세요.

    - 고객이 원하는 상품을 정확히 찾을 수 있도록, 후보 상품의 특징을 활용해 고객의 선호를 파악하는 질문을 작성하세요.
    - 현재 시즌: {season}

    후보 상품 목록:
    {products_text}


    요청사항:
    1) 후보 상품의 제목/설명에서 서로를 구분해주는 특징 키워드 3~5개를 추출하세요.
    - 기능/성능: 방수, 저소음, 고출력, 대용량, 저전력, 무선/유선, 규격(예: 27W, 128GB, 1.5L 등)
    - 재질/마감: 가죽/스테인리스/TPU/친환경 등
    - 구성/형태: 세트/단품, 접이식, 휴대용, 벽걸이, 2+1 구성 등
    - 호환/범용/시즌: 정품/호환, 규격·사이즈, 여름/겨울 등
    2) 그 특징을 활용해 고객 의도 확인과 선호 파악을 위한 '확인형 질문'을 반드시! 
    3) 상품 코드나 구체 모델명/스펙 나열은 금지합니다. 후보에서 추출한 '특징 키워드'만 요약해 언급하세요.
    4) 친근하면서도 전문적인 대화체를 유지하세요.
    5) 아래 예시는 참고만 하며, 그대로 복사하지 말고 실제 후보의 특징으로 대체하세요.

    답변 형식(단락 예시):
    심플한 구성이 더 좋으실까요? 사용하실 환경과 선호 가격대(프리미엄/실속), 디자인·재질, 그리고 본품/관련품 중 어떤 쪽을 찾으시는지도 알려주시면 더 정확히 추천해 드리겠습니다.
    
    **무조건 어떤 언어가 들어와도 {target_lang}로만 답변하세요**
    **무조건 답변은 한글은 200자 이내,영어는 230자 이내로 자세하게 작성하되 요약본으로 답변을 작성하세요.

    """

    print("target_lang->", target_lang)
    # - 예시 어조: "후보를 보니 [특징A/특징B/특징C] 옵션이 보이는데, 이런 조건이 필요하신가요?"
    # - 가격대(프리미엄/일반/실속), 디자인, 용도, 재질, 브랜드 선호, 계절감, 옵션 유무를 과도한 나열 없이 자연스럽게 묻으세요.

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
    
    # 🎯 LLM 리랭킹: 상위 10개 상품을 사용자 쿼리에 맞게 재정렬
    top_10_products = final_results[:10]
    
    # 리랭킹용 프롬프트 생성
    products_for_ranking = []
    for idx, item in enumerate(top_10_products):
        products_for_ranking.append(f"{idx}. {item['제목']}")
    
    # 🎯 리랭킹용 쿼리 선택: 누적쿼리 우선 사용
    if len(user_query_parts) > 1 and combined_query != query:
        ranking_query = combined_query  # 누적된 대화 맥락 활용
        query_type = "누적쿼리"
    else:
        ranking_query = query  # 단일 질문인 경우 원본 쿼리
        query_type = "원본쿼리"
    
    print(f"[LLM 리랭킹] {query_type} 사용: '{ranking_query}'")
    
    # 🔍 리랭킹 전 상품 제목들 출력
    print(f"\n{'='*60}")
    print(f"📋 [리랭킹 전] 상위 10개 상품 제목:")
    print(f"{'='*60}")
    for idx, item in enumerate(top_10_products):
        print(f"{idx}. {item['제목']}")
    print(f"{'='*60}")
    
    ranking_prompt = f"""사용자가 "{ranking_query}"를 검색했습니다.

다음 10개 상품을 사용자의 검색 의도에 가장 적합한 순서대로 0부터 9까지 번호를 매겨 재정렬해주세요.

상품 목록:
{chr(10).join(products_for_ranking)}

응답 형식: 0,1,2,3,4,5,6,7,8,9 (쉼표로 구분된 숫자만)
예시: 2,0,5,1,8,3,7,4,9,6

재정렬된 순서:"""

    try:
        # LLM으로 리랭킹 순서 받기
        rerank_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": ranking_prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        rerank_order = rerank_response.choices[0].message.content.strip()
        print(f"[LLM 리랭킹] 새로운 순서: {rerank_order}")
        
        # 순서 파싱 및 적용
        try:
            order_indices = [int(x.strip()) for x in rerank_order.split(',')]
            if len(order_indices) == 10 and all(0 <= x <= 9 for x in order_indices):
                # 리랭킹 적용
                reranked_products = [top_10_products[i] for i in order_indices]
                final_results[:10] = reranked_products  # 상위 10개만 교체
                print(f"[LLM 리랭킹] 성공적으로 재정렬됨")
                
                # 🔍 리랭킹 후 상품 제목들 출력
                print(f"\n{'='*60}")
                print(f"🎯 [리랭킹 후] 재정렬된 상위 10개 상품 제목:")
                print(f"{'='*60}")
                for idx, item in enumerate(reranked_products):
                    print(f"{idx}. {item['제목']}")
                print(f"{'='*60}")
                
            else:
                print(f"[LLM 리랭킹] 잘못된 형식, 원본 순서 유지")
                print(f"🔍 원본 순서 그대로 사용")
        except (ValueError, IndexError) as e:
            print(f"[LLM 리랭킹] 파싱 오류: {e}, 원본 순서 유지")
            
    except Exception as e:
        print(f"[LLM 리랭킹] 오류 발생: {e}, 원본 순서 유지")
        print(f"\n{'='*60}")
        print(f"⚠️ [리랭킹 실패] 원본 순서 그대로 사용:")
        print(f"{'='*60}")
        for idx, item in enumerate(top_10_products):
            print(f"{idx}. {item['제목']}")
        print(f"{'='*60}")
    
    print("\n🎯 최종 리랭킹된 상위 10개 상품:")
    for idx, item in enumerate(final_results[:10], 1):
        print(f"\n{idx}. {item['제목']}")
        print(f"   카테고리: {item['카테고리']}")
        print(f"   가격: {item['가격']:,}원")


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






if __name__ == '__main__':
    import uvicorn
    
    # 환경 변수에서 설정 가져오기
    host = '0.0.0.0'
    port = 8011
    debug = True
    
    print(f"🚀 FastAPI 서버 시작: {host}:{port} (debug={debug})")
    uvicorn.run("app:app", host=host, port=port, reload=debug)