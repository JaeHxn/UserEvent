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

from langdetect import detect
from collections import defaultdict, Counter
import math
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

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
from langchain.docstore.document import Document

#리랭킹 관련 임포트#
from transformers import AutoModelForSequenceClassification, AutoTokenizer




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
LLM_MODEL  = "gpt-4.1-mini"
EMB_MODEL  = "text-embedding-3-small"

# 클라이언트 및 래퍼
client    = OpenAIClient(api_key=API_KEY)
llm       = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)    # ← embedder 정의 추가
API_URL = "https://fb-narosu.duckdns.org"  # 예: http://114.110.135.96:8011

#사용자 이벤트 관리자 대시보드
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE_SECONDS", "10800")) # 3시간
ADMIN_COOKIE_NAME = os.getenv("ADMIN_COOKIE_NAME", "admin_session")

# 1) Milvus 서버에 먼저 연결
connections.connect(
    alias="default",
    host=MILVUS_HOST,    # 예: "114.110.135.96"
    port=MILVUS_PORT     # 예: "19530"
)
print("✅ Milvus에 연결되었습니다.")

collection_cat = Collection("ownerclan_category")
results = collection_cat.query(
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
    lang_code = detect(query)

    # ✅ Redis 세션 기록 불러오기 및 최신 입력 저장
    session_history = get_session_history(session_id)
    session_history.add_user_message(query)

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
        # 숫자 단위 정규화
        def normalize_price_units(text: str) -> str:
            # 한글 단위
            kr_unit_map = {
                "천": "000",
                "만": "0000",
                "십만": "00000",
                "백만": "000000",
                "천만": "0000000",
                "억": "00000000"
            }
            # 영어 단위
            en_unit_map = {
                "k": "000",
                "thousand": "000",
                "m": "000000",
                "million": "000000"
            }
            
            # 숫자가 없는 "만원" 처리
            text = re.sub(r'(?<!\d)만원', '10000원', text)
            
            # 한글 단위 변환 패턴 개선
            for unit, zeros in kr_unit_map.items():
                pattern = f'(\d+)\s*{unit}(?:원)?'  # "원" 옵션으로 처리
                text = re.sub(pattern, lambda m: f"{m.group(1)}{zeros}", text)
            
            # 영어 단위 변환 (대소문자 무관)
            text = text.lower()
            for unit, zeros in en_unit_map.items():
                pattern = f'(\d+)\s*{unit}'
                text = re.sub(pattern, lambda m: f"{m.group(1)}{zeros}", text)
            
            # 모든 숫자를 원 단위로 통일
            text = re.sub(r'(\d+)(?:won|dollars|usd)', r'\1원', text)
            
            print(f"[Debug] normalize_price_units 결과: {text}")  # 디버그 로그 추가
            return text

        # 쿼리 정규화 및 디버깅
        query = normalize_price_units(text.lower())
        print(f"[Debug] 정규화된 쿼리: {query}")

        # 가격 범위 패턴 (한글 + 영어)
        range_patterns = [
            # 한글 복합 범위 패턴
            r'(\d+)[^\d]*원?\s*이하\s*(\d+)[^\d]*원?\s*이상',  # "2만원 이하 1만원 이상"
            r'(\d+)[^\d]*원?\s*초과\s*(\d+)[^\d]*원?\s*미만',  # "1만원 초과 2만원 미만"
            r'(\d+)[^\d]*원?\s*(?:~|에서|부터)\s*(\d+)[^\d]*원?',
            # 영어 범위 패턴
            r'between\s*(\d+)\s*and\s*(\d+)(?:\s*원?)',
            r'from\s*(\d+)\s*to\s*(\d+)(?:\s*원?)',
            r'(\d+)\s*(?:to|-|~)\s*(\d+)(?:\s*원?)',
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
                    # "이하-이상" 패턴인 경우 순서 바꿔서 처리
                    if "이하" in pattern and "이상" in pattern:
                        max_price = int(m.group(1))  # 2만원 (이하)
                        min_price = int(m.group(2))  # 1만원 (이상)
                    else:
                        min_price = int(m.group(1))
                        max_price = int(m.group(2))
                    print(f"[Debug] 가격 범위 감지: {min_price}원 ~ {max_price}원")
                    return f"market_price >= {min_price} && market_price <= {max_price}"

            # 2. 단일 가격 검색 시도
            for pattern in single_patterns:
                m = re.search(pattern, query)
                if m:
                    # 한글 패턴 매칭
                    amount = int(m.group(1))
                    comp = m.group(2) if len(m.groups()) > 1 else None

                    # "원" 패턴은 "이상"으로 처리 (예: "1만원" → "1만원 이상")
                    if not comp:
                        if "원" in pattern:
                            print(f"[Debug] 단순 가격 감지: {amount}원 (이상으로 처리)")
                            return f"market_price >= {amount}"
                        continue

                    # 한글 연산자
                    if comp in op_map_kr:
                        price_op = op_map_kr[comp]
                        print(f"[Debug] 한글 가격 조건 감지: {amount}원 {comp}")
                        return f"market_price {price_op} {amount}"

                    # 영어 연산자
                    for op_text, op_symbol in op_map_en.items():
                        if op_text in query.lower():
                            print(f"[Debug] 영어 가격 조건 감지: {op_text} {amount}")
                            return f"market_price {op_symbol} {amount}"

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
        "목도리","스카프","머플러","장갑","내복","롱존","이너웨어",
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

    # 적용: 4계절 모두
    # if season in ("봄","여름","가을","겨울"):
    #     # sorted_cats는 이미 시즌 보정이 적용된 상태이므로 별도 정렬 불필요
    #     print("🛠 시즌 보정 적용된 카테고리 순위:")
    #     for i,(name,dist) in enumerate(sorted_cats,1):
    #         adj_score = get_adjusted_score(name, dist, season)
    #         print(f"  {i}. {name} | adj_L2={adj_score:.6f}")


    # 대화 이력 가져오기
    history_messages = [msg.content for msg in session_history.messages]
    conversation_context = "\n".join([f"이전 대화: {msg}" for msg in history_messages[-10:]]) if history_messages else "이전 대화 없음"

    system_prompt = (
        f"""System:
            당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
            입력 언어가 무엇이든 먼저 한국어로 의미 보존 번역을 수행합니다.
            
            [대화 컨텍스트]
            {conversation_context}를 잘보고 사용자가 무엇을 찾고자하는지 파악한 다음 그 키워드를 구성해서 다음작업을 진행하세요.!!


            [전처리 원칙]
            1) 사용자가 모호한 상위개념으로 묻는 경우(예: "한국식 과자", "여름 원피스", "세차용품") → 
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
        - “<대상>에게 좋은/에 좋은/맞는/추천”은 의도 핵심이므로 반드시 유지(필요 시 ‘추천’으로 표준화 가능)
        - 핵심 품목이 명확하지 않으면 원문 유지(축약·추정 금지)

            [출력 규칙(반드시 정확히 준수)]
            오직 두 줄만 출력, 따옴표 포함. 추가 설명/불릿/번호/코드블록 절대 금지.
            Raw Query: "<query>"
            Preprocessed Query: "<전처리된_쿼리(핵심 품목 + 유의미 속성만, ‘용’ 제거 후 표준형)>"
        """    
    )



    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ]
    )
    llm_response = resp.choices[0].message.content.strip()
    print("[Debug] LLM full response:\n", llm_response)  # ← 여기에!






    # print("session_history.messages:",session_history.messages)
    # print("session_history:",session_history)

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
    
    terms = extract_preprocessed(llm_response, query)
    
    preprocessed_query = strip_minus_terms(terms)

    print("[Debug] Preprocessed Query_Before ->", terms)
    print("[Debug] Preprocessed Query ->", preprocessed_query)









    # # --- 쿼리 임베딩 (L2 정규화) 카테고리 임베딩---
    q_vec = np.array(embedder.embed_query(preprocessed_query), dtype=np.float32)
    n = np.linalg.norm(q_vec)
    if np.isfinite(n) and n != 0.0:
        q_vec = q_vec / n
    print(f"[Debug] q_vec dim: {q_vec.shape}, norm: {np.linalg.norm(q_vec):.4f}")

    # --- ownerclan_category에서 L2로 Top5 카테고리 검색 (방법2에 해당하는 카테고리를 임베딩 벡터검색)---
    CAT_COLLECTION = "ownerclan_category"

    def get_top5_categories_from_embeddings(qv: np.ndarray):
        cat_col = Collection(CAT_COLLECTION)
        res = cat_col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type":"L2","params":{"nprobe":64}},
            limit=5,
            output_fields=["category_full"]
        )
        top5 = [(hit.entity.get("category_full"), float(hit.distance)) for hit in res[0]]
        print("🔎 카테고리 Top5 (L2, 작을수록 유사):")
        for i,(name,dist) in enumerate(top5,1):
            print(f"  {i}. {name} | L2={dist:.6f}")
        return top5

    # # 상위 카테고리에 대해 각각 상품 검색 (시즌 보정 적용)
    def get_adjusted_score(category_name: str, score: float, season: str) -> float:
        """시즌에 따른 점수 보정"""
        season_adj = _season_adjust(category_name, season)
        return score + season_adj

    def _has_any(text: str, words) -> bool:
        t = (text or "").lower()
        return any(w.lower() in t for w in words)

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
    # #                          방법 1 시작                                #
    # #################################################################


    # # 1) 전 카테고리 1000개 벡터 검색 후 점수 부여
    # print("\n[방법1] 1000개 벡터 검색 시작")
    # vector_hits_1000 = collection.search(
    #     data=[q_vec],
    #     anns_field="emb",
    #     param={"metric_type":"L2","params":{"nprobe":64}},
    #     limit=1000,  # 1000개 검색
    #     output_fields=[
    #         "product_code","category_code","category_name","market_product_name",
    #         "market_price","shipping_fee","shipping_type","max_quantity",
    #         "composite_options","image_url","manufacturer","model_name",
    #         "origin","keywords","description","return_shipping_fee",
    #     ]
    # )
    

    # print("\n[방법1] 벡터 검색 결과 1000개에서 직접 검색 및 점수 결합 시작")
    

    # # 1. 벡터 검색 결과에 점수 부여 (1000점 → 1점)
    # vector_items = []
    # for hits in vector_hits_1000:
    #     for idx, hit in enumerate(hits):
    #         item = _build_info_from_hit(hit)
    #         item['vector_match_score'] = 1000 - idx  # 1등: 1000점, 1000등: 1점
    #         vector_items.append(item)
    # print(f"[방법1] 벡터 검색 1000개 완료: {len(vector_items)}개")


    # # 편의 함수  (직접검색 점수 계산인데 아직 안씀..?)
    # def DirectSearch(user_text: str, items: List[Union[str, Dict[str, Any]]],
    #         fields: Optional[Iterable[str]] = None,
    #         near_threshold: float = 0.90,
    #         return_query_meta: bool = False) -> List[Dict[str, Any]]:
    #     ds = Ranker_DirectSearch(near_threshold=near_threshold)
    #     return ds.score_items(user_text, items, fields=fields, return_query_meta=return_query_meta)

    # tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]

    # ds = Ranker_DirectSearch()
    # qmeta = ds.prepare_query(preprocessed_query)
    # print("원문:", qmeta["original"])
    # print("정규화:", qmeta["normalized"])
    # print("토큰:", qmeta["tokens"])
    # print("정제쿼리(canonical):", qmeta["canonical_query"])

    # direct_items = []

    # # 1) 직접검색 매칭 점수(0~1000) 계산: 대상 = 벡터검색 결과 리스트 전체(vector_items)
    # for item in vector_items:
    #     title = item.get('제목', '')               # '제목'이 항상 있으면 item['제목'] 사용해도 됨
    #     item['direct_match_score'] = ds.score_text(preprocessed_query, title) # 0~1000점 연속값

    # # 2) 벡터검색 순위 점수 부여 (1000 → 1점)
    # for idx, item in enumerate(vector_items):
    #     item['vector_rank_score'] = 1000 - idx  # 1등: 1000점, 2등: 999점, ...

    # # 3) 직접검색 순위 점수 부여 (1000 → 1점)
    # direct_sorted = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for idx, item in enumerate(direct_sorted):
    #     item['direct_rank_score'] = 1000 - idx  # 1등: 1000점, 2등: 999점, ...

    # # 4) 최종 점수 계산
    # for item in vector_items:
    #     # 벡터중심 변환점수 (벡터 0.7 + 직접 0.3)
    #     item['vector_final_score'] = (0.7 * item['vector_rank_score'] + 0.3 * item['direct_rank_score'])
    #     # 직접중심 변환점수 (벡터 0.3 + 직접 0.7)
    #     item['direct_final_score'] = (0.3 * item['vector_rank_score'] + 0.7 * item['direct_rank_score'])
        
    # print("\n[방법1] 점수 정규화 및 최종 점수 계산 완료")
    # print("상위 5개 항목 점수 분포:")
    # sorted_items = sorted(vector_items, key=lambda x: x['direct_final_score'], reverse=True)[:5]
    # for idx, item in enumerate(sorted_items, 1):
    #     print(f"\n{idx}. {item['제목']}")
    #     print(f"   직접오리지널점수: {item['direct_match_score']:.1f}, 벡터매칭: {item['vector_match_score']:.1f}")
    #     print(f"   직접검색 1000점을 변환점수: {item['direct_rank_score']:.1f}")
    #     print(f"   벡터검색 1000점을 변환점수: {item['vector_rank_score']:.1f}")
    #     print(f"   최종점수: {item['direct_final_score']:.1f} (직접중심) / {item['vector_final_score']:.1f} (벡터중심)")


    # ranked = DirectSearch(preprocessed_query, vector_items, fields=("제목",))
    # print(f"{'Rank':>4}  {'Score':>5}  제목")
    # print("-"*48)
    # for i, r in enumerate(ranked[:30], 1):
    #     item = r["item"]
    #     title = item["제목"] if isinstance(item, dict) else str(item)
    #     print(f"{i:>4}  {r['direct_text_score']:>5}  {title}")


    # # direct_match_score 기준 내림차순 정렬 → 직접검색 "랭크 점수"(1000→1) 대입
    # direct_items = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for r_idx, it in enumerate(direct_items):
    #     it['direct_score'] = max(1, 1000 - r_idx)  # 1등 1000, 1000등 1

    # # direct_score가 없는 항목(이론상 없지만 안전용) 0으로 보정
    # for it in vector_items:
    #     if 'direct_score' not in it:
    #         it['direct_score'] = 0

    # print(f"[방법1] 직접 검색 매칭: {len(direct_items)}개")

    # # 3) 점수 통합 (벡터/직접오리지널점수 점수 결합)
    # for item in vector_items:
    #     # 벡터기반 선발
    #     item['vector_focused_score'] = 0.7*item['vector_match_score'] + 0.3*item['direct_score']  
    #     # 직접기반 선발
    #     item['direct_focused_score'] = 0.3*item['vector_match_score'] + 0.7*item['direct_score']
    # print("\n[방법1] 점수 계산 완료")

    # # === 최종 10개 선정 (직접 7 + 벡터 3) — 상품코드 중복 제거 ===
    # raw_candidates = []
    # seen_codes = set()







    # # 1. 벡터 검색 결과 1000개에 대한 순위 점수 부여 (1000점 → 1점)
    # for idx, item in enumerate(vector_items):
    #     item['vector_rank_score'] = max(1, 1000 - idx)  # 1등: 1000점, 1000등: 1점

    # # 2. 직접 검색 결과에 대한 순위 점수 부여 (1000점 → 1점)
    # direct_sorted = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for idx, item in enumerate(direct_sorted):
    #     item['direct_rank_score'] = max(1, 1000 - idx)  # 1등: 1000점, 마지막등: 1점
    
    # # 3. 최종 점수 계산
    # for item in vector_items:
    #     # 벡터 중심 점수 (벡터 0.7 + 직접 0.3)
    #     item['vector_final_score'] = (
    #         0.7 * item['vector_rank_score'] +
    #         0.3 * item['direct_rank_score']
    #     )
    #     # 직접 중심 점수 (벡터 0.3 + 직접 0.7)
    #     item['direct_final_score'] = (
    #         0.3 * item['vector_rank_score'] +
    #         0.7 * item['direct_rank_score']
    #     )
    
    # # 직접검색 점수가 같은 경우 벡터 점수로 2차 정렬하는 key 함수
    # def direct_sort_key(item):
    #     return (
    #         item['direct_final_score'],  # 1차 정렬: 직접검색 점수
    #         item['vector_final_score']   # 2차 정렬: 벡터 점수
    #     )
    
    # direct_sorted = sorted(vector_items, key=direct_sort_key, reverse=True)

    # print("\n[방법1] 직접검색 중심 상위 10개 점수 분포:")
    # for idx, item in enumerate(direct_sorted[:10], 1):
    #     print(f"{idx}. {item['제목']}")
    #     print(f"   벡터순위점수: {item['vector_rank_score']:.1f}")
    #     print(f"   직접순위점수: {item['direct_rank_score']:.1f}")
    #     print(f"   직접오리지날점수: {item['direct_match_score']:.1f}")
    #     print(f"   최종점수(직접중심): {item['direct_final_score']:.1f}")

    # # 직접검색 상위 3개 선정 (점수가 0인 항목 제외)
    # for item in direct_sorted:
    #     if len(raw_candidates) >= 3:
    #         break
    #     code = item.get('상품코드')
    #     if code and code not in seen_codes:
    #         # 직접 검색 점수가 0점이면 건너뛰기
    #         if item['direct_rank_score'] == 0:
    #             continue
                
    #         seen_codes.add(code)
    #         raw_candidates.append(item)
    #         print(f"\n[방법1] 직접검색 선정 {len(raw_candidates)}/3:")
    #         print(f"제목: {item['제목']}")
    #         print(f"점수: 벡터={item['vector_rank_score']:.1f}, "
    #               f"직접={item['direct_rank_score']:.1f}, "
    #               f"최종(직접)={item['direct_final_score']:.1f}, "
    #               f"최종(벡터)={item['vector_final_score']:.1f}")

    # # 5. 벡터 검색 중심으로 7개 선정 (중복 제외)
    # vector_sorted = sorted(vector_items, key=lambda x: x['vector_final_score'], reverse=True)

    # print("\n[방법1] 벡터검색 중심 상위 10개 점수 분포:")
    # for idx, item in enumerate(vector_sorted[:10], 1):
    #     print(f"{idx}. {item['제목']}")
    #     print(f"   벡터순위점수: {item['vector_rank_score']:.1f}")
    #     print(f"   직접오리지날점수: {item['direct_match_score']:.1f}")
    #     print(f"   직접순위점수: {item['direct_rank_score']:.1f}")
    #     print(f"   최종점수(벡터중심): {item['vector_final_score']:.1f}")

    # # 벡터검색에서 7개 선정 (직접검색과 중복 시 다음 순위로 대체)
    # vector_selections = []
    # current_index = 0
    
    # # 벡터검색 결과에서 7개를 채울 때까지 반복
    # while len(vector_selections) < 7 and current_index < len(vector_sorted):
    #     item = vector_sorted[current_index]
    #     code = item.get('상품코드')
        
    #     # 이미 직접검색에서 선정된 상품이면 다음 순위로
    #     if code in seen_codes:
    #         current_index += 1
    #         continue
            
    #     seen_codes.add(code)
    #     vector_selections.append(item)
    #     raw_candidates.append(item)
    #     print(f"\n[방법1] 벡터검색 선정 {len(vector_selections)}/7:")
    #     print(f"제목: {item['제목']}")
    #     print(f"점수: 벡터={item['vector_rank_score']:.1f}, "
    #           f"직접={item['direct_rank_score']:.1f}, "
    #           f"최종={item['vector_final_score']:.1f}")
        
    #     current_index += 1

    # # 모든 상품을 하나의 리스트로 순차적으로 수집
    # final_results = []
    # seen_codes = set()
    
    # # 1. 먼저 직접검색 상위 결과들을 3개씩 수집
    # direct_count = 0
    # for item in direct_sorted:
    #     if direct_count >= 15:  # 5세트 × 3개 = 15개
    #         break
    #     code = item.get('상품코드')
    #     if code and code not in seen_codes:
    #         if item['direct_rank_score'] == 0:  # 직접 검색 점수가 0점이면 건너뛰기
    #             continue
    #         seen_codes.add(code)
    #         final_results.append(item)
    #         direct_count += 1
            
    # # 2. 그 다음 벡터검색 결과들을 7개씩 수집
    # vector_count = 0
    # for item in vector_sorted:
    #     if vector_count >= 35:  # 5세트 × 7개 = 35개
    #         break
    #     code = item.get('상품코드')
    #     if code and code not in seen_codes:
    #         seen_codes.add(code)
    #         final_results.append(item)
    #         vector_count += 1
    
    # # 3. 10개씩 세트로 분할
    # all_sets = []
    # for i in range(0, len(final_results), 10):
    #     current_set = final_results[i:i+10]
    #     if len(current_set) == 10:  # 완전한 세트만 추가
    #         all_sets.append(current_set)
    
    # # 모든 세트 출력
    # print(f"\n[방법1] 전체 {len(all_sets)}개 세트 구성 완료")
    # for set_num, set_items in enumerate(all_sets, 1):
    #     print(f"\n=== 세트 {set_num} (총 {len(set_items)}개 상품) ===")
    #     for idx, item in enumerate(set_items, 1):
    #         print(f"\n{idx}. {item['제목']}")
    #         print(f"   카테고리: {item['카테고리']}")
    #         print(f"   가격: {item['가격']:,}원")
    #         print(f"   점수: 벡터={item['vector_rank_score']:.1f}, "
    #               f"직접={item['direct_rank_score']:.1f}, "
    #               f"최종={item['vector_final_score']:.1f}")
    
    # print(f"\n[방법1] 총 상품 수: {sum(len(s) for s in all_sets)}개")
    # print("\n[방법1] 최종 선정 10개 리스트:", *[(item['상품코드'], item['제목'], item['가격']) for item in all_sets[0]], sep="\n")
    # print("\n[방법1] 2번째 최종 선정 10개 리스트:", *[(item['상품코드'], item['제목'], item['가격']) for item in all_sets[1]], sep="\n")
    # print("\n[방법1] 3번째 최종 선정 10개 리스트:", *[(item['상품코드'], item['제목'], item['가격']) for item in all_sets[2]], sep="\n")






    # #################################################################
    # #                          방법 1 끝                             #
    # #################################################################







    # #################################################################
    # #                         NEW 방법 1 시작                                #
    # #################################################################
    ####메타 데이터의  카테고리와 임베딩 된 카테고리를 서로 매칭 시키기 위한 로직.
    cat_col = Collection("ownerclan_category")  # 스키마: id, category_full, embedding ...

    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", (s or "").strip().lower())

    # 질의 벡터(q_vec)로 카테고리 ANN 검색
    cat_hits = cat_col.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type":"L2","params":{"nprobe":32}},
        limit=200,
        output_fields=["id","category_full"]
    )



    # 카테고리명 기준의 랭크 테이블 (1등=1)
    cat_rank_by_name = {}
    rank_counter = 1
    for hits in cat_hits:
        for h in hits:
            cat_name = (h.get("category_full") 
                        or getattr(h, "category_full", None) 
                        or (hasattr(h, "fields") and h.fields.get("category_full")))
            if not cat_name: 
                continue
            key = _norm(cat_name)
            if key not in cat_rank_by_name:
                cat_rank_by_name[key] = rank_counter
                rank_counter += 1


    # 1) 전 카테고리 1000개 벡터 검색
    print("\n[방법1/RRF] 1000개 벡터 검색 시작")
    vector_hits_1000 = collection.search(
        data=[q_vec],
        anns_field="emb",
        param={"metric_type":"L2","params":{"nprobe":64}},
        limit=1000,
        expr=price_cond,  # 가격 조건 추가
        output_fields=[
            "product_code","category_code","category_name","market_product_name",
            "market_price","shipping_fee","shipping_type","max_quantity",
            "composite_options","image_url","manufacturer","model_name",
            "origin","keywords","description","return_shipping_fee",
        ]
    )
    print(f"[Debug] 가격 조건: {price_cond if price_cond else '제한 없음'}")

    
    # 검색 결과 -> dict 리스트
    vector_items = []
    for hits in vector_hits_1000:
        for idx, hit in enumerate(hits):
            item = _build_info_from_hit(hit)            # ✨ 제목/카테고리만 담긴 info (본문 X)
            item["vector_match_score"] = 1000 - idx     # 벡터 1등=1000, 1000등=1 (순위 신호로만 사용)
            vector_items.append(item)

    print(f"[방법1/RRF] 벡터 검색 1000개 완료: {len(vector_items)}개")

    # ---------- 직접매칭(문자열) 점수: '제목'과 '카테고리'만 ----------
    ds = Ranker_DirectSearch()
    for it in vector_items:
        it["direct_title_score"] = ds.score_text(preprocessed_query, it.get("제목", ""))
        it["direct_cate_score"]  = ds.score_text(preprocessed_query, it.get("카테고리", ""))

        # 카테고리 임베딩 순위(이름 기반 조회)
        cat_key = _norm(it.get("카테고리",""))
        it["cat_emb_rank"] = cat_rank_by_name.get(cat_key)  # 없으면 None

    

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
    cate_rank  = make_rank(vector_items, "direct_cate_score")

    # 1) RRF 점수 계산 전 결과 출력 (벡터 매칭 점수 및 직접 점수 출력)
    print("\n[방법1/RRF] RRF 점수 계산 전 결과:")
    for idx, item in enumerate(vector_items[:10], 1):  # 상위 10개 출력
        print(f"{idx}. {item['제목']} | {item['카테고리']} | 벡터 점수: {item.get('vector_match_score', 0)} | 제목 직접 점수: {item.get('direct_title_score', 0)} | 카테고리 직접점수: {item.get('direct_cate_score', 0)} | 카테고리 임베딩 점수: {item.get('cat_emb_rank')}")


    # ---------- RRF 합산 (가중치 없음) ----------
    def rrf_score(ranks, k=60):
        s = 0.0
        for r in ranks:
            if r is not None:
                s += 1.0 / (k + r)
        return s



    for it in vector_items:
        code = it.get("상품코드")
        rv = vector_rank.get(code)       # 벡터 순위
        rt = title_rank.get(code)        # 제목 직접매칭 순위
        rc = cate_rank.get(code)         # 카테고리 직접매칭 순위
        re_cat = it.get("cat_emb_rank")         # 카테고리 임베딩 순위 (문자열 매칭으로 조회된 정수)

        # 텍스트만 RRF
        it["rrf_text_only"]   = rrf_score([rt, rc])

        # 텍스트 + 카테고리 임베딩 RRF
        it["rrf_text_plus_catEmb"] = rrf_score([rt, rc, re_cat])

        # 벡터까지 포함 (최종 추천용)
        it["rrf_all"] = rrf_score([rv, rt, rc, re_cat])

    # 3) RRF 점수 계산 후 결과 출력 (RRF 계산 후 벡터와 점수를 출력)
    print("\n[방법1/RRF] RRF 점수 계산 후 결과:")
    for idx, item in enumerate(vector_items[:10], 1):  # 상위 10개 출력
        print(f"{idx}. {item['제목']} | {item['카테고리']} | RRF(텍스트만): {item['rrf_text_only']:.6f} | RRF(텍스트+카테고리): {item['rrf_text_plus_catEmb']:.6f} | RRF(전체): {item['rrf_all']:.6f}")





    # ---------- 최종 정렬: 벡터 포함 RRF(또는 텍스트만 RRF로 바꿔도 됨) ----------
    final_sorted = sorted(vector_items, key=lambda x: x["rrf_all"], reverse=True)

    final_results, seen_codes = [], set()
    for it in final_sorted:
        if len(final_results) >= 40:
            break
        code = it.get("상품코드")
        if code and code not in seen_codes:
            seen_codes.add(code)
            final_results.append(it)

    all_sets = []
    for i in range(0, len(final_results), 10):
        block = final_results[i:i+10]
        if len(block) == 10:
            all_sets.append(block)

    print(f"\n[방법1/RRF] 세트 수: {len(all_sets)}  (총 {sum(len(s) for s in all_sets)}개)")
    if all_sets:
        print("\n[방법1/RRF] 세트1 미리보기")
        for idx, it in enumerate(all_sets[0], 1):
            print(f"{idx}. {it['제목']} | {it['카테고리']} | {it['가격']:,}원 | RRF={it['rrf_all']:.6f}")


    # 4) 최종 결과 출력
    print("\n[방법1/RRF] 최종 결과:")
    for idx, item in enumerate(final_results[:10], 1):  # 상위 10개 출력
        print(f"{idx}. {item['제목']} | {item['카테고리']} | RRF: {item['rrf_all']:.6f} | 가격: {item['가격']:,}원")






    # #################################################################
    # #                         NEW 방법 1 끝                             #
    # #################################################################


    #################################################################
    #                          방법 2 시작                            #
    #################################################################
    
    # # 카테고리별 검색 수량 설정
    # CATEGORY_QUOTAS = {
    #     "Top1": 50,  # 총 50개
    #     "Top2": 30,  # 총 30개
    #     "Top3": 20   # 총 20개
    # }

    # # 방법2 1.1) 시즌 필터 적용 
    # raw_candidates = season_filter_items(raw_candidates, season)

    # # 3) 쿼리 임베딩으로 Top5 카테고리 검색
    # top5_cats = get_top5_categories_from_embeddings(q_vec)

    # print("\n🔎 (원본) 카테고리 Top5:")
    # for i,(name,dist) in enumerate(top5_cats,1):
    #     print(f"  {i}. {name} | L2={dist:.6f}")

    # # 시즌 보정 스코어 계산하여 정렬
    # adjusted_cats = [(name, dist, get_adjusted_score(name, dist, season)) 
    #                 for name, dist in top5_cats[:3]]
    
    # # 최종 점수로 정렬
    # sorted_cats = sorted(
    #     [(name, dist) for name, dist, _ in adjusted_cats],
    #     key=lambda x: get_adjusted_score(x[0], x[1], season)
    # )

    # print("\n🎯 시즌 보정 후 최종 카테고리 순위:")
    # name_1, name_2, name_3 = "", "", ""  # 상위 3개 카테고리 이름 저장용 변수
    
    # for i, (name, dist) in enumerate(sorted_cats, 1):
    #     adj_score = get_adjusted_score(name, dist, season)
        
    #     # 상위 3개 카테고리 이름 저장
    #     if i == 1:
    #         name_1 = name
    #     elif i == 2:
    #         name_2 = name
    #     elif i == 3:
    #         name_3 = name
            
    #     print(f"  {i}등. {name}")
    #     print(f"     - 원본 L2 거리: {dist:.6f}")
    #     print(f"     - 시즌 보정값: {_season_adjust(name, season):.6f}")
    #     print(f"     - 최종 점수: {adj_score:.6f}")
    #     print(f"     - 할당된 검색 수: {CATEGORY_QUOTAS[f'Top{i}']}개")
    


    # print(f"1위 카테고리 이름은 ? ->{name_1}")

    # print(f"2위 카테고리 이름은 ? ->{name_2}")

    # print(f"3위 카테고리 이름은 ? ->{name_3}")


    # # 방법2 카테고리 1등) 전 카테고리 50개 벡터 검색 후 점수 부여
    
    # print("\n[방법2] 50개 벡터 검색 시작")
    # vector_hits_50 = collection.search(
    #     data=[q_vec],
    #     anns_field="emb",
    #     param={"metric_type":"L2","params":{"nprobe":64}},
    #     limit=50,  # 50개 검색
    #     expr=f"category_name like '%{name_1}%'",
    #     output_fields=[
    #         "product_code","category_code","category_name","market_product_name",
    #         "market_price","shipping_fee","shipping_type","max_quantity",
    #         "composite_options","image_url","manufacturer","model_name",
    #         "origin","keywords","description","return_shipping_fee",
    #     ]
    # )
    # print("\n[방법2] 벡터 검색 결과 50개에서 직접 검색 및 점수 결합 시작")
    
    # # 1. 벡터 검색 결과에 점수 부여 (1000점 → 1점)
    # vector_items = []
    # for hits in vector_hits_50:
    #     for idx, hit in enumerate(hits):
    #         item = _build_info_from_hit(hit)
    #         item['vector_match_score'] = 1000 - idx  # 1등: 1000점, 1000등: 1점
    #         vector_items.append(item)
    # print(f"[방법2] 벡터 검색 50개 완료: {len(vector_items)}개")


    # # 편의 함수
    # def DirectSearch(user_text: str, items: List[Union[str, Dict[str, Any]]],
    #         fields: Optional[Iterable[str]] = None,
    #         near_threshold: float = 0.90,
    #         return_query_meta: bool = False) -> List[Dict[str, Any]]:
    #     ds = Ranker_DirectSearch(near_threshold=near_threshold)
    #     return ds.score_items(user_text, items, fields=fields, return_query_meta=return_query_meta)


    # ds = Ranker_DirectSearch()
    # qmeta = ds.prepare_query(preprocessed_query)
    # print("원문:", qmeta["original"])
    # print("정규화:", qmeta["normalized"])
    # print("토큰:", qmeta["tokens"])
    # print("정제쿼리(canonical):", qmeta["canonical_query"])

    # direct_items = []

    # # 1) 직접검색 매칭 점수(0~1000) 계산: 대상 = 벡터검색 결과 리스트 전체(vector_items)
    # for item in vector_items:
    #     title = item.get('제목', '')               # '제목'이 항상 있으면 item['제목'] 사용해도 됨
    #     item['direct_match_score'] = ds.score_text(preprocessed_query, title) # 0~1000점 연속값

    # # 2) 벡터검색 순위 점수 부여 (1000 → 1점)
    # for idx, item in enumerate(vector_items):
    #     item['vector_rank_score'] = 1000 - idx  # 1등: 1000점, 2등: 999점, ...

    # # 3) 직접검색 순위 점수 부여 (1000 → 1점)
    # direct_sorted = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for idx, item in enumerate(direct_sorted):
    #     item['direct_rank_score'] = 1000 - idx  # 1등: 1000점, 2등: 999점, ...

    # # 4) 최종 점수 계산
    # for item in vector_items:
    #     # 벡터중심 변환점수 (벡터 0.7 + 직접 0.3)
    #     item['vector_final_score'] = (0.7 * item['vector_rank_score'] + 0.3 * item['direct_rank_score'])
    #     # 직접중심 변환점수 (벡터 0.3 + 직접 0.7)
    #     item['direct_final_score'] = (0.3 * item['vector_rank_score'] + 0.7 * item['direct_rank_score'])
        
    # print("\n[방법2] 1등 카테고리 상품 점수 정규화 및 최종 점수 계산 완료")
    # print("상위 5개 항목 점수 분포:")
    # sorted_items = sorted(vector_items, key=lambda x: x['direct_final_score'], reverse=True)[:5]
    # for idx, item in enumerate(sorted_items, 1):
    #     print(f"\n{idx}. {item['제목']}")
    #     print(f"   직접오리지널점수: {item['direct_match_score']:.1f}, 벡터매칭: {item['vector_match_score']:.1f}")
    #     print(f"   직접검색 1000점을 변환점수: {item['direct_rank_score']:.1f}")
    #     print(f"   벡터검색 1000점을 변환점수: {item['vector_rank_score']:.1f}")
    #     print(f"   최종점수: {item['direct_final_score']:.1f} (직접중심) / {item['vector_final_score']:.1f} (벡터중심)")


    # ranked = DirectSearch(preprocessed_query, vector_items, fields=("제목",))
    # print(f"{'Rank':>4}  {'Score':>5}  제목")
    # print("-"*48)
    # for i, r in enumerate(ranked[:30], 1):
    #     item = r["item"]
    #     title = item["제목"] if isinstance(item, dict) else str(item)
    #     print(f"{i:>4}  {r['direct_text_score']:>5}  {title}")


    # # direct_match_score 기준 내림차순 정렬 → 직접검색 "랭크 점수"(1000→1) 대입
    # direct_items = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for r_idx, it in enumerate(direct_items):
    #     it['direct_score'] = max(1, 1000 - r_idx)  # 1등 1000, 1000등 1

    # # direct_score가 없는 항목 0으로 보정
    # for it in vector_items:
    #     if 'direct_score' not in it:
    #         it['direct_score'] = 0

    # print(f"[방법2] 직접 검색 매칭: {len(direct_items)}개")

    # # 3) 점수 통합 (벡터/직접오리지널점수 점수 결합)
    # for item in vector_items:
    #     # 벡터기반 선발
    #     item['vector_focused_score'] = 0.7*item['vector_match_score'] + 0.3*item['direct_score']  
    #     # 직접기반 선발
    #     item['direct_focused_score'] = 0.3*item['vector_match_score'] + 0.7*item['direct_score']
    # print("\n[방법2] 1등 카테고리 상품 점수 계산 완료")

    #     # 직접검색 점수가 같은 경우 벡터 점수로 2차 정렬하는 key 함수
    # def direct_sort_key(item):
    #     return (
    #         item['direct_final_score'],  # 1차 정렬: 직접검색 점수
    #         item['vector_final_score']   # 2차 정렬: 벡터 점수
    #     )
    
    # direct_sorted = sorted(vector_items, key=direct_sort_key, reverse=True)



    # # === 최종 5개 선정 (직접 2 + 벡터 3) — 상품코드 중복 제거 ===
    # raw_candidates = []
    # seen_codes = set()

    # # 1. 벡터 검색 결과 1000개에 대한 순위 점수 부여 (1000점 → 1점)
    # for idx, item in enumerate(vector_items):
    #     item['vector_rank_score'] = max(1, 1000 - idx)  # 1등: 1000점, 1000등: 1점

    # # 2. 직접 검색 결과에 대한 순위 점수 부여 (1000점 → 1점)
    # direct_sorted = sorted(vector_items, key=lambda x: x['direct_match_score'], reverse=True)
    # for idx, item in enumerate(direct_sorted):
    #     item['direct_rank_score'] = max(1, 1000 - idx)  # 1등: 1000점, 마지막등: 1점
    
    # # 3. 최종 점수 계산
    # for item in vector_items:
    #     # 벡터 중심 점수 (벡터 0.7 + 직접 0.3)
    #     item['vector_final_score'] = (
    #         0.7 * item['vector_rank_score'] +
    #         0.3 * item['direct_rank_score']
    #     )
    #     # 직접 중심 점수 (벡터 0.3 + 직접 0.7)
    #     item['direct_final_score'] = (
    #         0.3 * item['vector_rank_score'] +
    #         0.7 * item['direct_rank_score']
    #     )

    # # 4. 직접 검색 중심으로 2개 선정
    # raw_candidates = []
    # seen_codes = set()
    

    # print("\n[방법2] 직접검색 중심 상위 10개 점수 분포:")
    # for idx, item in enumerate(direct_sorted[:10], 1):
    #     print(f"{idx}. {item['제목']}")
    #     print(f"   벡터순위점수: {item['vector_rank_score']:.1f}")
    #     print(f"   직접순위점수: {item['direct_rank_score']:.1f}")
    #     print(f"   직접오리지날점수: {item['direct_match_score']:.1f}")
    #     print(f"   최종점수(직접중심): {item['direct_final_score']:.1f}")

    # # 직접검색 상위 2개 선정 (점수가 0인 항목 제외)
    # for item in direct_sorted:
    #     if len(raw_candidates) >= 2:
    #         break
    #     code = item.get('상품코드')
    #     if code and code not in seen_codes:
    #         # 직접 검색 점수가 0점이면 건너뛰기
    #         if item['direct_rank_score'] == 0:
    #             continue
                
    #         seen_codes.add(code)
    #         raw_candidates.append(item)
    #         print(f"\n[방법2] 직접검색 선정 {len(raw_candidates)}/2:")
    #         print(f"제목: {item['제목']}")
    #         print(f"점수: 벡터={item['vector_rank_score']:.1f}, "
    #               f"직접={item['direct_rank_score']:.1f}, "
    #               f"최종(직접)={item['direct_final_score']:.1f}, "
    #               f"최종(벡터)={item['vector_final_score']:.1f}")

    # # 5. 벡터 검색 중심으로 3개 선정 (중복 제외)
    # vector_sorted = sorted(vector_items, key=lambda x: x['vector_final_score'], reverse=True)

    # print("\n[방법2] 벡터검색 중심 상위 10개 점수 분포:")
    # for idx, item in enumerate(vector_sorted[:10], 1):
    #     print(f"{idx}. {item['제목']}")
    #     print(f"   벡터순위점수: {item['vector_rank_score']:.1f}")
    #     print(f"   직접오리지날점수: {item['direct_match_score']:.1f}")
    #     print(f"   직접순위점수: {item['direct_rank_score']:.1f}")
    #     print(f"   최종점수(벡터중심): {item['vector_final_score']:.1f}")

    # # 벡터검색에서 3개 선정 (직접검색과 중복 시 다음 순위로 대체)
    # vector_selections = []
    # current_index = 0

    # # 벡터검색 결과에서 3개를 채울 때까지 반복
    # while len(vector_selections) < 3 and current_index < len(vector_sorted):
    #     item = vector_sorted[current_index]
    #     code = item.get('상품코드')
        
    #     # 이미 직접검색에서 선정된 상품이면 다음 순위로
    #     if code in seen_codes:
    #         current_index += 1
    #         continue
            
    #     seen_codes.add(code)
    #     vector_selections.append(item)
    #     raw_candidates.append(item)
    #     print(f"\n[방법2] 벡터검색 선정 {len(vector_selections)}/3:")
    #     print(f"제목: {item['제목']}")
    #     print(f"점수: 벡터={item['vector_rank_score']:.1f}, "
    #           f"직접={item['direct_rank_score']:.1f}, "
    #           f"최종={item['vector_final_score']:.1f}")
        
    #     current_index += 1
















    # # 모든 상품을 하나의 리스트로 순차적으로 수집
    # all_products_v1 = []
    # seen_codes = set()

    # DIRECT_PER_SET = 2
    # VECTOR_PER_SET = 3
    # MAX_SETS = 5

    # di = 0  # direct_sorted 포인터
    # vi = 0  # vector_sorted 포인터
    # sets_built = 0

    # while sets_built < MAX_SETS and (di < len(direct_sorted) or vi < len(vector_sorted)):
    #     set_added = 0  # 이번 세트에 추가된 아이템 수

    #     # 1) 직접 2개
    #     d_added = 0
    #     while d_added < DIRECT_PER_SET and di < len(direct_sorted):
    #         it = direct_sorted[di]; di += 1
    #         code = it.get('상품코드')
    #         if not code or code in seen_codes:
    #             continue
    #         if it.get('direct_rank_score', 0) == 0:  # 네 기존 규칙 유지
    #             continue
    #         seen_codes.add(code)
    #         all_products_v1.append(it)
    #         d_added += 1
    #         set_added += 1

    #     # 2) 벡터 3개
    #     v_added = 0
    #     while v_added < VECTOR_PER_SET and vi < len(vector_sorted):
    #         it = vector_sorted[vi]; vi += 1
    #         code = it.get('상품코드')
    #         if not code or code in seen_codes:
    #             continue
    #         seen_codes.add(code)
    #         all_products_v1.append(it)
    #         v_added += 1
    #         set_added += 1

    #     # 3) 백필(옵션) — 세트가 5개 미만이면 반대 풀에서 채워 5개 맞추기 시도
    #     TARGET_SET_SIZE = DIRECT_PER_SET + VECTOR_PER_SET
    #     if set_added < TARGET_SET_SIZE:
    #         # 우선 벡터로 채우고, 그래도 부족하면 직접으로
    #         while set_added < TARGET_SET_SIZE and vi < len(vector_sorted):
    #             it = vector_sorted[vi]; vi += 1
    #             code = it.get('상품코드')
    #             if not code or code in seen_codes:
    #                 continue
    #             seen_codes.add(code)
    #             all_products_v1.append(it)
    #             set_added += 1

    #         while set_added < TARGET_SET_SIZE and di < len(direct_sorted):
    #             it = direct_sorted[di]; di += 1
    #             code = it.get('상품코드')
    #             if not code or code in seen_codes:
    #                 continue
    #             if it.get('direct_rank_score', 0) == 0:
    #                 continue
    #             seen_codes.add(code)
    #             all_products_v1.append(it)
    #             set_added += 1

    #     # 세트에 아무것도 못 담았으면 종료(무한루프 방지)
    #     if set_added == 0:
    #         break

    #     sets_built += 1

    # print(f"\n[방법2] 2.1 1등카테고리 상품 전체 {len(all_products_v1)}개 상품 수집 완료")

    # # 전체 수집된 상품 리스트 출력
    # print("\n[방법2] 2.1 1등카테고리 최종 선정된 상품 목록:")
    # for idx, item in enumerate(all_products_v1, 1):
    #     print(f"\n{idx}. {item['제목']}")
    #     print(f"   카테고리: {item.get('카테고리', '카테고리 정보 없음')}")
    #     print(f"   가격: {item.get('가격', 0):,}원")
    #     print(f"   벡터순위점수: {item['vector_rank_score']:.1f}")
    #     print(f"   직접오리지날점수: {item['direct_match_score']:.1f}")
    #     print(f"   직접순위점수: {item['direct_rank_score']:.1f}")
    #     print(f"   최종점수(직접중심): {item['direct_final_score']:.1f}")
    #     print(f"   최종점수(벡터중심): {item['vector_final_score']:.1f}")
    # print(f"=============================================================")
    # print(f"=============================================================")
    # print(f"=============================================================")






























###리랭커 구간 추가 방법1의 final_results를 실제 사용자 문장과 유사 비교해서 리랭커 실시 Start###

    def rerank_results(
        query: str,
        results: List[Dict],
        top_k: int = 40,
        title_key: str = "제목",
        cate_key: str = "카테고리",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 64,
        normalize: bool = True,   # softmax 대신 pair별 점수 정규화 옵션
        device: Optional[str] = None,  # "cuda" / "cpu" 강제 지정 가능
        use_fp16: Optional[bool] = None,  # None이면 자동; True면 반정밀
    ) -> List[Dict]:
        
        reranked = []
        return reranked

    

    print("리랭커 전 final_results 데이터 정보")
    for idx, item in enumerate(final_results[:40], 1):
        print(f"{idx}. {item['제목']} | {item['카테고리']} ")


    # # external_search_and_generate_response 함수 내에서 final_results 처리 전에 추가
    # final_results = rerank_results(
    #     query=query,  # 원본 사용자 쿼리 사용
    #     results=final_results,
    #     top_k=40
    # )
    # ###리랭커 구간 추가 방법1의 final_results를 실제 사용자 문장과 유사 비교해서 리랭커 실시 END###
    # print("리랭커 후 final_results 데이터 정보")
    # for idx, item in enumerate(final_results[:40], 1):
    #     print(f"{idx}. {item['제목']} | {item['카테고리']} ")

    #################################################################



    # (참고) products_text는 이미 f-string으로 잘 만들고 있음
    products_text = "\n".join([
        f"- 코드: {p['상품코드']} | 제목: {p['제목']} | 가격: {p['가격']:,}원 | 카테고리: {p['카테고리']}"
        for p in all_sets[0]
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
    2) 그 특징을 활용해 고객 의도 확인과 선호 파악을 위한 '확인형 질문'을 반드시! 한글은 200~250자,영어도 200자 내외 한 문단으로 작성하세요.
    - 예시 어조: "후보를 보니 [특징A/특징B/특징C] 옵션이 보이는데, 이런 조건이 필요하신가요?"
    - 본품인지 관련품·소모품·세트·서비스인지도 자연스럽게 확인하세요.
    - 가격대(프리미엄/일반/실속), 디자인, 용도, 재질, 브랜드 선호, 계절감, 옵션 유무를 과도한 나열 없이 자연스럽게 묻으세요.
    3) 상품 코드나 구체 모델명/스펙 나열은 금지합니다. 후보에서 추출한 '특징 키워드'만 요약해 언급하세요.
    4) 친근하면서도 전문적인 대화체를 유지하세요.
    5) 아래 예시는 참고만 하며, 그대로 복사하지 말고 실제 후보의 특징으로 대체하세요.

    답변 형식(단락 예시):
    심플한 구성이 더 좋으실까요? 사용하실 환경과 선호 가격대(프리미엄/실속), 디자인·재질, 그리고 본품/관련품 중 어떤 쪽을 찾으시는지도 알려주시면 더 정확히 추천해 드리겠습니다.
    
    **무조건 어떤 언어가 들어와도 {target_lang}로만 답변하세요!반드시**

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

    
 

    # 모든 상품을 하나의 리스트로 모으고 점수 기준으로 정렬
    seen_codes = set()

    # 방법1과 방법2의 결합
    # sorted_items = sorted(vector_items, key=lambda x: x['direct_final_score'], reverse=True)










    # 상위 40개 선택 (중복 제거)
    for item in final_results:
        if len(final_results) >= 40:  # 최대 40개로 제한
            break
            
        if item['상품코드'] not in seen_codes:
            seen_codes.add(item['상품코드'])
            final_results.append(item)

    # 결과 출력
    print(f"\n총 {len(final_results)}개의 상품이 최종 리스트에 저장되었습니다.")
    
    print("\n상위 10개 상품 상세 정보:")
    for idx, item in enumerate(final_results[:10], 1):
        print(f"\n{idx}. {item['제목']}")
        print(f"   카테고리: {item['카테고리']}")
        print(f"   가격: {item['가격']:,}원")
        # print(f"   직접오리지널점수 점수: {item['direct_match_score']:.1f}")
        # print(f"   최종점수: {item['direct_final_score']:.1f}")

    # 상품 캐시에 저장
    for info in final_results:
        PRODUCT_CACHE[info["상품코드"]] = info

    print("\n상품의 상세 정보:")
    for idx, info in enumerate(final_results[:40], start=1):
        PRODUCT_CACHE[info["상품코드"]] = info
        
        if idx % 10 == 0:  # 10개마다 한 번씩만 출력
            print(f"\n처리 중... {idx}/40개 완료")
        
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
        "results": final_results,  # 검색 결과 리스트 (방법 2가 앞쪽, 방법 1이 뒤쪽)
        "combined_message_text": clean,  # 사용자에게 보여줄 메시지
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
    clarify_answer: Optional[str] = None   # ← 추가: 재질문 답변(있으면 2턴째로 처리)

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
    uvicorn.run("app_방법1완료:app", host=host, port=port, reload=debug)