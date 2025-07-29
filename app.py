#ownerclan_weekly_0428_전치리 LLM(카테고리를 추측) 추가용
import numpy as np
from langchain_openai import OpenAI, OpenAIEmbeddings
from pymilvus import Collection, connections
import os
from openai import OpenAI as OpenAIClient      # 공식 OpenAI 클라이언트
import json
import re
import base64
import urllib
from langdetect import detect
from collections import defaultdict, Counter
import math

from typing import List, Dict, Any, Tuple
import time
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
import secrets
from user_events import event_manager
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ── 설정 ─────────────────────────────────────────────────────────
API_KEY    = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')                    # ← 환경변수에서 로드
COLLECTION = "ownerclan_weekly_0428"            # Milvus 컬렉션 이름
EMB_MODEL  = "text-embedding-3-small"
MILVUS_HOST = os.getenv('MILVUS_HOST', '114.110.135.96')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
LLM_MODEL  = "gpt-4.1-mini"

# 클라이언트 및 래퍼
client    = OpenAIClient(api_key=API_KEY)
llm       = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)    # ← embedder 정의 추가
API_URL = "https://fb-narosu.duckdns.org"  # 예: http://114.110.135.96:8011

#가격 조건 파싱
pattern = re.compile(r'(\d+)[^\d]*원\s*(이하|미만|이상|초과)')


# 1) Milvus 서버에 먼저 연결
connections.connect(
    alias="default",
    host=MILVUS_HOST,    # 예: "114.110.135.96"
    port=MILVUS_PORT     # 예: "19530"
)
print("✅ Milvus에 연결되었습니다.")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 세션을 위한 시크릿 키




def run_recommendation_pipeline(query, price_min, price_max):
    collection_cat = Collection("category_0710")
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

    def compute_top4_quota(
        candidates: List[Dict[str, Any]],
        max_total: int = 10,
        min_per_category: int = 1
    ) -> Dict[str, int]:
        """
        Top4 카테고리 자동 추출 & 비율 기반 quota 계산
        Returns: {카테고리: quota}
        """
        total = len(candidates)
        counts = Counter(item["카테고리"] for item in candidates)
        top4 = [cat for cat, _ in counts.most_common(4)]
        
        # 초기 quota 계산 (floor + 최소 보장)
        quotas = {
            cat: max(math.floor(counts[cat] / total * max_total), min_per_category)
            for cat in top4
        }
        
        # 부족·초과 보정
        diff = max_total - sum(quotas.values())
        if diff > 0:
            # 비중 큰 순서대로 +1
            for cat, _ in counts.most_common():
                if cat in quotas and diff > 0:
                    quotas[cat] += 1
                    diff -= 1
        elif diff < 0:
            # 비중 작은 순서대로 -1 (min 유지)
            for cat, _ in reversed(counts.most_common()):
                if cat in quotas and quotas[cat] > min_per_category and diff < 0:
                    quotas[cat] -= 1
                    diff += 1
        
        return quotas

    def filter_top4_candidates(
        candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Top4 카테고리만 필터링
        Returns: (filtered_candidates, top4_keys)
        """
        counts = Counter(item["카테고리"] for item in candidates)
        top4_keys = [cat for cat, _ in counts.most_common(4)]
        filtered = [item for item in candidates if item["카테고리"] in top4_keys]
        return filtered, top4_keys

    def prepare_recommendation(
        all_candidates: List[Dict[str, Any]],
        max_total: int = 20,
        min_per_category: int = 1
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
        """
        1) Top4 필터링
        2) quota 계산
        Returns: (filtered_candidates, quotas, top4_keys)
        """
        filtered, top4_keys = filter_top4_candidates(all_candidates)
        quotas = compute_top4_quota(filtered, max_total, min_per_category)
        return filtered, quotas, top4_keys

    def quota_to_text(quota: Dict[str, int]) -> str:
        return "\n".join([f'- {cat}: {q}개' for cat, q in quota.items()])

    def compute_category_proportions(
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        total = len(candidates)
        if total == 0:
            return {}
        counts = Counter(item["카테고리"] for item in candidates)
        return {cat: cnt / total for cat, cnt in counts.items()}


        

    ##############################################################################원본 사용자 쿼리
 
    print("[Debug] Raw query:", query)            # ← 여기에!
    lang_code = detect(query)

    print("[Debug] lang_code →", lang_code)   # ← 이 줄 추가!
    #가격을 이해하는 매핑
    m = pattern.search(query)
    if m:
        amount = int(m.group(1))
        comp  = m.group(2)
        # 부등호 매핑
        op_map = {"이하":"<=", "미만":"<", "이상":">=", "초과":">"}
        price_op = op_map[comp]
        price_cond = f"market_price {price_op} {amount}"
    else:
        # 디폴트: 제한 없음
        price_cond = None

    # 2) 언어 코드 → 사람말 매핑
    lang_map = {
        "ko": "한국어",
        "en": "English",
        "zh-cn": "中文",
        "ja": "日本語",
        "vi": "Tiếng Việt",  # 베트남어
        "th": "ไทย",        # 태국어
    }

    target_lang = lang_map.get(lang_code, "한국어")
    print("[Debug] Detected language →", target_lang)

    # LLM 전처리
    llm = OpenAI(
        api_key=API_KEY,
        model=LLM_MODEL,
        temperature=0
    )
    system_prompt = (
        f"""System:
    당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
    어떤 언어로 입력이 되든 반드시 한국어로 문장 의미에 맞게 번역 먼저 합니다.
    아래는 DB에서 로드된 **가능한 카테고리 목록**입니다.  
    모든 예측은 이 목록 안에서만 이루어져야 합니다:

    {categories}

    다음 순서대로 응답하세요:

    1) **전처리 단계**  
    - 사용자 원문(query)에서 오타 교정, 불용어 제거, 중복 표현 제거 
    - 핵심 키워드 하나를 뽑아 표준어/동의어로 치환한 뒤  
    - **“검색식(Search Query)”**으로 바로 쓸 수 있도록 포맷팅하세요.  

    2) **카테고리 예측 단계**  
    - 전처리된 쿼리(Preprocessed Query)를 바탕으로 직관적으로 최상위 카테고리 하나 예측

    3) **검색 결과 재정렬 단계**  
    - 이미 Milvus 벡터 검색을 통해 얻은 TOP N 결과 리스트(search_results)를 입력받아  
    - 각 결과의 메타데이터(id, 상품명, 카테고리, 가격, URL 등)를 활용해  
    - 2번에서 예측한 카테고리와 매칭되거나 인접한 결과를 우선 정렬하세요.

    4) **출력 형식**은 반드시 아래와 같습니다:

    Raw Query: "<query>"  
    Preprocessed Query: "<전처리된_쿼리>"
    Predicted Category: "<예측된_최상위_카테고리>"

        """    
    )

    if price_cond:
        system_prompt += f"⚠️ 사용자 요청 조건: 가격은 **{amount}원 {comp}** ({price_cond})인 상품만 고려하세요.\n\n"



    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ],
        temperature=0
    )
    llm_response = resp.choices[0].message.content.strip()
    print("[Debug] LLM full response:\n", llm_response)  # ← 여기에!


    #LLM 응답 파싱
    lines = [l.strip() for l in llm_response.splitlines() if l.strip()]
    preprocessed_query = next(
        l.split(":",1)[1].strip().strip('"')
        for l in lines if l.lower().startswith("preprocessed query")
    )
    predicted_category = next(
        l.split(":",1)[1].strip().strip('"')
        for l in lines if l.lower().startswith("predicted category")
    )
    # ← 여기에 한 줄 추가
    top_category = predicted_category.split(">")[0]

    print("[Debug] Preprocessed Query →", preprocessed_query)   # ← 여기에!
    print("[Debug] top_category →", top_category)   # ← 여기에!

    #최하위 카테고리
    lowest_subcategory = predicted_category.split(">")[-1]

    print("[Debug] lowest_subcategory →", lowest_subcategory)


    #쿼리 임베딩 생성
    q_vec = embedder.embed_query(preprocessed_query)
    print(f"[Debug] q_vec length: {len(q_vec)}, sample: {q_vec[:5]}")  # ← 여기에!

    # ① Stage1: 직접 문자열 검색 (boolean search)
    print("[Stage1] Direct name search 시작")

    # “남자용 향수” → ["남자", "향수"] 두 토큰으로 AND 검색
    tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
    query_expr = " && ".join(f'market_product_name like "%{tok}%"'
        for tok in tokens
    )

    print("[Debug] Stage1 expr:", query_expr)
    direct_hits = collection.query(
        expr=query_expr,
        limit=200,
        output_fields=[
            "product_code",
            "category_code",
            "category_name",
            "market_product_name",
            "market_price",
            "shipping_fee",
            "shipping_type",
            "max_quantity",
            "composite_options",
            "image_url",
            "manufacturer",
            "model_name",
            "origin",
            "keywords",
            "description",
            "return_shipping_fee",
        ]
    )
    print("[Stage1] Direct hits count:", len(direct_hits))
    # 샘플 출력
    for i, row in enumerate(direct_hits[:7], 1):
        print(f"  [Stage1 샘플 {i}]: 코드={row['product_code']}, 이름={row['market_product_name']}")


    print("\n[Stage2.5] 직접검색 results 구성 시작")  
    raw_candidates = []
    for row in direct_hits:
        # e = hit.entity
        # 본문 미리보기 링크
        try:
            html_raw = row.get("description", "") or ""
            html_cleaned = clean_html_content(html_raw)
            if isinstance(html_raw, bytes):
                html_raw = html_raw.decode("cp949")
            encoded_html = base64.b64encode(
                html_cleaned.encode("utf-8", errors="ignore")
            ).decode("utf-8")
            safe_html = urllib.parse.quote_plus(encoded_html)
            preview_url = f"{API_URL}/preview?html={safe_html}"
        except Exception as err:
            print(f"⚠️ 본문 처리 오류: {err}")
            preview_url = "https://naver.com"

        # 상품링크(fallback)
        product_link = row.get("product_link", "")
        if not product_link or product_link in ["링크 없음", "#", None]:
            product_link = preview_url

        # 옵션 파싱
        option_raw = str(row.get("composite_options", "")).strip()
        option_display = "없음"
        if option_raw.lower() not in ["", "nan"]:
            parsed = []
            for line in option_raw.splitlines():
                try:
                    name, extra, _ = line.split(",")
                    extra = int(float(extra))
                    parsed.append(
                        f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                    )
                except Exception:
                    parsed.append(line.strip())
            option_display = "\n".join(parsed)

        # 10개 한글 속성으로 딕셔너리 구성
        result_info = {
            "상품코드":     str(row.get("product_code", "없음")),
            "제목":        row.get("market_product_name", "제목 없음"),
            "가격":        convert_to_serializable(row.get("market_price", 0)),
            "배송비":      convert_to_serializable(row.get("shipping_fee", 0)),
            "이미지":      row.get("image_url", "이미지 없음"),
            "원산지":      row.get("origin", "정보 없음"),
            "상품링크":    product_link,
            "옵션":        option_display,
            "조합형옵션":  option_raw,
            "최대구매수량": convert_to_serializable(row.get("max_quantity", 0)),
            "카테고리":    row.get("category_name", "카테고리 없음"),
        }
        result_info_cleaned = {}
        for k, v in result_info.items():
            if isinstance(v, str):
                v = v.replace("\n", "").replace("\r", "").replace("\t", "")
            result_info_cleaned[k] = v
        raw_candidates.append(result_info_cleaned)

    # ② Stage2: 벡터 유사도 검색
    # expr = f'category_name like "%{top_category}%"'   #최상위 카테고리
    expr = f'category_name like "%{lowest_subcategory}%"'   #최하위 카테고리
    milvus_results = collection.search(
        data=[q_vec],
        anns_field="emb",  # ← 벡터 저장된 필드 이름
        param={"metric_type": "L2", "params": {"nprobe": 1536}},
        limit=200,
        expr=expr,                              # ← 이 줄 추가
        output_fields = [
        "product_code",
        "category_code",
        "category_name",
        "market_product_name",
        "market_price",
        "shipping_fee",
        "shipping_type",
        "max_quantity",
        "composite_options",
        "image_url",
        "manufacturer",
        "model_name",
        "origin",
        "keywords",
        "description",
        "return_shipping_fee",
    ]
    )
    print(f"[Stage2] Vector hits count: {len(milvus_results[0])}")
    # 샘플 출력



    for i, hit in enumerate(milvus_results[0][:7], 1):
        e = hit.entity
        print(f"  [Stage2 샘플 {i}]: 코드={e.get('product_code')}, 이름={e.get('market_product_name')}")

        
    #벡터 검색 담는 부분
    for hits in milvus_results:
        for hit in hits:
            e = hit.entity
            # 본문 미리보기 링크
            try:
                html_raw = e.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(
                    html_cleaned.encode("utf-8", errors="ignore")
                ).decode("utf-8")
                safe_html = urllib.parse.quote_plus(encoded_html)
                preview_url = f"{API_URL}/preview?html={safe_html}"
                # preview_url = f"{safe_html}"
            except Exception as err:
                print(f"⚠️ 본문 처리 오류: {err}")
                preview_url = "https://naver.com"

            # 상품링크(fallback)
            product_link = e.get("product_link", "")
            if not product_link or product_link in ["링크 없음", "#", None]:
                product_link = preview_url

            # 옵션 파싱
            option_raw = str(e.get("composite_options", "")).strip()
            option_display = "없음"
            if option_raw.lower() not in ["", "nan"]:
                parsed = []
                for line in option_raw.splitlines():
                    try:
                        name, extra, _ = line.split(",")
                        extra = int(float(extra))
                        parsed.append(
                            f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                        )
                    except Exception:
                        parsed.append(line.strip())
                option_display = "\n".join(parsed)

            # 10개 한글 속성으로 딕셔너리 구성
            result_info = {
                "상품코드":     str(e.get("product_code", "없음")),
                "제목":        e.get("market_product_name", "제목 없음"),
                "가격":        convert_to_serializable(e.get("market_price", 0)),
                "배송비":      convert_to_serializable(e.get("shipping_fee", 0)),
                "이미지":      e.get("image_url", "이미지 없음"),
                "원산지":      e.get("origin", "정보 없음"),
                "상품링크":    product_link,
                "옵션":        option_display,
                "조합형옵션":  option_raw,
                "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
                "카테고리":    e.get("category_name", "카테고리 없음"),
            }

            # 문자열 정리 후 리스트에 추가
            result_info_cleaned = {}
            for k, v in result_info.items():
                if isinstance(v, str):
                    v = v.replace("\n", "").replace("\r", "").replace("\t", "")
                result_info_cleaned[k] = v
            raw_candidates.append(result_info_cleaned)

            # 캐시에 안전 저장
            product_code = result_info_cleaned.get("상품코드")
            # if product_code and product_code != "없음":
                # PRODUCT_CACHE[product_code] = result_info_cleaned

    # 개수 및 샘플 확인
    print(f"[Stage2.5] raw_candidates count: {len(raw_candidates)}")
    for i, info in enumerate(raw_candidates[:3],1):
        print(f"  [raw_candidates 샘플 {i}]:", info)


    # Stage2.5 완료 후: 원본 보관
    original_candidates = raw_candidates.copy()

    # ① Top4 필터 + quota 계산
    filtered_cands, quotas, top4_keys = prepare_recommendation(
        all_candidates=original_candidates,
        max_total=10,
        min_per_category=1
    )

    print("🔝 Top4 카테고리:", top4_keys)
    print("🗂️ 카테고리별 quota:", quotas)

    total = len(raw_candidates)
    print(f"🔍 총 후보: {total}개")

    # 비율
    props = compute_category_proportions(filtered_cands)
    print("📊 카테고리별 비율:")
    for cat, ratio in props.items():
        print(f"  {cat}: {ratio*100:.1f}%")

        

    # quota (최종 5개 배정 기준 예시)
    quotas = compute_top4_quota(filtered_cands, max_total=10,min_per_category=1)
    print("🗂️ 카테고리별 추천 개수(quota):")
    for cat, q in quotas.items():
        print(f"  {cat}: {q}개")

    # ── 3) Prompt에 quota 가이드 추가 ────────────────────────────────
    def quota_to_text(quota: Dict[str, int]) -> str:
        return "\n".join([f"- {cat}: {num}개" for cat, num in quota.items()])

    quota_text = quota_to_text(quotas)
    print(f"quota_text ->   {quota_text}")
    # 후보 리스트(원본 전부)

    candidate_list = "\n".join(
        f"{i+1}. {c['제목']} [{c['카테고리']}]"
        for i, c in enumerate(filtered_cands)
    )

    print("[Stage4] LLM에 넘길 후보 리스트:\n", candidate_list[:300], "...")  # 앞부분만 출력

    rank_prompt = f"""
    **답변은 반드시 "{target_lang}"로 해주세요.**
    System: 당신은 쇼핑 검색 결과를 최종 랭킹하는 AI입니다.
    User Query: "{query}"
    # 예측된 카테고리: "{predicted_category}"
    # 아래 후보들은 모두 이 카테고리에 속합니다. 

    후보리스트 : {candidate_list}에는 이미 Top4 카테고리만 필터링된 상품들이 포함되어 있습니다.

    **지침:**
    1. **카테고리별 필터링 적용**  
    - {quota_text}에 명시된 각 카테고리별 할당량만큼, candidate_list에서 반드시 해당 카테고리 상품을 정확히 그 개수만큼 나열하세요.  
    - 예: “패션의류>남성의류>티셔츠: 4개”라면, 후보 리스트 중 해당 카테고리 상품 4개를 출력합니다.

    2. **추가 탐색 질문 생성 (최대 400자)**  
    - 나열된 상품 메타데이터(상품코드, 제목, 가격, 이미지 URL 등)를 종합하여, 사용자가 원하는 상품이 이 목록에 없을 경우 선택을 좁힐 수 있도록 구체적이고 자연스러운 질문을 **최소 400자 **로 작성하세요.

    3. **최종 선택**  
    - 제시된 후보 중 사용자의 의도에 가장 적합한 항목의 번호만을 JSON 배열 형태로 반환하세요.  
    - 반드시 **예시와 같은 형식**으로만 출력합니다:  
        
    json
        [1,2,3,4,5]


    """
    resp2 = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"system","content":rank_prompt}],
        temperature=0
    )
    selection = resp2.choices[0].message.content.strip()
    print("[Stage4] Raw LLM selection:", selection)

    # 1) 마크다운 제거
    clean = re.sub(r'.*?\n', '', selection).replace('```', '').strip()
    print("[Stage4] Cleaned selection:", clean)

    match = re.search(r'\[(?:\s*\d+\s*,?)+\s*\]', clean)
    if match:
        arr_text = match.group(0)
        try:
            chosen_idxs = json.loads(arr_text)
        except json.JSONDecodeError:
            chosen_idxs = []
    else:
        chosen_idxs = []

    # ── 여기에 추가 ──
    # ②.1 인덱스 범위 검증
    max_n = len(filtered_cands)
    valid_idxs = [i for i in chosen_idxs if 1 <= i <= max_n]
    if len(valid_idxs) < len(chosen_idxs):
        print(f"⚠️ 잘못된 인덱스 제거됨: {set(chosen_idxs) - set(valid_idxs)}")
    if not valid_idxs:
        print("⚠️ 유효 인덱스 없음, 상위 10개로 Fallback")
        valid_idxs = list(range(1, min(11, max_n+1)))
    chosen_idxs = valid_idxs
    print("[Stage4] Final chosen indices:", chosen_idxs)
    # ── 여기까지 추가 ──

    # 3) 최종 결과 매핑 → raw_candidates 기준
    final_results = [ filtered_cands[i-1] for i in chosen_idxs ]   #10개 제한 시키기
    print("\n✅ 최종 추천 상품:")

    # ★ 여기에 10개 이상이면 앞 10개만 사용하도록 자르기 ★
    if len(final_results) > 10:
        final_results = final_results[:10]

    for idx, info in enumerate(final_results, start=1):
        PRODUCT_CACHE[info["상품코드"]] = info
        
        print(f"\n[{idx}] {info['제목']}")
        print(f"   카테고리   : {info['카테고리']}")
        print(f"   상품코드   : {info['상품코드']}")
        print(f"   가격       : {info['가격']}원")
        print(f"   배송비     : {info['배송비']}원")
        print(f"   이미지     : {info['이미지']}")
        print(f"   원산지     : {info['원산지']}")
        print(f"   상품링크   : {info['상품링크']}")
        print(f"   옵션       : {info['옵션']}")
        print(f"   조합형옵션 : {info['조합형옵션']}")
        print(f"   최대구매수량: {info['최대구매수량']}")
    print(f"================================")
    print(f"PRODUCT_CACHE {PRODUCT_CACHE}")

    return final_results  # 템플릿에 넘길 리스트

@app.route('/track_view', methods=['POST'])
def track_product_view():
    """상품 상세보기 클릭 이벤트 추적"""
    try:
        data = request.get_json()
        product_code = data.get('product_code')
        product_data = data.get('product_data', {})
        
        # 세션 ID 생성 또는 가져오기
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
        
        session_id = session['session_id']
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        # 사용자 ID 생성 또는 가져오기
        user_id = event_manager.get_or_create_user(session_id, ip_address, user_agent)
        
        # 상품 조회 이벤트 기록
        view_id = event_manager.record_product_view(
            user_id=user_id,
            session_id=session_id,
            product_data=product_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {'status': 'success', 'view_id': view_id, 'user_id': user_id}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/user_stats')
def user_stats():
    """사용자 통계 정보"""
    try:
        if 'session_id' not in session:
            return {'status': 'error', 'message': '세션이 없습니다.'}
        
        session_id = session['session_id']
        user_id = event_manager.get_or_create_user(session_id)
        
        # 사용자별 통계
        user_views = event_manager.get_user_product_views(user_id, limit=10)
        user_searches = event_manager.get_user_search_history(user_id, limit=10)
        
        return {
            'status': 'success',
            'user_id': user_id,
            'recent_views': user_views,
            'recent_searches': user_searches
        }
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/popular_products')
def popular_products():
    """인기 상품 조회"""
    try:
        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        popular = event_manager.get_popular_products(days=days, limit=limit)
        return {'status': 'success', 'products': popular}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    results = None

    # 새로고침 감지 (Cache-Control 헤더 확인)
    cache_control = request.headers.get('Cache-Control', '')
    is_refresh = 'no-cache' in cache_control or request.headers.get('Pragma') == 'no-cache'

    if request.method == 'POST':
        query     = request.form.get('query', '').strip()
        price_min = request.form.get('price_min', type=int)
        price_max = request.form.get('price_max', type=int)

        if not query:
            error = '검색어를 입력하세요.'
            return render_template('index.html', error=error, results=None)
        else:
            try:
                # 세션 ID 생성 또는 가져오기
                if 'session_id' not in session:
                    session['session_id'] = secrets.token_hex(16)
                
                session_id = session['session_id']
                ip_address = request.remote_addr
                user_agent = request.headers.get('User-Agent', '')
                
                # 사용자 ID 생성 또는 가져오기
                user_id = event_manager.get_or_create_user(session_id, ip_address, user_agent)
                
                # 검색 결과 생성
                results = run_recommendation_pipeline(query, price_min, price_max)
                
                # 검색 결과 수 기록
                results_count = len(results) if results else 0
                event_manager.record_search_event(
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    price_min=price_min,
                    price_max=price_max,
                    results_count=results_count,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                # 결과와 함께 페이지 렌더링
                return render_template('index.html', error=error, results=results)
                
            except Exception as ex:
                error = f"추천 처리 중 오류 발생: {ex}"
                return render_template('index.html', error=error, results=None)

    # GET 요청 처리 - 새로고침이거나 일반 GET 요청인 경우 결과 초기화
    # 세션에서 검색 관련 데이터 정리
    session.pop('search_results', None)
    session.pop('search_query', None)
    session.pop('search_price_min', None)
    session.pop('search_price_max', None)
    
    return render_template('index.html', error=error, results=results)

if __name__ == '__main__':
    # 환경 변수에서 설정 가져오기
    host = '0.0.0.0'
    port = '7070'
    debug = False     # production 모드이므로 디버그 끔
    
    print(f"🚀 서버 시작: {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)







