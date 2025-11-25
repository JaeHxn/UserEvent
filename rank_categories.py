# rank_categories.py
from pymilvus import connections, Collection, utility
import numpy as np
import os
import re
from openai import OpenAI
import json
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
# ── 환경설정 ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ← 환경변수 사용
MILVUS_HOST    = os.getenv("MILVUS_HOST", "114.110.135.96")
MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")

COL_L2         = os.getenv("COL_L2", "category_embed_onerclean_l2_ivf")
EMB_MODEL      = os.getenv("EMB_MODEL", "text-embedding-3-large")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-5-nano-2025-08-07")

EMB_DIM        = int(os.getenv("EMB_DIM", "3072"))
TOPK_RETRIEVE  = int(os.getenv("TOPK_RETRIEVE", "100"))
TOPN_RETURN    = int(os.getenv("TOPN_RETURN", "3"))  # ✅ 딱 3개 반환
TOPK_PER_SEED  = int(os.getenv("TOPK_PER_SEED", "30"))     # ✅ 시드 카테고리당 30개

# ── 연결/클라이언트 ─────────────────────────────────────────────
def _get_milvus_collection():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if not utility.has_collection(COL_L2):
        raise RuntimeError(f"Milvus 컬렉션 없음: {COL_L2}")
    col = Collection(COL_L2)
    col.load()
    return col

_client = None
def _get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# ── 유틸 ─────────────────────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    client = _get_openai_client()
    
    # 카테고리 구분자 제거 및 자연스러운 텍스트로 변환
    clean_text = _clean_category_text(text)
    
    resp = client.embeddings.create(model=EMB_MODEL, input=[clean_text])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    return vec

def _clean_category_text(category_text: str) -> str:
    """
    카테고리 계층 구조를 자연스러운 검색 텍스트로 변환
    예: "화장품/미용>향수>여성향수" → "화장품 미용 향수 여성향수"
    """
    if not category_text:
        return ""
    
    # 구분자 제거하고 공백으로 대체
    cleaned = category_text.replace('/', ' ').replace('>', ' ')
    
    # 연속된 공백을 하나로 합치고 양쪽 공백 제거
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def _retrieve_candidates(col: Collection, query: str, topk: int = TOPK_RETRIEVE):
    qv = _embed(query)
    res = col.search(
        data=qv.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 128}},
        limit=topk,
        output_fields=["category_full"],
    )
    hits = res[0]
    # [(idx, label, distance)]
    out = []
    for i, h in enumerate(hits, 0):
        out.append((i, h.get("category_full"), float(h.distance)))


    # ✅ 상위 10개만 출력
    print(f"\n[벡터 검색] 상위 10개 카테고리 (총 {len(out)}개 중):")
    print("=" * 80)
    for i, (idx, label, dist) in enumerate(out[:10], 1):
        print(f"{i}. [{idx}] {label} (거리: {dist:.4f})")
    print("=" * 80)




    return out


def _retrieve_candidates_for_seeds(col: Collection, seed_categories, topk_per_seed: int = TOPK_PER_SEED):
    """
    시드 카테고리들(LLM Category Top3)을 각각 쿼리로 사용하여
    Milvus에서 후보 카테고리를 검색하고, 라벨 기준으로 통합/중복 제거.

    seed_categories: ["식품>과자/스낵>감자칩", "식품>과자/스낵>스낵", ...]
    반환: [(idx, label, distance), ...]
    """
    seed_categories = [s for s in (seed_categories or []) if s and str(s).strip()]
    if not seed_categories:
        return []

    all_hits = []
    for seed in seed_categories:
        print(f"\n[SeedSearch] 시드 카테고리 검색: '{seed}'")
        seed_hits = _retrieve_candidates(col, seed, topk_per_seed)
        all_hits.extend(seed_hits)

    # 라벨 기준으로 최솟값 거리만 유지
    merged = {}
    for _, label, dist in all_hits:
        if not label:
            continue
        if (label not in merged) or (dist < merged[label]):
            merged[label] = dist

    # 거리 기준 정렬 후 인덱스 재부여
    items = sorted(merged.items(), key=lambda x: x[1])
    out = [(i, label, dist) for i, (label, dist) in enumerate(items)]

    print(f"\n[SeedSearch] 시드 통합 후 후보 수: {len(out)}개")
    return out






def _make_catalog_prompt_lines(cands):
    # "0: 생활/건강>주방..." 형태
    return "\n".join(f"{i}: {label}" for i, label, _ in cands)

def _build_prompt(user_query: str, catalog_str: str) -> str:
    # 오로지 인덱스 쉼표열만 출력하도록 강제
    return (
        f"사용자 검색: {user_query}\n\n"
        "후보 카테고리 목록:\n"
        f"{catalog_str}\n\n"
        f"지시사항: 위 후보를 사용자 의도에 맞게 재정렬하여 제일 의도가 맞는 1등~3등 까지 순서대로 {TOPN_RETURN}개만 출력하세요.\n"
        "응답 형식(예시): 22,10,7\n"
        "답변:"
    )

def _parse_indices_to_labels(raw_text: str, candidates, topn: int):
    cand_len = len(candidates)
    nums = re.findall(r"\d+", raw_text or "")
    seen, idxs = set(), []
    for s in nums:
        try:
            i = int(s)
        except ValueError:
            continue
        if 0 <= i < cand_len and i not in seen:
            idxs.append(i); seen.add(i)
        if len(idxs) >= topn:
            break
    # 부족하면 원본순 보충
    if len(idxs) < topn:
        for i in range(cand_len):
            if i not in seen:
                idxs.append(i); seen.add(i)
            if len(idxs) >= topn:
                break
    idxs = idxs[:topn]
    return [candidates[i][1] for i in idxs]

def _llm_rerank(user_query: str, candidates):
    """
    숫자,쉼표만 받는 리랭킹 → 실패 시 폴백
    """
    client = _get_openai_client()
    catalog = _make_catalog_prompt_lines(candidates)
    prompt  = _build_prompt(user_query, catalog)

    raw_text = ""
    # 1) Responses API 시도
    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": "숫자와 쉼표만 출력하세요. 다른 텍스트 금지."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ],
                },
            ],
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        raw_text = (getattr(resp, "output_text", "") or "").strip()
    except Exception:
        raw_text = ""

    picked = []
    if raw_text:
        try:
            picked = _parse_indices_to_labels(raw_text, candidates, TOPN_RETURN)
        except Exception:
            picked = []

    # 2) 폴백: Chat Completions
    if not picked:
        try:
            cc = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Respond with ONLY numbers and commas. No other text allowed.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=32,
            )
            txt = (cc.choices[0].message.content or "").strip()
            picked = _parse_indices_to_labels(txt, candidates, TOPN_RETURN)
        except Exception:
            picked = []

    # 3) 최후 폴백: 원본 상위
    if not picked:
        picked = [lbl for _, lbl, _ in candidates[:TOPN_RETURN]]

    return picked

def get_top_categories(*args):
    """
    호출 형태:
      get_top_categories(query)
      get_top_categories(top1, top2, top3, query)

    요구사항(사용자 설명):
      - top1, top2, top3 카테고리(LLM 추출)를 각각 시드로 사용
      - 시드별로 벡터검색 상위 30개씩 (총 최대 90개) 그대로 합치기 (라벨 병합/중복 제거하지 않음)
      - 이 90개 후보 전체를 쿼리(query, 마지막 인자)로 LLM 리랭킹
      - 리랭킹 결과에서 상위 3개 최종 반환

    기존 _retrieve_candidates_for_seeds 는 라벨 병합을 수행하여 90개 → 축소될 수 있었음.
    여기서는 병합 없이 순수 30 * 시드개수 후보를 만들기 위해 새 로직 적용.
    """
    if not args:
        return []
    if len(args) == 1:
        # 단일 호출: 기존 방식 (쿼리만)
        query = args[0]
        if not query or not str(query).strip():
            return []
        col = _get_milvus_collection()
        cands = _retrieve_candidates(col, query, TOPK_RETRIEVE)
        return _final_rerank_and_select(query, cands)

    # 다중 인자: 마지막이 query, 앞이 시드들
    query = args[-1]
    seeds = [s for s in args[:-1] if isinstance(s, str) and s and s.strip() and s.lower() != 'none']
    if not query or not str(query).strip():
        return []
    col = _get_milvus_collection()

    all_hits = []
    print(f"[Debug] 시드 기반 RAW 후보 수집 (중복 유지): seeds={seeds}")
    for seed in seeds:
        try:
            seed_hits = _retrieve_candidates(col, seed, TOPK_PER_SEED)
            # 순서를 유지하며 그대로 추가
            all_hits.extend(seed_hits)
            print(f"[SeedRaw] '{seed}' → {len(seed_hits)}개")
        except Exception as e:
            print(f"[SeedRaw] 오류 '{seed}': {e}")

    # 시드 없을 경우 단일 쿼리 후보로 대체
    if not all_hits:
        print("[Debug] 시드 결과 없음 → 단일 쿼리 검색 폴백")
        all_hits = _retrieve_candidates(col, query, TOPK_RETRIEVE)

    if not all_hits:
        print("[Debug] 최종 후보 없음")
        return []

    # 인덱스 재부여 (LLM 입력 형태 통일: (idx, label, distance))
    cands = [(i, label, dist) for i, (orig_i, label, dist) in enumerate(all_hits)]
    return _final_rerank_and_select(query, cands)


def _final_rerank_and_select(rerank_query: str, candidates):
    """후보(최대 90개)를 rerank_query로 LLM 재정렬 후 TopN_RETURN 반환."""
    if not candidates:
        return []
    picked = _llm_rerank(rerank_query, candidates)
    cand_labels = [lbl for _, lbl, _ in candidates]
    uniq = []
    for p in picked:
        if p in cand_labels and p not in uniq:
            uniq.append(p)
        if len(uniq) >= TOPN_RETURN:
            break
    if len(uniq) < TOPN_RETURN:
        for lbl in cand_labels:
            if lbl not in uniq:
                uniq.append(lbl)
            if len(uniq) >= TOPN_RETURN:
                break
    final = uniq[:TOPN_RETURN]
    print(f"[CatRank] 최종 Top{len(final)}: {final}")
    return final

# ── CLI 사용 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    result = {
        "query": query,
        "categories": get_top_categories(query) if query else []
    }
    # 한 줄 JSON 출력
    print(json.dumps(result, ensure_ascii=False))
