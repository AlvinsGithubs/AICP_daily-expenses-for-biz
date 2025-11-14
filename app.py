# 2025-10-20-16 AI 기반 출장비 계산 도구 (v16.0 - Async, Dynamic Weights, Full Admin)
# --- 설치 안내 ---
# 1. 아래 명령으로 필요한 패키지를 설치하세요.
#    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
#
# 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.
import streamlit as st
import pandas as pd
import json
import os
import re
import fitz  # PyMuPDF 라이브러리
import openai
from dotenv import load_dotenv
import io
from datetime import datetime, timedelta
import time
import random
import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
from collections import Counter
from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import text


import altair as alt  # [신규 2] 고급 차트 라이브러리
import pydeck as pdk  # [신규 2] 고급 3D 지도 라이브러리

# [신규 2(지도)] 관련 임포트
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- [v20.0 DB연동] Streamlit Connection 초기화 ---
# Supabase (Postgres) DB에 연결합니다.
# 이 연결은 Streamlit의 Secrets에서 "connections.supabase_db" 설정을 자동으로 읽어옵니다.
conn = st.connection("supabase_db", type="sql", dialect="postgresql")

# --- [v20.0 DB연동] data_sources.menu_cache.py 기능 대체 ---
# 이제 'menu_cache.py' 파일은 더 이상 필요하지 않습니다.
def load_cached_menu_prices(
    city: str, 
    country: str, 
    neighborhood: Optional[str]
) -> List[Dict[str, Any]]:
    """DB에서 특정 위치의 메뉴 가격 샘플을 로드합니다."""
    city_lower = city.lower()
    country_lower = country.lower()
    
    query_base = "SELECT * FROM menu_cache WHERE LOWER(city) = :city AND LOWER(country) = :country"
    params = {"city": city_lower, "country": country_lower}

    if neighborhood:
        # 1순위: 세부 지역이 일치하는 데이터
        query_neighborhood = query_base + " AND LOWER(neighborhood) = :neighborhood"
        params_neighborhood = params.copy()
        params_neighborhood["neighborhood"] = neighborhood.lower().strip()
        df_neighborhood = conn.query(query_neighborhood, params=params_neighborhood)
        if not df_neighborhood.empty:
            return df_neighborhood.to_dict('records')

    # 2순위: 세부 지역이 비어있는 '도시 전체' 데이터
    query_city = query_base + " AND (neighborhood IS NULL OR neighborhood = '')"
    df_city = conn.query(query_city, params=params)
    return df_city.to_dict('records')

def load_all_cache() -> List[Dict[str, Any]]:
    """DB에서 *전체* 메뉴 캐시 목록을 로드합니다."""
    df = conn.query("SELECT * FROM menu_cache ORDER BY last_updated DESC, id DESC", ttl=5) # 5초 캐시
    return df.to_dict('records')

def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool:
    """새로운 캐시 항목 1개를 DB에 추가합니다."""
    try:
        new_entry["last_updated"] = datetime.now().date()
        # DataFrame으로 변환하여 insert (to_dict('records')와 형식을 맞춤)
        df_new = pd.DataFrame([new_entry]) 
        conn.insert("menu_cache", df_new)
        return True
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")
        return False

def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool:
    """(삭제 시 사용) *전체* 메뉴 캐시 목록을 DB에 덮어씁니다."""
    try:
        # 1. 기존 데이터 모두 삭제
        with conn.session as s:
            s.execute(text("DELETE FROM menu_cache"))
            s.commit()
            
        # 2. 새 목록으로 삽입 (비어있지 않다면)
        if all_samples:
            df_new = pd.DataFrame(all_samples)
            # id, created_at 등 자동 생성 컬럼은 DataFrame에 없을 수 있으므로 제외
            df_new = df_new.drop(columns=["id", "created_at"], errors='ignore') 
            conn.insert("menu_cache", df_new)
        return True
    except Exception as e:
        st.error(f"DB 업데이트 실패: {e}")
        return False

MENU_CACHE_ENABLED = True # DB를 사용하므로 항상 활성화
# --- [v20.0 DB연동] 끝 ---

# --- 초기 환경 설정 ---

# .env 파일에서 환경 변수 로드
load_dotenv()

# Maximum number of AI calls per analysis
NUM_AI_CALLS = 10
# --- Weight configuration (sum should remain 1.0) ---
DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
_WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


def weight_config_path() -> str:
    return os.path.join(DATA_DIR, "weight_config.json")



def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
    try:
        un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
    except (TypeError, ValueError):
        un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
    try:
        ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
    except (TypeError, ValueError):
        ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

    total = un_raw + ai_raw
    if total <= 0:
        return dict(DEFAULT_WEIGHT_CONFIG)

    un_norm = max(0.0, min(1.0, un_raw / total))
    ai_norm = max(0.0, min(1.0, ai_raw / total))

    total_norm = un_norm + ai_norm
    if total_norm == 0:
        return dict(DEFAULT_WEIGHT_CONFIG)
    return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# --- [v20.0 DB연동] Weight Config 함수 (DB 기반) ---
def _get_config_from_db(config_key: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """DB의 app_config 테이블에서 설정을 읽어옵니다."""
    try:
        result = conn.query(
            "SELECT config_value FROM app_config WHERE config_key = :key",
            params={"key": config_key},
            ttl=10 # 10초 캐시
        )
        if result.empty:
            return default_config
        
        db_value = result.iloc[0]["config_value"]
        if isinstance(db_value, str): # DB가 JSON을 문자열로 반환할 경우
            return json.loads(db_value)
        return db_value
    except Exception:
        return default_config

def _save_config_to_db(config_key: str, config_value: Dict[str, Any]) -> None:
    """DB의 app_config 테이블에 설정을 저장(업데이트)합니다."""
    try:
        # PostgreSQL의 JSONB 타입을 위해 json.dumps 사용
        config_json = json.dumps(config_value) 
        
        # 'UPSERT' (Update or Insert) 쿼리
        with conn.session as s:
            s.execute(text(
                f"""
                INSERT INTO app_config (config_key, config_value, last_updated)
                VALUES (:key, :value, :now)
                ON CONFLICT (config_key) 
                DO UPDATE SET config_value = :value, last_updated = :now
                """
            ), params={"key": config_key, "value": config_json, "now": datetime.now()})
            s.commit()
    except Exception as e:
        st.error(f"DB 설정 저장 실패: {e}")

def get_weight_config() -> Dict[str, float]:
    """DB에서 가중치 설정을 불러옵니다."""
    config = _get_config_from_db("weight_config", DEFAULT_WEIGHT_CONFIG)
    normalized = _normalize_weight_config(config)
    
    # st.session_state에도 저장 (기존 로직과 호환성 유지)
    try:
        st.session_state["weight_config"] = normalized
    except Exception:
        pass
    return normalized

def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
    """DB와 세션의 가중치 설정을 업데이트합니다."""
    config = {"un_weight": un_weight, "ai_weight": ai_weight}
    normalized = _normalize_weight_config(config)
    _save_config_to_db("weight_config", normalized)
    
    try:
        st.session_state["weight_config"] = normalized
    except Exception:
        pass
    return normalized
# --- [v20.0 DB연동] 끝 ---
def load_weight_config() -> Dict[str, float]:
    """
    (Backwards compatibility)
    예전 파일 기반 함수 이름을 그대로 쓰되,
    실제 구현은 DB 기반 get_weight_config()를 사용합니다.
    """
    return get_weight_config()



# 분석 결과를 저장할 디렉터리 경로


def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
    """Return markdown-friendly bullets for cached menu/reference entries."""
    lines_out: List[str] = []
    if not menu_samples:
        return lines_out

    for sample in menu_samples[:max_items]:
        if not isinstance(sample, dict):
            continue

        name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

        url = None
        for key in ("url", "link", "source_url", "href"):
            value = sample.get(key)
            if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
                url = value
                break

        details: List[str] = []
        price = sample.get("price")
        if isinstance(price, (int, float)):
            currency = sample.get("currency") or "USD"
            details.append(f"{currency} {price}")
        elif isinstance(price, str) and price.strip():
            details.append(price.strip())

        category = sample.get("category")
        if category:
            details.append(str(category))

        last_updated = sample.get("last_updated")
        if last_updated:
            details.append(f"updated {last_updated}")

        detail_text = ", ".join(details)
        label = f"[{name}]({url})" if url else name

        if detail_text:
            lines_out.append(f"{label} - {detail_text}")
        else:
            lines_out.append(label)

    return lines_out


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
    "show_un_basis": True,
    "show_ai_estimate": True,
    "show_weighted_result": True,
    "show_ai_market_detail": True,
    "show_provenance": True,
    "show_menu_samples": True,
}
EMPLOYEE_SECTION_LABELS = [
    ("show_un_basis", "UN-DSA 기준 카드"),
    ("show_ai_estimate", "AI 시장 추정 카드"),
    ("show_weighted_result", "가중 평균 결과 카드"),
    ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
    ("show_provenance", "AI 산출 근거(JSON)"),
    ("show_menu_samples", "레퍼런스 메뉴 표"),
]
_UI_SETTINGS_CACHE: Dict[str, Any] = {}


CARD_STYLES = {
    "primary": {
        # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
        "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
        "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
        "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
        "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
    },
    "secondary": {
        # [수정] Streamlit 테마 변수 사용
        "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
        "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
        "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
        "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
    },
    "muted": {
        # [수정] Streamlit 테마 변수 사용
        "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
        "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
        "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
        "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
    },
}


def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
    style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
    # [수정] 캡션에 스타일 적용
    caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
    card_html = f"""
    <div style="{style['container']}">
        <div style="{style['title']}">{title}</div>
        <div style="{style['value']}">{value}</div>
        {caption_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
    style = CARD_STYLES["primary"]
    card_html = f"""
    <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
        <div style="{style['title']}">Estimated Total Per Diem ({level_label})</div>
        <div style="{style['value']}">$ {total:,}</div>
        <div style="{style['caption']}">
            <span style='font-size:0.95rem;opacity:0.8;'>Calculation</span><br/>
            $ {daily:,} × {days} days × {term_label} (×{multiplier:.2f})
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
    normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
    if isinstance(sections, dict):
        for key in normalized:
            normalized[key] = bool(sections.get(key, normalized[key]))
    return normalized

def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure UI settings include expected keys with correct types."""
    normalized = dict(DEFAULT_UI_SETTINGS)
    raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
    normalized["show_employee_tab"] = bool(raw_visibility)
    normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
    return normalized

# --- [v20.0 DB연동] UI Settings 함수 (DB 기반) ---
def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """DB에 UI 설정을 저장합니다."""
    normalized = _normalize_ui_settings(settings)
    _save_config_to_db("ui_settings", normalized)
    global _UI_SETTINGS_CACHE
    _UI_SETTINGS_CACHE = dict(normalized)
    return normalized

def load_ui_settings(force: bool = False) -> Dict[str, Any]:
    """DB에서 UI 설정을 로드합니다."""
    global _UI_SETTINGS_CACHE
    if _UI_SETTINGS_CACHE and not force:
        return dict(_UI_SETTINGS_CACHE)
    
    config = _get_config_from_db("ui_settings", DEFAULT_UI_SETTINGS)
    normalized = _normalize_ui_settings(config)
    
    _UI_SETTINGS_CACHE = dict(normalized)
    return dict(normalized)
# --- [v20.0 DB연동] 끝 ---

JOB_LEVEL_RATIOS = {
    "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6": 1.00,
    "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
}

TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
DEFAULT_TRIP_LENGTH = ["Short-term", "Long-term"]
LONG_TERM_THRESHOLD_DAYS = 30
SHORT_TERM_MULTIPLIER = 1.0
LONG_TERM_MULTIPLIER = 1.05
TRIP_TERM_LABELS = {"Short-term": "Short-term", "Long-term": "Long-term"}


def classify_trip_duration(days: int) -> Tuple[str, float]:
    """Return trip term classification and multiplier based on duration in days."""
    if days >= LONG_TERM_THRESHOLD_DAYS:
        return "Long-term", LONG_TERM_MULTIPLIER
    return "Short-term", SHORT_TERM_MULTIPLIER

DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
    {"region": "North America", "city": "Nassau", "country": "Bahamas"},
    {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
    {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
    {"region": "North America", "city": "Seattle", "country": "USA"},
    {"region": "North America", "city": "Florida", "country": "USA"},
    {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
    {"region": "North America", "city": "Toronto", "country": "Canada"},
    {"region": "Europe", "city": "Valletta", "country": "Malta"},
    {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
    {"region": "Europe", "city": "Dublin", "country": "Ireland"},
    {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
    {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
    {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
    {"region": "Europe", "city": "San Remo", "country": "Italy"},
    {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
    {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
    {"region": "Europe", "city": "Paris", "country": "France"},
    {"region": "Europe", "city": "Provence", "country": "France"},
    {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
    {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
    {"region": "Asia", "city": "Manila", "country": "Philippines"},
    {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
    {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
    {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
    {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
    {"region": "Others", "city": "Sydney", "country": "Australia"},
    {"region": "Others", "city": "Rosario", "country": "Argentina"},
    {"region": "Others", "city": "Marrakech", "country": "Morocco"},
    {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
]


def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """대상 도시 항목에 기본값을 채워 넣는다."""
    entry = dict(entry)
    entry.setdefault("region", "Others")
    entry.setdefault("neighborhood", "")
    entry.setdefault("hotel_cluster", "")
    entry.setdefault("trip_lengths", DEFAULT_TRIP_LENGTH.copy())
    return entry


# --- [v20.0 DB연동] Target Cities 함수 (DB 기반) ---
def load_target_city_entries() -> List[Dict[str, Any]]:
    """DB에서 모든 도시 목록을 로드하되, 비어 있으면 기본값으로 폴백."""
    df = conn.query("SELECT * FROM target_cities ORDER BY region, country, city", ttl=10)
    if df.empty:
        # DB 비어 있으면 코드에 박혀 있는 DEFAULT_TARGET_CITY_ENTRIES 사용
        return [normalize_target_entry(e) for e in DEFAULT_TARGET_CITY_ENTRIES]
    return df.to_dict('records')

def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
    """(수정/삭제 시 사용) 전체 도시 목록을 DB에 덮어씁니다."""
    try:
        # 1. 기존 데이터 모두 삭제
        with conn.session as s:
            s.execute(text("DELETE FROM target_cities"))
            s.commit()
            
        # 2. 새 목록으로 삽입 (비어있지 않다면)
        if entries:
            df_new = pd.DataFrame(entries)
            # DB가 자동 생성하는 'id', 'created_at'은 삽입 시 제외
            df_new = df_new.drop(columns=["id", "created_at"], errors='ignore') 
            conn.insert("target_cities", df_new)
    except Exception as e:
        st.error(f"DB 도시 목록 저장 실패: {e}")

TARGET_CITIES_ENTRIES = load_target_city_entries() # 앱 시작 시 DB에서 로드

def get_target_city_entries() -> List[Dict[str, Any]]:
    if "target_cities_entries" in st.session_state:
        return st.session_state["target_cities_entries"]
    # DB에서 로드한 최신본을 세션 상태에 저장
    st.session_state["target_cities_entries"] = load_target_city_entries()
    return st.session_state["target_cities_entries"]

def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
    # 1. DB에 영구 저장
    save_target_city_entries(entries) 
    # 2. 현재 세션 상태에도 즉시 반영
    st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
# --- [v20.0 DB연동] 끝 ---


def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    entries = entries or get_target_city_entries()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry.get("region", "Others"), []).append(entry)
    return grouped


def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    entries = entries or get_target_city_entries()
    return [normalize_target_entry(entry) for entry in entries]

# 도시 이름 별칭 매핑
CITY_ALIASES = {
    "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
    "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
    "kuala lumpur": "kuala lumpur"
}

# --- 도시 메타데이터 및 시즌 설정 ---

SEASON_BANDS = [
    {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
    {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
    {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
    {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
]

CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
    ("las vegas", "usa"): [
        {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
        {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
    ],
    ("seoul", "korea, republic of"): [
        {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
        {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
    ],
    ("barcelona", "spain"): [
        {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
    ],
}


def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
    key = (city.lower(), country.lower())
    for entry in get_target_city_entries():
        if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
            return {
                "neighborhood": entry.get("neighborhood"),
                "hotel_cluster": entry.get("hotel_cluster"),
            }
    return {"neighborhood": None, "hotel_cluster": None}


def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
    """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
    month = datetime.now().month
    city_key = (city.lower(), country.lower())
    overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
    for override in overrides:
        if month in override["months"]:
            return {
                "label": override["label"],
                "factor": override["factor"],
                "source": "city_override",
            }

    for band in SEASON_BANDS:
        if month in band["months"]:
            return {
                "label": band["label"],
                "factor": band["factor"],
                "source": "global_profile",
            }

    return {"label": "Standard", "factor": 1.0, "source": "default"}


# --- [신규 1] aggregate_ai_totals 함수 수정 ---
# (이상치 제거 + 변동계수(VC) 계산)
def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
    """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
    if not totals:
        return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

    sorted_totals = sorted(totals)
    if len(sorted_totals) >= 4:
        try:
            q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
        except (ValueError, StatisticsError):  # type: ignore[name-defined]
            filtered = sorted_totals
    else:
        filtered = sorted_totals

    if not filtered:
        filtered = sorted_totals

    removed_values: List[int] = []
    filtered_counter = Counter(filtered)
    for value in sorted_totals:
        if filtered_counter[value]:
            filtered_counter[value] -= 1
        else:
            removed_values.append(value)

    computed_mean = mean(filtered) if filtered else None
    
    # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
    variation_coeff = None
    if filtered and computed_mean and computed_mean > 0:
        if len(filtered) > 1:
            try:
                computed_stdev = stdev(filtered)
                variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
            except StatisticsError:
                variation_coeff = 0.0 # 모든 값이 동일
        else:
            variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

    return {
        "used_values": filtered,
        "removed_values": removed_values,
        "mean_raw": computed_mean,
        "mean": round(computed_mean) if computed_mean is not None else None,
        "variation_coeff": variation_coeff # <-- AI 일관성 점수
    }

# --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
def get_dynamic_weights(
    variation_coeff: Optional[float], 
    admin_weights: Dict[str, float]
) -> Dict[str, Any]:
    """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
    # 관리자 설정값을 기본값으로 사용
    base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
    if variation_coeff is None:
        # AI 데이터가 없으면 UN 100%
        return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
    if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
        # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
        dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
        source = f"High AI Consistency (VC: {variation_coeff:.2f})"
    elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
        # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
        dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
        source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
    else:
        # 5% ~ 15% 사이: 관리자 설정값 유지
        dynamic_ai_weight = base_ai_weight
        source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

    final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
    final_un_weight = 1.0 - final_ai_weight
    
    return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# --- 핵심 로직 함수 ---

def parse_pdf_to_text(uploaded_file):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(4, len(doc)):
        full_text += doc[page_num].get_text("text") + "\n\n"
    return full_text

# --- [v20.0 DB연동] Report History 함수 (DB 기반) ---
def get_history_files() -> List[str]:
    """DB에서 과거 보고서 '이름' 목록을 최신순으로 가져옵니다."""
    df = conn.query("SELECT name FROM analysis_reports ORDER BY created_at DESC", ttl=10) # 10초 캐시
    return df['name'].tolist()

def save_report_data(data):
    """분석 결과를 DB(JSONB)에 저장합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.json"  # 이름은 고유 식별자로 사용

    try:
        # JSON 문자열로 먼저 변환
        report_json = json.dumps(data)

        # conn.insert 를 이용해 간단하게 INSERT
        df = pd.DataFrame([{
            "name": filename,
            "report_data": report_json,  # JSONB 컬럼이어도 문자열을 넣으면 Postgres가 캐스팅해 줍니다.
            # created_at 은 테이블 DEFAULT NOW() 가 있으니 굳이 넣지 않아도 됨
        }])

        conn.insert("analysis_reports", df)

    except Exception as e:
        st.error(f"DB 보고서 저장 실패: {e}")

def load_report_data(filename):
    """DB에서 특정 보고서(JSON)를 이름으로 불러옵니다."""
    try:
        result = conn.query(
            "SELECT report_data FROM analysis_reports WHERE name = :name",
            params={"name": filename},
            ttl=3600 # 1시간 캐시
        )
        if result.empty:
            return None
        
        db_value = result.iloc[0]["report_data"]
        
        if isinstance(db_value, str): # DB가 JSON을 문자열로 반환할 경우
            data = json.loads(db_value)
        else: # 이미 dict/list로 반환된 경우
            data = db_value
            
        return _sanitize_report_data(data)
    except Exception as e:
        st.error(f"DB 보고서 로드 실패: {e}")
        return None
# --- [v20.0 DB연동] 끝 ---


def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    cities = data.get("cities")
    if isinstance(cities, list):
        for city in cities:
            if isinstance(city, dict):
                city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
    return data




def build_tsv_conversion_prompt():
    return """
[Task]
Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
[Guidelines]
1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
[Output Format]
Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
Country	Area	First 60 Days US$	Room as % of DSA
USA (US Dollar)	Washington D.C.	403	57
"""


def call_openai_for_tsv_conversion(pdf_chunk, api_key):
    client = openai.OpenAI(api_key=api_key)
    system_prompt = build_tsv_conversion_prompt()
    user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
        tsv_content = response.choices[0].message.content
        if "```" in tsv_content:
            tsv_content = tsv_content.split('```')[1].strip()
            if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
        return tsv_content
    except Exception as e:
        st.error(f"OpenAI API request failed: {e}")
        return None

def process_tsv_data(tsv_content):
    try:
        df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
        df['Country'] = df['Country'].ffill()
        df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
        df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
        df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
        df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
        df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
        df = df.astype({'TotalDSA': int, 'RoomPct': int})
    except Exception as e:
        st.error(f"TSV processing error: {e}")
        return None

    all_target_cities = get_all_target_cities()
    final_cities_data = []
    for target in all_target_cities:
        city_data = {
            "city": target["city"],
            "country_display": target["country"],
            "notes": "",
            "neighborhood": target.get("neighborhood"),
            "hotel_cluster": target.get("hotel_cluster"),
            "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
        }
        found_row = None

        # --- [수정] un_dsa_substitute 값 안전하게 처리 ---
        sub_raw = target.get("un_dsa_substitute")

        sub_value = None
        if isinstance(sub_raw, dict):
            sub_value = sub_raw
        elif isinstance(sub_raw, str) and sub_raw.strip():
            # JSON 문자열로 들어온 경우 대비
            try:
                sub_value = json.loads(sub_raw)
            except Exception:
                sub_value = None

        if (
            isinstance(sub_value, dict)
            and sub_value.get("city")
            and sub_value.get("country")
        ):
            # 유효한 대체 도시 정보가 있을 때만 사용
            search_target = sub_value
            is_substitute = True
        else:
            # 없으면 원래 도시 사용
            search_target = target
            is_substitute = False
        # --- [수정 끝] ---

        country_df = df[
            df["Country"].str.contains(search_target["country"], case=False, na=False)
        ]
        if not country_df.empty:
            target_city_lower = search_target["city"].lower()
            target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
            exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
            non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
            if not non_special_rate.empty:
                found_row = non_special_rate.iloc[0]
                city_data["notes"] = "Exact city match"
            elif not exact_match.empty:
                found_row = exact_match.iloc[0]
                city_data["notes"] = "Exact city match (special rate possible)"
            if found_row is None:
                elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
                if not elsewhere_match.empty:
                    found_row = elsewhere_match.iloc[0]
                    city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
        if is_substitute and found_row is not None:
            city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
        if found_row is not None:
            total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
            if 0 < total_dsa and 0 <= room_pct <= 100:
                per_diem = round(total_dsa * (1 - room_pct / 100))
                city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
            else: city_data["un"] = {"status": "not_found"}
        else:
            city_data["un"] = {"status": "not_found"}
            if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
        city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
        final_cities_data.append(city_data)
    return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
async def get_market_data_from_ai_async(
    client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
    city: str,
    country: str,
    source_name: str = "",
    context: Optional[Dict[str, Optional[str]]] = None,
    season_context: Optional[Dict[str, Any]] = None,
    menu_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
    context = context or {}
    season_context = season_context or {}
    menu_samples = menu_samples or []

    request_id = random.randint(10000, 99999)
    called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # --- (내부 헬퍼 함수들은 기존과 동일) ---
    def _build_location_block() -> str:
        lines: List[str] = []
        if context.get("neighborhood"):
            lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
        if context.get("hotel_cluster"):
            lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
        return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

    def _build_menu_block() -> str:
        if not menu_samples:
            return "- No direct venue menu data available; use standard mid-range venues."
        snippets = []
        for sample in menu_samples[:5]:
            vendor = sample.get("vendor") or sample.get("name") or "Venue"
            category = sample.get("category") or "General"
            price = sample.get("price")
            currency = sample.get("currency", "USD")
            last_updated = sample.get("last_updated")
            if price is None:
                continue
            tail = f" (last updated {last_updated})" if last_updated else ""
            snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
        if not snippets:
            return "- No direct venue menu data available; use standard mid-range venues."
        return "Menu price signals:\n" + "\n".join(snippets)

    location_block = _build_location_block()
    menu_block = _build_menu_block()
    season_label = season_context.get("label", "Standard")
    season_factor = season_context.get("factor", 1.0)
    season_source = season_context.get("source", "global_profile")
    # --- (프롬프트 구성은 기존과 동일) ---
    prompt = f"""
You are a corporate travel cost analyst. Request ID: {request_id}.
Location context:
{location_block}
Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
{menu_block}

For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

{{
  "food": {{
    "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
    "value": <integer>
  }},
  "transport": {{
    "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
    "value": <integer>
  }},
  "misc": {{
    "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
    "value": <integer>
  }}
}}
"""

    try:
        # --- [수정] 비동기 API 호출로 변경 ---
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        # --- [수정] 끝 ---
        
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)

        food = data.get("food", {}).get("value")
        transport = data.get("transport", {}).get("value")
        misc = data.get("misc", {}).get("value")

        food_val = food if isinstance(food, int) else 0
        transport_val = transport if isinstance(transport, int) else 0
        misc_val = misc if isinstance(misc, int) else 0

        meta = {
            "source_name": source_name,
            "request_id": request_id,
            "prompt": prompt.strip(),
            "response_raw": raw_content,
            "called_at": called_at,
            "season_context": season_context,
            "location_context": context,
            "menu_samples_used": menu_samples[:5],
        }

        if food_val == 0 and transport_val == 0 and misc_val == 0:
            return {
                "status": "na",
                "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
                "meta": meta,
            }

        total = food_val + transport_val + misc_val
        notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
        return {
            "food": food_val,
            "transport": transport_val,
            "misc": misc_val,
            "total": total,
            "status": "ok",
            "notes": notes,
            "meta": meta,
        }

    except Exception as e:
        return {
            "status": "na",
            "notes": f"{source_name} AI data extraction failed: {e}",
            "meta": {
                "source_name": source_name,
                "request_id": request_id,
                "prompt": prompt.strip(),
                "called_at": called_at,
                "season_context": season_context,
                "location_context": context,
                "menu_samples_used": menu_samples[:5],
                "error": str(e),
            },
        }
# --- [개선 1] 끝 ---

def generate_markdown_report(report_data):
    md = f"# Business Travel Daily Allowance Report\n\n"
    md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
    weights_cfg = load_weight_config()
    md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

    valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
    if valid_allowances:
        md += "## 1. Summary\n\n"
        md += (
            f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
            f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
        )

    md += "## 2. City Details\n\n"
    table_data = []
    all_reference_links: Set[str] = set()
    all_target_cities = get_all_target_cities()
    report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
    for target in all_target_cities:
        city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
        if city_data:
            un_data = city_data.get('un', {})
            ai_summary = city_data.get('ai_summary', {})
            season_context = city_data.get('season_context', {})

            un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
            final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
            delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
            ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
            ai_runs_used = ai_summary.get('successful_runs', 0)
            ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
            removed_totals = ai_summary.get('removed_totals') or []
            reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
            # [신규 1] 동적 가중치 적용 사유
            weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

            for link in reference_links:
                if isinstance(link, str) and link.strip():
                    all_reference_links.add(link.strip())

            row = {
                'City': city_data.get('city', 'N/A'),
                'Country': city_data.get('country_display', 'N/A'),
                'UN-DSA (1 day)': un_val,
                'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
                'AI runs used': f"{ai_runs_used}/{ai_attempts}",
                'Season label': season_context.get('label', 'Standard'),
                'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
                'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
            }
            for j in range(1, NUM_AI_CALLS + 1):
                market_data = city_data.get(f"market_data_{j}", {})
                md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
                row[f"AI run {j}"] = md_val

            row.update({
                'Final allowance': final_val,
                'Delta vs UN (%)': delta,
                'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
                'Notes': city_data.get('notes', ''),
            })
            table_data.append(row)

    df = pd.DataFrame(table_data)
    md += df.to_markdown(index=False)
    md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

    md += (
        "---\n"
        "## 3. Methodology\n\n"
        "1. **Baseline (UN-DSA)**\n"
        "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
        "   - Normalize the data as TSV to align city/country names.\n\n"
        "2. **Market data (AI)**\n"
        "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
        "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
        "3. **Post-processing**\n"
        "   - Remove outliers via the IQR rule and compute averages.\n"
        "   - Apply season factors and blend with UN-DSA using configured weights.\n"
        "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
        "   - Multiply by grade ratios to produce per-level allowances.\n\n"
        "---\n"
        "## 4. Sources\n\n"
        "- UN-DSA Circular (International Civil Service Commission)\n"
        "- Mercer Cost of Living (2025 edition)\n"
        "- Numbeo Cost of Living Index (2025 snapshot)\n"
        "- Expatistan Cost of Living Guide\n"
    )

    return md




# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide")
st.title("AICP: NSUS GROUP Per Diem Calculation & Inquiry System")

if 'latest_analysis_result' not in st.session_state:
    st.session_state.latest_analysis_result = None
if 'target_cities_entries' not in st.session_state:
    st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
if 'weight_config' not in st.session_state:
    st.session_state.weight_config = load_weight_config()
else:
    st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

ui_settings = load_ui_settings()
stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
if "employee_tab_visibility" not in st.session_state:
    st.session_state.employee_tab_visibility = stored_employee_tab_visible
employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
if "employee_sections_visibility" not in st.session_state:
    st.session_state.employee_sections_visibility = section_visibility_default
else:
    st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
employee_sections_visibility = st.session_state.employee_sections_visibility


# --- [Improvement 3 & New 2] Tab structure change (v18.0) ---
tab_definitions = [
    "📊 Executive Dashboard", # [New 2] Dashboard tab added
]

if employee_tab_visible:
    tab_definitions.append("💵 Per Diem Inquiry (Employee)")

# Split admin tab into two
tab_definitions.append("📈 Report Analysis (Admin)")
tab_definitions.append("🛠️ System Settings (Admin)")

tabs = st.tabs(tab_definitions)

# Assign tab variables
dashboard_tab = tabs[0]
tab_index_offset = 1

if employee_tab_visible:
    employee_tab = tabs[tab_index_offset]
    admin_analysis_tab = tabs[tab_index_offset + 1]
    admin_config_tab = tabs[tab_index_offset + 2]
    tab_index_offset += 1
else:
    employee_tab = None
    admin_analysis_tab = tabs[tab_index_offset]
    admin_config_tab = tabs[tab_index_offset + 1]

# --- [End of modification] ---

with dashboard_tab:
    st.header("Global Cost Dashboard")
    st.info("Visualizes the global business trip cost status based on the latest report data.")

    try:
        alt.theme.enable("streamlit")
    except Exception:
        pass 

    history_files = get_history_files()
    if not history_files:
        st.warning("No data to display. Please run AI analysis at least once in the 'Report Analysis' tab.")
    else:
        latest_report_file = history_files[0]
        st.subheader(f"Reference Report: `{latest_report_file}`")
        
        report_data = load_report_data(latest_report_file)
        config_entries = get_target_city_entries()
        
        if not report_data or 'cities' not in report_data or not config_entries:
            st.error("Failed to load data.")
        else:
            # 1. Prepare DataFrame (Report + Coordinates)
            df_report = pd.DataFrame(report_data['cities'])
            df_config = pd.DataFrame(config_entries)
            
            df_merged = pd.merge(
                df_report,
                df_config,
                left_on=["city", "country_display"],
                right_on=["city", "country"],
                suffixes=("_report", "_config")
            )
            
            required_map_cols = ['city', 'country', 'lat', 'lon', 'final_allowance']
            
            if not all(col in df_merged.columns for col in ['lat', 'lon']):
                st.warning(
                    "Coordinate (lat/lon) data for the map is missing. 🗺️\n\n"
                    "**Solution:** Go to the '🛠️ System Settings (Admin)' tab and press the [Auto-complete all city coordinates] button."
                )
                map_data = pd.DataFrame(columns=required_map_cols)
            else:
                map_data = df_merged.copy()
                map_data = map_data[required_map_cols]
                map_data.dropna(subset=['lat', 'lon', 'final_allowance'], inplace=True)
                map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
                map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
                map_data.dropna(subset=['lat', 'lon'], inplace=True)

            if map_data.empty:
                st.caption("No data to display on the map. (Check if coordinates were generated.)")
            else:
                # 2. Calculate color (R,G,B) and size based on cost
                min_cost = map_data['final_allowance'].min()
                max_cost = map_data['final_allowance'].max()
                range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

                def get_color_and_size(cost):
                    norm_cost = (cost - min_cost) / range_cost
                    r = int(255 * norm_cost)
                    g = int(255 * (1 - norm_cost))
                    b = 0
                    size = 50000 + (norm_cost * 450000)
                    return [r, g, b, 160], size

                color_size = map_data['final_allowance'].apply(get_color_and_size)
                map_data['color'] = [item[0] for item in color_size]
                map_data['size'] = [item[1] for item in color_size]

                # 3. Create Pydeck chart
                view_state = pdk.ViewState(
                    latitude=map_data['lat'].mean(),
                    longitude=map_data['lon'].mean(),
                    zoom=0.5,
                    pitch=0,
                    bearing=0
                )

                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_radius='size',
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    radius_scale=0.5,
                    get_line_color=[255, 255, 255, 100],
                    get_line_width=10000,
                )

                tooltip = {
                    "html": "<b>{city}, {country}</b><br/>"
                            "Final Allowance: <b>${final_allowance}</b>",
                    "style": { "color": "white", "backgroundColor": "#1e3c72" }
                }
                
                r = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip
                )

                map_col, legend_col = st.columns([4, 1])

                with map_col:
                    st.pydeck_chart(r, use_container_width=True)

                with legend_col:
                    st.write("##### Legend (Cost)")
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
                        <span style="margin-left: 10px;">High Cost (~${max_cost:,.0f})</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(127, 127, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
                        <span style="margin-left: 10px;">Medium Cost</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
                        <span style="margin-left: 10px;">Low Cost (~${min_cost:,.0f})</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("The larger the circle and the redder the color, the higher the cost of the city.")

            # 4. (Apply Idea 1) Top 10 Charts
            st.divider()
            col1, col2 = st.columns(2)
            
            if 'final_allowance' in df_merged.columns:
                with col1:
                    st.write("##### 💰 Top 10 High Cost Cities (AI Final)")
                    top_10_cost_df = df_merged.nlargest(10, 'final_allowance')[['city', 'final_allowance']].reset_index(drop=True)
                    
                    average_cost = df_merged['final_allowance'].mean()
                    
                    # --- [v19.1 Hotfix] Add 'average' column for tooltip ---
                    top_10_cost_df['average'] = average_cost
                    
                    base_cost = alt.Chart(top_10_cost_df).encode(
                        x=alt.X('city', sort=None, title="City", axis=alt.Axis(labelAngle=-45)), 
                        y=alt.Y('final_allowance', title="Final Allowance ($)", axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('city', title="City"),
                            alt.Tooltip('final_allowance', title="Final Allowance ($)", format='$,.0f'),
                            alt.Tooltip('average', title="Overall Average", format='$,.0f') # <-- Modified
                        ]
                    )
                    
                    bars_cost = base_cost.mark_bar(color="#0D6EFD").encode()
                    
                    rule_cost = alt.Chart(pd.DataFrame({'average_cost': [average_cost]})).mark_rule(
                        color='gray', strokeDash=[3, 3] # [v19.1] Change line color
                    ).encode(
                        y=alt.Y('average_cost', title=''),
                        tooltip=[alt.Tooltip('average_cost', title="Overall Average", format='$,.0f')] 
                    )
                    
                    chart_cost = (bars_cost + rule_cost).properties(
                        background='transparent',
                        title=f"Overall Average: ${average_cost:,.0f}" 
                    ).interactive()
                    st.altair_chart(chart_cost, use_container_width=True)
                
                with col2:
                    st.write("##### ⚠️ Top 10 High Volatility Cities (AI Confidence)")
                    df_report_vc = pd.DataFrame(report_data['cities'])
                    df_report_vc['vc'] = df_report_vc['ai_summary'].apply(lambda x: x.get('ai_consistency_vc') if isinstance(x, dict) else None)
                    df_report_vc.dropna(subset=['vc'], inplace=True)
                    
                    if df_report_vc.empty:
                        st.info("Volatility (VC) data is missing. (AI analysis with the latest version is required)")
                    else:
                        top_10_vc_df = df_report_vc.nlargest(10, 'vc')[['city', 'vc']].reset_index(drop=True)
                        
                        average_vc = df_report_vc['vc'].mean()

                        # --- [v19.1 Hotfix] Add 'average' column for tooltip ---
                        top_10_vc_df['average'] = average_vc
                        
                        base_vc = alt.Chart(top_10_vc_df).encode(
                            x=alt.X('city', sort=None, title="City", axis=alt.Axis(labelAngle=-45)), 
                            y=alt.Y('vc', title="Variation Coefficient (VC)", axis=alt.Axis(format='%')),
                            tooltip=[
                                alt.Tooltip('city', title="City"),
                                alt.Tooltip('vc', title="Variation Coefficient (VC)", format='.2%'),
                                alt.Tooltip('average', title="Overall Average", format='.2%') # <-- Modified
                            ]
                        )
                        
                        bars_vc = base_vc.mark_bar(color="#DC3545").encode()
                        
                        rule_vc = alt.Chart(pd.DataFrame({'average_vc': [average_vc]})).mark_rule(
                            color='gray', strokeDash=[3, 3] # [v19.1] Change line color
                        ).encode(
                            y=alt.Y('average_vc', title=''),
                            tooltip=[alt.Tooltip('average_vc', title="Overall Average", format='.2%')]
                        )
                        
                        chart_vc = (bars_vc + rule_vc).properties(
                            background='transparent',
                            title=f"Overall Average: {average_vc:.2%}"
                        ).interactive()
                        st.altair_chart(chart_vc, use_container_width=True)
                        st.caption("The higher the volatility (VC), the less confident the AI is in its price estimation for the city.")
            else:
                st.warning("No 'final_allowance' data to display the chart.")

if employee_tab is not None:
    with employee_tab:
        st.header("Per Diem Inquiry by City")
        history_files = get_history_files()
        if not history_files:
            st.info("Please analyze a PDF in the 'Report Analysis' tab first.")
        else:
            if "selected_report_file" not in st.session_state:
                st.session_state["selected_report_file"] = history_files[0]
            if st.session_state["selected_report_file"] not in history_files:
                st.session_state["selected_report_file"] = history_files[0]
            selected_file = st.session_state["selected_report_file"]
            report_data = load_report_data(selected_file)
            if report_data and 'cities' in report_data and report_data['cities']:
                cities_df = pd.DataFrame(report_data['cities'])
                target_entries = get_target_city_entries()
                countries = sorted({entry['country'] for entry in target_entries})

                
                col_country, col_city = st.columns(2)
                with col_country:
                    selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
                    sel_country = st.selectbox("Country:", selectable_countries, key=f"country_{selected_file}")
                filtered_cities_all = sorted({
                    entry['city'] for entry in target_entries if entry['country'] == sel_country
                })
                with col_city:
                    if filtered_cities_all:
                        sel_city = st.selectbox("City:", filtered_cities_all, key=f"city_{selected_file}")
                    else:
                        sel_city = None
                        st.warning("There are no registered cities for the selected country.")

                col_start, col_end, col_level = st.columns([1, 1, 1])
                with col_start:
                    trip_start = st.date_input(
                        "Trip Start Date",
                        value=datetime.today().date(),
                        key=f"trip_start_{selected_file}",
                    )
                with col_end:
                    trip_end = st.date_input(
                        "Trip End Date",
                        value=datetime.today().date() + timedelta(days=4),
                        key=f"trip_end_{selected_file}",
                    )
                with col_level:
                    sel_level = st.selectbox("Job Level:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

                if isinstance(trip_start, datetime):
                    trip_start = trip_start.date()
                if isinstance(trip_end, datetime):
                    trip_end = trip_end.date()

                trip_valid = trip_end >= trip_start
                if not trip_valid:
                    st.error("The end date must be on or after the start date.")
                    trip_days = 0 # Set to 0
                    trip_term = "Short-term"
                    trip_multiplier = SHORT_TERM_MULTIPLIER
                    trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
                else:
                    trip_days = (trip_end - trip_start).days + 1
                    trip_term, trip_multiplier = classify_trip_duration(trip_days)
                    trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
                    st.caption(f"Auto-classified trip type: {trip_term_label} · {trip_days}-day trip")

                if sel_city:
                    filtered_trip_cities = []
                    for entry in target_entries:
                        if entry['country'] != sel_country or entry['city'] != sel_city:
                            continue
                        if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
                            continue
                        filtered_trip_cities.append(entry['city'])
                    if trip_valid and not filtered_trip_cities:
                        st.warning("No city data available for this period. Adjust the trip type to 'Short-term' or check city settings.")
                        sel_city = None

                if trip_valid and sel_city and sel_level and trip_days is not None:
                    city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
                    final_allowance = city_data.get('final_allowance')
                    st.subheader(f"{sel_country} - {sel_city} Results")
                    if final_allowance:
                        level_ratio = JOB_LEVEL_RATIOS[sel_level]
                        adjusted_daily_allowance = round(final_allowance * trip_multiplier)
                        level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
                        trip_total_allowance = level_daily_allowance * trip_days
                        
                        # [New 2] Employee tab total card
                        render_primary_summary(
                            f"{sel_level.split(' ')[0]}",
                            trip_total_allowance,
                            level_daily_allowance,
                            trip_days,
                            trip_term_label,
                            trip_multiplier
                        )
                    else:
                        st.metric(f"{sel_level.split(' ')[0]} Daily Recommended Per Diem", "No Amount")

                    menu_samples = city_data.get('menu_samples') or []

                    detail_cards_visible = any([
                        employee_sections_visibility["show_un_basis"],
                        employee_sections_visibility["show_ai_estimate"],
                        employee_sections_visibility["show_weighted_result"],
                        employee_sections_visibility["show_ai_market_detail"],
                    ])
                    extra_content_visible = (
                        employee_sections_visibility["show_provenance"]
                        or (employee_sections_visibility["show_menu_samples"] and menu_samples)
                    )

                    if detail_cards_visible or extra_content_visible:
                        st.markdown("---")
                        st.write("**Basis of Calculation (Daily Rate)**")
                        un_data = city_data.get('un', {})
                        ai_summary = city_data.get('ai_summary', {})
                        season_context = city_data.get('season_context', {})

                        ai_avg = ai_summary.get('season_adjusted_mean_rounded')
                        ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
                        ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
                        removed_totals = ai_summary.get('removed_totals') or []
                        season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
                        season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

                        ai_notes_parts = [f"Success {ai_runs}/{ai_attempts} runs"]
                        if removed_totals:
                            ai_notes_parts.append(f"Outliers {removed_totals}")
                        if season_label:
                            ai_notes_parts.append(f"Season {season_label} ×{season_factor}")
                        ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "No AI Data"
                        
                        # [New 1] Reason for applying dynamic weights
                        weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
                        weights_source = weights_info.get("source", "N/A")
                        un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
                        ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
                        weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | Reason: {weights_source}"

                        un_base = None
                        un_display = None
                        if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
                            un_base = un_data['per_diem_excl_lodging']
                            un_display = round(un_base * trip_multiplier)

                        ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
                        weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

                        first_row_keys = []
                        if employee_sections_visibility["show_un_basis"]:
                            first_row_keys.append("un")
                        if employee_sections_visibility["show_ai_estimate"]:
                            first_row_keys.append("ai")
                        if employee_sections_visibility["show_weighted_result"]:
                            first_row_keys.append("weighted")

                        if first_row_keys:
                            first_row_cols = st.columns(len(first_row_keys))
                            for key, col in zip(first_row_keys, first_row_cols):
                                with col:
                                    if key == "un":
                                        un_caption = f"Short-term base $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
                                        if trip_term == "Long-term" and un_base is not None:
                                            un_caption = f"Short-term $ {un_base:,} → Long-term $ {un_display:,}"
                                        render_stat_card("UN-DSA Basis", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
                                    elif key == "ai":
                                        ai_caption_base = f"Short-term base $ {ai_avg:,}" if ai_avg is not None else ""
                                        if trip_term == "Long-term" and ai_avg is not None:
                                            ai_caption_base = f"Short-term $ {ai_avg:,} → Long-term $ {ai_display:,}"
                                        ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
                                        render_stat_card("AI Market Estimate (Seasonal Adj.)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
                                    else: # key == "weighted"
                                        weighted_caption = weight_caption
                                        if trip_term == "Long-term" and final_allowance is not None:
                                            weighted_caption = f"Short-term $ {final_allowance:,} → Long-term $ {weighted_display:,} | {weight_caption}"
                                        render_stat_card("Weighted Average Result", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

                        # [New 2] Detailed cost breakdown (merged with show_ai_market_detail logic)
                        if employee_sections_visibility["show_ai_market_detail"]:
                            st.markdown("<br>", unsafe_allow_html=True) # line break
                            
                            mean_food = ai_summary.get("mean_food", 0)
                            mean_trans = ai_summary.get("mean_transport", 0)
                            mean_misc = ai_summary.get("mean_misc", 0)
                            
                            # Apply long-term/seasonal rates
                            food_display = round(mean_food * season_factor * trip_multiplier)
                            trans_display = round(mean_trans * season_factor * trip_multiplier)
                            misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
                            st.write("###### AI Estimate Details (Daily Rate)")
                            col_f, col_t, col_m = st.columns(3)
                            with col_f:
                                render_stat_card("Est. Food", f"$ {food_display:,}", f"Short-term base: $ {round(mean_food)}", "muted")
                            with col_t:
                                render_stat_card("Est. Transport", f"$ {trans_display:,}", f"Short-term base: $ {round(mean_trans)}", "muted")
                            with col_m:
                                render_stat_card("Est. Misc", f"$ {misc_display:,}", f"Short-term base: $ {round(mean_misc)}", "muted")
                        
                        # [Improvement 3] The show_weighted_result card is redundant, so the block below is removed
                        # (Original second_row_keys logic removed)

                        if employee_sections_visibility["show_provenance"]:
                            with st.expander("AI provenance & prompts"):
                                provenance_payload = {
                                    "season_context": season_context,
                                    "ai_summary": ai_summary,
                                    "ai_runs": city_data.get('ai_provenance', []),
                                    "reference_links": build_reference_link_lines(menu_samples, max_items=8),
                                    "weights": weights_info,
                                }
                                st.json(provenance_payload)

                        if employee_sections_visibility["show_menu_samples"] and menu_samples:
                            with st.expander("Reference menu samples"):
                                link_lines = build_reference_link_lines(menu_samples, max_items=8)
                                if link_lines:
                                    st.markdown("**Direct links**")
                                    for link_line in link_lines:
                                        st.markdown(f"- {link_line}")
                                    st.markdown("---")
                                st.table(pd.DataFrame(menu_samples))
                    else:
                        st.info("The administrator has hidden the detailed calculation basis.")

# --- [Improvement 2] Changed admin_tab -> admin_analysis_tab ---
with admin_analysis_tab:
    
    # [Improvement 2] Load ADMIN_ACCESS_CODE and check .env
    ACCESS_CODE_KEY = "admin_access_code_valid"
    ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # Load from .env

    if not ACCESS_CODE_VALUE:
        st.error("Security Error: 'ADMIN_ACCESS_CODE' is not set in the .env file. Please stop the app and set the .env file.")
        st.stop()
    
    if not st.session_state.get(ACCESS_CODE_KEY, False):
        with st.form("admin_access_form"):
            input_code = st.text_input("Access Code", type="password")
            submitted = st.form_submit_button("Enter")
        if submitted:
            if input_code == ACCESS_CODE_VALUE:
                st.session_state[ACCESS_CODE_KEY] = True
                st.success("Access granted.")
                st.rerun() # [Improvement 3] Rerun on success
            else:
                st.error("The Access Code is incorrect.")
                st.stop() # [Improvement 3] Stop on failure
        else:
            st.stop() # [Improvement 3] Stop before form submission

    # --- [Improvement 3] "Report Version Management" feature (analysis_sub_tab) ---
    st.subheader("Report Version Management")
    history_files = get_history_files()
    if history_files:
        if "selected_report_file" not in st.session_state:
            st.session_state["selected_report_file"] = history_files[0]
        if st.session_state["selected_report_file"] not in history_files:
            st.session_state["selected_report_file"] = history_files[0]
        default_index = history_files.index(st.session_state["selected_report_file"])
        selected_file = st.selectbox("Select the active report version:", history_files, index=default_index, key="admin_report_file_select")
        st.session_state["selected_report_file"] = selected_file
    else:
        st.info("No reports have been generated.")

    # --- [New 4] Past Report Comparison feature (analysis_sub_tab) ---
    st.divider()
    st.subheader("Compare Past Reports")
    if len(history_files) < 2:
        st.info("At least 2 reports are required for comparison.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            file_a = st.selectbox("Base Report (A)", history_files, index=1, key="compare_a")
        with col_b:
            file_b = st.selectbox("Comparison Report (B)", history_files, index=0, key="compare_b")
        
        if st.button("Compare Reports"):
            if file_a == file_b:
                st.warning("You must select two different reports.")
            else:
                with st.spinner("Comparing reports..."):
                    data_a = load_report_data(file_a)
                    data_b = load_report_data(file_b)
                    
                    if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
                        df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
                        df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
                        df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
                        report_a_label = file_a.split('report_')[-1].split('.')[0]
                        report_b_label = file_b.split('report_')[-1].split('.')[0]

                        df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
                        df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
                        df_merged["Change ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
                        # Prevent division by zero
                        df_merged["Change (%)"] = (df_merged["Change ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
                        st.dataframe(df_merged[[
                            "city", "country_display", 
                            f"A ({report_a_label})", 
                            f"B ({report_b_label})", 
                            "Change ($)", "Change (%)"
                        ]].style.format({"Change (%)": "{:,.1f}%", "Change ($)": "{:,.0f}"}), width="stretch")
                    else:
                        st.error("Failed to load report files.")
    
    # --- [Improvement 3] "UN-DSA (PDF) Analysis" feature (analysis_sub_tab) ---
    st.divider()
    st.subheader("UN-DSA (PDF) Analysis & AI Execution")
    st.warning(f"Note that the AI will be called {NUM_AI_CALLS} times, which will consume time and cost. (Improvement 1: Async processing for faster speed)")
    uploaded_file = st.file_uploader("Upload UN-DSA PDF file.", type="pdf")

    # --- [Improvement 1] Async AI analysis execution logic ---
    if uploaded_file and st.button("Run AI Analysis", type="primary"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("Please set OPENAI_API_KEY in the .env file.")
        else:
            st.session_state.latest_analysis_result = None
            
            # --- Define async execution function ---
            async def run_analysis(progress_bar, openai_api_key):
                progress_bar.progress(0, text="Extracting PDF text...")
                full_text = parse_pdf_to_text(uploaded_file)
                
                CHUNK_SIZE = 15000
                text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
                all_tsv_lines = []
                analysis_failed = False
                
                for i, chunk in enumerate(text_chunks):
                    progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV converting... ({i+1}/{len(text_chunks)})")
                    chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
                    if chunk_tsv:
                        lines = chunk_tsv.strip().split('\n')
                        if not all_tsv_lines:
                            all_tsv_lines.extend(lines)
                        else:
                            all_tsv_lines.extend(lines[1:])
                    else:
                        analysis_failed = True
                        break
                
                if analysis_failed:
                    st.error("Failed to convert PDF->TSV.")
                    progress_bar.empty()
                    return

                processed_data = process_tsv_data("\n".join(all_tsv_lines))
                if not processed_data:
                    st.error("Failed to process TSV data.")
                    progress_bar.empty()
                    return

                # Create async OpenAI client
                client = openai.AsyncOpenAI(api_key=openai_api_key)
                
                total_cities = len(processed_data["cities"])
                all_tasks = [] # List to hold all AI call tasks

                # 1. Pre-create all AI call tasks for all cities
                for city_data in processed_data["cities"]:
                    city_name, country_name = city_data["city"], city_data["country_display"]
                    city_context = {
                        "neighborhood": city_data.get("neighborhood"),
                        "hotel_cluster": city_data.get("hotel_cluster"),
                    }
                    season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
                    menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
                    city_data["menu_samples"] = menu_samples
                    city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
                    city_tasks = []
                    for j in range(1, NUM_AI_CALLS + 1):
                        task = get_market_data_from_ai_async(
                            client, city_name, country_name, f"Run {j}",
                            context=city_context, season_context=season_context, menu_samples=menu_samples
                        )
                        city_tasks.append(task)
                    
                    all_tasks.append(city_tasks) # [ [City1-10runs], [City2-10runs], ... ]

                # 2. Execute all tasks asynchronously and collect results
                city_index = 0
                for city_tasks in all_tasks:
                    city_data = processed_data["cities"][city_index]
                    city_name = city_data["city"]
                    progress_text = f"Calculating AI estimates... ({city_index+1}/{total_cities}) {city_name}"
                    progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
                    # Run 10 tasks for this city concurrently
                    try:
                        market_results = await asyncio.gather(*city_tasks)
                    except Exception as e:
                        st.error(f"Async error during {city_name} analysis: {e}")
                        market_results = [] # Handle failure

                    # 3. Process results
                    ai_totals_source: List[int] = []
                    ai_meta_runs: List[Dict[str, Any]] = []
                    
                    # [New 2] Lists for detailed cost breakdown
                    ai_food: List[int] = []
                    ai_transport: List[int] = []
                    ai_misc: List[int] = []

                    for j, market_result in enumerate(market_results, 1):
                        city_data[f"market_data_{j}"] = market_result
                        if market_result.get("status") == 'ok' and market_result.get("total") is not None:
                            ai_totals_source.append(market_result["total"])
                            # [New 2] Add detailed costs
                            ai_food.append(market_result.get("food", 0))
                            ai_transport.append(market_result.get("transport", 0))
                            ai_misc.append(market_result.get("misc", 0))
                        
                        if "meta" in market_result:
                            ai_meta_runs.append(market_result["meta"])
                    
                    city_data["ai_provenance"] = ai_meta_runs

                    # 4. Calculate final allowance
                    final_allowance = None
                    un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
                    un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

                    ai_stats = aggregate_ai_totals(ai_totals_source)
                    season_factor = (season_context or {}).get("factor", 1.0)
                    ai_base_mean = ai_stats.get("mean_raw")
                    ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
                    # [New 1] Calculate dynamic weights
                    admin_weights = get_weight_config() # Load admin settings
                    ai_vc_score = ai_stats.get("variation_coeff")
                    
                    if un_per_diem is not None:
                        weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
                    else:
                        # If no UN data, use AI 100%
                        weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
                    city_data["ai_summary"] = {
                        "raw_totals": ai_totals_source,
                        "used_totals": ai_stats.get("used_values", []),
                        "removed_totals": ai_stats.get("removed_values", []),
                        "mean_base": ai_base_mean,
                        "mean_base_rounded": ai_stats.get("mean"),
                        
                        "ai_consistency_vc": ai_vc_score, # [New 1]
                        
                        "mean_food": mean(ai_food) if ai_food else 0, # [New 2]
                        "mean_transport": mean(ai_transport) if ai_transport else 0, # [New 2]
                        "mean_misc": mean(ai_misc) if ai_misc else 0, # [New 2]

                        "season_factor": season_factor,
                        "season_label": (season_context or {}).get("label"),
                        "season_adjusted_mean_raw": ai_season_adjusted,
                        "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
                        "successful_runs": len(ai_stats.get("used_values", [])),
                        "attempted_runs": NUM_AI_CALLS,
                        "reference_links": city_data.get("reference_links", []),
                        "weighted_average_components": {
                            "un_per_diem": un_per_diem,
                            "ai_season_adjusted": ai_season_adjusted,
                            "weights": weights_cfg, # [New 1] Save dynamic weights
                        },
                    }

                    # [New 1] Calculate final value with dynamic weights
                    if un_per_diem is not None and ai_season_adjusted is not None:
                        weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
                        final_allowance = round(weighted_average)
                    elif un_per_diem is not None:
                        final_allowance = round(un_per_diem)
                    elif ai_season_adjusted is not None:
                        final_allowance = round(ai_season_adjusted)

                    city_data["final_allowance"] = final_allowance

                    if final_allowance and un_per_diem and un_per_diem > 0:
                        city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
                    else:
                        city_data["delta_vs_un_pct"] = "N/A"
                    
                    city_index += 1 # Next city

                save_report_data(processed_data)
                st.session_state.latest_analysis_result = processed_data
                st.success("AI analysis completed.")
                progress_bar.empty()
                st.rerun()
            
            # --- Async execution ---
            with st.spinner("Processing PDF and running AI analysis... (Takes approx. 10-30 seconds)"):
                progress_bar = st.progress(0, text="Starting analysis...")
                asyncio.run(run_analysis(progress_bar, openai_api_key))

    # --- [Improvement 3] "Latest Analysis Summary" feature (analysis_sub_tab) ---
    if st.session_state.latest_analysis_result:
        st.markdown("---")
        st.subheader("Latest Analysis Summary")
        df_data = []
        for city in st.session_state.latest_analysis_result['cities']:
            row = {
                'City': city.get('city', 'N/A'),
                'Country': city.get('country_display', 'N/A'),
                'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
            }
            for j in range(1, NUM_AI_CALLS + 1):
                row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

            # --- [HOTFIX] Prevent ArrowInvalid Error ---
            delta_val = city.get('delta_vs_un_pct')
            if isinstance(delta_val, (int, float)):
                delta_display = f"{delta_val:.0f}%" # Change number to string format like "12%"
            else:
                delta_display = "N/A" # Already "N/A" string
            # --- [HOTFIX] End ---
                
            row.update({
                'Final Allowance': city.get('final_allowance'),
                'Delta (%)': delta_display, # <-- Use modified string value
                'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
                'Notes': city.get('notes', ''),
            })
            df_data.append(row)

        st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- Added use_container_width (change to width='stretch' if needed)
        with st.expander("View generated markdown report"):
            st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
with admin_config_tab:
    # 암호 확인 (필수)
    if not st.session_state.get(ACCESS_CODE_KEY, False):
        st.error("Access Code가 필요합니다. '보고서 분석 (Admin)' 탭에서 먼저 로그인해주세요.")
        st.stop()
        
    # --- [v19.3] 도시 편집/캐시 관리가 공유할 도시 목록을 탭 상단에서 정의 ---
    current_entries = get_target_city_entries()
    options = {
        f"{entry['region']} | {entry['country']} | {entry['city']}": idx
        for idx, entry in enumerate(current_entries)
    }
    sorted_labels = list(options.keys())
    
    # --- 콜백 함수 1: '도시 편집' 폼 동기화 ---
    def _sync_edit_form_from_selection():
        if "edit_city_selector" not in st.session_state or not st.session_state.edit_city_selector:
             # st.session_state.edit_city_selector가 비어있거나 None일 때
             if sorted_labels:
                 st.session_state.edit_city_selector = sorted_labels[0]
             else:
                 return # 도시가 하나도 없으면 중단
             
        selected_idx = options[st.session_state.edit_city_selector]
        selected_entry = current_entries[selected_idx]
        
        st.session_state.edit_region = selected_entry.get("region", "")
        st.session_state.edit_city = selected_entry.get("city", "")
        st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
        st.session_state.edit_country = selected_entry.get("country", "")
        st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
        
        existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
        st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
        
        sub_data = selected_entry.get("un_dsa_substitute") or {}
        st.session_state.edit_sub_city = sub_data.get("city", "")
        st.session_state.edit_sub_country = sub_data.get("country", "")

    # --- [v19.3] 콜백 함수 2: '캐시 추가' 폼 동기화 ---
    def _sync_cache_form_from_selection():
        selected_label = st.session_state.get("cache_city_selector") # get()으로 오류 방지
        
        if selected_label in options: # 'options' dict를 공유
            selected_idx = options[selected_label]
            selected_entry = current_entries[selected_idx]
            st.session_state.new_cache_country = selected_entry.get("country", "")
            st.session_state.new_cache_city = selected_entry.get("city", "")
            st.session_state.new_cache_neighborhood = selected_entry.get("neighborhood", "")
        else: # (placeholder 선택 시)
            st.session_state.new_cache_country = ""
            st.session_state.new_cache_city = ""
            st.session_state.new_cache_neighborhood = ""
        
        # 나머지 필드는 항상 기본값으로 초기화
        st.session_state.new_cache_vendor = ""
        st.session_state.new_cache_category = "Food"
        st.session_state.new_cache_price = 0.0
        st.session_state.new_cache_currency = "USD"
        st.session_state.new_cache_url = ""

    # --- [v19.3 핫픽스] 콜백 함수 3: '캐시 저장' 로직 ---
    def handle_cache_submit():
        # 1. 유효성 검사
        if (not st.session_state.new_cache_country or 
            not st.session_state.new_cache_city or 
            not st.session_state.new_cache_vendor):
            st.error("국가, 도시, 장소/상품명은 필수입니다.")
            return # 여기서 중단 (폼 값 유지됨)

        # 2. 새 항목 생성
        new_entry = {
            "country": st.session_state.new_cache_country.strip(),
            "city": st.session_state.new_cache_city.strip(),
            "neighborhood": st.session_state.new_cache_neighborhood.strip(),
            "vendor": st.session_state.new_cache_vendor.strip(),
            "category": st.session_state.new_cache_category,
            "price": st.session_state.new_cache_price,
            "currency": st.session_state.new_cache_currency.strip().upper(),
            "url": st.session_state.new_cache_url.strip(),
        }
        
        # 3. 파일에 저장
        if add_menu_cache_entry(new_entry):
            st.success(f"'{new_entry['vendor']}' 항목을 캐시에 추가했습니다.")
            
            # 4. (중요) 폼 리셋: session_state 값들을 수동으로 초기화
            # 이 로직은 on_click 콜백 내부에서 실행되므로 안전합니다.
            st.session_state.new_cache_country = ""
            st.session_state.new_cache_city = ""
            st.session_state.new_cache_neighborhood = ""
            st.session_state.new_cache_vendor = ""
            st.session_state.new_cache_category = "Food"
            st.session_state.new_cache_price = 0.0
            st.session_state.new_cache_currency = "USD"
            st.session_state.new_cache_url = ""
            st.session_state.cache_city_selector = None # 드롭다운도 리셋
            
            # st.rerun()은 on_click 콜백이 끝나면 자동으로 호출되므로 명시적으로 호출할 필요 없음
        else:
            st.error("캐시 항목 추가에 실패했습니다.")
    # --- [v19.3 핫픽스] 끝 ---

    st.subheader("직원용 탭 노출")
    visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key 이름 변경
    if visibility_toggle != stored_employee_tab_visible:
        updated_settings = dict(ui_settings)
        updated_settings["show_employee_tab"] = visibility_toggle
        updated_settings["employee_sections"] = employee_sections_visibility
        save_ui_settings(updated_settings)
        ui_settings = updated_settings
        st.session_state.employee_tab_visibility = visibility_toggle # 세션 상태에도 반영
        st.success("직원용 탭 노출 상태가 업데이트되었습니다. (새로고침 시 적용)")
        time.sleep(1) # 유저가 메시지를 읽을 시간을 줌
        st.rerun()

    st.subheader("직원 화면 노출 설정")
    section_toggle_values: Dict[str, bool] = {}
    for section_key, label in EMPLOYEE_SECTION_LABELS:
        current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
        section_toggle_values[section_key] = st.toggle(
            label,
            value=current_value,
            key=f"employee_section_toggle_{section_key}",
        )
    if section_toggle_values != employee_sections_visibility:
        updated_settings = dict(ui_settings)
        updated_settings["employee_sections"] = section_toggle_values
        save_ui_settings(updated_settings)
        ui_settings["employee_sections"] = section_toggle_values
        st.session_state.employee_sections_visibility = section_toggle_values
        employee_sections_visibility = section_toggle_values
        st.success("직원 화면 노출 설정이 업데이트되었습니다.")
        time.sleep(1)
        st.rerun()

    st.divider()
    st.subheader("비중 설정 (기본값)")
    st.info("이제 이 설정은 '동적 가중치' 로직의 기본값으로 사용됩니다. AI 응답이 불안정하면 자동으로 AI 비중이 낮아집니다.")
    current_weights = get_weight_config()
    st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
    with st.form("weight_config_form"):
        un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
        ai_weight_preview = max(0.0, 1.0 - un_weight_input)
        st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
        st.caption("Weights are normalised to sum to 1.0 when saved.")
        weight_submit = st.form_submit_button("Save weights")
    if weight_submit:
        updated = update_weight_config(un_weight_input, ai_weight_preview)
        st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
        st.rerun()

    st.divider()
    st.header("목표 도시 관리 (target_cities_config.json)")
    entries_df = pd.DataFrame(get_target_city_entries())
    if not entries_df.empty:
        entries_display = entries_df.copy()
        # trip_lengths를 보기 쉽게 문자열로 변환
        entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
        st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], width='stretch') # [v19.3] 경고 수정
    else:
        st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

    # --- [신규 2] 도시 좌표 자동 완성 기능 (새로 추가) ---
    st.divider()
    st.subheader("도시 좌표 관리")
    
    if st.button("모든 도시 좌표(Lat/Lon) 자동 완성", help="target_cities_config.json의 모든 도시를 대상으로 좌표가 없는 도시에 대해 geopy를 호출해 좌표를 자동 저장합니다."):
        
        # 1. 지오코더 초기화
        try:
            geolocator = Nominatim(user_agent=f"aicp_app_{random.randint(1000,9999)}")
        except Exception as e:
            st.error(f"Geopy(Nominatim) 초기화 실패: {e}")
            st.stop()

        # 2. 도시 목록 로드
        current_entries = get_target_city_entries()
        entries_to_update = [e for e in current_entries if not e.get('lat') or not e.get('lon')]
        
        if not entries_to_update:
            st.success("모든 도시에 이미 좌표가 설정되어 있습니다. (업데이트 불필요)")
            st.stop()
            
        st.info(f"총 {len(current_entries)}개 도시 중, 좌표가 없는 {len(entries_to_update)}개 도시에 대한 좌표를 불러옵니다...")
        
        progress_bar = st.progress(0, text="좌표 자동 완성 시작...")
        success_count = 0
        fail_count = 0
        
        with st.spinner("도시 좌표를 하나씩 불러오는 중... (시간이 걸릴 수 있습니다)"):
            for i, entry in enumerate(entries_to_update):
                city = entry['city']
                country = entry['country']
                query = f"{city}, {country}"
                
                try:
                    # 3. API 호출
                    location = geolocator.geocode(query, timeout=5)
                    time.sleep(1) # (중요) Nominatim의 API 제한(초당 1회) 준수
                    
                    if location:
                        # 4. 원본 entry에 lat/lon 추가
                        entry['lat'] = location.latitude
                        entry['lon'] = location.longitude
                        st.toast(f"✅ 성공: {query} ({location.latitude:.4f}, {location.longitude:.4f})", icon="🌍")
                        success_count += 1
                    else:
                        st.toast(f"⚠️ 실패: {query}의 좌표를 찾을 수 없습니다.", icon="❓")
                        fail_count += 1
                        
                except (GeocoderTimedOut, GeocoderUnavailable):
                    st.toast(f"❌ 오류: {query} 요청 시간 초과. 잠시 후 다시 시도하세요.", icon="🔥")
                    fail_count += 1
                except Exception as e:
                    st.toast(f"❌ 오류: {query} ({e})", icon="🔥")
                    fail_count += 1
                
                progress_bar.progress((i + 1) / len(entries_to_update), text=f"처리 중: {query}")

        # 5. 전체 파일 저장
        set_target_city_entries(current_entries) # (save_target_city_entries 호출 포함)
        
        st.success(f"좌표 자동 완성 완료! (성공: {success_count} / 실패: {fail_count})")
        st.rerun()
    # --- [신규 2] 끝 ---


    existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
    st.subheader("신규 도시 추가")
    with st.form("add_target_city_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            region_options = existing_regions + ["기타 (직접 입력)"]
            region_choice = st.selectbox("지역", region_options, key="add_region_choice")
            new_region = ""
            if region_choice == "기타 (직접 입력)":
                new_region = st.text_input("새 지역 이름", key="add_region_text")
        with col_b:
            trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

        col_c, col_d = st.columns(2)
        with col_c:
            city_name = st.text_input("도시", key="add_city")
            neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
        with col_d:
            country_name = st.text_input("국가", key="add_country")
            hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

        with st.expander("UN-DSA 대체 도시 (선택)"):
            substitute_city = st.text_input("대체 도시", key="add_sub_city")
            substitute_country = st.text_input("대체 국가", key="add_sub_country")

        add_submitted = st.form_submit_button("추가")

    if add_submitted:
        region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
        if not region_value or not city_name.strip() or not country_name.strip():
            st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
        else:
            current_entries = get_target_city_entries()
            canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
            duplicate_exists = any(
                (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
                for entry in current_entries
            )
            if duplicate_exists:
                st.warning("동일한 항목이 이미 등록되어 있습니다.")
            else:
                new_entry = {
                    "region": region_value,
                    "country": country_name.strip(),
                    "city": city_name.strip(),
                    "neighborhood": neighborhood.strip(),
                    "hotel_cluster": hotel_cluster.strip(),
                    "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
                }
                if substitute_city.strip() and substitute_country.strip():
                    new_entry["un_dsa_substitute"] = {
                        "city": substitute_city.strip(),
                        "country": substitute_country.strip(),
                    }
                current_entries.append(new_entry)
                set_target_city_entries(current_entries)
                st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
                st.rerun()

    st.subheader("기존 도시 편집/삭제")
    
    if current_entries:
        # 드롭다운(Selectbox)에 on_change 콜백 연결
        selected_label = st.selectbox(
            "편집할 도시를 선택하세요", 
            sorted_labels, 
            key="edit_city_selector",
            on_change=_sync_edit_form_from_selection
        )

        # 페이지 첫 로드 시 폼을 채우기 위한 초기화
        if "edit_region" not in st.session_state:
            _sync_edit_form_from_selection()

        # 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
        with st.form("edit_target_city_form"):
            col_e, col_f = st.columns(2)
            with col_e:
                region_edit = st.text_input("지역", key="edit_region")
                city_edit = st.text_input("도시", key="edit_city")
                neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
            with col_f:
                country_edit = st.text_input("국가", key="edit_country")
                hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

            trip_lengths_edit = st.multiselect(
                "출장 기간",
                TRIP_LENGTH_OPTIONS,
                key="edit_trip_lengths", 
            )

            with st.expander("UN-DSA 대체 도시 (선택)"):
                sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
                sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                update_btn = st.form_submit_button("변경사항 저장")
            with col_btn2:
                delete_btn = st.form_submit_button("삭제", type="secondary")

        # 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
        if update_btn:
            if (not st.session_state.edit_region.strip() or 
                not st.session_state.edit_city.strip() or 
                not st.session_state.edit_country.strip()):
                st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
            else:
                selected_idx = options[st.session_state.edit_city_selector]
                current_entries[selected_idx] = {
                    "region": st.session_state.edit_region.strip(),
                    "country": st.session_state.edit_country.strip(),
                    "city": st.session_state.edit_city.strip(),
                    "neighborhood": st.session_state.edit_neighborhood.strip(),
                    "hotel_cluster": st.session_state.edit_hotel.strip(),
                    "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
                }
                if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
                    current_entries[selected_idx]["un_dsa_substitute"] = {
                        "city": st.session_state.edit_sub_city.strip(),
                        "country": st.session_state.edit_sub_country.strip(),
                    }
                else:
                    current_entries[selected_idx].pop("un_dsa_substitute", None)

                set_target_city_entries(current_entries)
                st.success("수정을 완료했습니다.")
                st.rerun()
        
        if delete_btn:
            selected_idx = options[st.session_state.edit_city_selector]
            del current_entries[selected_idx]
            set_target_city_entries(current_entries)
            st.warning("선택한 항목을 삭제했습니다.")
            st.rerun()
    else:
        st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

    # --- [신규 3] '데이터 캐시 관리' UI 추가 ---
    st.divider()
    st.header("데이터 캐시 관리 (Menu Cache)")

    if not MENU_CACHE_ENABLED:
        st.error("`data_sources/menu_cache.py` 파일 로드에 실패하여 이 기능을 사용할 수 없습니다.")
    else:
        st.info("AI가 도시 물가 추정 시 참고할 실제 메뉴/가격 데이터를 관리합니다. (AI 분석 정확도 향상)")

        # 1. 새 캐시 항목 추가 폼
        st.subheader("신규 캐시 항목 추가")
        
        st.selectbox(
            "도시 선택 (자동 채우기):", 
            sorted_labels,  # 탭 상단에서 정의한 변수
            key="cache_city_selector",
            on_change=_sync_cache_form_from_selection, # 새로 만든 콜백
            index=None,
            placeholder="도시를 선택하면 국가, 도시, 세부 지역이 자동 입력됩니다."
        )

        # 페이지 첫 로드 시 캐시 폼 초기화
        if "new_cache_country" not in st.session_state:
            _sync_cache_form_from_selection() # 빈 값으로 초기화
        
        # --- [v19.3 핫픽스] clear_on_submit=False, on_click 콜백 사용 ---
        with st.form("add_menu_cache_form"): # clear_on_submit 제거
            st.write("AI 분석에 사용할 참고 가격 정보를 입력합니다. (예: 레스토랑 메뉴, 택시비 고지 등)")
            c1, c2 = st.columns(2)
            with c1:
                new_cache_country = st.text_input("국가 (Country)", key="new_cache_country", help="예: Philippines")
                new_cache_city = st.text_input("도시 (City)", key="new_cache_city", help="예: Manila")
                new_cache_neighborhood = st.text_input("세부 지역 (Neighborhood) (선택)", key="new_cache_neighborhood", help="예: Makati (비워두면 도시 전체에 적용)")
                new_cache_vendor = st.text_input("장소/상품명 (Vendor)", key="new_cache_vendor", help="예: Jollibee (C3, Ayala Ave)")
            with c2:
                new_cache_category = st.selectbox("카테고리 (Category)", ["Food", "Transport", "Misc"], key="new_cache_category")
                new_cache_price = st.number_input("가격 (Price)", min_value=0.0, step=0.01, key="new_cache_price")
                new_cache_currency = st.text_input("통화 (Currency)", value="USD", key="new_cache_currency", help="예: PHP, USD")
                new_cache_url = st.text_input("출처 URL (Source URL) (선택)", key="new_cache_url")
            
            # [v19.3] on_click 콜백으로 저장/초기화 로직 실행
            add_cache_submitted = st.form_submit_button(
                "신규 캐시 항목 저장",
                on_click=handle_cache_submit # <-- 핵심 수정
            )
        # --- [v19.3 핫픽스] 끝 ---

        # 2. 기존 캐시 항목 조회 및 삭제
        st.subheader("기존 캐시 항목 조회 및 삭제")
        all_cache_data = load_all_cache() # menu_cache.py의 함수
        
        if not all_cache_data:
            st.info("현재 저장된 캐시 데이터가 없습니다.")
        else:
            df_cache = pd.DataFrame(all_cache_data)
            # [v19.3] 경고 수정
            st.dataframe(df_cache[[
                "country", "city", "neighborhood", "vendor", 
                "category", "price", "currency", "last_updated", "url"
            ]], width='stretch')

            # 삭제 기능
            st.markdown("---")
            st.write("##### 캐시 항목 삭제")
            
            delete_options_map = {
                f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
                for idx, entry in enumerate(reversed(all_cache_data))
            }
            delete_labels = list(delete_options_map.keys())
            
            label_to_delete = st.selectbox("삭제할 캐시 항목을 선택하세요:", delete_labels, index=None, placeholder="삭제할 항목 선택...")
            
            if label_to_delete and st.button(f"'{label_to_delete}' 항목 삭제", type="primary"):
                original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
                entry_to_delete = all_cache_data.pop(original_list_index)
                
                if save_cached_menu_prices(all_cache_data):
                    st.success(f"'{entry_to_delete.get('vendor')}' 항목을 삭제했습니다.")
                    st.rerun()
                else:
                    st.error("캐시 삭제에 실패했습니다.")
    
    # --- [신규 3] UI 끝 ---
    


# # 2025-11-13 스트림릿 정식 배포 전
# # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# # 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
# from typing import Any, Dict, List, Optional, Set, Tuple

# import altair as alt  # [신규 2] 고급 차트 라이브러리
# import pydeck as pdk  # [신규 2] 고급 3D 지도 라이브러리

# # [신규 3] menu_cache 임포트 (파일이 없으면 이 기능은 작동하지 않음)
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
# try:
#     from data_sources.menu_cache import (
#         load_cached_menu_prices, 
#         load_all_cache, 
#         add_menu_cache_entry, 
#         save_cached_menu_prices
#     )
#     MENU_CACHE_ENABLED = True
# except ImportError:
#     st.warning("`data_sources/menu_cache.py` 파일을 찾을 수 없습니다. '데이터 캐시 관리' 기능이 비활성화됩니다.")
#     # (기존 함수들을 임시로 정의)
#     def load_cached_menu_prices(city: str, country: str, neighborhood: Optional[str]) -> List[Dict[str, Any]]: return []
#     def load_all_cache() -> List[Dict[str, Any]]: return []
#     def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool: return False
#     def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool: return False
#     MENU_CACHE_ENABLED = False


# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
#         "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
#     },
#     "secondary": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
#     "muted": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
#     # [수정] 캡션에 스타일 적용
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">Estimated Total Per Diem ({level_label})</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>Calculation</span><br/>
#             $ {daily:,} × {days} days × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term", "Long-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "Short-term", "Long-term": "Long-term"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry.setdefault("trip_lengths", DEFAULT_TRIP_LENGTH.copy())
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# # --- [신규 1] aggregate_ai_totals 함수 수정 ---
# # (이상치 제거 + 변동계수(VC) 계산)
# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
    
#     # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
#     variation_coeff = None
#     if filtered and computed_mean and computed_mean > 0:
#         if len(filtered) > 1:
#             try:
#                 computed_stdev = stdev(filtered)
#                 variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
#             except StatisticsError:
#                 variation_coeff = 0.0 # 모든 값이 동일
#         else:
#             variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#         "variation_coeff": variation_coeff # <-- AI 일관성 점수
#     }

# # --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
# def get_dynamic_weights(
#     variation_coeff: Optional[float], 
#     admin_weights: Dict[str, float]
# ) -> Dict[str, Any]:
#     """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
#     # 관리자 설정값을 기본값으로 사용
#     base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
#     if variation_coeff is None:
#         # AI 데이터가 없으면 UN 100%
#         return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
#     if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
#         # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
#         dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
#         source = f"High AI Consistency (VC: {variation_coeff:.2f})"
#     elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
#         # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
#         dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
#         source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
#     else:
#         # 5% ~ 15% 사이: 관리자 설정값 유지
#         dynamic_ai_weight = base_ai_weight
#         source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

#     final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
#     final_un_weight = 1.0 - final_ai_weight
    
#     return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# # --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
# async def get_market_data_from_ai_async(
#     client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
#     city: str,
#     country: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     # --- (내부 헬퍼 함수들은 기존과 동일) ---
#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")
#     # --- (프롬프트 구성은 기존과 동일) ---
#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         # --- [수정] 비동기 API 호출로 변경 ---
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         # --- [수정] 끝 ---
        
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }
# # --- [개선 1] 끝 ---

# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
#             # [신규 1] 동적 가중치 적용 사유
#             weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#                 'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- Streamlit UI Configuration ---
# st.set_page_config(layout="wide")
# st.title("AICP: NSUS GROUP Per Diem Calculation & Inquiry System")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility


# # --- [Improvement 3 & New 2] Tab structure change (v18.0) ---
# tab_definitions = [
#     "📊 Executive Dashboard", # [New 2] Dashboard tab added
# ]

# if employee_tab_visible:
#     tab_definitions.append("💵 Per Diem Inquiry (Employee)")

# # Split admin tab into two
# tab_definitions.append("📈 Report Analysis (Admin)")
# tab_definitions.append("🛠️ System Settings (Admin)")

# tabs = st.tabs(tab_definitions)

# # Assign tab variables
# dashboard_tab = tabs[0]
# tab_index_offset = 1

# if employee_tab_visible:
#     employee_tab = tabs[tab_index_offset]
#     admin_analysis_tab = tabs[tab_index_offset + 1]
#     admin_config_tab = tabs[tab_index_offset + 2]
#     tab_index_offset += 1
# else:
#     employee_tab = None
#     admin_analysis_tab = tabs[tab_index_offset]
#     admin_config_tab = tabs[tab_index_offset + 1]
# # --- [End of modification] ---


# # --- [New 2] "Executive Dashboard" Tab (Newly Added) ---
# # with dashboard_tab:
# #     st.header("Global Cost Dashboard")
# #     st.info("Visualizes the global business trip cost status based on the latest report data.")

# #     # --- [v18.5 Fix] Set Altair theme to auto-detect Streamlit theme ---
# #     try:
# #         alt.theme.enable("streamlit")
# #     except Exception:
# #         # (If library conflict, apply manual theme as before - omitted here)
# #         pass 

# #     history_files = get_history_files()
# #     if not history_files:
# #         st.warning("No data to display. Please run AI analysis at least once in the 'Report Analysis' tab.")
# #     else:
# #         latest_report_file = history_files[0]
# #         st.subheader(f"Reference Report: `{latest_report_file}`")
        
# #         report_data = load_report_data(latest_report_file)
# #         config_entries = get_target_city_entries()
        
# #         if not report_data or 'cities' not in report_data or not config_entries:
# #             st.error("Failed to load data.")
# #         else:
# #             # 1. Prepare DataFrame (Report + Coordinates)
# #             df_report = pd.DataFrame(report_data['cities'])
# #             df_config = pd.DataFrame(config_entries)
            
# #             df_merged = pd.merge(
# #                 df_report,
# #                 df_config,
# #                 left_on=["city", "country_display"],
# #                 right_on=["city", "country"],
# #                 suffixes=("_report", "_config")
# #             )
            
# #             required_map_cols = ['city', 'country', 'lat', 'lon', 'final_allowance']
            
# #             if not all(col in df_merged.columns for col in ['lat', 'lon']):
# #                 st.warning(
# #                     "Coordinate (lat/lon) data for the map is missing. 🗺️\n\n"
# #                     "**Solution:** Go to the '🛠️ System Settings (Admin)' tab and press the [Auto-complete all city coordinates] button."
# #                 )
# #                 map_data = pd.DataFrame(columns=required_map_cols)
# #             else:
# #                 map_data = df_merged.copy()
# #                 map_data = map_data[required_map_cols]
# #                 map_data.dropna(subset=['lat', 'lon', 'final_allowance'], inplace=True)
# #                 map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
# #                 map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
# #                 map_data.dropna(subset=['lat', 'lon'], inplace=True)

# #             if map_data.empty:
# #                 st.caption("No data to display on the map. (Check if coordinates were generated.)")
# #             else:
# #                 # 2. Calculate color (R,G,B) and size based on cost
# #                 min_cost = map_data['final_allowance'].min()
# #                 max_cost = map_data['final_allowance'].max()
# #                 range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

# #                 def get_color_and_size(cost):
# #                     norm_cost = (cost - min_cost) / range_cost
# #                     r = int(255 * norm_cost)
# #                     g = int(255 * (1 - norm_cost))
# #                     b = 0
# #                     size = 50000 + (norm_cost * 450000)
# #                     return [r, g, b, 160], size

# #                 color_size = map_data['final_allowance'].apply(get_color_and_size)
# #                 map_data['color'] = [item[0] for item in color_size]
# #                 map_data['size'] = [item[1] for item in color_size]

# #                 # 3. Create Pydeck chart
# #                 view_state = pdk.ViewState(
# #                     latitude=map_data['lat'].mean(),
# #                     longitude=map_data['lon'].mean(),
# #                     zoom=0.5,
# #                     pitch=0,
# #                     bearing=0
# #                 )

# #                 layer = pdk.Layer(
# #                     'ScatterplotLayer',
# #                     data=map_data,
# #                     get_position='[lon, lat]',
# #                     get_color='color',
# #                     get_radius='size',
# #                     pickable=True,
# #                     opacity=0.8,
# #                     stroked=True,
# #                     filled=True,
# #                     radius_scale=0.5,
# #                     get_line_color=[255, 255, 255, 100],
# #                     get_line_width=10000,
# #                 )

# #                 tooltip = {
# #                     "html": "<b>{city}, {country}</b><br/>"
# #                             "Final Allowance: <b>${final_allowance}</b>",
# #                     "style": { "color": "white", "backgroundColor": "#1e3c72" }
# #                 }
                
# #                 r = pdk.Deck(
# #                     layers=[layer],
# #                     initial_view_state=view_state,
# #                     # --- [v18.5 Fix] Removed map_style to use Streamlit default (auto-detect theme) ---
# #                     tooltip=tooltip
# #                 )

# #                 map_col, legend_col = st.columns([4, 1])

# #                 with map_col:
# #                     st.pydeck_chart(r, use_container_width=True)

# #                 with legend_col:
# #                     st.write("##### Legend (Cost)")
# #                     st.markdown(f"""
# #                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">High Cost (~${max_cost:,.0f})</span>
# #                     </div>
# #                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(127, 127, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">Medium Cost</span>
# #                     </div>
# #                     <div style="display: flex; align-items: center;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">Low Cost (~${min_cost:,.0f})</span>
# #                     </div>
# #                     """, unsafe_allow_html=True)
# #                     st.caption("The larger the circle and the redder the color, the higher the cost of the city.")

# #                 # 4. (Apply Idea 1) Top 10 Charts
# #                 st.divider()
# #                 col1, col2 = st.columns(2)
                
# #                 if 'final_allowance' in df_merged.columns:
# #                     with col1:
# #                         st.write("##### 💰 Top 10 High Cost Cities (AI Final)")
# #                         top_10_cost_df = df_merged.nlargest(10, 'final_allowance')[['city', 'final_allowance']].reset_index(drop=True)
                        
# #                         # --- [v18.5 Fix] Altair Chart (Theme auto-manages text color) ---
# #                         chart_cost = alt.Chart(top_10_cost_df).mark_bar(
# #                             color="#0D6EFD"
# #                         ).encode(
# #                             x=alt.X('city', sort=None, title="City"),
# #                             y=alt.Y('final_allowance', title="Final Allowance ($)"),
# #                             tooltip=[
# #                                 alt.Tooltip('city', title="City"),
# #                                 alt.Tooltip('final_allowance', title="Final Allowance ($)", format='$,.0f')
# #                             ]
# #                         ).properties(
# #                             background='transparent' # Transparent background
# #                         ).interactive()
# #                         st.altair_chart(chart_cost, use_container_width=True)
                
# #                     with col2:
# #                         st.write("##### ⚠️ Top 10 High Volatility Cities (AI Confidence)")
# #                         df_report_vc = pd.DataFrame(report_data['cities'])
# #                         df_report_vc['vc'] = df_report_vc['ai_summary'].apply(lambda x: x.get('ai_consistency_vc') if isinstance(x, dict) else None)
# #                         df_report_vc.dropna(subset=['vc'], inplace=True)
                        
# #                         if df_report_vc.empty:
# #                             st.info("Volatility (VC) data is missing. (AI analysis with the latest version is required)")
# #                         else:
# #                             top_10_vc_df = df_report_vc.nlargest(10, 'vc')[['city', 'vc']].reset_index(drop=True)
                            
# #                             # --- [v18.5 Fix] Altair Chart (Theme auto-manages text color) ---
# #                             chart_vc = alt.Chart(top_10_vc_df).mark_bar(
# #                                 color="#DC3545"
# #                             ).encode(
# #                                 x=alt.X('city', sort=None, title="City"),
# #                                 y=alt.Y('vc', title="Variation Coefficient (VC)", axis=alt.Axis(format='%')),
# #                                 tooltip=[
# #                                     alt.Tooltip('city', title="City"),
# #                                     alt.Tooltip('vc', title="Variation Coefficient (VC)", format='.2%')
# #                                 ]
# #                             ).properties(
# #                                 background='transparent' # Transparent background
# #                             ).interactive()
# #                             st.altair_chart(chart_vc, use_container_width=True)
# #                             st.caption("The higher the volatility (VC), the less confident the AI is in its price estimation for the city.")
# #                 else:
# #                     st.warning("No 'final_allowance' data to display the chart.")
# with dashboard_tab:
#     st.header("Global Cost Dashboard")
#     st.info("Visualizes the global business trip cost status based on the latest report data.")

#     try:
#         alt.theme.enable("streamlit")
#     except Exception:
#         pass 

#     history_files = get_history_files()
#     if not history_files:
#         st.warning("No data to display. Please run AI analysis at least once in the 'Report Analysis' tab.")
#     else:
#         latest_report_file = history_files[0]
#         st.subheader(f"Reference Report: `{latest_report_file}`")
        
#         report_data = load_report_data(latest_report_file)
#         config_entries = get_target_city_entries()
        
#         if not report_data or 'cities' not in report_data or not config_entries:
#             st.error("Failed to load data.")
#         else:
#             # 1. Prepare DataFrame (Report + Coordinates)
#             df_report = pd.DataFrame(report_data['cities'])
#             df_config = pd.DataFrame(config_entries)
            
#             df_merged = pd.merge(
#                 df_report,
#                 df_config,
#                 left_on=["city", "country_display"],
#                 right_on=["city", "country"],
#                 suffixes=("_report", "_config")
#             )
            
#             required_map_cols = ['city', 'country', 'lat', 'lon', 'final_allowance']
            
#             if not all(col in df_merged.columns for col in ['lat', 'lon']):
#                 st.warning(
#                     "Coordinate (lat/lon) data for the map is missing. 🗺️\n\n"
#                     "**Solution:** Go to the '🛠️ System Settings (Admin)' tab and press the [Auto-complete all city coordinates] button."
#                 )
#                 map_data = pd.DataFrame(columns=required_map_cols)
#             else:
#                 map_data = df_merged.copy()
#                 map_data = map_data[required_map_cols]
#                 map_data.dropna(subset=['lat', 'lon', 'final_allowance'], inplace=True)
#                 map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
#                 map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
#                 map_data.dropna(subset=['lat', 'lon'], inplace=True)

#             if map_data.empty:
#                 st.caption("No data to display on the map. (Check if coordinates were generated.)")
#             else:
#                 # 2. Calculate color (R,G,B) and size based on cost
#                 min_cost = map_data['final_allowance'].min()
#                 max_cost = map_data['final_allowance'].max()
#                 range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

#                 def get_color_and_size(cost):
#                     norm_cost = (cost - min_cost) / range_cost
#                     r = int(255 * norm_cost)
#                     g = int(255 * (1 - norm_cost))
#                     b = 0
#                     size = 50000 + (norm_cost * 450000)
#                     return [r, g, b, 160], size

#                 color_size = map_data['final_allowance'].apply(get_color_and_size)
#                 map_data['color'] = [item[0] for item in color_size]
#                 map_data['size'] = [item[1] for item in color_size]

#                 # 3. Create Pydeck chart
#                 view_state = pdk.ViewState(
#                     latitude=map_data['lat'].mean(),
#                     longitude=map_data['lon'].mean(),
#                     zoom=0.5,
#                     pitch=0,
#                     bearing=0
#                 )

#                 layer = pdk.Layer(
#                     'ScatterplotLayer',
#                     data=map_data,
#                     get_position='[lon, lat]',
#                     get_color='color',
#                     get_radius='size',
#                     pickable=True,
#                     opacity=0.8,
#                     stroked=True,
#                     filled=True,
#                     radius_scale=0.5,
#                     get_line_color=[255, 255, 255, 100],
#                     get_line_width=10000,
#                 )

#                 tooltip = {
#                     "html": "<b>{city}, {country}</b><br/>"
#                             "Final Allowance: <b>${final_allowance}</b>",
#                     "style": { "color": "white", "backgroundColor": "#1e3c72" }
#                 }
                
#                 r = pdk.Deck(
#                     layers=[layer],
#                     initial_view_state=view_state,
#                     tooltip=tooltip
#                 )

#                 map_col, legend_col = st.columns([4, 1])

#                 with map_col:
#                     st.pydeck_chart(r, use_container_width=True)

#                 with legend_col:
#                     st.write("##### Legend (Cost)")
#                     st.markdown(f"""
#                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">High Cost (~${max_cost:,.0f})</span>
#                     </div>
#                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(127, 127, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">Medium Cost</span>
#                     </div>
#                     <div style="display: flex; align-items: center;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">Low Cost (~${min_cost:,.0f})</span>
#                     </div>
#                     """, unsafe_allow_html=True)
#                     st.caption("The larger the circle and the redder the color, the higher the cost of the city.")

#             # 4. (Apply Idea 1) Top 10 Charts
#             st.divider()
#             col1, col2 = st.columns(2)
            
#             if 'final_allowance' in df_merged.columns:
#                 with col1:
#                     st.write("##### 💰 Top 10 High Cost Cities (AI Final)")
#                     top_10_cost_df = df_merged.nlargest(10, 'final_allowance')[['city', 'final_allowance']].reset_index(drop=True)
                    
#                     average_cost = df_merged['final_allowance'].mean()
                    
#                     # --- [v19.1 Hotfix] Add 'average' column for tooltip ---
#                     top_10_cost_df['average'] = average_cost
                    
#                     base_cost = alt.Chart(top_10_cost_df).encode(
#                         x=alt.X('city', sort=None, title="City", axis=alt.Axis(labelAngle=-45)), 
#                         y=alt.Y('final_allowance', title="Final Allowance ($)", axis=alt.Axis(format='$,.0f')),
#                         tooltip=[
#                             alt.Tooltip('city', title="City"),
#                             alt.Tooltip('final_allowance', title="Final Allowance ($)", format='$,.0f'),
#                             alt.Tooltip('average', title="Overall Average", format='$,.0f') # <-- Modified
#                         ]
#                     )
                    
#                     bars_cost = base_cost.mark_bar(color="#0D6EFD").encode()
                    
#                     rule_cost = alt.Chart(pd.DataFrame({'average_cost': [average_cost]})).mark_rule(
#                         color='gray', strokeDash=[3, 3] # [v19.1] Change line color
#                     ).encode(
#                         y=alt.Y('average_cost', title=''),
#                         tooltip=[alt.Tooltip('average_cost', title="Overall Average", format='$,.0f')] 
#                     )
                    
#                     chart_cost = (bars_cost + rule_cost).properties(
#                         background='transparent',
#                         title=f"Overall Average: ${average_cost:,.0f}" 
#                     ).interactive()
#                     st.altair_chart(chart_cost, use_container_width=True)
                
#                 with col2:
#                     st.write("##### ⚠️ Top 10 High Volatility Cities (AI Confidence)")
#                     df_report_vc = pd.DataFrame(report_data['cities'])
#                     df_report_vc['vc'] = df_report_vc['ai_summary'].apply(lambda x: x.get('ai_consistency_vc') if isinstance(x, dict) else None)
#                     df_report_vc.dropna(subset=['vc'], inplace=True)
                    
#                     if df_report_vc.empty:
#                         st.info("Volatility (VC) data is missing. (AI analysis with the latest version is required)")
#                     else:
#                         top_10_vc_df = df_report_vc.nlargest(10, 'vc')[['city', 'vc']].reset_index(drop=True)
                        
#                         average_vc = df_report_vc['vc'].mean()

#                         # --- [v19.1 Hotfix] Add 'average' column for tooltip ---
#                         top_10_vc_df['average'] = average_vc
                        
#                         base_vc = alt.Chart(top_10_vc_df).encode(
#                             x=alt.X('city', sort=None, title="City", axis=alt.Axis(labelAngle=-45)), 
#                             y=alt.Y('vc', title="Variation Coefficient (VC)", axis=alt.Axis(format='%')),
#                             tooltip=[
#                                 alt.Tooltip('city', title="City"),
#                                 alt.Tooltip('vc', title="Variation Coefficient (VC)", format='.2%'),
#                                 alt.Tooltip('average', title="Overall Average", format='.2%') # <-- Modified
#                             ]
#                         )
                        
#                         bars_vc = base_vc.mark_bar(color="#DC3545").encode()
                        
#                         rule_vc = alt.Chart(pd.DataFrame({'average_vc': [average_vc]})).mark_rule(
#                             color='gray', strokeDash=[3, 3] # [v19.1] Change line color
#                         ).encode(
#                             y=alt.Y('average_vc', title=''),
#                             tooltip=[alt.Tooltip('average_vc', title="Overall Average", format='.2%')]
#                         )
                        
#                         chart_vc = (bars_vc + rule_vc).properties(
#                             background='transparent',
#                             title=f"Overall Average: {average_vc:.2%}"
#                         ).interactive()
#                         st.altair_chart(chart_vc, use_container_width=True)
#                         st.caption("The higher the volatility (VC), the less confident the AI is in its price estimation for the city.")
#             else:
#                 st.warning("No 'final_allowance' data to display the chart.")

# if employee_tab is not None:
#     with employee_tab:
#         st.header("Per Diem Inquiry by City")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("Please analyze a PDF in the 'Report Analysis' tab first.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("Country:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("City:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("There are no registered cities for the selected country.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "Trip Start Date",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "Trip End Date",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("Job Level:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("The end date must be on or after the start date.")
#                     trip_days = 0 # Set to 0
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"Auto-classified trip type: {trip_term_label} · {trip_days}-day trip")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("No city data available for this period. Adjust the trip type to 'Short-term' or check city settings.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level and trip_days is not None:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} Results")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
                        
#                         # [New 2] Employee tab total card
#                         render_primary_summary(
#                             f"{sel_level.split(' ')[0]}",
#                             trip_total_allowance,
#                             level_daily_allowance,
#                             trip_days,
#                             trip_term_label,
#                             trip_multiplier
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} Daily Recommended Per Diem", "No Amount")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**Basis of Calculation (Daily Rate)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"Success {ai_runs}/{ai_attempts} runs"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"Outliers {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"Season {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "No AI Data"
                        
#                         # [New 1] Reason for applying dynamic weights
#                         weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
#                         weights_source = weights_info.get("source", "N/A")
#                         un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
#                         ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
#                         weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | Reason: {weights_source}"

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         un_caption = f"Short-term base $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
#                                         if trip_term == "Long-term" and un_base is not None:
#                                             un_caption = f"Short-term $ {un_base:,} → Long-term $ {un_display:,}"
#                                         render_stat_card("UN-DSA Basis", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
#                                     elif key == "ai":
#                                         ai_caption_base = f"Short-term base $ {ai_avg:,}" if ai_avg is not None else ""
#                                         if trip_term == "Long-term" and ai_avg is not None:
#                                             ai_caption_base = f"Short-term $ {ai_avg:,} → Long-term $ {ai_display:,}"
#                                         ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
#                                         render_stat_card("AI Market Estimate (Seasonal Adj.)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
#                                     else: # key == "weighted"
#                                         weighted_caption = weight_caption
#                                         if trip_term == "Long-term" and final_allowance is not None:
#                                             weighted_caption = f"Short-term $ {final_allowance:,} → Long-term $ {weighted_display:,} | {weight_caption}"
#                                         render_stat_card("Weighted Average Result", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

#                         # [New 2] Detailed cost breakdown (merged with show_ai_market_detail logic)
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             st.markdown("<br>", unsafe_allow_html=True) # line break
                            
#                             mean_food = ai_summary.get("mean_food", 0)
#                             mean_trans = ai_summary.get("mean_transport", 0)
#                             mean_misc = ai_summary.get("mean_misc", 0)
                            
#                             # Apply long-term/seasonal rates
#                             food_display = round(mean_food * season_factor * trip_multiplier)
#                             trans_display = round(mean_trans * season_factor * trip_multiplier)
#                             misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
#                             st.write("###### AI Estimate Details (Daily Rate)")
#                             col_f, col_t, col_m = st.columns(3)
#                             with col_f:
#                                 render_stat_card("Est. Food", f"$ {food_display:,}", f"Short-term base: $ {round(mean_food)}", "muted")
#                             with col_t:
#                                 render_stat_card("Est. Transport", f"$ {trans_display:,}", f"Short-term base: $ {round(mean_trans)}", "muted")
#                             with col_m:
#                                 render_stat_card("Est. Misc", f"$ {misc_display:,}", f"Short-term base: $ {round(mean_misc)}", "muted")
                        
#                         # [Improvement 3] The show_weighted_result card is redundant, so the block below is removed
#                         # (Original second_row_keys logic removed)

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_info,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("The administrator has hidden the detailed calculation basis.")

# # --- [Improvement 2] Changed admin_tab -> admin_analysis_tab ---
# with admin_analysis_tab:
    
#     # [Improvement 2] Load ADMIN_ACCESS_CODE and check .env
#     ACCESS_CODE_KEY = "admin_access_code_valid"
#     ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # Load from .env

#     if not ACCESS_CODE_VALUE:
#         st.error("Security Error: 'ADMIN_ACCESS_CODE' is not set in the .env file. Please stop the app and set the .env file.")
#         st.stop()
    
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#                 st.rerun() # [Improvement 3] Rerun on success
#             else:
#                 st.error("The Access Code is incorrect.")
#                 st.stop() # [Improvement 3] Stop on failure
#         else:
#             st.stop() # [Improvement 3] Stop before form submission

#     # --- [Improvement 3] "Report Version Management" feature (analysis_sub_tab) ---
#     st.subheader("Report Version Management")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("Select the active report version:", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("No reports have been generated.")

#     # --- [New 4] Past Report Comparison feature (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("Compare Past Reports")
#     if len(history_files) < 2:
#         st.info("At least 2 reports are required for comparison.")
#     else:
#         col_a, col_b = st.columns(2)
#         with col_a:
#             file_a = st.selectbox("Base Report (A)", history_files, index=1, key="compare_a")
#         with col_b:
#             file_b = st.selectbox("Comparison Report (B)", history_files, index=0, key="compare_b")
        
#         if st.button("Compare Reports"):
#             if file_a == file_b:
#                 st.warning("You must select two different reports.")
#             else:
#                 with st.spinner("Comparing reports..."):
#                     data_a = load_report_data(file_a)
#                     data_b = load_report_data(file_b)
                    
#                     if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
#                         df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
#                         df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
#                         df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
#                         report_a_label = file_a.split('report_')[-1].split('.')[0]
#                         report_b_label = file_b.split('report_')[-1].split('.')[0]

#                         df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
#                         df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
#                         df_merged["Change ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
#                         # Prevent division by zero
#                         df_merged["Change (%)"] = (df_merged["Change ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
#                         st.dataframe(df_merged[[
#                             "city", "country_display", 
#                             f"A ({report_a_label})", 
#                             f"B ({report_b_label})", 
#                             "Change ($)", "Change (%)"
#                         ]].style.format({"Change (%)": "{:,.1f}%", "Change ($)": "{:,.0f}"}), width="stretch")
#                     else:
#                         st.error("Failed to load report files.")
    
#     # --- [Improvement 3] "UN-DSA (PDF) Analysis" feature (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("UN-DSA (PDF) Analysis & AI Execution")
#     st.warning(f"Note that the AI will be called {NUM_AI_CALLS} times, which will consume time and cost. (Improvement 1: Async processing for faster speed)")
#     uploaded_file = st.file_uploader("Upload UN-DSA PDF file.", type="pdf")

#     # --- [Improvement 1] Async AI analysis execution logic ---
#     if uploaded_file and st.button("Run AI Analysis", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error("Please set OPENAI_API_KEY in the .env file.")
#         else:
#             st.session_state.latest_analysis_result = None
            
#             # --- Define async execution function ---
#             async def run_analysis(progress_bar, openai_api_key):
#                 progress_bar.progress(0, text="Extracting PDF text...")
#                 full_text = parse_pdf_to_text(uploaded_file)
                
#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []
#                 analysis_failed = False
                
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV converting... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break
                
#                 if analysis_failed:
#                     st.error("Failed to convert PDF->TSV.")
#                     progress_bar.empty()
#                     return

#                 processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                 if not processed_data:
#                     st.error("Failed to process TSV data.")
#                     progress_bar.empty()
#                     return

#                 # Create async OpenAI client
#                 client = openai.AsyncOpenAI(api_key=openai_api_key)
                
#                 total_cities = len(processed_data["cities"])
#                 all_tasks = [] # List to hold all AI call tasks

#                 # 1. Pre-create all AI call tasks for all cities
#                 for city_data in processed_data["cities"]:
#                     city_name, country_name = city_data["city"], city_data["country_display"]
#                     city_context = {
#                         "neighborhood": city_data.get("neighborhood"),
#                         "hotel_cluster": city_data.get("hotel_cluster"),
#                     }
#                     season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                     menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
#                     city_data["menu_samples"] = menu_samples
#                     city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
#                     city_tasks = []
#                     for j in range(1, NUM_AI_CALLS + 1):
#                         task = get_market_data_from_ai_async(
#                             client, city_name, country_name, f"Run {j}",
#                             context=city_context, season_context=season_context, menu_samples=menu_samples
#                         )
#                         city_tasks.append(task)
                    
#                     all_tasks.append(city_tasks) # [ [City1-10runs], [City2-10runs], ... ]

#                 # 2. Execute all tasks asynchronously and collect results
#                 city_index = 0
#                 for city_tasks in all_tasks:
#                     city_data = processed_data["cities"][city_index]
#                     city_name = city_data["city"]
#                     progress_text = f"Calculating AI estimates... ({city_index+1}/{total_cities}) {city_name}"
#                     progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
#                     # Run 10 tasks for this city concurrently
#                     try:
#                         market_results = await asyncio.gather(*city_tasks)
#                     except Exception as e:
#                         st.error(f"Async error during {city_name} analysis: {e}")
#                         market_results = [] # Handle failure

#                     # 3. Process results
#                     ai_totals_source: List[int] = []
#                     ai_meta_runs: List[Dict[str, Any]] = []
                    
#                     # [New 2] Lists for detailed cost breakdown
#                     ai_food: List[int] = []
#                     ai_transport: List[int] = []
#                     ai_misc: List[int] = []

#                     for j, market_result in enumerate(market_results, 1):
#                         city_data[f"market_data_{j}"] = market_result
#                         if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                             ai_totals_source.append(market_result["total"])
#                             # [New 2] Add detailed costs
#                             ai_food.append(market_result.get("food", 0))
#                             ai_transport.append(market_result.get("transport", 0))
#                             ai_misc.append(market_result.get("misc", 0))
                        
#                         if "meta" in market_result:
#                             ai_meta_runs.append(market_result["meta"])
                    
#                     city_data["ai_provenance"] = ai_meta_runs

#                     # 4. Calculate final allowance
#                     final_allowance = None
#                     un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
#                     un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

#                     ai_stats = aggregate_ai_totals(ai_totals_source)
#                     season_factor = (season_context or {}).get("factor", 1.0)
#                     ai_base_mean = ai_stats.get("mean_raw")
#                     ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
#                     # [New 1] Calculate dynamic weights
#                     admin_weights = get_weight_config() # Load admin settings
#                     ai_vc_score = ai_stats.get("variation_coeff")
                    
#                     if un_per_diem is not None:
#                         weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
#                     else:
#                         # If no UN data, use AI 100%
#                         weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
#                     city_data["ai_summary"] = {
#                         "raw_totals": ai_totals_source,
#                         "used_totals": ai_stats.get("used_values", []),
#                         "removed_totals": ai_stats.get("removed_values", []),
#                         "mean_base": ai_base_mean,
#                         "mean_base_rounded": ai_stats.get("mean"),
                        
#                         "ai_consistency_vc": ai_vc_score, # [New 1]
                        
#                         "mean_food": mean(ai_food) if ai_food else 0, # [New 2]
#                         "mean_transport": mean(ai_transport) if ai_transport else 0, # [New 2]
#                         "mean_misc": mean(ai_misc) if ai_misc else 0, # [New 2]

#                         "season_factor": season_factor,
#                         "season_label": (season_context or {}).get("label"),
#                         "season_adjusted_mean_raw": ai_season_adjusted,
#                         "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                         "successful_runs": len(ai_stats.get("used_values", [])),
#                         "attempted_runs": NUM_AI_CALLS,
#                         "reference_links": city_data.get("reference_links", []),
#                         "weighted_average_components": {
#                             "un_per_diem": un_per_diem,
#                             "ai_season_adjusted": ai_season_adjusted,
#                             "weights": weights_cfg, # [New 1] Save dynamic weights
#                         },
#                     }

#                     # [New 1] Calculate final value with dynamic weights
#                     if un_per_diem is not None and ai_season_adjusted is not None:
#                         weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                         final_allowance = round(weighted_average)
#                     elif un_per_diem is not None:
#                         final_allowance = round(un_per_diem)
#                     elif ai_season_adjusted is not None:
#                         final_allowance = round(ai_season_adjusted)

#                     city_data["final_allowance"] = final_allowance

#                     if final_allowance and un_per_diem and un_per_diem > 0:
#                         city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                     else:
#                         city_data["delta_vs_un_pct"] = "N/A"
                    
#                     city_index += 1 # Next city

#                 save_report_data(processed_data)
#                 st.session_state.latest_analysis_result = processed_data
#                 st.success("AI analysis completed.")
#                 progress_bar.empty()
#                 st.rerun()
            
#             # --- Async execution ---
#             with st.spinner("Processing PDF and running AI analysis... (Takes approx. 10-30 seconds)"):
#                 progress_bar = st.progress(0, text="Starting analysis...")
#                 asyncio.run(run_analysis(progress_bar, openai_api_key))

#     # --- [Improvement 3] "Latest Analysis Summary" feature (analysis_sub_tab) ---
#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             # --- [HOTFIX] Prevent ArrowInvalid Error ---
#             delta_val = city.get('delta_vs_un_pct')
#             if isinstance(delta_val, (int, float)):
#                 delta_display = f"{delta_val:.0f}%" # Change number to string format like "12%"
#             else:
#                 delta_display = "N/A" # Already "N/A" string
#             # --- [HOTFIX] End ---
                
#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': delta_display, # <-- Use modified string value
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- Added use_container_width (change to width='stretch' if needed)
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# # --- [Improvement 3] "System Settings" Tab (admin_config_tab) ---
# with admin_config_tab:
#     # Check password (required)
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         st.error("Access Code required. Please log in on the 'Report Analysis (Admin)' tab first.")
#         st.stop()
        
#     # --- [v19.3] Define shared city list at the top of the tab for city editing/cache management ---
#     current_entries = get_target_city_entries()
#     options = {
#         f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#         for idx, entry in enumerate(current_entries)
#     }
#     sorted_labels = list(options.keys())
    
#     # --- Callback Function 1: Sync 'Edit City' form ---
#     def _sync_edit_form_from_selection():
#         if "edit_city_selector" not in st.session_state or not st.session_state.edit_city_selector:
#             st.session_state.edit_city_selector = sorted_labels[0]
            
#         selected_idx = options[st.session_state.edit_city_selector]
#         selected_entry = current_entries[selected_idx]
        
#         st.session_state.edit_region = selected_entry.get("region", "")
#         st.session_state.edit_city = selected_entry.get("city", "")
#         st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#         st.session_state.edit_country = selected_entry.get("country", "")
#         st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
        
#         existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#         st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
        
#         sub_data = selected_entry.get("un_dsa_substitute") or {}
#         st.session_state.edit_sub_city = sub_data.get("city", "")
#         st.session_state.edit_sub_country = sub_data.get("country", "")

#     # --- [v19.3] Callback Function 2: Sync 'Add Cache' form ---
#     def _sync_cache_form_from_selection():
#         selected_label = st.session_state.get("cache_city_selector") # Use get() to prevent errors
        
#         if selected_label in options: # Share the 'options' dict
#             selected_idx = options[selected_label]
#             selected_entry = current_entries[selected_idx]
#             st.session_state.new_cache_country = selected_entry.get("country", "")
#             st.session_state.new_cache_city = selected_entry.get("city", "")
#             st.session_state.new_cache_neighborhood = selected_entry.get("neighborhood", "")
#         else: # (If placeholder is selected)
#             st.session_state.new_cache_country = ""
#             st.session_state.new_cache_city = ""
#             st.session_state.new_cache_neighborhood = ""
        
#         # Always initialize remaining fields to default
#         st.session_state.new_cache_vendor = ""
#         st.session_state.new_cache_category = "Food"
#         st.session_state.new_cache_price = 0.0
#         st.session_state.new_cache_currency = "USD"
#         st.session_state.new_cache_url = ""

#     # --- [v19.3 Hotfix] Callback Function 3: 'Save Cache' logic ---
#     def handle_cache_submit():
#         # 1. Validation
#         if (not st.session_state.new_cache_country or 
#             not st.session_state.new_cache_city or 
#             not st.session_state.new_cache_vendor):
#             st.error("Country, City, and Vendor/Item Name are required.")
#             return # Stop here (form values are kept)

#         # 2. Create new entry
#         new_entry = {
#             "country": st.session_state.new_cache_country.strip(),
#             "city": st.session_state.new_cache_city.strip(),
#             "neighborhood": st.session_state.new_cache_neighborhood.strip(),
#             "vendor": st.session_state.new_cache_vendor.strip(),
#             "category": st.session_state.new_cache_category,
#             "price": st.session_state.new_cache_price,
#             "currency": st.session_state.new_cache_currency.strip().upper(),
#             "url": st.session_state.new_cache_url.strip(),
#         }
        
#         # 3. Save to file
#         if add_menu_cache_entry(new_entry):
#             st.success(f"Added '{new_entry['vendor']}' entry to cache.")
            
#             # 4. (Important) Reset form: Manually initialize session_state values
#             st.session_state.new_cache_country = ""
#             st.session_state.new_cache_city = ""
#             st.session_state.new_cache_neighborhood = ""
#             st.session_state.new_cache_vendor = ""
#             st.session_state.new_cache_category = "Food"
#             st.session_state.new_cache_price = 0.0
#             st.session_state.new_cache_currency = "USD"
#             st.session_state.new_cache_url = ""
#             st.session_state.cache_city_selector = None # Reset dropdown as well
            
#             # st.rerun() is called automatically after on_click callback finishes, no need to call explicitly
#         else:
#             st.error("Failed to add cache entry.")
#     # --- [v19.3 Hotfix] End ---

#     st.subheader("Employee Tab Visibility")
#     visibility_toggle = st.toggle("Show Employee Tab", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key name changed
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.session_state.employee_tab_visibility = visibility_toggle # Reflect in session state as well
#         st.success("Employee tab visibility updated. (Applies on refresh)")
#         time.sleep(1) # Give user time to read the message
#         st.rerun()

#     st.subheader("Employee View Section Settings")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("Employee view section settings updated.")
#         time.sleep(1)
#         st.rerun()

#     st.divider()
#     st.subheader("Weight Settings (Default)")
#     st.info("This setting is now used as the default for the 'Dynamic Weight' logic. If AI responses are unstable, the AI weight will be automatically lowered.")
#     current_weights = get_weight_config()
#     st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
#         st.rerun()

#     st.divider()
#     st.header("Target City Management (target_cities_config.json)")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         # Convert trip_lengths to a readable string
#         entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], width='stretch') # [v19.3] Fix warning
#     else:
#         st.info("No target cities registered. Please add a new entry below.")

#     # --- [New 2] Auto-complete City Coordinates feature (Newly Added) ---
#     st.divider()
#     st.subheader("City Coordinate Management")
    
#     if st.button("Auto-complete all city coordinates (Lat/Lon)", help="Calls geopy to auto-save coordinates for all cities in target_cities_config.json that are missing them."):
        
#         # 1. Initialize geocoder
#         try:
#             geolocator = Nominatim(user_agent=f"aicp_app_{random.randint(1000,9999)}")
#         except Exception as e:
#             st.error(f"Geopy(Nominatim) initialization failed: {e}")
#             st.stop()

#         # 2. Load city list
#         current_entries = get_target_city_entries()
#         entries_to_update = [e for e in current_entries if not e.get('lat') or not e.get('lon')]
        
#         if not entries_to_update:
#             st.success("All cities already have coordinates set. (No update needed)")
#             st.stop()
            
#         st.info(f"Out of {len(current_entries)} total cities, fetching coordinates for {len(entries_to_update)} cities that are missing them...")
        
#         progress_bar = st.progress(0, text="Starting coordinate auto-completion...")
#         success_count = 0
#         fail_count = 0
        
#         with st.spinner("Fetching city coordinates one by one... (This may take a while)"):
#             for i, entry in enumerate(entries_to_update):
#                 city = entry['city']
#                 country = entry['country']
#                 query = f"{city}, {country}"
                
#                 try:
#                     # 3. API Call
#                     location = geolocator.geocode(query, timeout=5)
#                     time.sleep(1) # (Important) Adhere to Nominatim's API limit (1 req/sec)
                    
#                     if location:
#                         # 4. Add lat/lon to the original entry
#                         entry['lat'] = location.latitude
#                         entry['lon'] = location.longitude
#                         st.toast(f"✅ Success: {query} ({location.latitude:.4f}, {location.longitude:.4f})", icon="🌍")
#                         success_count += 1
#                     else:
#                         st.toast(f"⚠️ Failed: Could not find coordinates for {query}.", icon="❓")
#                         fail_count += 1
                        
#                 except (GeocoderTimedOut, GeocoderUnavailable):
#                     st.toast(f"❌ Error: Request for {query} timed out. Please try again later.", icon="🔥")
#                     fail_count += 1
#                 except Exception as e:
#                     st.toast(f"❌ Error: {query} ({e})", icon="🔥")
#                     fail_count += 1
                
#                 progress_bar.progress((i + 1) / len(entries_to_update), text=f"Processing: {query}")

#         # 5. Save the entire file
#         set_target_city_entries(current_entries) # (Includes call to save_target_city_entries)
        
#         st.success(f"Coordinate auto-completion finished! (Success: {success_count} / Failed: {fail_count})")
#         st.rerun()
#     # --- [New 2] End ---


#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("Add New City")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["Other (Enter manually)"]
#             region_choice = st.selectbox("Region", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "Other (Enter manually)":
#                 new_region = st.text_input("New Region Name", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("Trip Duration", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("City", key="add_city")
#             neighborhood = st.text_input("Neighborhood (Optional)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("Country", key="add_country")
#             hotel_cluster = st.text_input("Recommended Hotel Cluster (Optional)", key="add_hotel_cluster")

#         with st.expander("UN-DSA Substitute City (Optional)"):
#             substitute_city = st.text_input("Substitute City", key="add_sub_city")
#             substitute_country = st.text_input("Substitute Country", key="add_sub_country")

#         add_submitted = st.form_submit_button("Add")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "Other (Enter manually)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("Region, Country, and City are required fields.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("An identical entry already exists.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"Added {region_value} - {city_name.strip()} entry.")
#                 st.rerun()

#     st.subheader("Edit/Delete Existing City")
    
#     if current_entries:
#         # Connect on_change callback to the Selectbox
#         selected_label = st.selectbox(
#             "Select city to edit", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection
#         )

#         # Initial fill for the form on page load
#         if "edit_region" not in st.session_state:
#             _sync_edit_form_from_selection()

#         # Remove 'value=' from form widgets and use only 'key='
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 region_edit = st.text_input("Region", key="edit_region")
#                 city_edit = st.text_input("City", key="edit_city")
#                 neighborhood_edit = st.text_input("Neighborhood (Optional)", key="edit_neighborhood")
#             with col_f:
#                 country_edit = st.text_input("Country", key="edit_country")
#                 hotel_cluster_edit = st.text_input("Recommended Hotel Cluster (Optional)", key="edit_hotel")

#             trip_lengths_edit = st.multiselect(
#                 "Trip Duration",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", 
#             )

#             with st.expander("UN-DSA Substitute City (Optional)"):
#                 sub_city_edit = st.text_input("Substitute City", key="edit_sub_city")
#                 sub_country_edit = st.text_input("Substitute Country", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("Save Changes")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("Delete", type="secondary")

#         # Modify save/delete logic to read values from session_state
#         if update_btn:
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("Region, Country, and City are required fields.")
#             else:
#                 selected_idx = options[st.session_state.edit_city_selector]
#                 current_entries[selected_idx] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[selected_idx]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[selected_idx].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("Update complete.")
#                 st.rerun()
        
#         if delete_btn:
#             selected_idx = options[st.session_state.edit_city_selector]
#             del current_entries[selected_idx]
#             set_target_city_entries(current_entries)
#             st.warning("Deleted the selected item.")
#             st.rerun()
#     else:
#         st.info("No target cities registered, so there is nothing to edit.")

#     # --- [New 3] Add 'Data Cache Management' UI ---
#     st.divider()
#     st.header("Data Cache Management (Menu Cache)")

#     if not MENU_CACHE_ENABLED:
#         st.error("Failed to load `data_sources/menu_cache.py`. This feature is unavailable.")
#     else:
#         st.info("Manage actual menu/price data for AI to reference when estimating city prices. (Improves AI analysis accuracy)")

#         # 1. Add new cache item form
#         st.subheader("Add New Cache Item")
        
#         st.selectbox(
#             "Select City (Auto-fill):", 
#             sorted_labels, # Variable defined at the top of the tab
#             key="cache_city_selector",
#             on_change=_sync_cache_form_from_selection, # Newly created callback
#             index=None,
#             placeholder="Select a city to auto-fill Country, City, and Neighborhood."
#         )

#         # Initialize cache form on page load
#         if "new_cache_country" not in st.session_state:
#             _sync_cache_form_from_selection() # Initialize with empty values
        
#         # --- [v19.3 Hotfix] Remove clear_on_submit=True, use on_click callback ---
#         with st.form("add_menu_cache_form"):
#             st.write("Enter reference price information for AI analysis. (e.g., restaurant menu, taxi fare notice, etc.)")
#             c1, c2 = st.columns(2)
#             with c1:
#                 new_cache_country = st.text_input("Country", key="new_cache_country", help="e.g.: Philippines")
#                 new_cache_city = st.text_input("City", key="new_cache_city", help="e.g.: Manila")
#                 new_cache_neighborhood = st.text_input("Neighborhood (Optional)", key="new_cache_neighborhood", help="e.g.: Makati (applies to whole city if left blank)")
#                 new_cache_vendor = st.text_input("Vendor/Item Name", key="new_cache_vendor", help="e.g.: Jollibee (C3, Ayala Ave)")
#             with c2:
#                 new_cache_category = st.selectbox("Category", ["Food", "Transport", "Misc"], key="new_cache_category")
#                 new_cache_price = st.number_input("Price", min_value=0.0, step=0.01, key="new_cache_price")
#                 new_cache_currency = st.text_input("Currency", value="USD", key="new_cache_currency", help="e.g.: PHP, USD")
#                 new_cache_url = st.text_input("Source URL (Optional)", key="new_cache_url")
            
#             # [v19.3] Execute save/init logic via on_click callback
#             add_cache_submitted = st.form_submit_button(
#                 "Save New Cache Item",
#                 on_click=handle_cache_submit
#             )
#         # --- [v19.3 Hotfix] End ---

#         # 2. View and delete existing cache items
#         st.subheader("View/Delete Existing Cache Items")
#         all_cache_data = load_all_cache() # Function from menu_cache.py
        
#         if not all_cache_data:
#             st.info("No cached data is currently saved.")
#         else:
#             df_cache = pd.DataFrame(all_cache_data)
#             st.dataframe(df_cache[[
#                 "country", "city", "neighborhood", "vendor", 
#                 "category", "price", "currency", "last_updated", "url"
#             ]], width='stretch') # [v19.3] Fix warning

#             # Delete function
#             st.markdown("---")
#             st.write("##### Delete Cache Item")
            
#             delete_options_map = {
#                 f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
#                 for idx, entry in enumerate(reversed(all_cache_data))
#             }
#             delete_labels = list(delete_options_map.keys())
            
#             label_to_delete = st.selectbox("Select cache item to delete:", delete_labels, index=None, placeholder="Select item to delete...")
            
#             if label_to_delete and st.button(f"Delete '{label_to_delete}' item", type="primary"):
#                 original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
#                 entry_to_delete = all_cache_data.pop(original_list_index)
                
#                 if save_cached_menu_prices(all_cache_data):
#                     st.success(f"Deleted '{entry_to_delete.get('vendor')}' item.")
#                     st.rerun()
#                 else:
#                     st.error("Failed to delete cache item.")
    

# 2025-11-11 전체 영문버전 업데이트 전
# # 2025-10-20-16 AI 기반 출장비 계산 도구 (v16.0 - Async, Dynamic Weights, Full Admin)
# # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# # 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
# from typing import Any, Dict, List, Optional, Set, Tuple

# import altair as alt  # [신규 2] 고급 차트 라이브러리
# import pydeck as pdk  # [신규 2] 고급 3D 지도 라이브러리

# # [신규 3] menu_cache 임포트 (파일이 없으면 이 기능은 작동하지 않음)
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
# try:
#     from data_sources.menu_cache import (
#         load_cached_menu_prices, 
#         load_all_cache, 
#         add_menu_cache_entry, 
#         save_cached_menu_prices
#     )
#     MENU_CACHE_ENABLED = True
# except ImportError:
#     st.warning("`data_sources/menu_cache.py` 파일을 찾을 수 없습니다. '데이터 캐시 관리' 기능이 비활성화됩니다.")
#     # (기존 함수들을 임시로 정의)
#     def load_cached_menu_prices(city: str, country: str, neighborhood: Optional[str]) -> List[Dict[str, Any]]: return []
#     def load_all_cache() -> List[Dict[str, Any]]: return []
#     def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool: return False
#     def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool: return False
#     MENU_CACHE_ENABLED = False


# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
#         "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
#     },
#     "secondary": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
#     "muted": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
#     # [수정] 캡션에 스타일 적용
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">{level_label} 기준 예상 일비 총액</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>계산식</span><br/>
#             $ {daily:,} × {days}일 일정 × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term", "Long-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "숏텀", "Long-term": "롱텀"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry.setdefault("trip_lengths", DEFAULT_TRIP_LENGTH.copy())
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# # --- [신규 1] aggregate_ai_totals 함수 수정 ---
# # (이상치 제거 + 변동계수(VC) 계산)
# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
    
#     # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
#     variation_coeff = None
#     if filtered and computed_mean and computed_mean > 0:
#         if len(filtered) > 1:
#             try:
#                 computed_stdev = stdev(filtered)
#                 variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
#             except StatisticsError:
#                 variation_coeff = 0.0 # 모든 값이 동일
#         else:
#             variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#         "variation_coeff": variation_coeff # <-- AI 일관성 점수
#     }

# # --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
# def get_dynamic_weights(
#     variation_coeff: Optional[float], 
#     admin_weights: Dict[str, float]
# ) -> Dict[str, Any]:
#     """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
#     # 관리자 설정값을 기본값으로 사용
#     base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
#     if variation_coeff is None:
#         # AI 데이터가 없으면 UN 100%
#         return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
#     if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
#         # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
#         dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
#         source = f"High AI Consistency (VC: {variation_coeff:.2f})"
#     elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
#         # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
#         dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
#         source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
#     else:
#         # 5% ~ 15% 사이: 관리자 설정값 유지
#         dynamic_ai_weight = base_ai_weight
#         source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

#     final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
#     final_un_weight = 1.0 - final_ai_weight
    
#     return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# # --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
# async def get_market_data_from_ai_async(
#     client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
#     city: str,
#     country: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     # --- (내부 헬퍼 함수들은 기존과 동일) ---
#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")
#     # --- (프롬프트 구성은 기존과 동일) ---
#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         # --- [수정] 비동기 API 호출로 변경 ---
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         # --- [수정] 끝 ---
        
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }
# # --- [개선 1] 끝 ---

# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
#             # [신규 1] 동적 가중치 적용 사유
#             weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#                 'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- 스트림릿 UI 구성 ---
# st.set_page_config(layout="wide")
# st.title("AICP: 출장 일비 계산 & 조회 시스템 (v16.0 - Async & Dynamic)")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility


# # --- [개선 3 & 신규 2] 탭 구조 변경 (v18.0) ---
# tab_definitions = [
#     "📊 Executive Dashboard", # [신규 2] 대시보드 탭 추가
# ]

# if employee_tab_visible:
#     tab_definitions.append("💵 일비 조회 (직원용)")

# # 관리자 탭을 2개로 분리
# tab_definitions.append("📈 보고서 분석 (Admin)")
# tab_definitions.append("🛠️ 시스템 설정 (Admin)")

# tabs = st.tabs(tab_definitions)

# # 탭 변수 할당
# dashboard_tab = tabs[0]
# tab_index_offset = 1

# if employee_tab_visible:
#     employee_tab = tabs[tab_index_offset]
#     admin_analysis_tab = tabs[tab_index_offset + 1]
#     admin_config_tab = tabs[tab_index_offset + 2]
#     tab_index_offset += 1
# else:
#     employee_tab = None
#     admin_analysis_tab = tabs[tab_index_offset]
#     admin_config_tab = tabs[tab_index_offset + 1]
# # --- [수정] 끝 ---


# # --- [신규 2] "Executive Dashboard" 탭 (새로 추가) ---
# # with dashboard_tab:
# #     st.header("Global Cost Dashboard")
# #     st.info("최신 보고서 데이터를 기반으로 전 세계 출장 비용 현황을 시각화합니다.")

# #     # --- [v18.5 수정] Altair 테마가 Streamlit 테마를 자동 감지하도록 설정 ---
# #     try:
# #         alt.theme.enable("streamlit")
# #     except Exception:
# #          # (라이브러리 충돌 시, 이전처럼 수동 테마 적용 - 여기서는 생략)
# #         pass 

# #     history_files = get_history_files()
# #     if not history_files:
# #         st.warning("표시할 데이터가 없습니다. 먼저 '보고서 분석' 탭에서 AI 분석을 1회 이상 실행해주세요.")
# #     else:
# #         latest_report_file = history_files[0]
# #         st.subheader(f"기준 보고서: `{latest_report_file}`")
        
# #         report_data = load_report_data(latest_report_file)
# #         config_entries = get_target_city_entries()
        
# #         if not report_data or 'cities' not in report_data or not config_entries:
# #             st.error("데이터 로드에 실패했습니다.")
# #         else:
# #             # 1. 데이터프레임 준비 (보고서 + 좌표)
# #             df_report = pd.DataFrame(report_data['cities'])
# #             df_config = pd.DataFrame(config_entries)
            
# #             df_merged = pd.merge(
# #                 df_report,
# #                 df_config,
# #                 left_on=["city", "country_display"],
# #                 right_on=["city", "country"],
# #                 suffixes=("_report", "_config")
# #             )
            
# #             required_map_cols = ['city', 'country', 'lat', 'lon', 'final_allowance']
            
# #             if not all(col in df_merged.columns for col in ['lat', 'lon']):
# #                 st.warning(
# #                     "지도에 표시할 좌표(lat/lon) 데이터가 없습니다. 🗺️\n\n"
# #                     "**해결 방법:** '🛠️ 시스템 설정 (Admin)' 탭으로 이동하여 [모든 도시 좌표 자동 완성] 버튼을 눌러주세요."
# #                 )
# #                 map_data = pd.DataFrame(columns=required_map_cols)
# #             else:
# #                 map_data = df_merged.copy()
# #                 map_data = map_data[required_map_cols]
# #                 map_data.dropna(subset=['lat', 'lon', 'final_allowance'], inplace=True)
# #                 map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
# #                 map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
# #                 map_data.dropna(subset=['lat', 'lon'], inplace=True)

# #             if map_data.empty:
# #                 st.caption("지도에 표시할 데이터가 없습니다. (좌표가 생성되었는지 확인하세요.)")
# #             else:
# #                 # 2. 비용에 따른 색상(R,G,B) 및 크기(Size) 계산
# #                 min_cost = map_data['final_allowance'].min()
# #                 max_cost = map_data['final_allowance'].max()
# #                 range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

# #                 def get_color_and_size(cost):
# #                     norm_cost = (cost - min_cost) / range_cost
# #                     r = int(255 * norm_cost)
# #                     g = int(255 * (1 - norm_cost))
# #                     b = 0
# #                     size = 50000 + (norm_cost * 450000)
# #                     return [r, g, b, 160], size

# #                 color_size = map_data['final_allowance'].apply(get_color_and_size)
# #                 map_data['color'] = [item[0] for item in color_size]
# #                 map_data['size'] = [item[1] for item in color_size]

# #                 # 3. Pydeck 차트 생성
# #                 view_state = pdk.ViewState(
# #                     latitude=map_data['lat'].mean(),
# #                     longitude=map_data['lon'].mean(),
# #                     zoom=0.5,
# #                     pitch=0,
# #                     bearing=0
# #                 )

# #                 layer = pdk.Layer(
# #                     'ScatterplotLayer',
# #                     data=map_data,
# #                     get_position='[lon, lat]',
# #                     get_color='color',
# #                     get_radius='size',
# #                     pickable=True,
# #                     opacity=0.8,
# #                     stroked=True,
# #                     filled=True,
# #                     radius_scale=0.5,
# #                     get_line_color=[255, 255, 255, 100],
# #                     get_line_width=10000,
# #                 )

# #                 tooltip = {
# #                     "html": "<b>{city}, {country}</b><br/>"
# #                             "최종 수당: <b>${final_allowance}</b>",
# #                     "style": { "color": "white", "backgroundColor": "#1e3c72" }
# #                 }
                
# #                 r = pdk.Deck(
# #                     layers=[layer],
# #                     initial_view_state=view_state,
# #                     # --- [v18.5 수정] map_style을 삭제하여 Streamlit 기본값(테마 자동 감지) 사용 ---
# #                     tooltip=tooltip
# #                 )

# #                 map_col, legend_col = st.columns([4, 1])

# #                 with map_col:
# #                     st.pydeck_chart(r, use_container_width=True)

# #                 with legend_col:
# #                     st.write("##### Legend (비용)")
# #                     st.markdown(f"""
# #                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">고비용 (~${max_cost:,.0f})</span>
# #                     </div>
# #                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(127, 127, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">중간 비용</span>
# #                     </div>
# #                     <div style="display: flex; align-items: center;">
# #                         <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
# #                         <span style="margin-left: 10px;">저비용 (~${min_cost:,.0f})</span>
# #                     </div>
# #                     """, unsafe_allow_html=True)
# #                     st.caption("원의 크기와 색상(붉은색)이 클수록 비용이 높은 도시입니다.")

# #             # 4. (아이디어 1 적용) Top 10 차트
# #             st.divider()
# #             col1, col2 = st.columns(2)
            
# #             if 'final_allowance' in df_merged.columns:
# #                 with col1:
# #                     st.write("##### 💰 Top 10 고비용 도시 (AI 최종안)")
# #                     top_10_cost_df = df_merged.nlargest(10, 'final_allowance')[['city', 'final_allowance']].reset_index(drop=True)
                    
# #                     # --- [v18.5 수정] Altair 차트 (테마가 글자색을 자동 관리) ---
# #                     chart_cost = alt.Chart(top_10_cost_df).mark_bar(
# #                         color="#0D6EFD"
# #                     ).encode(
# #                         x=alt.X('city', sort=None, title="도시"),
# #                         y=alt.Y('final_allowance', title="최종 수당 ($)"),
# #                         tooltip=[
# #                             alt.Tooltip('city', title="도시"),
# #                             alt.Tooltip('final_allowance', title="최종 수당 ($)", format='$,.0f')
# #                         ]
# #                     ).properties(
# #                         background='transparent' # 배경 투명화
# #                     ).interactive()
# #                     st.altair_chart(chart_cost, use_container_width=True)
            
# #                 with col2:
# #                     st.write("##### ⚠️ Top 10 변동성 높은 도시 (AI 신뢰도)")
# #                     df_report_vc = pd.DataFrame(report_data['cities'])
# #                     df_report_vc['vc'] = df_report_vc['ai_summary'].apply(lambda x: x.get('ai_consistency_vc') if isinstance(x, dict) else None)
# #                     df_report_vc.dropna(subset=['vc'], inplace=True)
                    
# #                     if df_report_vc.empty:
# #                         st.info("변동성(VC) 데이터가 없습니다. (최신 버전으로 AI 분석 필요)")
# #                     else:
# #                         top_10_vc_df = df_report_vc.nlargest(10, 'vc')[['city', 'vc']].reset_index(drop=True)
                        
# #                         # --- [v18.5 수정] Altair 차트 (테마가 글자색을 자동 관리) ---
# #                         chart_vc = alt.Chart(top_10_vc_df).mark_bar(
# #                             color="#DC3545"
# #                         ).encode(
# #                             x=alt.X('city', sort=None, title="도시"),
# #                             y=alt.Y('vc', title="변동 계수 (VC)", axis=alt.Axis(format='%')),
# #                             tooltip=[
# #                                 alt.Tooltip('city', title="도시"),
# #                                 alt.Tooltip('vc', title="변동 계수 (VC)", format='.2%')
# #                             ]
# #                         ).properties(
# #                             background='transparent' # 배경 투명화
# #                         ).interactive()
# #                         st.altair_chart(chart_vc, use_container_width=True)
# #                         st.caption("변동성(VC)이 높을수록 AI가 가격 추정을 확신하지 못하는 도시입니다.")
# #             else:
# #                 st.warning("차트를 표시할 'final_allowance' 데이터가 없습니다.")
# with dashboard_tab:
#     st.header("Global Cost Dashboard")
#     st.info("최신 보고서 데이터를 기반으로 전 세계 출장 비용 현황을 시각화합니다.")

#     try:
#         alt.theme.enable("streamlit")
#     except Exception:
#         pass 

#     history_files = get_history_files()
#     if not history_files:
#         st.warning("표시할 데이터가 없습니다. 먼저 '보고서 분석' 탭에서 AI 분석을 1회 이상 실행해주세요.")
#     else:
#         latest_report_file = history_files[0]
#         st.subheader(f"기준 보고서: `{latest_report_file}`")
        
#         report_data = load_report_data(latest_report_file)
#         config_entries = get_target_city_entries()
        
#         if not report_data or 'cities' not in report_data or not config_entries:
#             st.error("데이터 로드에 실패했습니다.")
#         else:
#             # 1. 데이터프레임 준비 (보고서 + 좌표)
#             df_report = pd.DataFrame(report_data['cities'])
#             df_config = pd.DataFrame(config_entries)
            
#             df_merged = pd.merge(
#                 df_report,
#                 df_config,
#                 left_on=["city", "country_display"],
#                 right_on=["city", "country"],
#                 suffixes=("_report", "_config")
#             )
            
#             required_map_cols = ['city', 'country', 'lat', 'lon', 'final_allowance']
            
#             if not all(col in df_merged.columns for col in ['lat', 'lon']):
#                 st.warning(
#                     "지도에 표시할 좌표(lat/lon) 데이터가 없습니다. 🗺️\n\n"
#                     "**해결 방법:** '🛠️ 시스템 설정 (Admin)' 탭으로 이동하여 [모든 도시 좌표 자동 완성] 버튼을 눌러주세요."
#                 )
#                 map_data = pd.DataFrame(columns=required_map_cols)
#             else:
#                 map_data = df_merged.copy()
#                 map_data = map_data[required_map_cols]
#                 map_data.dropna(subset=['lat', 'lon', 'final_allowance'], inplace=True)
#                 map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
#                 map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
#                 map_data.dropna(subset=['lat', 'lon'], inplace=True)

#             if map_data.empty:
#                 st.caption("지도에 표시할 데이터가 없습니다. (좌표가 생성되었는지 확인하세요.)")
#             else:
#                 # 2. 비용에 따른 색상(R,G,B) 및 크기(Size) 계산
#                 min_cost = map_data['final_allowance'].min()
#                 max_cost = map_data['final_allowance'].max()
#                 range_cost = max_cost - min_cost if max_cost > min_cost else 1.0

#                 def get_color_and_size(cost):
#                     norm_cost = (cost - min_cost) / range_cost
#                     r = int(255 * norm_cost)
#                     g = int(255 * (1 - norm_cost))
#                     b = 0
#                     size = 50000 + (norm_cost * 450000)
#                     return [r, g, b, 160], size

#                 color_size = map_data['final_allowance'].apply(get_color_and_size)
#                 map_data['color'] = [item[0] for item in color_size]
#                 map_data['size'] = [item[1] for item in color_size]

#                 # 3. Pydeck 차트 생성
#                 view_state = pdk.ViewState(
#                     latitude=map_data['lat'].mean(),
#                     longitude=map_data['lon'].mean(),
#                     zoom=0.5,
#                     pitch=0,
#                     bearing=0
#                 )

#                 layer = pdk.Layer(
#                     'ScatterplotLayer',
#                     data=map_data,
#                     get_position='[lon, lat]',
#                     get_color='color',
#                     get_radius='size',
#                     pickable=True,
#                     opacity=0.8,
#                     stroked=True,
#                     filled=True,
#                     radius_scale=0.5,
#                     get_line_color=[255, 255, 255, 100],
#                     get_line_width=10000,
#                 )

#                 tooltip = {
#                     "html": "<b>{city}, {country}</b><br/>"
#                             "최종 수당: <b>${final_allowance}</b>",
#                     "style": { "color": "white", "backgroundColor": "#1e3c72" }
#                 }
                
#                 r = pdk.Deck(
#                     layers=[layer],
#                     initial_view_state=view_state,
#                     tooltip=tooltip
#                 )

#                 map_col, legend_col = st.columns([4, 1])

#                 with map_col:
#                     st.pydeck_chart(r, use_container_width=True)

#                 with legend_col:
#                     st.write("##### Legend (비용)")
#                     st.markdown(f"""
#                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">고비용 (~${max_cost:,.0f})</span>
#                     </div>
#                     <div style="display: flex; align-items: center; margin-bottom: 5px;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(127, 127, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">중간 비용</span>
#                     </div>
#                     <div style="display: flex; align-items: center;">
#                         <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0, 0.8); border-radius: 50%; border: 1px solid #FFF;"></div>
#                         <span style="margin-left: 10px;">저비용 (~${min_cost:,.0f})</span>
#                     </div>
#                     """, unsafe_allow_html=True)
#                     st.caption("원의 크기와 색상(붉은색)이 클수록 비용이 높은 도시입니다.")

#             # 4. (아이디어 1 적용) Top 10 차트
#             st.divider()
#             col1, col2 = st.columns(2)
            
#             if 'final_allowance' in df_merged.columns:
#                 with col1:
#                     st.write("##### 💰 Top 10 고비용 도시 (AI 최종안)")
#                     top_10_cost_df = df_merged.nlargest(10, 'final_allowance')[['city', 'final_allowance']].reset_index(drop=True)
                    
#                     average_cost = df_merged['final_allowance'].mean()
                    
#                     # --- [v19.1 핫픽스] 툴팁용 'average' 컬럼 추가 ---
#                     top_10_cost_df['average'] = average_cost
                    
#                     base_cost = alt.Chart(top_10_cost_df).encode(
#                         x=alt.X('city', sort=None, title="도시", axis=alt.Axis(labelAngle=-45)), 
#                         y=alt.Y('final_allowance', title="최종 수당 ($)", axis=alt.Axis(format='$,.0f')),
#                         tooltip=[
#                             alt.Tooltip('city', title="도시"),
#                             alt.Tooltip('final_allowance', title="최종 수당 ($)", format='$,.0f'),
#                             alt.Tooltip('average', title="전체 평균", format='$,.0f') # <-- 수정됨
#                         ]
#                     )
                    
#                     bars_cost = base_cost.mark_bar(color="#0D6EFD").encode()
                    
#                     rule_cost = alt.Chart(pd.DataFrame({'average_cost': [average_cost]})).mark_rule(
#                         color='gray', strokeDash=[3, 3] # [v19.1] 선 색상 변경
#                     ).encode(
#                         y=alt.Y('average_cost', title=''),
#                         tooltip=[alt.Tooltip('average_cost', title="전체 평균", format='$,.0f')] 
#                     )
                    
#                     chart_cost = (bars_cost + rule_cost).properties(
#                         background='transparent',
#                         title=f"전체 평균: ${average_cost:,.0f}" 
#                     ).interactive()
#                     st.altair_chart(chart_cost, use_container_width=True)
            
#                 with col2:
#                     st.write("##### ⚠️ Top 10 변동성 높은 도시 (AI 신뢰도)")
#                     df_report_vc = pd.DataFrame(report_data['cities'])
#                     df_report_vc['vc'] = df_report_vc['ai_summary'].apply(lambda x: x.get('ai_consistency_vc') if isinstance(x, dict) else None)
#                     df_report_vc.dropna(subset=['vc'], inplace=True)
                    
#                     if df_report_vc.empty:
#                         st.info("변동성(VC) 데이터가 없습니다. (최신 버전으로 AI 분석 필요)")
#                     else:
#                         top_10_vc_df = df_report_vc.nlargest(10, 'vc')[['city', 'vc']].reset_index(drop=True)
                        
#                         average_vc = df_report_vc['vc'].mean()

#                         # --- [v19.1 핫픽스] 툴팁용 'average' 컬럼 추가 ---
#                         top_10_vc_df['average'] = average_vc
                        
#                         base_vc = alt.Chart(top_10_vc_df).encode(
#                             x=alt.X('city', sort=None, title="도시", axis=alt.Axis(labelAngle=-45)), 
#                             y=alt.Y('vc', title="변동 계수 (VC)", axis=alt.Axis(format='%')),
#                             tooltip=[
#                                 alt.Tooltip('city', title="도시"),
#                                 alt.Tooltip('vc', title="변동 계수 (VC)", format='.2%'),
#                                 alt.Tooltip('average', title="전체 평균", format='.2%') # <-- 수정됨
#                             ]
#                         )
                        
#                         bars_vc = base_vc.mark_bar(color="#DC3545").encode()
                        
#                         rule_vc = alt.Chart(pd.DataFrame({'average_vc': [average_vc]})).mark_rule(
#                             color='gray', strokeDash=[3, 3] # [v19.1] 선 색상 변경
#                         ).encode(
#                             y=alt.Y('average_vc', title=''),
#                             tooltip=[alt.Tooltip('average_vc', title="전체 평균", format='.2%')]
#                         )
                        
#                         chart_vc = (bars_vc + rule_vc).properties(
#                             background='transparent',
#                             title=f"전체 평균: {average_vc:.2%}"
#                         ).interactive()
#                         st.altair_chart(chart_vc, use_container_width=True)
#                         st.caption("변동성(VC)이 높을수록 AI가 가격 추정을 확신하지 못하는 도시입니다.")
#             else:
#                 st.warning("차트를 표시할 'final_allowance' 데이터가 없습니다.")

# if employee_tab is not None:
#     with employee_tab:
#         st.header("도시별 출장 일비 조회")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("도시:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("선택한 국가에 등록된 도시가 없습니다.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "출장 시작일",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "출장 종료일",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("종료일은 시작일 이후여야 합니다.")
#                     trip_days = 0 # 0으로 설정
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"자동 분류된 출장 유형: {trip_term_label} · {trip_days}일 일정")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("이 기간에 해당하는 도시 데이터가 없습니다. 출장 유형을 '숏텀'으로 조정하거나 도시 설정을 확인하세요.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level and trip_days is not None:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} 결과")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
                        
#                         # [신규 2] 직원 탭 총액 카드
#                         render_primary_summary(
#                             f"{sel_level.split(' ')[0]}",
#                             trip_total_allowance,
#                             level_daily_allowance,
#                             trip_days,
#                             trip_term_label,
#                             trip_multiplier
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} 일일 권장 일비", "금액 없음")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**세부 산출 근거 (일비 기준)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"제외값 {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"
                        
#                         # [신규 1] 동적 가중치 적용 사유
#                         weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
#                         weights_source = weights_info.get("source", "N/A")
#                         un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
#                         ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
#                         weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | 사유: {weights_source}"

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         un_caption = f"숏텀 기준 $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
#                                         if trip_term == "Long-term" and un_base is not None:
#                                             un_caption = f"숏텀 $ {un_base:,} → 롱텀 $ {un_display:,}"
#                                         render_stat_card("UN-DSA 기준", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
#                                     elif key == "ai":
#                                         ai_caption_base = f"숏텀 기준 $ {ai_avg:,}" if ai_avg is not None else ""
#                                         if trip_term == "Long-term" and ai_avg is not None:
#                                             ai_caption_base = f"숏텀 $ {ai_avg:,} → 롱텀 $ {ai_display:,}"
#                                         ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
#                                         render_stat_card("AI 시장 추정 (시즌 보정)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
#                                     else: # key == "weighted"
#                                         weighted_caption = weight_caption
#                                         if trip_term == "Long-term" and final_allowance is not None:
#                                             weighted_caption = f"숏텀 $ {final_allowance:,} → 롱텀 $ {weighted_display:,} | {weight_caption}"
#                                         render_stat_card("가중 평균 결과", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

#                         # [신규 2] 비용 항목별 상세 내역 (show_ai_market_detail과 로직 통합)
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             st.markdown("<br>", unsafe_allow_html=True) # 줄 간격
                            
#                             mean_food = ai_summary.get("mean_food", 0)
#                             mean_trans = ai_summary.get("mean_transport", 0)
#                             mean_misc = ai_summary.get("mean_misc", 0)
                            
#                             # 롱텀/시즌 요율 적용
#                             food_display = round(mean_food * season_factor * trip_multiplier)
#                             trans_display = round(mean_trans * season_factor * trip_multiplier)
#                             misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
#                             st.write("###### AI 추정 상세 내역 (일비 기준)")
#                             col_f, col_t, col_m = st.columns(3)
#                             with col_f:
#                                 render_stat_card("예상 식비 (Food)", f"$ {food_display:,}", f"숏텀 기준: $ {round(mean_food)}", "muted")
#                             with col_t:
#                                 render_stat_card("예상 교통비 (Transport)", f"$ {trans_display:,}", f"숏텀 기준: $ {round(mean_trans)}", "muted")
#                             with col_m:
#                                 render_stat_card("예상 기타 (Misc)", f"$ {misc_display:,}", f"숏텀 기준: $ {round(mean_misc)}", "muted")
                        
#                         # [개선 3] show_weighted_result 카드가 중복되므로, 아래 블록은 제거
#                         # (기존 second_row_keys 로직 제거)

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_info,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("관리자가 세부 산출 근거를 숨겼습니다.")

# # --- [개선 2] admin_tab -> admin_analysis_tab 으로 변경 ---
# with admin_analysis_tab:
    
#     # [개선 2] ADMIN_ACCESS_CODE 로드 및 .env 체크
#     ACCESS_CODE_KEY = "admin_access_code_valid"
#     ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # .env에서 로드

#     if not ACCESS_CODE_VALUE:
#         st.error("보안 오류: .env 파일에 'ADMIN_ACCESS_CODE'가 설정되지 않았습니다. 앱을 중지하고 .env 파일을 설정해주세요.")
#         st.stop()
    
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#                 st.rerun() # [개선 3] 성공 시 새로고침
#             else:
#                 st.error("Access Code가 올바르지 않습니다.")
#                 st.stop() # [개선 3] 실패 시 중지
#         else:
#             st.stop() # [개선 3] 폼 제출 전 중지

#     # --- [개선 3] "보고서 버전 관리" 기능 (analysis_sub_tab) ---
#     st.subheader("보고서 버전 관리")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("활성 보고서 버전을 선택하세요:", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("생성된 보고서가 없습니다.")

#     # --- [신규 4] 과거 보고서 비교 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("과거 보고서 비교")
#     if len(history_files) < 2:
#         st.info("비교할 보고서가 2개 이상 필요합니다.")
#     else:
#         col_a, col_b = st.columns(2)
#         with col_a:
#             file_a = st.selectbox("기준 보고서 (A)", history_files, index=1, key="compare_a")
#         with col_b:
#             file_b = st.selectbox("비교 보고서 (B)", history_files, index=0, key="compare_b")
        
#         if st.button("보고서 비교하기"):
#             if file_a == file_b:
#                 st.warning("서로 다른 보고서를 선택해야 합니다.")
#             else:
#                 with st.spinner("보고서 비교 중..."):
#                     data_a = load_report_data(file_a)
#                     data_b = load_report_data(file_b)
                    
#                     if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
#                         df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
#                         df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
#                         df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
#                         report_a_label = file_a.split('report_')[-1].split('.')[0]
#                         report_b_label = file_b.split('report_')[-1].split('.')[0]

#                         df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
#                         df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
#                         df_merged["변동액 ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
#                         # 0으로 나누기 방지
#                         df_merged["변동률 (%)"] = (df_merged["변동액 ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
#                         st.dataframe(df_merged[[
#                             "city", "country_display", 
#                             f"A ({report_a_label})", 
#                             f"B ({report_b_label})", 
#                             "변동액 ($)", "변동률 (%)"
#                         ]].style.format({"변동률 (%)": "{:,.1f}%", "변동액 ($)": "{:,.0f}"}), width="stretch")
#                     else:
#                         st.error("보고서 파일을 불러오는 데 실패했습니다.")
    
#     # --- [개선 3] "UN-DSA (PDF) 분석" 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("UN-DSA (PDF) 분석 및 AI 실행")
#     st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요. (개선 1: 비동기 처리로 속도 향상)")
#     uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")

#     # --- [개선 1] 비동기 AI 분석 실행 로직 ---
#     if uploaded_file and st.button("AI 분석 실행", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
#         else:
#             st.session_state.latest_analysis_result = None
            
#             # --- 비동기 실행 함수 정의 ---
#             async def run_analysis(progress_bar, openai_api_key):
#                 progress_bar.progress(0, text="PDF 텍스트 추출 중...")
#                 full_text = parse_pdf_to_text(uploaded_file)
                
#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []
#                 analysis_failed = False
                
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break
                
#                 if analysis_failed:
#                     st.error("PDF->TSV 변환에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                 if not processed_data:
#                     st.error("TSV 데이터 처리에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 # 비동기 OpenAI 클라이언트 생성
#                 client = openai.AsyncOpenAI(api_key=openai_api_key)
                
#                 total_cities = len(processed_data["cities"])
#                 all_tasks = [] # 모든 AI 호출 작업을 담을 리스트

#                 # 1. 모든 도시에 대한 모든 AI 호출 작업을 미리 생성
#                 for city_data in processed_data["cities"]:
#                     city_name, country_name = city_data["city"], city_data["country_display"]
#                     city_context = {
#                         "neighborhood": city_data.get("neighborhood"),
#                         "hotel_cluster": city_data.get("hotel_cluster"),
#                     }
#                     season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                     menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
#                     city_data["menu_samples"] = menu_samples
#                     city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
#                     city_tasks = []
#                     for j in range(1, NUM_AI_CALLS + 1):
#                         task = get_market_data_from_ai_async(
#                             client, city_name, country_name, f"Run {j}",
#                             context=city_context, season_context=season_context, menu_samples=menu_samples
#                         )
#                         city_tasks.append(task)
                    
#                     all_tasks.append(city_tasks) # [ [도시1-10회], [도시2-10회], ... ]

#                 # 2. 모든 작업을 비동기로 실행하고 결과 수집
#                 city_index = 0
#                 for city_tasks in all_tasks:
#                     city_data = processed_data["cities"][city_index]
#                     city_name = city_data["city"]
#                     progress_text = f"AI 추정치 계산 중... ({city_index+1}/{total_cities}) {city_name}"
#                     progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
#                     # 해당 도시의 10개 작업을 동시에 실행
#                     try:
#                         market_results = await asyncio.gather(*city_tasks)
#                     except Exception as e:
#                         st.error(f"{city_name} 분석 중 비동기 오류: {e}")
#                         market_results = [] # 실패 처리

#                     # 3. 결과 처리
#                     ai_totals_source: List[int] = []
#                     ai_meta_runs: List[Dict[str, Any]] = []
                    
#                     # [신규 2] 비용 항목별 상세 내역을 위한 리스트
#                     ai_food: List[int] = []
#                     ai_transport: List[int] = []
#                     ai_misc: List[int] = []

#                     for j, market_result in enumerate(market_results, 1):
#                         city_data[f"market_data_{j}"] = market_result
#                         if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                             ai_totals_source.append(market_result["total"])
#                             # [신규 2] 상세 비용 추가
#                             ai_food.append(market_result.get("food", 0))
#                             ai_transport.append(market_result.get("transport", 0))
#                             ai_misc.append(market_result.get("misc", 0))
                        
#                         if "meta" in market_result:
#                             ai_meta_runs.append(market_result["meta"])
                    
#                     city_data["ai_provenance"] = ai_meta_runs

#                     # 4. 최종 수당 계산
#                     final_allowance = None
#                     un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
#                     un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

#                     ai_stats = aggregate_ai_totals(ai_totals_source)
#                     season_factor = (season_context or {}).get("factor", 1.0)
#                     ai_base_mean = ai_stats.get("mean_raw")
#                     ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
#                     # [신규 1] 동적 가중치 계산
#                     admin_weights = get_weight_config() # 관리자 설정 로드
#                     ai_vc_score = ai_stats.get("variation_coeff")
                    
#                     if un_per_diem is not None:
#                         weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
#                     else:
#                         # UN 데이터 없으면 AI 100%
#                         weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
#                     city_data["ai_summary"] = {
#                         "raw_totals": ai_totals_source,
#                         "used_totals": ai_stats.get("used_values", []),
#                         "removed_totals": ai_stats.get("removed_values", []),
#                         "mean_base": ai_base_mean,
#                         "mean_base_rounded": ai_stats.get("mean"),
                        
#                         "ai_consistency_vc": ai_vc_score, # [신규 1]
                        
#                         "mean_food": mean(ai_food) if ai_food else 0, # [신규 2]
#                         "mean_transport": mean(ai_transport) if ai_transport else 0, # [신규 2]
#                         "mean_misc": mean(ai_misc) if ai_misc else 0, # [신규 2]

#                         "season_factor": season_factor,
#                         "season_label": (season_context or {}).get("label"),
#                         "season_adjusted_mean_raw": ai_season_adjusted,
#                         "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                         "successful_runs": len(ai_stats.get("used_values", [])),
#                         "attempted_runs": NUM_AI_CALLS,
#                         "reference_links": city_data.get("reference_links", []),
#                         "weighted_average_components": {
#                             "un_per_diem": un_per_diem,
#                             "ai_season_adjusted": ai_season_adjusted,
#                             "weights": weights_cfg, # [신규 1] 동적 가중치 저장
#                         },
#                     }

#                     # [신규 1] 동적 가중치로 최종값 계산
#                     if un_per_diem is not None and ai_season_adjusted is not None:
#                         weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                         final_allowance = round(weighted_average)
#                     elif un_per_diem is not None:
#                         final_allowance = round(un_per_diem)
#                     elif ai_season_adjusted is not None:
#                         final_allowance = round(ai_season_adjusted)

#                     city_data["final_allowance"] = final_allowance

#                     if final_allowance and un_per_diem and un_per_diem > 0:
#                         city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                     else:
#                         city_data["delta_vs_un_pct"] = "N/A"
                    
#                     city_index += 1 # 다음 도시로

#                 save_report_data(processed_data)
#                 st.session_state.latest_analysis_result = processed_data
#                 st.success("AI analysis completed.")
#                 progress_bar.empty()
#                 st.rerun()
            
#             # --- 비동기 실행 ---
#             with st.spinner("PDF 처리 및 AI 분석을 실행합니다. (약 10~30초 소요)"):
#                 progress_bar = st.progress(0, text="분석 시작...")
#                 asyncio.run(run_analysis(progress_bar, openai_api_key))

#     # --- [개선 3] "Latest Analysis Summary" 기능 (analysis_sub_tab) ---
#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             # --- [HOTFIX] ArrowInvalid Error 방지 ---
#             delta_val = city.get('delta_vs_un_pct')
#             if isinstance(delta_val, (int, float)):
#                 delta_display = f"{delta_val:.0f}%" # 숫자를 "12%" 형태의 문자열로 변경
#             else:
#                 delta_display = "N/A" # 이미 "N/A" 문자열
#             # --- [HOTFIX] End ---
                
#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': delta_display, # <-- 수정된 문자열 값 사용
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- use_container_width 추가 (필요시 width='stretch'로 변경)
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# # --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
# with admin_config_tab:
#     # 암호 확인 (필수)
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         st.error("Access Code가 필요합니다. '보고서 분석 (Admin)' 탭에서 먼저 로그인해주세요.")
#         st.stop()
        
#     # --- [v19.3] 도시 편집/캐시 관리가 공유할 도시 목록을 탭 상단에서 정의 ---
#     current_entries = get_target_city_entries()
#     options = {
#         f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#         for idx, entry in enumerate(current_entries)
#     }
#     sorted_labels = list(options.keys())
    
#     # --- 콜백 함수 1: '도시 편집' 폼 동기화 ---
#     def _sync_edit_form_from_selection():
#         if "edit_city_selector" not in st.session_state or not st.session_state.edit_city_selector:
#              st.session_state.edit_city_selector = sorted_labels[0]
             
#         selected_idx = options[st.session_state.edit_city_selector]
#         selected_entry = current_entries[selected_idx]
        
#         st.session_state.edit_region = selected_entry.get("region", "")
#         st.session_state.edit_city = selected_entry.get("city", "")
#         st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#         st.session_state.edit_country = selected_entry.get("country", "")
#         st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
        
#         existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#         st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
        
#         sub_data = selected_entry.get("un_dsa_substitute") or {}
#         st.session_state.edit_sub_city = sub_data.get("city", "")
#         st.session_state.edit_sub_country = sub_data.get("country", "")

#     # --- [v19.3] 콜백 함수 2: '캐시 추가' 폼 동기화 ---
#     def _sync_cache_form_from_selection():
#         selected_label = st.session_state.get("cache_city_selector") # get()으로 오류 방지
        
#         if selected_label in options: # 'options' dict를 공유
#             selected_idx = options[selected_label]
#             selected_entry = current_entries[selected_idx]
#             st.session_state.new_cache_country = selected_entry.get("country", "")
#             st.session_state.new_cache_city = selected_entry.get("city", "")
#             st.session_state.new_cache_neighborhood = selected_entry.get("neighborhood", "")
#         else: # (placeholder 선택 시)
#             st.session_state.new_cache_country = ""
#             st.session_state.new_cache_city = ""
#             st.session_state.new_cache_neighborhood = ""
        
#         # 나머지 필드는 항상 기본값으로 초기화
#         st.session_state.new_cache_vendor = ""
#         st.session_state.new_cache_category = "Food"
#         st.session_state.new_cache_price = 0.0
#         st.session_state.new_cache_currency = "USD"
#         st.session_state.new_cache_url = ""

#     # --- [v19.3 핫픽스] 콜백 함수 3: '캐시 저장' 로직 ---
#     def handle_cache_submit():
#         # 1. 유효성 검사
#         if (not st.session_state.new_cache_country or 
#             not st.session_state.new_cache_city or 
#             not st.session_state.new_cache_vendor):
#             st.error("국가, 도시, 장소/상품명은 필수입니다.")
#             return # 여기서 중단 (폼 값 유지됨)

#         # 2. 새 항목 생성
#         new_entry = {
#             "country": st.session_state.new_cache_country.strip(),
#             "city": st.session_state.new_cache_city.strip(),
#             "neighborhood": st.session_state.new_cache_neighborhood.strip(),
#             "vendor": st.session_state.new_cache_vendor.strip(),
#             "category": st.session_state.new_cache_category,
#             "price": st.session_state.new_cache_price,
#             "currency": st.session_state.new_cache_currency.strip().upper(),
#             "url": st.session_state.new_cache_url.strip(),
#         }
        
#         # 3. 파일에 저장
#         if add_menu_cache_entry(new_entry):
#             st.success(f"'{new_entry['vendor']}' 항목을 캐시에 추가했습니다.")
            
#             # 4. (중요) 폼 리셋: session_state 값들을 수동으로 초기화
#             st.session_state.new_cache_country = ""
#             st.session_state.new_cache_city = ""
#             st.session_state.new_cache_neighborhood = ""
#             st.session_state.new_cache_vendor = ""
#             st.session_state.new_cache_category = "Food"
#             st.session_state.new_cache_price = 0.0
#             st.session_state.new_cache_currency = "USD"
#             st.session_state.new_cache_url = ""
#             st.session_state.cache_city_selector = None # 드롭다운도 리셋
            
#             # st.rerun()은 on_click 콜백이 끝나면 자동으로 호출되므로 명시적으로 호출할 필요 없음
#         else:
#             st.error("캐시 항목 추가에 실패했습니다.")
#     # --- [v19.3 핫픽스] 끝 ---

#     st.subheader("직원용 탭 노출")
#     visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key 이름 변경
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.session_state.employee_tab_visibility = visibility_toggle # 세션 상태에도 반영
#         st.success("직원용 탭 노출 상태가 업데이트되었습니다. (새로고침 시 적용)")
#         time.sleep(1) # 유저가 메시지를 읽을 시간을 줌
#         st.rerun()

#     st.subheader("직원 화면 노출 설정")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("직원 화면 노출 설정이 업데이트되었습니다.")
#         time.sleep(1)
#         st.rerun()

#     st.divider()
#     st.subheader("비중 설정 (기본값)")
#     st.info("이제 이 설정은 '동적 가중치' 로직의 기본값으로 사용됩니다. AI 응답이 불안정하면 자동으로 AI 비중이 낮아집니다.")
#     current_weights = get_weight_config()
#     st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
#         st.rerun()

#     st.divider()
#     st.header("목표 도시 관리 (target_cities_config.json)")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         # trip_lengths를 보기 쉽게 문자열로 변환
#         entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], width='stretch') # [v19.3] 경고 수정
#     else:
#         st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

#     # --- [신규 2] 도시 좌표 자동 완성 기능 (새로 추가) ---
#     st.divider()
#     st.subheader("도시 좌표 관리")
    
#     if st.button("모든 도시 좌표(Lat/Lon) 자동 완성", help="target_cities_config.json의 모든 도시를 대상으로 좌표가 없는 도시에 대해 geopy를 호출해 좌표를 자동 저장합니다."):
        
#         # 1. 지오코더 초기화
#         try:
#             geolocator = Nominatim(user_agent=f"aicp_app_{random.randint(1000,9999)}")
#         except Exception as e:
#             st.error(f"Geopy(Nominatim) 초기화 실패: {e}")
#             st.stop()

#         # 2. 도시 목록 로드
#         current_entries = get_target_city_entries()
#         entries_to_update = [e for e in current_entries if not e.get('lat') or not e.get('lon')]
        
#         if not entries_to_update:
#             st.success("모든 도시에 이미 좌표가 설정되어 있습니다. (업데이트 불필요)")
#             st.stop()
            
#         st.info(f"총 {len(current_entries)}개 도시 중, 좌표가 없는 {len(entries_to_update)}개 도시에 대한 좌표를 불러옵니다...")
        
#         progress_bar = st.progress(0, text="좌표 자동 완성 시작...")
#         success_count = 0
#         fail_count = 0
        
#         with st.spinner("도시 좌표를 하나씩 불러오는 중... (시간이 걸릴 수 있습니다)"):
#             for i, entry in enumerate(entries_to_update):
#                 city = entry['city']
#                 country = entry['country']
#                 query = f"{city}, {country}"
                
#                 try:
#                     # 3. API 호출
#                     location = geolocator.geocode(query, timeout=5)
#                     time.sleep(1) # (중요) Nominatim의 API 제한(초당 1회) 준수
                    
#                     if location:
#                         # 4. 원본 entry에 lat/lon 추가
#                         entry['lat'] = location.latitude
#                         entry['lon'] = location.longitude
#                         st.toast(f"✅ 성공: {query} ({location.latitude:.4f}, {location.longitude:.4f})", icon="🌍")
#                         success_count += 1
#                     else:
#                         st.toast(f"⚠️ 실패: {query}의 좌표를 찾을 수 없습니다.", icon="❓")
#                         fail_count += 1
                        
#                 except (GeocoderTimedOut, GeocoderUnavailable):
#                     st.toast(f"❌ 오류: {query} 요청 시간 초과. 잠시 후 다시 시도하세요.", icon="🔥")
#                     fail_count += 1
#                 except Exception as e:
#                     st.toast(f"❌ 오류: {query} ({e})", icon="🔥")
#                     fail_count += 1
                
#                 progress_bar.progress((i + 1) / len(entries_to_update), text=f"처리 중: {query}")

#         # 5. 전체 파일 저장
#         set_target_city_entries(current_entries) # (save_target_city_entries 호출 포함)
        
#         st.success(f"좌표 자동 완성 완료! (성공: {success_count} / 실패: {fail_count})")
#         st.rerun()
#     # --- [신규 2] 끝 ---


#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("신규 도시 추가")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["기타 (직접 입력)"]
#             region_choice = st.selectbox("지역", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "기타 (직접 입력)":
#                 new_region = st.text_input("새 지역 이름", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("도시", key="add_city")
#             neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("국가", key="add_country")
#             hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

#         with st.expander("UN-DSA 대체 도시 (선택)"):
#             substitute_city = st.text_input("대체 도시", key="add_sub_city")
#             substitute_country = st.text_input("대체 국가", key="add_sub_country")

#         add_submitted = st.form_submit_button("추가")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("동일한 항목이 이미 등록되어 있습니다.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
#                 st.rerun()

#     st.subheader("기존 도시 편집/삭제")
    
#     if current_entries:
#         # 드롭다운(Selectbox)에 on_change 콜백 연결
#         selected_label = st.selectbox(
#             "편집할 도시를 선택하세요", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection
#         )

#         # 페이지 첫 로드 시 폼을 채우기 위한 초기화
#         if "edit_region" not in st.session_state:
#             _sync_edit_form_from_selection()

#         # 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 region_edit = st.text_input("지역", key="edit_region")
#                 city_edit = st.text_input("도시", key="edit_city")
#                 neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
#             with col_f:
#                 country_edit = st.text_input("국가", key="edit_country")
#                 hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

#             trip_lengths_edit = st.multiselect(
#                 "출장 기간",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", 
#             )

#             with st.expander("UN-DSA 대체 도시 (선택)"):
#                 sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
#                 sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("변경사항 저장")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("삭제", type="secondary")

#         # 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
#         if update_btn:
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#             else:
#                 selected_idx = options[st.session_state.edit_city_selector]
#                 current_entries[selected_idx] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[selected_idx]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[selected_idx].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("수정을 완료했습니다.")
#                 st.rerun()
        
#         if delete_btn:
#             selected_idx = options[st.session_state.edit_city_selector]
#             del current_entries[selected_idx]
#             set_target_city_entries(current_entries)
#             st.warning("선택한 항목을 삭제했습니다.")
#             st.rerun()
#     else:
#         st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

#     # --- [신규 3] '데이터 캐시 관리' UI 추가 ---
#     st.divider()
#     st.header("데이터 캐시 관리 (Menu Cache)")

#     if not MENU_CACHE_ENABLED:
#         st.error("`data_sources/menu_cache.py` 파일 로드에 실패하여 이 기능을 사용할 수 없습니다.")
#     else:
#         st.info("AI가 도시 물가 추정 시 참고할 실제 메뉴/가격 데이터를 관리합니다. (AI 분석 정확도 향상)")

#         # 1. 새 캐시 항목 추가 폼
#         st.subheader("신규 캐시 항목 추가")
        
#         st.selectbox(
#             "도시 선택 (자동 채우기):", 
#             sorted_labels,  # 탭 상단에서 정의한 변수
#             key="cache_city_selector",
#             on_change=_sync_cache_form_from_selection, # 새로 만든 콜백
#             index=None,
#             placeholder="도시를 선택하면 국가, 도시, 세부 지역이 자동 입력됩니다."
#         )

#         # 페이지 첫 로드 시 캐시 폼 초기화
#         if "new_cache_country" not in st.session_state:
#             _sync_cache_form_from_selection() # 빈 값으로 초기화
        
#         # --- [v19.3 핫픽스] clear_on_submit=True 제거, on_click 콜백 사용 ---
#         with st.form("add_menu_cache_form"):
#             st.write("AI 분석에 사용할 참고 가격 정보를 입력합니다. (예: 레스토랑 메뉴, 택시비 고지 등)")
#             c1, c2 = st.columns(2)
#             with c1:
#                 new_cache_country = st.text_input("국가 (Country)", key="new_cache_country", help="예: Philippines")
#                 new_cache_city = st.text_input("도시 (City)", key="new_cache_city", help="예: Manila")
#                 new_cache_neighborhood = st.text_input("세부 지역 (Neighborhood) (선택)", key="new_cache_neighborhood", help="예: Makati (비워두면 도시 전체에 적용)")
#                 new_cache_vendor = st.text_input("장소/상품명 (Vendor)", key="new_cache_vendor", help="예: Jollibee (C3, Ayala Ave)")
#             with c2:
#                 new_cache_category = st.selectbox("카테고리 (Category)", ["Food", "Transport", "Misc"], key="new_cache_category")
#                 new_cache_price = st.number_input("가격 (Price)", min_value=0.0, step=0.01, key="new_cache_price")
#                 new_cache_currency = st.text_input("통화 (Currency)", value="USD", key="new_cache_currency", help="예: PHP, USD")
#                 new_cache_url = st.text_input("출처 URL (Source URL) (선택)", key="new_cache_url")
            
#             # [v19.3] on_click 콜백으로 저장/초기화 로직 실행
#             add_cache_submitted = st.form_submit_button(
#                 "신규 캐시 항목 저장",
#                 on_click=handle_cache_submit
#             )
#         # --- [v19.3 핫픽스] 끝 ---

#         # 2. 기존 캐시 항목 조회 및 삭제
#         st.subheader("기존 캐시 항목 조회 및 삭제")
#         all_cache_data = load_all_cache() # menu_cache.py의 함수
        
#         if not all_cache_data:
#             st.info("현재 저장된 캐시 데이터가 없습니다.")
#         else:
#             df_cache = pd.DataFrame(all_cache_data)
#             st.dataframe(df_cache[[
#                 "country", "city", "neighborhood", "vendor", 
#                 "category", "price", "currency", "last_updated", "url"
#             ]], width='stretch') # [v19.3] 경고 수정

#             # 삭제 기능
#             st.markdown("---")
#             st.write("##### 캐시 항목 삭제")
            
#             delete_options_map = {
#                 f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
#                 for idx, entry in enumerate(reversed(all_cache_data))
#             }
#             delete_labels = list(delete_options_map.keys())
            
#             label_to_delete = st.selectbox("삭제할 캐시 항목을 선택하세요:", delete_labels, index=None, placeholder="삭제할 항목 선택...")
            
#             if label_to_delete and st.button(f"'{label_to_delete}' 항목 삭제", type="primary"):
#                 original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
#                 entry_to_delete = all_cache_data.pop(original_list_index)
                
#                 if save_cached_menu_prices(all_cache_data):
#                     st.success(f"'{entry_to_delete.get('vendor')}' 항목을 삭제했습니다.")
#                     st.rerun()
#                 else:
#                     st.error("캐시 삭제에 실패했습니다.")
    
    # --- [신규 3] UI 끝 ---

# 2025-11-06 와우포인트(맵) 개선 전
#  # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# # 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
# from typing import Any, Dict, List, Optional, Set, Tuple

# # [신규 3] menu_cache 임포트 (파일이 없으면 이 기능은 작동하지 않음)
# try:
#     from data_sources.menu_cache import (
#         load_cached_menu_prices, 
#         load_all_cache, 
#         add_menu_cache_entry, 
#         save_cached_menu_prices
#     )
#     MENU_CACHE_ENABLED = True
# except ImportError:
#     st.warning("`data_sources/menu_cache.py` 파일을 찾을 수 없습니다. '데이터 캐시 관리' 기능이 비활성화됩니다.")
#     # (기존 함수들을 임시로 정의)
#     def load_cached_menu_prices(city: str, country: str, neighborhood: Optional[str]) -> List[Dict[str, Any]]: return []
#     def load_all_cache() -> List[Dict[str, Any]]: return []
#     def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool: return False
#     def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool: return False
#     MENU_CACHE_ENABLED = False


# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
#         "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
#     },
#     "secondary": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
#     "muted": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
#     # [수정] 캡션에 스타일 적용
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">{level_label} 기준 예상 일비 총액</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>계산식</span><br/>
#             $ {daily:,} × {days}일 일정 × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6)": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term", "Long-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "숏텀", "Long-term": "롱텀"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry.setdefault("trip_lengths", DEFAULT_TRIP_LENGTH.copy())
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# # --- [신규 1] aggregate_ai_totals 함수 수정 ---
# # (이상치 제거 + 변동계수(VC) 계산)
# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
    
#     # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
#     variation_coeff = None
#     if filtered and computed_mean and computed_mean > 0:
#         if len(filtered) > 1:
#             try:
#                 computed_stdev = stdev(filtered)
#                 variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
#             except StatisticsError:
#                 variation_coeff = 0.0 # 모든 값이 동일
#         else:
#             variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#         "variation_coeff": variation_coeff # <-- AI 일관성 점수
#     }

# # --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
# def get_dynamic_weights(
#     variation_coeff: Optional[float], 
#     admin_weights: Dict[str, float]
# ) -> Dict[str, Any]:
#     """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
#     # 관리자 설정값을 기본값으로 사용
#     base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
#     if variation_coeff is None:
#         # AI 데이터가 없으면 UN 100%
#         return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
#     if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
#         # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
#         dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
#         source = f"High AI Consistency (VC: {variation_coeff:.2f})"
#     elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
#         # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
#         dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
#         source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
#     else:
#         # 5% ~ 15% 사이: 관리자 설정값 유지
#         dynamic_ai_weight = base_ai_weight
#         source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

#     final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
#     final_un_weight = 1.0 - final_ai_weight
    
#     return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# # --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
# async def get_market_data_from_ai_async(
#     client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
#     city: str,
#     country: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     # --- (내부 헬퍼 함수들은 기존과 동일) ---
#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")
#     # --- (프롬프트 구성은 기존과 동일) ---
#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         # --- [수정] 비동기 API 호출로 변경 ---
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         # --- [수정] 끝 ---
        
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }
# # --- [개선 1] 끝 ---

# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
#             # [신규 1] 동적 가중치 적용 사유
#             weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#                 'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- 스트림릿 UI 구성 ---
# st.set_page_config(layout="wide")
# st.title("AICP: 출장 일비 계산 & 조회 시스템 (v16.0 - Async & Dynamic)")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility


# # --- [개선 3] 탭 구조 변경 ---
# tab_definitions = []
# if employee_tab_visible:
#     tab_definitions.append("💵 일비 조회 (직원용)")

# # 관리자 탭을 2개로 분리
# tab_definitions.append("📈 보고서 분석 (Admin)")
# tab_definitions.append("🛠️ 시스템 설정 (Admin)")

# tabs = st.tabs(tab_definitions)

# employee_tab = None
# admin_analysis_tab = None
# admin_config_tab = None

# if employee_tab_visible:
#     employee_tab = tabs[0]
#     admin_analysis_tab = tabs[1]
#     admin_config_tab = tabs[2]
# else:
#     admin_analysis_tab = tabs[0]
#     admin_config_tab = tabs[1]
# # --- [개선 3] 끝 ---


# if employee_tab is not None:
#     with employee_tab:
#         st.header("도시별 출장 일비 조회")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("도시:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("선택한 국가에 등록된 도시가 없습니다.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "출장 시작일",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "출장 종료일",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("종료일은 시작일 이후여야 합니다.")
#                     trip_days = 0 # 0으로 설정
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"자동 분류된 출장 유형: {trip_term_label} · {trip_days}일 일정")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("이 기간에 해당하는 도시 데이터가 없습니다. 출장 유형을 '숏텀'으로 조정하거나 도시 설정을 확인하세요.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level and trip_days is not None:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} 결과")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
                        
#                         # [신규 2] 직원 탭 총액 카드
#                         render_primary_summary(
#                             f"{sel_level.split(' ')[0]}",
#                             trip_total_allowance,
#                             level_daily_allowance,
#                             trip_days,
#                             trip_term_label,
#                             trip_multiplier
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} 일일 권장 일비", "금액 없음")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**세부 산출 근거 (일비 기준)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"제외값 {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"
                        
#                         # [신규 1] 동적 가중치 적용 사유
#                         weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
#                         weights_source = weights_info.get("source", "N/A")
#                         un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
#                         ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
#                         weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | 사유: {weights_source}"

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         un_caption = f"숏텀 기준 $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
#                                         if trip_term == "Long-term" and un_base is not None:
#                                             un_caption = f"숏텀 $ {un_base:,} → 롱텀 $ {un_display:,}"
#                                         render_stat_card("UN-DSA 기준", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
#                                     elif key == "ai":
#                                         ai_caption_base = f"숏텀 기준 $ {ai_avg:,}" if ai_avg is not None else ""
#                                         if trip_term == "Long-term" and ai_avg is not None:
#                                             ai_caption_base = f"숏텀 $ {ai_avg:,} → 롱텀 $ {ai_display:,}"
#                                         ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
#                                         render_stat_card("AI 시장 추정 (시즌 보정)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
#                                     else: # key == "weighted"
#                                         weighted_caption = weight_caption
#                                         if trip_term == "Long-term" and final_allowance is not None:
#                                             weighted_caption = f"숏텀 $ {final_allowance:,} → 롱텀 $ {weighted_display:,} | {weight_caption}"
#                                         render_stat_card("가중 평균 결과", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

#                         # [신규 2] 비용 항목별 상세 내역 (show_ai_market_detail과 로직 통합)
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             st.markdown("<br>", unsafe_allow_html=True) # 줄 간격
                            
#                             mean_food = ai_summary.get("mean_food", 0)
#                             mean_trans = ai_summary.get("mean_transport", 0)
#                             mean_misc = ai_summary.get("mean_misc", 0)
                            
#                             # 롱텀/시즌 요율 적용
#                             food_display = round(mean_food * season_factor * trip_multiplier)
#                             trans_display = round(mean_trans * season_factor * trip_multiplier)
#                             misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
#                             st.write("###### AI 추정 상세 내역 (일비 기준)")
#                             col_f, col_t, col_m = st.columns(3)
#                             with col_f:
#                                 render_stat_card("예상 식비 (Food)", f"$ {food_display:,}", f"숏텀 기준: $ {round(mean_food)}", "muted")
#                             with col_t:
#                                 render_stat_card("예상 교통비 (Transport)", f"$ {trans_display:,}", f"숏텀 기준: $ {round(mean_trans)}", "muted")
#                             with col_m:
#                                 render_stat_card("예상 기타 (Misc)", f"$ {misc_display:,}", f"숏텀 기준: $ {round(mean_misc)}", "muted")
                        
#                         # [개선 3] show_weighted_result 카드가 중복되므로, 아래 블록은 제거
#                         # (기존 second_row_keys 로직 제거)

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_info,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("관리자가 세부 산출 근거를 숨겼습니다.")

# # --- [개선 2] admin_tab -> admin_analysis_tab 으로 변경 ---
# with admin_analysis_tab:
    
#     # [개선 2] ADMIN_ACCESS_CODE 로드 및 .env 체크
#     ACCESS_CODE_KEY = "admin_access_code_valid"
#     ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # .env에서 로드

#     if not ACCESS_CODE_VALUE:
#         st.error("보안 오류: .env 파일에 'ADMIN_ACCESS_CODE'가 설정되지 않았습니다. 앱을 중지하고 .env 파일을 설정해주세요.")
#         st.stop()
    
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#                 st.rerun() # [개선 3] 성공 시 새로고침
#             else:
#                 st.error("Access Code가 올바르지 않습니다.")
#                 st.stop() # [개선 3] 실패 시 중지
#         else:
#             st.stop() # [개선 3] 폼 제출 전 중지

#     # --- [개선 3] "보고서 버전 관리" 기능 (analysis_sub_tab) ---
#     st.subheader("보고서 버전 관리")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("활성 보고서 버전을 선택하세요:", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("생성된 보고서가 없습니다.")

#     # --- [신규 4] 과거 보고서 비교 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("과거 보고서 비교")
#     if len(history_files) < 2:
#         st.info("비교할 보고서가 2개 이상 필요합니다.")
#     else:
#         col_a, col_b = st.columns(2)
#         with col_a:
#             file_a = st.selectbox("기준 보고서 (A)", history_files, index=1, key="compare_a")
#         with col_b:
#             file_b = st.selectbox("비교 보고서 (B)", history_files, index=0, key="compare_b")
        
#         if st.button("보고서 비교하기"):
#             if file_a == file_b:
#                 st.warning("서로 다른 보고서를 선택해야 합니다.")
#             else:
#                 with st.spinner("보고서 비교 중..."):
#                     data_a = load_report_data(file_a)
#                     data_b = load_report_data(file_b)
                    
#                     if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
#                         df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
#                         df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
#                         df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
#                         report_a_label = file_a.split('report_')[-1].split('.')[0]
#                         report_b_label = file_b.split('report_')[-1].split('.')[0]

#                         df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
#                         df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
#                         df_merged["변동액 ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
#                         # 0으로 나누기 방지
#                         df_merged["변동률 (%)"] = (df_merged["변동액 ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
#                         st.dataframe(df_merged[[
#                             "city", "country_display", 
#                             f"A ({report_a_label})", 
#                             f"B ({report_b_label})", 
#                             "변동액 ($)", "변동률 (%)"
#                         ]].style.format({"변동률 (%)": "{:,.1f}%", "변동액 ($)": "{:,.0f}"}), width="stretch")
#                     else:
#                         st.error("보고서 파일을 불러오는 데 실패했습니다.")
    
#     # --- [개선 3] "UN-DSA (PDF) 분석" 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("UN-DSA (PDF) 분석 및 AI 실행")
#     st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요. (개선 1: 비동기 처리로 속도 향상)")
#     uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")

#     # --- [개선 1] 비동기 AI 분석 실행 로직 ---
#     if uploaded_file and st.button("AI 분석 실행", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
#         else:
#             st.session_state.latest_analysis_result = None
            
#             # --- 비동기 실행 함수 정의 ---
#             async def run_analysis(progress_bar, openai_api_key):
#                 progress_bar.progress(0, text="PDF 텍스트 추출 중...")
#                 full_text = parse_pdf_to_text(uploaded_file)
                
#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []
#                 analysis_failed = False
                
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break
                
#                 if analysis_failed:
#                     st.error("PDF->TSV 변환에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                 if not processed_data:
#                     st.error("TSV 데이터 처리에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 # 비동기 OpenAI 클라이언트 생성
#                 client = openai.AsyncOpenAI(api_key=openai_api_key)
                
#                 total_cities = len(processed_data["cities"])
#                 all_tasks = [] # 모든 AI 호출 작업을 담을 리스트

#                 # 1. 모든 도시에 대한 모든 AI 호출 작업을 미리 생성
#                 for city_data in processed_data["cities"]:
#                     city_name, country_name = city_data["city"], city_data["country_display"]
#                     city_context = {
#                         "neighborhood": city_data.get("neighborhood"),
#                         "hotel_cluster": city_data.get("hotel_cluster"),
#                     }
#                     season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                     menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
#                     city_data["menu_samples"] = menu_samples
#                     city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
#                     city_tasks = []
#                     for j in range(1, NUM_AI_CALLS + 1):
#                         task = get_market_data_from_ai_async(
#                             client, city_name, country_name, f"Run {j}",
#                             context=city_context, season_context=season_context, menu_samples=menu_samples
#                         )
#                         city_tasks.append(task)
                    
#                     all_tasks.append(city_tasks) # [ [도시1-10회], [도시2-10회], ... ]

#                 # 2. 모든 작업을 비동기로 실행하고 결과 수집
#                 city_index = 0
#                 for city_tasks in all_tasks:
#                     city_data = processed_data["cities"][city_index]
#                     city_name = city_data["city"]
#                     progress_text = f"AI 추정치 계산 중... ({city_index+1}/{total_cities}) {city_name}"
#                     progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
#                     # 해당 도시의 10개 작업을 동시에 실행
#                     try:
#                         market_results = await asyncio.gather(*city_tasks)
#                     except Exception as e:
#                         st.error(f"{city_name} 분석 중 비동기 오류: {e}")
#                         market_results = [] # 실패 처리

#                     # 3. 결과 처리
#                     ai_totals_source: List[int] = []
#                     ai_meta_runs: List[Dict[str, Any]] = []
                    
#                     # [신규 2] 비용 항목별 상세 내역을 위한 리스트
#                     ai_food: List[int] = []
#                     ai_transport: List[int] = []
#                     ai_misc: List[int] = []

#                     for j, market_result in enumerate(market_results, 1):
#                         city_data[f"market_data_{j}"] = market_result
#                         if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                             ai_totals_source.append(market_result["total"])
#                             # [신규 2] 상세 비용 추가
#                             ai_food.append(market_result.get("food", 0))
#                             ai_transport.append(market_result.get("transport", 0))
#                             ai_misc.append(market_result.get("misc", 0))
                        
#                         if "meta" in market_result:
#                             ai_meta_runs.append(market_result["meta"])
                    
#                     city_data["ai_provenance"] = ai_meta_runs

#                     # 4. 최종 수당 계산
#                     final_allowance = None
#                     un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
#                     un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

#                     ai_stats = aggregate_ai_totals(ai_totals_source)
#                     season_factor = (season_context or {}).get("factor", 1.0)
#                     ai_base_mean = ai_stats.get("mean_raw")
#                     ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
#                     # [신규 1] 동적 가중치 계산
#                     admin_weights = get_weight_config() # 관리자 설정 로드
#                     ai_vc_score = ai_stats.get("variation_coeff")
                    
#                     if un_per_diem is not None:
#                         weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
#                     else:
#                         # UN 데이터 없으면 AI 100%
#                         weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
#                     city_data["ai_summary"] = {
#                         "raw_totals": ai_totals_source,
#                         "used_totals": ai_stats.get("used_values", []),
#                         "removed_totals": ai_stats.get("removed_values", []),
#                         "mean_base": ai_base_mean,
#                         "mean_base_rounded": ai_stats.get("mean"),
                        
#                         "ai_consistency_vc": ai_vc_score, # [신규 1]
                        
#                         "mean_food": mean(ai_food) if ai_food else 0, # [신규 2]
#                         "mean_transport": mean(ai_transport) if ai_transport else 0, # [신규 2]
#                         "mean_misc": mean(ai_misc) if ai_misc else 0, # [신규 2]

#                         "season_factor": season_factor,
#                         "season_label": (season_context or {}).get("label"),
#                         "season_adjusted_mean_raw": ai_season_adjusted,
#                         "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                         "successful_runs": len(ai_stats.get("used_values", [])),
#                         "attempted_runs": NUM_AI_CALLS,
#                         "reference_links": city_data.get("reference_links", []),
#                         "weighted_average_components": {
#                             "un_per_diem": un_per_diem,
#                             "ai_season_adjusted": ai_season_adjusted,
#                             "weights": weights_cfg, # [신규 1] 동적 가중치 저장
#                         },
#                     }

#                     # [신규 1] 동적 가중치로 최종값 계산
#                     if un_per_diem is not None and ai_season_adjusted is not None:
#                         weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                         final_allowance = round(weighted_average)
#                     elif un_per_diem is not None:
#                         final_allowance = round(un_per_diem)
#                     elif ai_season_adjusted is not None:
#                         final_allowance = round(ai_season_adjusted)

#                     city_data["final_allowance"] = final_allowance

#                     if final_allowance and un_per_diem and un_per_diem > 0:
#                         city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                     else:
#                         city_data["delta_vs_un_pct"] = "N/A"
                    
#                     city_index += 1 # 다음 도시로

#                 save_report_data(processed_data)
#                 st.session_state.latest_analysis_result = processed_data
#                 st.success("AI analysis completed.")
#                 progress_bar.empty()
#                 st.rerun()
            
#             # --- 비동기 실행 ---
#             with st.spinner("PDF 처리 및 AI 분석을 실행합니다. (약 10~30초 소요)"):
#                 progress_bar = st.progress(0, text="분석 시작...")
#                 asyncio.run(run_analysis(progress_bar, openai_api_key))

#     # --- [개선 3] "Latest Analysis Summary" 기능 (analysis_sub_tab) ---
#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             # --- [HOTFIX] ArrowInvalid Error 방지 ---
#             delta_val = city.get('delta_vs_un_pct')
#             if isinstance(delta_val, (int, float)):
#                 delta_display = f"{delta_val:.0f}%" # 숫자를 "12%" 형태의 문자열로 변경
#             else:
#                 delta_display = "N/A" # 이미 "N/A" 문자열
#             # --- [HOTFIX] End ---
                
#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': delta_display, # <-- 수정된 문자열 값 사용
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- use_container_width 추가 (필요시 width='stretch'로 변경)
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# # --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
# # --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
# with admin_config_tab:
#     # 암호 확인 (필수)
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         st.error("Access Code가 필요합니다. '보고서 분석 (Admin)' 탭에서 먼저 로그인해주세요.")
#         st.stop()
        
#     # --- [패치 v17.1] 도시 편집/캐시 관리가 공유할 도시 목록을 탭 상단에서 정의 ---
#     current_entries = get_target_city_entries()
#     options = {
#         f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#         for idx, entry in enumerate(current_entries)
#     }
#     sorted_labels = list(options.keys())
    
#     # --- 콜백 함수 1: '도시 편집' 폼 동기화 ---
#     def _sync_edit_form_from_selection():
#         if "edit_city_selector" not in st.session_state:
#              st.session_state.edit_city_selector = sorted_labels[0]
             
#         selected_idx = options[st.session_state.edit_city_selector]
#         selected_entry = current_entries[selected_idx]
        
#         st.session_state.edit_region = selected_entry.get("region", "")
#         st.session_state.edit_city = selected_entry.get("city", "")
#         st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#         st.session_state.edit_country = selected_entry.get("country", "")
#         st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
        
#         existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#         st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
        
#         sub_data = selected_entry.get("un_dsa_substitute") or {}
#         st.session_state.edit_sub_city = sub_data.get("city", "")
#         st.session_state.edit_sub_country = sub_data.get("country", "")

#     # --- [패치 v17.1] 콜백 함수 2: '캐시 추가' 폼 동기화 ---
#     def _sync_cache_form_from_selection():
#         selected_label = st.session_state.get("cache_city_selector") # get()으로 오류 방지
        
#         if selected_label in options: # 'options' dict를 공유
#             selected_idx = options[selected_label]
#             selected_entry = current_entries[selected_idx]
#             st.session_state.new_cache_country = selected_entry.get("country", "")
#             st.session_state.new_cache_city = selected_entry.get("city", "")
#             st.session_state.new_cache_neighborhood = selected_entry.get("neighborhood", "")
#         else: # (placeholder 선택 시)
#             st.session_state.new_cache_country = ""
#             st.session_state.new_cache_city = ""
#             st.session_state.new_cache_neighborhood = ""
        
#         # 나머지 필드는 항상 기본값으로 초기화
#         st.session_state.new_cache_vendor = ""
#         st.session_state.new_cache_category = "Food"
#         st.session_state.new_cache_price = 0.0
#         st.session_state.new_cache_currency = "USD"
#         st.session_state.new_cache_url = ""
#     # --- [패치 v17.1] 끝 ---

#     st.subheader("직원용 탭 노출")
#     visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key 이름 변경
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.session_state.employee_tab_visibility = visibility_toggle # 세션 상태에도 반영
#         st.success("직원용 탭 노출 상태가 업데이트되었습니다. (새로고침 시 적용)")
#         time.sleep(1) # 유저가 메시지를 읽을 시간을 줌
#         st.rerun()

#     st.subheader("직원 화면 노출 설정")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("직원 화면 노출 설정이 업데이트되었습니다.")
#         time.sleep(1)
#         st.rerun()

#     st.divider()
#     st.subheader("비중 설정 (기본값)")
#     st.info("이제 이 설정은 '동적 가중치' 로직의 기본값으로 사용됩니다. AI 응답이 불안정하면 자동으로 AI 비중이 낮아집니다.")
#     current_weights = get_weight_config()
#     st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
#         st.rerun()

#     st.divider()
#     st.header("목표 도시 관리 (target_cities_config.json)")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         # trip_lengths를 보기 쉽게 문자열로 변환
#         entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], width='stretch')
#     else:
#         st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("신규 도시 추가")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["기타 (직접 입력)"]
#             region_choice = st.selectbox("지역", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "기타 (직접 입력)":
#                 new_region = st.text_input("새 지역 이름", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("도시", key="add_city")
#             neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("국가", key="add_country")
#             hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

#         with st.expander("UN-DSA 대체 도시 (선택)"):
#             substitute_city = st.text_input("대체 도시", key="add_sub_city")
#             substitute_country = st.text_input("대체 국가", key="add_sub_country")

#         add_submitted = st.form_submit_button("추가")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("동일한 항목이 이미 등록되어 있습니다.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
#                 st.rerun()

#     st.subheader("기존 도시 편집/삭제")
#     # current_entries = get_target_city_entries() # 탭 상단으로 이동
    
#     if current_entries:
#         # options = { ... } # 탭 상단으로 이동
#         # sorted_labels = list(options.keys()) # 탭 상단으로 이동
#         # def _sync_edit_form_from_selection(): # 탭 상단으로 이동

#         # 드롭다운(Selectbox)에 on_change 콜백 연결
#         selected_label = st.selectbox(
#             "편집할 도시를 선택하세요", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection
#         )

#         # 페이지 첫 로드 시 폼을 채우기 위한 초기화
#         if "edit_region" not in st.session_state:
#             _sync_edit_form_from_selection()

#         # 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 region_edit = st.text_input("지역", key="edit_region")
#                 city_edit = st.text_input("도시", key="edit_city")
#                 neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
#             with col_f:
#                 country_edit = st.text_input("국가", key="edit_country")
#                 hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

#             trip_lengths_edit = st.multiselect(
#                 "출장 기간",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", 
#             )

#             with st.expander("UN-DSA 대체 도시 (선택)"):
#                 sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
#                 sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("변경사항 저장")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("삭제", type="secondary")

#         # 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
#         if update_btn:
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#             else:
#                 selected_idx = options[st.session_state.edit_city_selector]
#                 current_entries[selected_idx] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[selected_idx]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[selected_idx].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("수정을 완료했습니다.")
#                 st.rerun()
        
#         if delete_btn:
#             selected_idx = options[st.session_state.edit_city_selector]
#             del current_entries[selected_idx]
#             set_target_city_entries(current_entries)
#             st.warning("선택한 항목을 삭제했습니다.")
#             st.rerun()
#     else:
#         st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

#     # --- [신규 3] '데이터 캐시 관리' UI 추가 ---
#     st.divider()
#     st.header("데이터 캐시 관리 (Menu Cache)")

#     if not MENU_CACHE_ENABLED:
#         st.error("`data_sources/menu_cache.py` 파일 로드에 실패하여 이 기능을 사용할 수 없습니다.")
#     else:
#         st.info("AI가 도시 물가 추정 시 참고할 실제 메뉴/가격 데이터를 관리합니다. (AI 분석 정확도 향상)")

#         # 1. 새 캐시 항목 추가 폼
#         st.subheader("신규 캐시 항목 추가")
        
#         # --- [패치 v17.1] 도시 자동 완성을 위한 드롭다운 추가 ---
#         st.selectbox(
#             "도시 선택 (자동 채우기):", 
#             sorted_labels,  # 탭 상단에서 정의한 변수
#             key="cache_city_selector",
#             on_change=_sync_cache_form_from_selection, # 새로 만든 콜백
#             index=None,
#             placeholder="도시를 선택하면 국가, 도시, 세부 지역이 자동 입력됩니다."
#         )

#         # 페이지 첫 로드 시 캐시 폼 초기화
#         if "new_cache_country" not in st.session_state:
#             _sync_cache_form_from_selection() # 빈 값으로 초기화
#         # --- [패치 v17.1] 끝 ---
        
#         with st.form("add_menu_cache_form", clear_on_submit=True):
#             st.write("AI 분석에 사용할 참고 가격 정보를 입력합니다. (예: 레스토랑 메뉴, 택시비 고지 등)")
#             c1, c2 = st.columns(2)
#             with c1:
#                 # --- [패치 v17.1] 모든 입력 위젯에 key=... 추가 ---
#                 new_cache_country = st.text_input("국가 (Country)", key="new_cache_country", help="예: Philippines")
#                 new_cache_city = st.text_input("도시 (City)", key="new_cache_city", help="예: Manila")
#                 new_cache_neighborhood = st.text_input("세부 지역 (Neighborhood) (선택)", key="new_cache_neighborhood", help="예: Makati (비워두면 도시 전체에 적용)")
#                 new_cache_vendor = st.text_input("장소/상품명 (Vendor)", key="new_cache_vendor", help="예: Jollibee (C3, Ayala Ave)")
#             with c2:
#                 new_cache_category = st.selectbox("카테고리 (Category)", ["Food", "Transport", "Misc"], key="new_cache_category")
#                 new_cache_price = st.number_input("가격 (Price)", min_value=0.0, step=0.01, key="new_cache_price")
#                 new_cache_currency = st.text_input("통화 (Currency)", value="USD", key="new_cache_currency", help="예: PHP, USD")
#                 new_cache_url = st.text_input("출처 URL (Source URL) (선택)", key="new_cache_url")
            
#             add_cache_submitted = st.form_submit_button("신규 캐시 항목 저장")

#             if add_cache_submitted:
#                 # --- [패치 v17.1] 로컬 변수 대신 st.session_state에서 값을 읽어옴 ---
#                 if (not st.session_state.new_cache_country or 
#                     not st.session_state.new_cache_city or 
#                     not st.session_state.new_cache_vendor):
#                     st.error("국가, 도시, 장소/상품명은 필수입니다.")
#                 else:
#                     new_entry = {
#                         "country": st.session_state.new_cache_country.strip(),
#                         "city": st.session_state.new_cache_city.strip(),
#                         "neighborhood": st.session_state.new_cache_neighborhood.strip(),
#                         "vendor": st.session_state.new_cache_vendor.strip(),
#                         "category": st.session_state.new_cache_category,
#                         "price": st.session_state.new_cache_price,
#                         "currency": st.session_state.new_cache_currency.strip().upper(),
#                         "url": st.session_state.new_cache_url.strip(),
#                     }
                    
#                     # menu_cache.py의 함수를 호출하여 항목 추가
#                     if add_menu_cache_entry(new_entry):
#                         st.success(f"'{new_entry['vendor']}' 항목을 캐시에 추가했습니다.")
#                         # 폼이 clear_on_submit=True이므로 폼 값을 다시 초기화
#                         _sync_cache_form_from_selection()
#                         st.rerun()
#                     else:
#                         st.error("캐시 항목 추가에 실패했습니다.")

#         # 2. 기존 캐시 항목 조회 및 삭제
#         st.subheader("기존 캐시 항목 조회 및 삭제")
#         all_cache_data = load_all_cache() # menu_cache.py의 함수
        
#         if not all_cache_data:
#             st.info("현재 저장된 캐시 데이터가 없습니다.")
#         else:
#             df_cache = pd.DataFrame(all_cache_data)
#             st.dataframe(df_cache[[
#                 "country", "city", "neighborhood", "vendor", 
#                 "category", "price", "currency", "last_updated", "url"
#             ]], width='stretch') # [HOTFIX] width='stretch' 적용

#             # 삭제 기능
#             st.markdown("---")
#             st.write("##### 캐시 항목 삭제")
            
#             # 삭제할 항목을 식별할 수 있는 고유한 레이블 생성 (최신 항목이 위로)
#             delete_options_map = {
#                 f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
#                 for idx, entry in enumerate(reversed(all_cache_data)) # reversed()로 최신 항목이 먼저 보이게
#             }
#             delete_labels = list(delete_options_map.keys())
            
#             label_to_delete = st.selectbox("삭제할 캐시 항목을 선택하세요:", delete_labels, index=None, placeholder="삭제할 항목 선택...")
            
#             if label_to_delete and st.button(f"'{label_to_delete}' 항목 삭제", type="primary"):
#                 # 거꾸로 매핑된 인덱스를 실제 인덱스로 변환
#                 original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
#                 entry_to_delete = all_cache_data.pop(original_list_index)
                
#                 # menu_cache.py의 함수를 호출하여 전체 파일 저장
#                 if save_cached_menu_prices(all_cache_data):
#                     st.success(f"'{entry_to_delete.get('vendor')}' 항목을 삭제했습니다.")
#                     st.rerun()
#                 else:
#                     st.error("캐시 삭제에 실패했습니다.")
    
#     # --- [신규 3] UI 끝 ---

# 202511-05
# # 2025-10-20-16 AI 기반 출장비 계산 도구 (v16.0 - Async, Dynamic Weights, Full Admin)
# # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# # 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
# from typing import Any, Dict, List, Optional, Set, Tuple

# # [신규 3] menu_cache 임포트 (파일이 없으면 이 기능은 작동하지 않음)
# try:
#     from data_sources.menu_cache import (
#         load_cached_menu_prices, 
#         load_all_cache, 
#         add_menu_cache_entry, 
#         save_cached_menu_prices
#     )
#     MENU_CACHE_ENABLED = True
# except ImportError:
#     st.warning("`data_sources/menu_cache.py` 파일을 찾을 수 없습니다. '데이터 캐시 관리' 기능이 비활성화됩니다.")
#     # (기존 함수들을 임시로 정의)
#     def load_cached_menu_prices(city: str, country: str, neighborhood: Optional[str]) -> List[Dict[str, Any]]: return []
#     def load_all_cache() -> List[Dict[str, Any]]: return []
#     def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool: return False
#     def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool: return False
#     MENU_CACHE_ENABLED = False


# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
#         "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
#     },
#     "secondary": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
#     "muted": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
#     # [수정] 캡션에 스타일 적용
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">{level_label} 기준 예상 일비 총액</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>계산식</span><br/>
#             $ {daily:,} × {days}일 일정 × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6)": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "숏텀", "Long-term": "롱텀"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# # --- [신규 1] aggregate_ai_totals 함수 수정 ---
# # (이상치 제거 + 변동계수(VC) 계산)
# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
    
#     # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
#     variation_coeff = None
#     if filtered and computed_mean and computed_mean > 0:
#         if len(filtered) > 1:
#             try:
#                 computed_stdev = stdev(filtered)
#                 variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
#             except StatisticsError:
#                 variation_coeff = 0.0 # 모든 값이 동일
#         else:
#             variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#         "variation_coeff": variation_coeff # <-- AI 일관성 점수
#     }

# # --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
# def get_dynamic_weights(
#     variation_coeff: Optional[float], 
#     admin_weights: Dict[str, float]
# ) -> Dict[str, Any]:
#     """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
#     # 관리자 설정값을 기본값으로 사용
#     base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
#     if variation_coeff is None:
#         # AI 데이터가 없으면 UN 100%
#         return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
#     if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
#         # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
#         dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
#         source = f"High AI Consistency (VC: {variation_coeff:.2f})"
#     elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
#         # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
#         dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
#         source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
#     else:
#         # 5% ~ 15% 사이: 관리자 설정값 유지
#         dynamic_ai_weight = base_ai_weight
#         source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

#     final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
#     final_un_weight = 1.0 - final_ai_weight
    
#     return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# # --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
# async def get_market_data_from_ai_async(
#     client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
#     city: str,
#     country: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     # --- (내부 헬퍼 함수들은 기존과 동일) ---
#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")
#     # --- (프롬프트 구성은 기존과 동일) ---
#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         # --- [수정] 비동기 API 호출로 변경 ---
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         # --- [수정] 끝 ---
        
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }
# # --- [개선 1] 끝 ---

# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
#             # [신규 1] 동적 가중치 적용 사유
#             weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#                 'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- 스트림릿 UI 구성 ---
# st.set_page_config(layout="wide")
# st.title("AICP: 출장 일비 계산 & 조회 시스템 (v16.0 - Async & Dynamic)")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility


# # --- [개선 3] 탭 구조 변경 ---
# tab_definitions = []
# if employee_tab_visible:
#     tab_definitions.append("💵 일비 조회 (직원용)")

# # 관리자 탭을 2개로 분리
# tab_definitions.append("📈 보고서 분석 (Admin)")
# tab_definitions.append("🛠️ 시스템 설정 (Admin)")

# tabs = st.tabs(tab_definitions)

# employee_tab = None
# admin_analysis_tab = None
# admin_config_tab = None

# if employee_tab_visible:
#     employee_tab = tabs[0]
#     admin_analysis_tab = tabs[1]
#     admin_config_tab = tabs[2]
# else:
#     admin_analysis_tab = tabs[0]
#     admin_config_tab = tabs[1]
# # --- [개선 3] 끝 ---


# if employee_tab is not None:
#     with employee_tab:
#         st.header("도시별 출장 일비 조회")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("도시:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("선택한 국가에 등록된 도시가 없습니다.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "출장 시작일",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "출장 종료일",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("종료일은 시작일 이후여야 합니다.")
#                     trip_days = 0 # 0으로 설정
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"자동 분류된 출장 유형: {trip_term_label} · {trip_days}일 일정")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("이 기간에 해당하는 도시 데이터가 없습니다. 출장 유형을 '숏텀'으로 조정하거나 도시 설정을 확인하세요.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level and trip_days is not None:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} 결과")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
                        
#                         # [신규 2] 직원 탭 총액 카드
#                         render_primary_summary(
#                             f"{sel_level.split(' ')[0]}",
#                             trip_total_allowance,
#                             level_daily_allowance,
#                             trip_days,
#                             trip_term_label,
#                             trip_multiplier
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} 일일 권장 일비", "금액 없음")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**세부 산출 근거 (일비 기준)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"제외값 {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"
                        
#                         # [신규 1] 동적 가중치 적용 사유
#                         weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
#                         weights_source = weights_info.get("source", "N/A")
#                         un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
#                         ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
#                         weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | 사유: {weights_source}"

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         un_caption = f"숏텀 기준 $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
#                                         if trip_term == "Long-term" and un_base is not None:
#                                             un_caption = f"숏텀 $ {un_base:,} → 롱텀 $ {un_display:,}"
#                                         render_stat_card("UN-DSA 기준", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
#                                     elif key == "ai":
#                                         ai_caption_base = f"숏텀 기준 $ {ai_avg:,}" if ai_avg is not None else ""
#                                         if trip_term == "Long-term" and ai_avg is not None:
#                                             ai_caption_base = f"숏텀 $ {ai_avg:,} → 롱텀 $ {ai_display:,}"
#                                         ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
#                                         render_stat_card("AI 시장 추정 (시즌 보정)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
#                                     else: # key == "weighted"
#                                         weighted_caption = weight_caption
#                                         if trip_term == "Long-term" and final_allowance is not None:
#                                             weighted_caption = f"숏텀 $ {final_allowance:,} → 롱텀 $ {weighted_display:,} | {weight_caption}"
#                                         render_stat_card("가중 평균 결과", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

#                         # [신규 2] 비용 항목별 상세 내역 (show_ai_market_detail과 로직 통합)
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             st.markdown("<br>", unsafe_allow_html=True) # 줄 간격
                            
#                             mean_food = ai_summary.get("mean_food", 0)
#                             mean_trans = ai_summary.get("mean_transport", 0)
#                             mean_misc = ai_summary.get("mean_misc", 0)
                            
#                             # 롱텀/시즌 요율 적용
#                             food_display = round(mean_food * season_factor * trip_multiplier)
#                             trans_display = round(mean_trans * season_factor * trip_multiplier)
#                             misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
#                             st.write("###### AI 추정 상세 내역 (일비 기준)")
#                             col_f, col_t, col_m = st.columns(3)
#                             with col_f:
#                                 render_stat_card("예상 식비 (Food)", f"$ {food_display:,}", f"숏텀 기준: $ {round(mean_food)}", "muted")
#                             with col_t:
#                                 render_stat_card("예상 교통비 (Transport)", f"$ {trans_display:,}", f"숏텀 기준: $ {round(mean_trans)}", "muted")
#                             with col_m:
#                                 render_stat_card("예상 기타 (Misc)", f"$ {misc_display:,}", f"숏텀 기준: $ {round(mean_misc)}", "muted")
                        
#                         # [개선 3] show_weighted_result 카드가 중복되므로, 아래 블록은 제거
#                         # (기존 second_row_keys 로직 제거)

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_info,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("관리자가 세부 산출 근거를 숨겼습니다.")

# # --- [개선 2] admin_tab -> admin_analysis_tab 으로 변경 ---
# with admin_analysis_tab:
    
#     # [개선 2] ADMIN_ACCESS_CODE 로드 및 .env 체크
#     ACCESS_CODE_KEY = "admin_access_code_valid"
#     ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # .env에서 로드

#     if not ACCESS_CODE_VALUE:
#         st.error("보안 오류: .env 파일에 'ADMIN_ACCESS_CODE'가 설정되지 않았습니다. 앱을 중지하고 .env 파일을 설정해주세요.")
#         st.stop()
    
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#                 st.rerun() # [개선 3] 성공 시 새로고침
#             else:
#                 st.error("Access Code가 올바르지 않습니다.")
#                 st.stop() # [개선 3] 실패 시 중지
#         else:
#             st.stop() # [개선 3] 폼 제출 전 중지

#     # --- [개선 3] "보고서 버전 관리" 기능 (analysis_sub_tab) ---
#     st.subheader("보고서 버전 관리")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("활성 보고서 버전을 선택하세요:", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("생성된 보고서가 없습니다.")

#     # --- [신규 4] 과거 보고서 비교 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("과거 보고서 비교")
#     if len(history_files) < 2:
#         st.info("비교할 보고서가 2개 이상 필요합니다.")
#     else:
#         col_a, col_b = st.columns(2)
#         with col_a:
#             file_a = st.selectbox("기준 보고서 (A)", history_files, index=1, key="compare_a")
#         with col_b:
#             file_b = st.selectbox("비교 보고서 (B)", history_files, index=0, key="compare_b")
        
#         if st.button("보고서 비교하기"):
#             if file_a == file_b:
#                 st.warning("서로 다른 보고서를 선택해야 합니다.")
#             else:
#                 with st.spinner("보고서 비교 중..."):
#                     data_a = load_report_data(file_a)
#                     data_b = load_report_data(file_b)
                    
#                     if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
#                         df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
#                         df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
#                         df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
#                         report_a_label = file_a.split('report_')[-1].split('.')[0]
#                         report_b_label = file_b.split('report_')[-1].split('.')[0]

#                         df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
#                         df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
#                         df_merged["변동액 ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
#                         # 0으로 나누기 방지
#                         df_merged["변동률 (%)"] = (df_merged["변동액 ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
#                         st.dataframe(df_merged[[
#                             "city", "country_display", 
#                             f"A ({report_a_label})", 
#                             f"B ({report_b_label})", 
#                             "변동액 ($)", "변동률 (%)"
#                         ]].style.format({"변동률 (%)": "{:,.1f}%", "변동액 ($)": "{:,.0f}"}), width="stretch")
#                     else:
#                         st.error("보고서 파일을 불러오는 데 실패했습니다.")
    
#     # --- [개선 3] "UN-DSA (PDF) 분석" 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("UN-DSA (PDF) 분석 및 AI 실행")
#     st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요. (개선 1: 비동기 처리로 속도 향상)")
#     uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")

#     # --- [개선 1] 비동기 AI 분석 실행 로직 ---
#     if uploaded_file and st.button("AI 분석 실행", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
#         else:
#             st.session_state.latest_analysis_result = None
            
#             # --- 비동기 실행 함수 정의 ---
#             async def run_analysis(progress_bar, openai_api_key):
#                 progress_bar.progress(0, text="PDF 텍스트 추출 중...")
#                 full_text = parse_pdf_to_text(uploaded_file)
                
#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []
#                 analysis_failed = False
                
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break
                
#                 if analysis_failed:
#                     st.error("PDF->TSV 변환에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                 if not processed_data:
#                     st.error("TSV 데이터 처리에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 # 비동기 OpenAI 클라이언트 생성
#                 client = openai.AsyncOpenAI(api_key=openai_api_key)
                
#                 total_cities = len(processed_data["cities"])
#                 all_tasks = [] # 모든 AI 호출 작업을 담을 리스트

#                 # 1. 모든 도시에 대한 모든 AI 호출 작업을 미리 생성
#                 for city_data in processed_data["cities"]:
#                     city_name, country_name = city_data["city"], city_data["country_display"]
#                     city_context = {
#                         "neighborhood": city_data.get("neighborhood"),
#                         "hotel_cluster": city_data.get("hotel_cluster"),
#                     }
#                     season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                     menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
#                     city_data["menu_samples"] = menu_samples
#                     city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
#                     city_tasks = []
#                     for j in range(1, NUM_AI_CALLS + 1):
#                         task = get_market_data_from_ai_async(
#                             client, city_name, country_name, f"Run {j}",
#                             context=city_context, season_context=season_context, menu_samples=menu_samples
#                         )
#                         city_tasks.append(task)
                    
#                     all_tasks.append(city_tasks) # [ [도시1-10회], [도시2-10회], ... ]

#                 # 2. 모든 작업을 비동기로 실행하고 결과 수집
#                 city_index = 0
#                 for city_tasks in all_tasks:
#                     city_data = processed_data["cities"][city_index]
#                     city_name = city_data["city"]
#                     progress_text = f"AI 추정치 계산 중... ({city_index+1}/{total_cities}) {city_name}"
#                     progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
#                     # 해당 도시의 10개 작업을 동시에 실행
#                     try:
#                         market_results = await asyncio.gather(*city_tasks)
#                     except Exception as e:
#                         st.error(f"{city_name} 분석 중 비동기 오류: {e}")
#                         market_results = [] # 실패 처리

#                     # 3. 결과 처리
#                     ai_totals_source: List[int] = []
#                     ai_meta_runs: List[Dict[str, Any]] = []
                    
#                     # [신규 2] 비용 항목별 상세 내역을 위한 리스트
#                     ai_food: List[int] = []
#                     ai_transport: List[int] = []
#                     ai_misc: List[int] = []

#                     for j, market_result in enumerate(market_results, 1):
#                         city_data[f"market_data_{j}"] = market_result
#                         if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                             ai_totals_source.append(market_result["total"])
#                             # [신규 2] 상세 비용 추가
#                             ai_food.append(market_result.get("food", 0))
#                             ai_transport.append(market_result.get("transport", 0))
#                             ai_misc.append(market_result.get("misc", 0))
                        
#                         if "meta" in market_result:
#                             ai_meta_runs.append(market_result["meta"])
                    
#                     city_data["ai_provenance"] = ai_meta_runs

#                     # 4. 최종 수당 계산
#                     final_allowance = None
#                     un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
#                     un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

#                     ai_stats = aggregate_ai_totals(ai_totals_source)
#                     season_factor = (season_context or {}).get("factor", 1.0)
#                     ai_base_mean = ai_stats.get("mean_raw")
#                     ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
#                     # [신규 1] 동적 가중치 계산
#                     admin_weights = get_weight_config() # 관리자 설정 로드
#                     ai_vc_score = ai_stats.get("variation_coeff")
                    
#                     if un_per_diem is not None:
#                         weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
#                     else:
#                         # UN 데이터 없으면 AI 100%
#                         weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
#                     city_data["ai_summary"] = {
#                         "raw_totals": ai_totals_source,
#                         "used_totals": ai_stats.get("used_values", []),
#                         "removed_totals": ai_stats.get("removed_values", []),
#                         "mean_base": ai_base_mean,
#                         "mean_base_rounded": ai_stats.get("mean"),
                        
#                         "ai_consistency_vc": ai_vc_score, # [신규 1]
                        
#                         "mean_food": mean(ai_food) if ai_food else 0, # [신규 2]
#                         "mean_transport": mean(ai_transport) if ai_transport else 0, # [신규 2]
#                         "mean_misc": mean(ai_misc) if ai_misc else 0, # [신규 2]

#                         "season_factor": season_factor,
#                         "season_label": (season_context or {}).get("label"),
#                         "season_adjusted_mean_raw": ai_season_adjusted,
#                         "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                         "successful_runs": len(ai_stats.get("used_values", [])),
#                         "attempted_runs": NUM_AI_CALLS,
#                         "reference_links": city_data.get("reference_links", []),
#                         "weighted_average_components": {
#                             "un_per_diem": un_per_diem,
#                             "ai_season_adjusted": ai_season_adjusted,
#                             "weights": weights_cfg, # [신규 1] 동적 가중치 저장
#                         },
#                     }

#                     # [신규 1] 동적 가중치로 최종값 계산
#                     if un_per_diem is not None and ai_season_adjusted is not None:
#                         weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                         final_allowance = round(weighted_average)
#                     elif un_per_diem is not None:
#                         final_allowance = round(un_per_diem)
#                     elif ai_season_adjusted is not None:
#                         final_allowance = round(ai_season_adjusted)

#                     city_data["final_allowance"] = final_allowance

#                     if final_allowance and un_per_diem and un_per_diem > 0:
#                         city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                     else:
#                         city_data["delta_vs_un_pct"] = "N/A"
                    
#                     city_index += 1 # 다음 도시로

#                 save_report_data(processed_data)
#                 st.session_state.latest_analysis_result = processed_data
#                 st.success("AI analysis completed.")
#                 progress_bar.empty()
#                 st.rerun()
            
#             # --- 비동기 실행 ---
#             with st.spinner("PDF 처리 및 AI 분석을 실행합니다. (약 10~30초 소요)"):
#                 progress_bar = st.progress(0, text="분석 시작...")
#                 asyncio.run(run_analysis(progress_bar, openai_api_key))

#     # --- [개선 3] "Latest Analysis Summary" 기능 (analysis_sub_tab) ---
#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             # --- [HOTFIX] ArrowInvalid Error 방지 ---
#             delta_val = city.get('delta_vs_un_pct')
#             if isinstance(delta_val, (int, float)):
#                 delta_display = f"{delta_val:.0f}%" # 숫자를 "12%" 형태의 문자열로 변경
#             else:
#                 delta_display = "N/A" # 이미 "N/A" 문자열
#             # --- [HOTFIX] End ---
                
#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': delta_display, # <-- 수정된 문자열 값 사용
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- use_container_width 추가 (필요시 width='stretch'로 변경)
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# # --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
# with admin_config_tab:
#     # 암호 확인 (필수)
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         st.error("Access Code가 필요합니다.")
#         st.stop()
        
#     st.subheader("직원용 탭 노출")
#     visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key 이름 변경
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.session_state.employee_tab_visibility = visibility_toggle # 세션 상태에도 반영
#         st.success("직원용 탭 노출 상태가 업데이트되었습니다. (새로고침 시 적용)")

#     st.subheader("직원 화면 노출 설정")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("직원 화면 노출 설정이 업데이트되었습니다.")

#     st.divider()
#     st.subheader("비중 설정 (기본값)")
#     st.info("이제 이 설정은 '동적 가중치' 로직의 기본값으로 사용됩니다. AI 응답이 불안정하면 자동으로 AI 비중이 낮아집니다.")
#     current_weights = get_weight_config()
#     st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
#         st.rerun()

#     st.divider()
#     st.header("목표 도시 관리")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         # trip_lengths를 보기 쉽게 문자열로 변환
#         entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], use_container_width=True)
#     else:
#         st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("신규 도시 추가")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["기타 (직접 입력)"]
#             region_choice = st.selectbox("지역", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "기타 (직접 입력)":
#                 new_region = st.text_input("새 지역 이름", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("도시", key="add_city")
#             neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("국가", key="add_country")
#             hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

#         with st.expander("UN-DSA 대체 도시 (선택)"):
#             substitute_city = st.text_input("대체 도시", key="add_sub_city")
#             substitute_country = st.text_input("대체 국가", key="add_sub_country")

#         add_submitted = st.form_submit_button("추가")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("동일한 항목이 이미 등록되어 있습니다.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
#                 st.rerun()

#     st.subheader("기존 도시 편집/삭제")
#     current_entries = get_target_city_entries()
    
#     if current_entries:
#         # 1. 드롭다운 옵션 구성
#         options = {
#             f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#             for idx, entry in enumerate(current_entries)
#         }
#         sorted_labels = list(options.keys())

#         # 2. on_change 콜백 함수 정의 (위젯보다 먼저)
#         def _sync_edit_form_from_selection():
#             # 드롭다운에서 현재 선택된 값을 가져옴
#             if "edit_city_selector" not in st.session_state:
#                  st.session_state.edit_city_selector = sorted_labels[0]
                 
#             selected_idx = options[st.session_state.edit_city_selector]
#             selected_entry = current_entries[selected_idx]
            
#             # session_state의 값을 선택된 도시의 데이터로 강제 업데이트
#             st.session_state.edit_region = selected_entry.get("region", "")
#             st.session_state.edit_city = selected_entry.get("city", "")
#             st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#             st.session_state.edit_country = selected_entry.get("country", "")
#             st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
            
#             # 출장 기간 (trip_lengths) 설정
#             existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#             st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
            
#             # UN-DSA 대체 도시 설정
#             sub_data = selected_entry.get("un_dsa_substitute") or {}
#             st.session_state.edit_sub_city = sub_data.get("city", "")
#             st.session_state.edit_sub_country = sub_data.get("country", "")

#         # 3. 드롭다운(Selectbox)에 on_change 콜백 연결
#         selected_label = st.selectbox(
#             "편집할 도시를 선택하세요", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection  # <-- [수정] 콜백 함수 연결
#         )

#         # 4. 페이지 첫 로드 시 폼을 채우기 위한 초기화
#         if "edit_region" not in st.session_state:
#             # 첫 로드 시, selectbox의 기본값(첫 번째 항목)에 맞춰 폼을 채움
#             _sync_edit_form_from_selection()

#         # 5. 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 # [수정] value=... 제거
#                 region_edit = st.text_input("지역", key="edit_region")
#                 city_edit = st.text_input("도시", key="edit_city")
#                 neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
#             with col_f:
#                 # [수정] value=... 제거
#                 country_edit = st.text_input("국가", key="edit_country")
#                 hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

#             # [수정] default=... 대신 key=... 사용
#             trip_lengths_edit = st.multiselect(
#                 "출장 기간",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", # 'default' 대신 'key'로 상태 관리
#             )

#             with st.expander("UN-DSA 대체 도시 (선택)"):
#                 # [수정] value=... 제거
#                 sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
#                 sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("변경사항 저장")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("삭제", type="secondary")

#         # 6. 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
#         if update_btn:
#             # [수정] 위젯 변수(region_edit) 대신 st.session_state에서 직접 값을 읽음
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#             else:
#                 current_entries[options[selected_label]] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[options[selected_label]]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[options[selected_label]].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("수정을 완료했습니다.")
#                 st.rerun()  # <-- [확인] 이미 올바르게 수정되어 있음
        
#         if delete_btn:
#             del current_entries[options[selected_label]]
#             set_target_city_entries(current_entries)
#             st.warning("선택한 항목을 삭제했습니다.")
#             st.rerun() # <-- [확인] 이미 올바르게 수정되어 있음

#     else:
#         st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

#     # --- [신규 3] '데이터 캐시 관리' UI 추가 ---
#     st.divider()
#     st.header("데이터 캐시 관리 (Menu Cache)")

#     if not MENU_CACHE_ENABLED:
#         st.error("`data_sources/menu_cache.py` 파일 로드에 실패하여 이 기능을 사용할 수 없습니다.")
#     else:
#         st.info("AI가 도시 물가 추정 시 참고할 실제 메뉴/가격 데이터를 관리합니다. (AI 분석 정확도 향상)")

#         # 1. 새 캐시 항목 추가 폼
#         st.subheader("신규 캐시 항목 추가")
#         with st.form("add_menu_cache_form", clear_on_submit=True):
#             st.write("AI 분석에 사용할 참고 가격 정보를 입력합니다. (예: 레스토랑 메뉴, 택시비 고지 등)")
#             c1, c2 = st.columns(2)
#             with c1:
#                 new_cache_country = st.text_input("국가 (Country)", help="예: Philippines")
#                 new_cache_city = st.text_input("도시 (City)", help="예: Manila")
#                 new_cache_neighborhood = st.text_input("세부 지역 (Neighborhood) (선택)", help="예: Makati (비워두면 도시 전체에 적용)")
#                 new_cache_vendor = st.text_input("장소/상품명 (Vendor)", help="예: Jollibee (C3, Ayala Ave)")
#             with c2:
#                 new_cache_category = st.selectbox("카테고리 (Category)", ["Food", "Transport", "Misc"])
#                 new_cache_price = st.number_input("가격 (Price)", min_value=0.0, step=0.01)
#                 new_cache_currency = st.text_input("통화 (Currency)", value="USD", help="예: PHP, USD")
#                 new_cache_url = st.text_input("출처 URL (Source URL) (선택)")
            
#             add_cache_submitted = st.form_submit_button("신규 캐시 항목 저장")

#             if add_cache_submitted:
#                 if not new_cache_country or not new_cache_city or not new_cache_vendor:
#                     st.error("국가, 도시, 장소/상품명은 필수입니다.")
#                 else:
#                     new_entry = {
#                         "country": new_cache_country.strip(),
#                         "city": new_cache_city.strip(),
#                         "neighborhood": new_cache_neighborhood.strip(),
#                         "vendor": new_cache_vendor.strip(),
#                         "category": new_cache_category,
#                         "price": new_cache_price,
#                         "currency": new_cache_currency.strip().upper(),
#                         "url": new_cache_url.strip(),
#                     }
#                     # menu_cache.py의 함수를 호출하여 항목 추가
#                     if add_menu_cache_entry(new_entry):
#                         st.success(f"'{new_cache_vendor}' 항목을 캐시에 추가했습니다.")
#                         st.rerun()
#                     else:
#                         st.error("캐시 항목 추가에 실패했습니다.")

#         # 2. 기존 캐시 항목 조회 및 삭제
#         st.subheader("기존 캐시 항목 조회 및 삭제")
#         all_cache_data = load_all_cache() # menu_cache.py의 함수
        
#         if not all_cache_data:
#             st.info("현재 저장된 캐시 데이터가 없습니다.")
#         else:
#             df_cache = pd.DataFrame(all_cache_data)
#             st.dataframe(df_cache[[
#                 "country", "city", "neighborhood", "vendor", 
#                 "category", "price", "currency", "last_updated", "url"
#             ]], use_container_width=True)

#             # 삭제 기능
#             st.markdown("---")
#             st.write("##### 캐시 항목 삭제")
            
#             # 삭제할 항목을 식별할 수 있는 고유한 레이블 생성 (최신 항목이 위로)
#             delete_options_map = {
#                 f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
#                 for idx, entry in enumerate(reversed(all_cache_data)) # reversed()로 최신 항목이 먼저 보이게
#             }
#             delete_labels = list(delete_options_map.keys())
            
#             label_to_delete = st.selectbox("삭제할 캐시 항목을 선택하세요:", delete_labels, index=None, placeholder="삭제할 항목 선택...")
            
#             if label_to_delete and st.button(f"'{label_to_delete}' 항목 삭제", type="primary"):
#                 # 거꾸로 매핑된 인덱스를 실제 인덱스로 변환
#                 original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
#                 entry_to_delete = all_cache_data.pop(original_list_index)
                
#                 # menu_cache.py의 함수를 호출하여 전체 파일 저장
#                 if save_cached_menu_prices(all_cache_data):
#                     st.success(f"'{entry_to_delete.get('vendor')}' 항목을 삭제했습니다.")
#                     st.rerun()
#                 else:
#                     st.error("캐시 삭제에 실패했습니다.")
    
#     # --- [신규 3] UI 끝 ---

#     st.divider() # <-- 이것이 '비중 설정' 섹션과 구분하는 선입니다.
#     st.subheader("비중 설정 (기본값)")
#     # ... (이후 비중 설정 폼 코드가 이어짐) ...




# # 2025-10-20-16 AI 기반 출장비 계산 도구 (v16.0 - Async, Dynamic Weights, Full Admin)
# # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv httpx
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.
# # 3. .env 파일에 ADMIN_ACCESS_CODE="<비밀번호>"를 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# import asyncio  # [개선 1] 비동기 처리를 위한 라이브러리
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles, stdev  # [신규 1] stdev 추가
# from typing import Any, Dict, List, Optional, Set, Tuple

# # [신규 3] menu_cache 임포트 (파일이 없으면 이 기능은 작동하지 않음)
# try:
#     from data_sources.menu_cache import (
#         load_cached_menu_prices, 
#         load_all_cache, 
#         add_menu_cache_entry, 
#         save_cached_menu_prices
#     )
#     MENU_CACHE_ENABLED = True
# except ImportError:
#     st.warning("`data_sources/menu_cache.py` 파일을 찾을 수 없습니다. '데이터 캐시 관리' 기능이 비활성화됩니다.")
#     # (기존 함수들을 임시로 정의)
#     def load_cached_menu_prices(city: str, country: str, neighborhood: Optional[str]) -> List[Dict[str, Any]]: return []
#     def load_all_cache() -> List[Dict[str, Any]]: return []
#     def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool: return False
#     def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool: return False
#     MENU_CACHE_ENABLED = False


# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드 (중복)"), # [신규 2] 중복된 카드
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         # 이 스타일은 커스텀 색상을 유지합니다 (양쪽 모드에서 동일하게 보임)
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem; color: #ffffff;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem; color: #ffffff;",
#         "caption": "font-size:1.1rem;opacity:0.95; color: #ffffff;",
#     },
#     "secondary": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--secondary-background-color); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
#     "muted": {
#         # [수정] Streamlit 테마 변수 사용
#         "container": "padding:1.1rem;border-radius:14px;background-color: var(--gray-100); border: 1px solid var(--gray-300);",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem; color: var(--text-color);",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem; color: var(--text-color);",
#         "caption": "font-size:0.85rem; color: var(--gray-600);", # 캡션은 회색 계열 사용
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
    
#     # [수정] 캡션에 스타일 적용
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
    
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">{level_label} 기준 예상 일비 총액</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>계산식</span><br/>
#             $ {daily:,} × {days}일 일정 × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6)": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "숏텀", "Long-term": "롱텀"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# # --- [신규 1] aggregate_ai_totals 함수 수정 ---
# # (이상치 제거 + 변동계수(VC) 계산)
# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균 및 변동 계수(VC)를 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean_raw": None, "mean": None, "variation_coeff": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
    
#     # --- [신규 1] AI 일관성 점수 (변동 계수) 계산 ---
#     variation_coeff = None
#     if filtered and computed_mean and computed_mean > 0:
#         if len(filtered) > 1:
#             try:
#                 computed_stdev = stdev(filtered)
#                 variation_coeff = computed_stdev / computed_mean # 변동 계수 = 표준편차 / 평균
#             except StatisticsError:
#                 variation_coeff = 0.0 # 모든 값이 동일
#         else:
#             variation_coeff = 0.0 # 값이 하나뿐이면 변동 없음

#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#         "variation_coeff": variation_coeff # <-- AI 일관성 점수
#     }

# # --- [신규 1] 동적 가중치 계산 함수 (새로 추가) ---
# def get_dynamic_weights(
#     variation_coeff: Optional[float], 
#     admin_weights: Dict[str, float]
# ) -> Dict[str, Any]:
#     """AI 일관성(VC)에 따라 관리자가 설정한 가중치를 동적으로 보정합니다."""
    
#     # 관리자 설정값을 기본값으로 사용
#     base_ai_weight = admin_weights.get("ai_weight", 0.5)
    
#     if variation_coeff is None:
#         # AI 데이터가 없으면 UN 100%
#         return {"un_weight": 1.0, "ai_weight": 0.0, "source": "No AI Data"}
        
#     if variation_coeff <= 0.05: # 5% 이하: 매우 일관됨
#         # AI 신뢰도 상향 (관리자 설정치에서 최대 0.7까지)
#         dynamic_ai_weight = min(base_ai_weight + 0.2, 0.7)
#         source = f"High AI Consistency (VC: {variation_coeff:.2f})"
#     elif variation_coeff >= 0.15: # 15% 이상: 매우 불안정
#         # AI 신뢰도 하향 (관리자 설정치에서 최소 0.3까지)
#         dynamic_ai_weight = max(base_ai_weight - 0.2, 0.3)
#         source = f"Low AI Consistency (VC: {variation_coeff:.2f})"
#     else:
#         # 5% ~ 15% 사이: 관리자 설정값 유지
#         dynamic_ai_weight = base_ai_weight
#         source = f"Standard (Admin Default) (VC: {variation_coeff:.2f})"

#     final_ai_weight = max(0.0, min(1.0, dynamic_ai_weight))
#     final_un_weight = 1.0 - final_ai_weight
    
#     return {"un_weight": final_un_weight, "ai_weight": final_ai_weight, "source": source}


# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# # --- [개선 1] AI 호출 함수를 비동기(async) 버전으로 교체 ---
# async def get_market_data_from_ai_async(
#     client: openai.AsyncOpenAI,  # <-- Async 클라이언트를 받음
#     city: str,
#     country: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """[비동기 버전] AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     # --- (내부 헬퍼 함수들은 기존과 동일) ---
#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")
#     # --- (프롬프트 구성은 기존과 동일) ---
#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         # --- [수정] 비동기 API 호출로 변경 ---
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         # --- [수정] 끝 ---
        
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }
# # --- [개선 1] 끝 ---

# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
            
#             # [신규 1] 동적 가중치 적용 사유
#             weight_source = ai_summary.get("weighted_average_components", {}).get("weights", {}).get("source", "N/A")

#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#                 'Weight Logic': weight_source, # [신규 1] 동적 가중치 사유 추가
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - [신규 1] **Dynamic Weighting**: AI-generated data consistency (Variation Coefficient) is measured. If AI results are highly consistent (VC <= 5%), AI weight is increased. If highly inconsistent (VC >= 15%), AI weight is decreased. Otherwise, admin-set defaults are used.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- 스트림릿 UI 구성 ---
# st.set_page_config(layout="wide")
# st.title("AICP: 출장 일비 계산 & 조회 시스템 (v16.0 - Async & Dynamic)")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility


# # --- [개선 3] 탭 구조 변경 ---
# tab_definitions = []
# if employee_tab_visible:
#     tab_definitions.append("💵 일비 조회 (직원용)")

# # 관리자 탭을 2개로 분리
# tab_definitions.append("📈 보고서 분석 (Admin)")
# tab_definitions.append("🛠️ 시스템 설정 (Admin)")

# tabs = st.tabs(tab_definitions)

# employee_tab = None
# admin_analysis_tab = None
# admin_config_tab = None

# if employee_tab_visible:
#     employee_tab = tabs[0]
#     admin_analysis_tab = tabs[1]
#     admin_config_tab = tabs[2]
# else:
#     admin_analysis_tab = tabs[0]
#     admin_config_tab = tabs[1]
# # --- [개선 3] 끝 ---


# if employee_tab is not None:
#     with employee_tab:
#         st.header("도시별 출장 일비 조회")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("도시:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("선택한 국가에 등록된 도시가 없습니다.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "출장 시작일",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "출장 종료일",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("종료일은 시작일 이후여야 합니다.")
#                     trip_days = 0 # 0으로 설정
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"자동 분류된 출장 유형: {trip_term_label} · {trip_days}일 일정")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("이 기간에 해당하는 도시 데이터가 없습니다. 출장 유형을 '숏텀'으로 조정하거나 도시 설정을 확인하세요.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level and trip_days is not None:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} 결과")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
                        
#                         # [신규 2] 직원 탭 총액 카드
#                         render_primary_summary(
#                             f"{sel_level.split(' ')[0]}",
#                             trip_total_allowance,
#                             level_daily_allowance,
#                             trip_days,
#                             trip_term_label,
#                             trip_multiplier
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} 일일 권장 일비", "금액 없음")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**세부 산출 근거 (일비 기준)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"제외값 {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"
                        
#                         # [신규 1] 동적 가중치 적용 사유
#                         weights_info = ai_summary.get("weighted_average_components", {}).get("weights", {})
#                         weights_source = weights_info.get("source", "N/A")
#                         un_weight_pct = f"{weights_info.get('un_weight', 0.5):.0%}"
#                         ai_weight_pct = f"{weights_info.get('ai_weight', 0.5):.0%}"
#                         weight_caption = f"Blend: UN-DSA ({un_weight_pct}) + AI ({ai_weight_pct}) | 사유: {weights_source}"

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         un_caption = f"숏텀 기준 $ {un_base:,}" if un_base is not None else city_data.get("notes", "")
#                                         if trip_term == "Long-term" and un_base is not None:
#                                             un_caption = f"숏텀 $ {un_base:,} → 롱텀 $ {un_display:,}"
#                                         render_stat_card("UN-DSA 기준", f"$ {un_display:,}" if un_display is not None else "N/A", un_caption, "secondary")
                                    
#                                     elif key == "ai":
#                                         ai_caption_base = f"숏텀 기준 $ {ai_avg:,}" if ai_avg is not None else ""
#                                         if trip_term == "Long-term" and ai_avg is not None:
#                                             ai_caption_base = f"숏텀 $ {ai_avg:,} → 롱텀 $ {ai_display:,}"
#                                         ai_full_caption = f"{ai_notes} | {ai_caption_base}".strip(" | ")
#                                         render_stat_card("AI 시장 추정 (시즌 보정)", f"$ {ai_display:,}" if ai_display is not None else "N/A", ai_full_caption, "secondary")
                                    
#                                     else: # key == "weighted"
#                                         weighted_caption = weight_caption
#                                         if trip_term == "Long-term" and final_allowance is not None:
#                                             weighted_caption = f"숏텀 $ {final_allowance:,} → 롱텀 $ {weighted_display:,} | {weight_caption}"
#                                         render_stat_card("가중 평균 결과", f"$ {weighted_display:,}" if weighted_display is not None else "N/A", weighted_caption, "secondary")

#                         # [신규 2] 비용 항목별 상세 내역 (show_ai_market_detail과 로직 통합)
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             st.markdown("<br>", unsafe_allow_html=True) # 줄 간격
                            
#                             mean_food = ai_summary.get("mean_food", 0)
#                             mean_trans = ai_summary.get("mean_transport", 0)
#                             mean_misc = ai_summary.get("mean_misc", 0)
                            
#                             # 롱텀/시즌 요율 적용
#                             food_display = round(mean_food * season_factor * trip_multiplier)
#                             trans_display = round(mean_trans * season_factor * trip_multiplier)
#                             misc_display = round(mean_misc * season_factor * trip_multiplier)
                            
#                             st.write("###### AI 추정 상세 내역 (일비 기준)")
#                             col_f, col_t, col_m = st.columns(3)
#                             with col_f:
#                                 render_stat_card("예상 식비 (Food)", f"$ {food_display:,}", f"숏텀 기준: $ {round(mean_food)}", "muted")
#                             with col_t:
#                                 render_stat_card("예상 교통비 (Transport)", f"$ {trans_display:,}", f"숏텀 기준: $ {round(mean_trans)}", "muted")
#                             with col_m:
#                                 render_stat_card("예상 기타 (Misc)", f"$ {misc_display:,}", f"숏텀 기준: $ {round(mean_misc)}", "muted")
                        
#                         # [개선 3] show_weighted_result 카드가 중복되므로, 아래 블록은 제거
#                         # (기존 second_row_keys 로직 제거)

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_info,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("관리자가 세부 산출 근거를 숨겼습니다.")

# # --- [개선 2] admin_tab -> admin_analysis_tab 으로 변경 ---
# with admin_analysis_tab:
    
#     # [개선 2] ADMIN_ACCESS_CODE 로드 및 .env 체크
#     ACCESS_CODE_KEY = "admin_access_code_valid"
#     ACCESS_CODE_VALUE = os.getenv("ADMIN_ACCESS_CODE") # .env에서 로드

#     if not ACCESS_CODE_VALUE:
#         st.error("보안 오류: .env 파일에 'ADMIN_ACCESS_CODE'가 설정되지 않았습니다. 앱을 중지하고 .env 파일을 설정해주세요.")
#         st.stop()
    
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#                 st.rerun() # [개선 3] 성공 시 새로고침
#             else:
#                 st.error("Access Code가 올바르지 않습니다.")
#                 st.stop() # [개선 3] 실패 시 중지
#         else:
#             st.stop() # [개선 3] 폼 제출 전 중지

#     # --- [개선 3] "보고서 버전 관리" 기능 (analysis_sub_tab) ---
#     st.subheader("보고서 버전 관리")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("활성 보고서 버전을 선택하세요:", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("생성된 보고서가 없습니다.")

#     # --- [신규 4] 과거 보고서 비교 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("과거 보고서 비교")
#     if len(history_files) < 2:
#         st.info("비교할 보고서가 2개 이상 필요합니다.")
#     else:
#         col_a, col_b = st.columns(2)
#         with col_a:
#             file_a = st.selectbox("기준 보고서 (A)", history_files, index=1, key="compare_a")
#         with col_b:
#             file_b = st.selectbox("비교 보고서 (B)", history_files, index=0, key="compare_b")
        
#         if st.button("보고서 비교하기"):
#             if file_a == file_b:
#                 st.warning("서로 다른 보고서를 선택해야 합니다.")
#             else:
#                 with st.spinner("보고서 비교 중..."):
#                     data_a = load_report_data(file_a)
#                     data_b = load_report_data(file_b)
                    
#                     if data_a and data_b and 'cities' in data_a and 'cities' in data_b:
#                         df_a = pd.DataFrame(data_a['cities'])[['city', 'country_display', 'final_allowance']]
#                         df_b = pd.DataFrame(data_b['cities'])[['city', 'country_display', 'final_allowance']]
                        
#                         df_merged = pd.merge(df_a, df_b, on=["city", "country_display"], suffixes=("_A", "_B"))
                        
#                         report_a_label = file_a.split('report_')[-1].split('.')[0]
#                         report_b_label = file_b.split('report_')[-1].split('.')[0]

#                         df_merged[f"A ({report_a_label})"] = df_merged["final_allowance_A"]
#                         df_merged[f"B ({report_b_label})"] = df_merged["final_allowance_B"]
                        
#                         df_merged["변동액 ($)"] = df_merged["final_allowance_B"] - df_merged["final_allowance_A"]
                        
#                         # 0으로 나누기 방지
#                         df_merged["변동률 (%)"] = (df_merged["변동액 ($)"] / df_merged["final_allowance_A"].replace(0, pd.NA)) * 100
                        
#                         st.dataframe(df_merged[[
#                             "city", "country_display", 
#                             f"A ({report_a_label})", 
#                             f"B ({report_b_label})", 
#                             "변동액 ($)", "변동률 (%)"
#                         ]].style.format({"변동률 (%)": "{:,.1f}%", "변동액 ($)": "{:,.0f}"}), width="stretch")
#                     else:
#                         st.error("보고서 파일을 불러오는 데 실패했습니다.")
    
#     # --- [개선 3] "UN-DSA (PDF) 분석" 기능 (analysis_sub_tab) ---
#     st.divider()
#     st.subheader("UN-DSA (PDF) 분석 및 AI 실행")
#     st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요. (개선 1: 비동기 처리로 속도 향상)")
#     uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")

#     # --- [개선 1] 비동기 AI 분석 실행 로직 ---
#     if uploaded_file and st.button("AI 분석 실행", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
#         else:
#             st.session_state.latest_analysis_result = None
            
#             # --- 비동기 실행 함수 정의 ---
#             async def run_analysis(progress_bar, openai_api_key):
#                 progress_bar.progress(0, text="PDF 텍스트 추출 중...")
#                 full_text = parse_pdf_to_text(uploaded_file)
                
#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []
#                 analysis_failed = False
                
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break
                
#                 if analysis_failed:
#                     st.error("PDF->TSV 변환에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                 if not processed_data:
#                     st.error("TSV 데이터 처리에 실패했습니다.")
#                     progress_bar.empty()
#                     return

#                 # 비동기 OpenAI 클라이언트 생성
#                 client = openai.AsyncOpenAI(api_key=openai_api_key)
                
#                 total_cities = len(processed_data["cities"])
#                 all_tasks = [] # 모든 AI 호출 작업을 담을 리스트

#                 # 1. 모든 도시에 대한 모든 AI 호출 작업을 미리 생성
#                 for city_data in processed_data["cities"]:
#                     city_name, country_name = city_data["city"], city_data["country_display"]
#                     city_context = {
#                         "neighborhood": city_data.get("neighborhood"),
#                         "hotel_cluster": city_data.get("hotel_cluster"),
#                     }
#                     season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                     menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                    
#                     city_data["menu_samples"] = menu_samples
#                     city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
                    
#                     city_tasks = []
#                     for j in range(1, NUM_AI_CALLS + 1):
#                         task = get_market_data_from_ai_async(
#                             client, city_name, country_name, f"Run {j}",
#                             context=city_context, season_context=season_context, menu_samples=menu_samples
#                         )
#                         city_tasks.append(task)
                    
#                     all_tasks.append(city_tasks) # [ [도시1-10회], [도시2-10회], ... ]

#                 # 2. 모든 작업을 비동기로 실행하고 결과 수집
#                 city_index = 0
#                 for city_tasks in all_tasks:
#                     city_data = processed_data["cities"][city_index]
#                     city_name = city_data["city"]
#                     progress_text = f"AI 추정치 계산 중... ({city_index+1}/{total_cities}) {city_name}"
#                     progress_bar.progress((city_index + 1) / max(total_cities, 1), text=progress_text)
                    
#                     # 해당 도시의 10개 작업을 동시에 실행
#                     try:
#                         market_results = await asyncio.gather(*city_tasks)
#                     except Exception as e:
#                         st.error(f"{city_name} 분석 중 비동기 오류: {e}")
#                         market_results = [] # 실패 처리

#                     # 3. 결과 처리
#                     ai_totals_source: List[int] = []
#                     ai_meta_runs: List[Dict[str, Any]] = []
                    
#                     # [신규 2] 비용 항목별 상세 내역을 위한 리스트
#                     ai_food: List[int] = []
#                     ai_transport: List[int] = []
#                     ai_misc: List[int] = []

#                     for j, market_result in enumerate(market_results, 1):
#                         city_data[f"market_data_{j}"] = market_result
#                         if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                             ai_totals_source.append(market_result["total"])
#                             # [신규 2] 상세 비용 추가
#                             ai_food.append(market_result.get("food", 0))
#                             ai_transport.append(market_result.get("transport", 0))
#                             ai_misc.append(market_result.get("misc", 0))
                        
#                         if "meta" in market_result:
#                             ai_meta_runs.append(market_result["meta"])
                    
#                     city_data["ai_provenance"] = ai_meta_runs

#                     # 4. 최종 수당 계산
#                     final_allowance = None
#                     un_per_diem_raw = city_data.get("un", {}).get("per_diem_excl_lodging")
#                     un_per_diem = float(un_per_diem_raw) if isinstance(un_per_diem_raw, (int, float)) else None

#                     ai_stats = aggregate_ai_totals(ai_totals_source)
#                     season_factor = (season_context or {}).get("factor", 1.0)
#                     ai_base_mean = ai_stats.get("mean_raw")
#                     ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
                    
#                     # [신규 1] 동적 가중치 계산
#                     admin_weights = get_weight_config() # 관리자 설정 로드
#                     ai_vc_score = ai_stats.get("variation_coeff")
                    
#                     if un_per_diem is not None:
#                         weights_cfg = get_dynamic_weights(ai_vc_score, admin_weights)
#                     else:
#                         # UN 데이터 없으면 AI 100%
#                         weights_cfg = {"un_weight": 0.0, "ai_weight": 1.0, "source": "AI Only (UN-DSA Missing)"}
                    
#                     city_data["ai_summary"] = {
#                         "raw_totals": ai_totals_source,
#                         "used_totals": ai_stats.get("used_values", []),
#                         "removed_totals": ai_stats.get("removed_values", []),
#                         "mean_base": ai_base_mean,
#                         "mean_base_rounded": ai_stats.get("mean"),
                        
#                         "ai_consistency_vc": ai_vc_score, # [신규 1]
                        
#                         "mean_food": mean(ai_food) if ai_food else 0, # [신규 2]
#                         "mean_transport": mean(ai_transport) if ai_transport else 0, # [신규 2]
#                         "mean_misc": mean(ai_misc) if ai_misc else 0, # [신규 2]

#                         "season_factor": season_factor,
#                         "season_label": (season_context or {}).get("label"),
#                         "season_adjusted_mean_raw": ai_season_adjusted,
#                         "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                         "successful_runs": len(ai_stats.get("used_values", [])),
#                         "attempted_runs": NUM_AI_CALLS,
#                         "reference_links": city_data.get("reference_links", []),
#                         "weighted_average_components": {
#                             "un_per_diem": un_per_diem,
#                             "ai_season_adjusted": ai_season_adjusted,
#                             "weights": weights_cfg, # [신규 1] 동적 가중치 저장
#                         },
#                     }

#                     # [신규 1] 동적 가중치로 최종값 계산
#                     if un_per_diem is not None and ai_season_adjusted is not None:
#                         weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                         final_allowance = round(weighted_average)
#                     elif un_per_diem is not None:
#                         final_allowance = round(un_per_diem)
#                     elif ai_season_adjusted is not None:
#                         final_allowance = round(ai_season_adjusted)

#                     city_data["final_allowance"] = final_allowance

#                     if final_allowance and un_per_diem and un_per_diem > 0:
#                         city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                     else:
#                         city_data["delta_vs_un_pct"] = "N/A"
                    
#                     city_index += 1 # 다음 도시로

#                 save_report_data(processed_data)
#                 st.session_state.latest_analysis_result = processed_data
#                 st.success("AI analysis completed.")
#                 progress_bar.empty()
#                 st.rerun()
            
#             # --- 비동기 실행 ---
#             with st.spinner("PDF 처리 및 AI 분석을 실행합니다. (약 10~30초 소요)"):
#                 progress_bar = st.progress(0, text="분석 시작...")
#                 asyncio.run(run_analysis(progress_bar, openai_api_key))

#     # --- [개선 3] "Latest Analysis Summary" 기능 (analysis_sub_tab) ---
#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             # --- [HOTFIX] ArrowInvalid Error 방지 ---
#             delta_val = city.get('delta_vs_un_pct')
#             if isinstance(delta_val, (int, float)):
#                 delta_display = f"{delta_val:.0f}%" # 숫자를 "12%" 형태의 문자열로 변경
#             else:
#                 delta_display = "N/A" # 이미 "N/A" 문자열
#             # --- [HOTFIX] End ---
                
#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': delta_display, # <-- 수정된 문자열 값 사용
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data), use_container_width=True) # <-- use_container_width 추가 (필요시 width='stretch'로 변경)
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))

# # --- [개선 3] "시스템 설정" 탭 (admin_config_tab) ---
# with admin_config_tab:
#     # 암호 확인 (필수)
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         st.error("Access Code가 필요합니다.")
#         st.stop()
        
#     st.subheader("직원용 탭 노출")
#     visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility_toggle") # Key 이름 변경
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.session_state.employee_tab_visibility = visibility_toggle # 세션 상태에도 반영
#         st.success("직원용 탭 노출 상태가 업데이트되었습니다. (새로고침 시 적용)")

#     st.subheader("직원 화면 노출 설정")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS.get(section_key, True))
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("직원 화면 노출 설정이 업데이트되었습니다.")

#     st.divider()
#     st.subheader("비중 설정 (기본값)")
#     st.info("이제 이 설정은 '동적 가중치' 로직의 기본값으로 사용됩니다. AI 응답이 불안정하면 자동으로 AI 비중이 낮아집니다.")
#     current_weights = get_weight_config()
#     st.caption(f"Current Admin Default -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")
#         st.rerun()

#     st.divider()
#     st.header("목표 도시 관리")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         # trip_lengths를 보기 쉽게 문자열로 변환
#         entries_display["trip_lengths"] = entries_display["trip_lengths"].apply(lambda x: ', '.join(x) if isinstance(x, list) else DEFAULT_TRIP_LENGTH[0])
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], use_container_width=True)
#     else:
#         st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("신규 도시 추가")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["기타 (직접 입력)"]
#             region_choice = st.selectbox("지역", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "기타 (직접 입력)":
#                 new_region = st.text_input("새 지역 이름", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("도시", key="add_city")
#             neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("국가", key="add_country")
#             hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

#         with st.expander("UN-DSA 대체 도시 (선택)"):
#             substitute_city = st.text_input("대체 도시", key="add_sub_city")
#             substitute_country = st.text_input("대체 국가", key="add_sub_country")

#         add_submitted = st.form_submit_button("추가")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("동일한 항목이 이미 등록되어 있습니다.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
#                 st.rerun()

#     st.subheader("기존 도시 편집/삭제")
#     current_entries = get_target_city_entries()
    
#     if current_entries:
#         # 1. 드롭다운 옵션 구성
#         options = {
#             f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#             for idx, entry in enumerate(current_entries)
#         }
#         sorted_labels = list(options.keys())

#         # 2. on_change 콜백 함수 정의 (위젯보다 먼저)
#         def _sync_edit_form_from_selection():
#             # 드롭다운에서 현재 선택된 값을 가져옴
#             if "edit_city_selector" not in st.session_state:
#                  st.session_state.edit_city_selector = sorted_labels[0]
                 
#             selected_idx = options[st.session_state.edit_city_selector]
#             selected_entry = current_entries[selected_idx]
            
#             # session_state의 값을 선택된 도시의 데이터로 강제 업데이트
#             st.session_state.edit_region = selected_entry.get("region", "")
#             st.session_state.edit_city = selected_entry.get("city", "")
#             st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#             st.session_state.edit_country = selected_entry.get("country", "")
#             st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
            
#             # 출장 기간 (trip_lengths) 설정
#             existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#             st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
            
#             # UN-DSA 대체 도시 설정
#             sub_data = selected_entry.get("un_dsa_substitute") or {}
#             st.session_state.edit_sub_city = sub_data.get("city", "")
#             st.session_state.edit_sub_country = sub_data.get("country", "")

#         # 3. 드롭다운(Selectbox)에 on_change 콜백 연결
#         selected_label = st.selectbox(
#             "편집할 도시를 선택하세요", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection  # <-- [수정] 콜백 함수 연결
#         )

#         # 4. 페이지 첫 로드 시 폼을 채우기 위한 초기화
#         if "edit_region" not in st.session_state:
#             # 첫 로드 시, selectbox의 기본값(첫 번째 항목)에 맞춰 폼을 채움
#             _sync_edit_form_from_selection()

#         # 5. 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 # [수정] value=... 제거
#                 region_edit = st.text_input("지역", key="edit_region")
#                 city_edit = st.text_input("도시", key="edit_city")
#                 neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
#             with col_f:
#                 # [수정] value=... 제거
#                 country_edit = st.text_input("국가", key="edit_country")
#                 hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

#             # [수정] default=... 대신 key=... 사용
#             trip_lengths_edit = st.multiselect(
#                 "출장 기간",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", # 'default' 대신 'key'로 상태 관리
#             )

#             with st.expander("UN-DSA 대체 도시 (선택)"):
#                 # [수정] value=... 제거
#                 sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
#                 sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("변경사항 저장")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("삭제", type="secondary")

#         # 6. 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
#         if update_btn:
#             # [수정] 위젯 변수(region_edit) 대신 st.session_state에서 직접 값을 읽음
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#             else:
#                 current_entries[options[selected_label]] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[options[selected_label]]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[options[selected_label]].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("수정을 완료했습니다.")
#                 st.rerun()  # <-- [확인] 이미 올바르게 수정되어 있음
        
#         if delete_btn:
#             del current_entries[options[selected_label]]
#             set_target_city_entries(current_entries)
#             st.warning("선택한 항목을 삭제했습니다.")
#             st.rerun() # <-- [확인] 이미 올바르게 수정되어 있음

#     else:
#         st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

#     # --- [신규 3] '데이터 캐시 관리' UI 추가 ---
#     st.divider()
#     st.header("데이터 캐시 관리 (Menu Cache)")

#     if not MENU_CACHE_ENABLED:
#         st.error("`data_sources/menu_cache.py` 파일 로드에 실패하여 이 기능을 사용할 수 없습니다.")
#     else:
#         st.info("AI가 도시 물가 추정 시 참고할 실제 메뉴/가격 데이터를 관리합니다. (AI 분석 정확도 향상)")

#         # 1. 새 캐시 항목 추가 폼
#         st.subheader("신규 캐시 항목 추가")
#         with st.form("add_menu_cache_form", clear_on_submit=True):
#             st.write("AI 분석에 사용할 참고 가격 정보를 입력합니다. (예: 레스토랑 메뉴, 택시비 고지 등)")
#             c1, c2 = st.columns(2)
#             with c1:
#                 new_cache_country = st.text_input("국가 (Country)", help="예: Philippines")
#                 new_cache_city = st.text_input("도시 (City)", help="예: Manila")
#                 new_cache_neighborhood = st.text_input("세부 지역 (Neighborhood) (선택)", help="예: Makati (비워두면 도시 전체에 적용)")
#                 new_cache_vendor = st.text_input("장소/상품명 (Vendor)", help="예: Jollibee (C3, Ayala Ave)")
#             with c2:
#                 new_cache_category = st.selectbox("카테고리 (Category)", ["Food", "Transport", "Misc"])
#                 new_cache_price = st.number_input("가격 (Price)", min_value=0.0, step=0.01)
#                 new_cache_currency = st.text_input("통화 (Currency)", value="USD", help="예: PHP, USD")
#                 new_cache_url = st.text_input("출처 URL (Source URL) (선택)")
            
#             add_cache_submitted = st.form_submit_button("신규 캐시 항목 저장")

#             if add_cache_submitted:
#                 if not new_cache_country or not new_cache_city or not new_cache_vendor:
#                     st.error("국가, 도시, 장소/상품명은 필수입니다.")
#                 else:
#                     new_entry = {
#                         "country": new_cache_country.strip(),
#                         "city": new_cache_city.strip(),
#                         "neighborhood": new_cache_neighborhood.strip(),
#                         "vendor": new_cache_vendor.strip(),
#                         "category": new_cache_category,
#                         "price": new_cache_price,
#                         "currency": new_cache_currency.strip().upper(),
#                         "url": new_cache_url.strip(),
#                     }
#                     # menu_cache.py의 함수를 호출하여 항목 추가
#                     if add_menu_cache_entry(new_entry):
#                         st.success(f"'{new_cache_vendor}' 항목을 캐시에 추가했습니다.")
#                         st.rerun()
#                     else:
#                         st.error("캐시 항목 추가에 실패했습니다.")

#         # 2. 기존 캐시 항목 조회 및 삭제
#         st.subheader("기존 캐시 항목 조회 및 삭제")
#         all_cache_data = load_all_cache() # menu_cache.py의 함수
        
#         if not all_cache_data:
#             st.info("현재 저장된 캐시 데이터가 없습니다.")
#         else:
#             df_cache = pd.DataFrame(all_cache_data)
#             st.dataframe(df_cache[[
#                 "country", "city", "neighborhood", "vendor", 
#                 "category", "price", "currency", "last_updated", "url"
#             ]], use_container_width=True)

#             # 삭제 기능
#             st.markdown("---")
#             st.write("##### 캐시 항목 삭제")
            
#             # 삭제할 항목을 식별할 수 있는 고유한 레이블 생성 (최신 항목이 위로)
#             delete_options_map = {
#                 f"[{entry.get('last_updated', '...')} / {entry.get('city', '...')}] {entry.get('vendor', '...')} ({entry.get('price', '...')})": idx
#                 for idx, entry in enumerate(reversed(all_cache_data)) # reversed()로 최신 항목이 먼저 보이게
#             }
#             delete_labels = list(delete_options_map.keys())
            
#             label_to_delete = st.selectbox("삭제할 캐시 항목을 선택하세요:", delete_labels, index=None, placeholder="삭제할 항목 선택...")
            
#             if label_to_delete and st.button(f"'{label_to_delete}' 항목 삭제", type="primary"):
#                 # 거꾸로 매핑된 인덱스를 실제 인덱스로 변환
#                 original_list_index = (len(all_cache_data) - 1) - delete_options_map[label_to_delete]
                
#                 entry_to_delete = all_cache_data.pop(original_list_index)
                
#                 # menu_cache.py의 함수를 호출하여 전체 파일 저장
#                 if save_cached_menu_prices(all_cache_data):
#                     st.success(f"'{entry_to_delete.get('vendor')}' 항목을 삭제했습니다.")
#                     st.rerun()
#                 else:
#                     st.error("캐시 삭제에 실패했습니다.")
    
#     # --- [신규 3] UI 끝 ---

#     st.divider() # <-- 이것이 '비중 설정' 섹션과 구분하는 선입니다.
#     st.subheader("비중 설정 (기본값)")
#     # ... (이후 비중 설정 폼 코드가 이어짐) ...


# # 2025-10-20-14 AI 기반 출장비 계산 도구 (고급 분석 모델 적용)
# # --- 설치 안내 ---
# # 1. 아래 명령으로 필요한 패키지를 설치하세요.
# #    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv
# #
# # 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.

# import streamlit as st
# import pandas as pd
# import json
# import os
# import re
# import fitz  # PyMuPDF 라이브러리
# import openai
# from dotenv import load_dotenv
# import io
# from datetime import datetime, timedelta
# import time
# import random
# from collections import Counter
# from statistics import StatisticsError, mean, quantiles
# from typing import Any, Dict, List, Optional, Set, Tuple

# from data_sources.menu_cache import load_cached_menu_prices

# # --- 초기 환경 설정 ---

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Maximum number of AI calls per analysis
# NUM_AI_CALLS = 10
# # --- Weight configuration (sum should remain 1.0) ---
# DEFAULT_WEIGHT_CONFIG = {"un_weight": 0.5, "ai_weight": 0.5}
# _WEIGHT_CONFIG_CACHE: Dict[str, float] = {}


# def weight_config_path() -> str:
#     return os.path.join(DATA_DIR, "weight_config.json")



# def _normalize_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Ensure weights are floats that sum to 1.0 (defaults fall back to 0.5 / 0.5)."""
#     try:
#         un_raw = float(config.get("un_weight", DEFAULT_WEIGHT_CONFIG["un_weight"]))
#     except (TypeError, ValueError):
#         un_raw = DEFAULT_WEIGHT_CONFIG["un_weight"]
#     try:
#         ai_raw = float(config.get("ai_weight", DEFAULT_WEIGHT_CONFIG["ai_weight"]))
#     except (TypeError, ValueError):
#         ai_raw = DEFAULT_WEIGHT_CONFIG["ai_weight"]

#     total = un_raw + ai_raw
#     if total <= 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)

#     un_norm = max(0.0, min(1.0, un_raw / total))
#     ai_norm = max(0.0, min(1.0, ai_raw / total))

#     total_norm = un_norm + ai_norm
#     if total_norm == 0:
#         return dict(DEFAULT_WEIGHT_CONFIG)
#     return {"un_weight": un_norm / total_norm, "ai_weight": ai_norm / total_norm}


# def save_weight_config(config: Dict[str, Any]) -> Dict[str, float]:
#     """Persist weight configuration to disk and update the in-memory cache."""
#     normalized = _normalize_weight_config(config)
#     with open(weight_config_path(), "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)

#     global _WEIGHT_CONFIG_CACHE
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return normalized


# def load_weight_config(force: bool = False) -> Dict[str, float]:
#     """Load weight configuration from disk (or defaults when missing)."""
#     global _WEIGHT_CONFIG_CACHE
#     if _WEIGHT_CONFIG_CACHE and not force:
#         return dict(_WEIGHT_CONFIG_CACHE)

#     if not os.path.exists(weight_config_path()):
#         normalized = save_weight_config(DEFAULT_WEIGHT_CONFIG)
#         return dict(normalized)

#     try:
#         with open(weight_config_path(), "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("Weight config must be a JSON object")
#     except Exception:
#         data = DEFAULT_WEIGHT_CONFIG

#     normalized = _normalize_weight_config(data)
#     _WEIGHT_CONFIG_CACHE = dict(normalized)
#     return dict(normalized)


# def get_weight_config() -> Dict[str, float]:
#     """Return the active weight configuration, favouring session state if available."""
#     try:
#         session_config = st.session_state.get("weight_config")  # type: ignore[attr-defined]
#     except RuntimeError:
#         session_config = None

#     if session_config:
#         normalized = _normalize_weight_config(session_config)
#         try:
#             st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#         except RuntimeError:
#             pass
#         return normalized

#     config = load_weight_config()
#     try:
#         st.session_state["weight_config"] = config  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return config


# def update_weight_config(un_weight: float, ai_weight: float) -> Dict[str, float]:
#     """Update weights both in session and on disk."""
#     config = {"un_weight": un_weight, "ai_weight": ai_weight}
#     normalized = save_weight_config(config)
#     try:
#         st.session_state["weight_config"] = normalized  # type: ignore[attr-defined]
#     except RuntimeError:
#         pass
#     return normalized


# # 분석 결과를 저장할 디렉터리 경로


# def build_reference_link_lines(menu_samples: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
#     """Return markdown-friendly bullets for cached menu/reference entries."""
#     lines_out: List[str] = []
#     if not menu_samples:
#         return lines_out

#     for sample in menu_samples[:max_items]:
#         if not isinstance(sample, dict):
#             continue

#         name = str(sample.get("vendor") or sample.get("name") or sample.get("title") or sample.get("source") or "Reference")

#         url = None
#         for key in ("url", "link", "source_url", "href"):
#             value = sample.get(key)
#             if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
#                 url = value
#                 break

#         details: List[str] = []
#         price = sample.get("price")
#         if isinstance(price, (int, float)):
#             currency = sample.get("currency") or "USD"
#             details.append(f"{currency} {price}")
#         elif isinstance(price, str) and price.strip():
#             details.append(price.strip())

#         category = sample.get("category")
#         if category:
#             details.append(str(category))

#         last_updated = sample.get("last_updated")
#         if last_updated:
#             details.append(f"updated {last_updated}")

#         detail_text = ", ".join(details)
#         label = f"[{name}]({url})" if url else name

#         if detail_text:
#             lines_out.append(f"{label} - {detail_text}")
#         else:
#             lines_out.append(label)

#     return lines_out


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# UI_SETTINGS_FILE = os.path.join(DATA_DIR, "ui_settings.json")
# DEFAULT_UI_SETTINGS = {"show_employee_tab": True}
# EMPLOYEE_SECTION_DEFAULTS: Dict[str, bool] = {
#     "show_un_basis": True,
#     "show_ai_estimate": True,
#     "show_weighted_result": True,
#     "show_ai_market_detail": True,
#     "show_provenance": True,
#     "show_menu_samples": True,
# }
# EMPLOYEE_SECTION_LABELS = [
#     ("show_un_basis", "UN-DSA 기준 카드"),
#     ("show_ai_estimate", "AI 시장 추정 카드"),
#     ("show_weighted_result", "가중 평균 결과 카드"),
#     ("show_ai_market_detail", "AI Market Estimate 카드"),
#     ("show_provenance", "AI 산출 근거(JSON)"),
#     ("show_menu_samples", "레퍼런스 메뉴 표"),
# ]
# _UI_SETTINGS_CACHE: Dict[str, Any] = {}


# CARD_STYLES = {
#     "primary": {
#         "container": "margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;",
#         "title": "font-size:1rem;opacity:0.85;margin-bottom:0.4rem;",
#         "value": "font-size:2.6rem;font-weight:800;letter-spacing:0.02em;margin-bottom:0.5rem;",
#         "caption": "font-size:1.1rem;opacity:0.95;",
#     },
#     "secondary": {
#         "container": "padding:1.1rem;border-radius:14px;background:rgba(30,60,114,0.08);border:1px solid rgba(30,60,114,0.12);color:#0f1e3d;",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem;",
#         "value": "font-size:1.55rem;font-weight:700;margin-bottom:0.3rem;",
#         "caption": "font-size:0.85rem;opacity:0.75;",
#     },
#     "muted": {
#         "container": "padding:1.1rem;border-radius:14px;background:#f5f7fb;border:1px solid #d8deec;color:#495063;",
#         "title": "font-size:0.95rem;font-weight:600;margin-bottom:0.35rem;",
#         "value": "font-size:1.45rem;font-weight:700;margin-bottom:0.3rem;",
#         "caption": "font-size:0.85rem;opacity:0.7;",
#     },
# }


# def render_stat_card(title: str, value: str, caption: str = "", variant: str = "secondary") -> None:
#     style = CARD_STYLES.get(variant, CARD_STYLES["secondary"])
#     caption_html = f"<div style='{style['caption']}'>{caption}</div>" if caption else ""
#     card_html = f"""
#     <div style="{style['container']}">
#         <div style="{style['title']}">{title}</div>
#         <div style="{style['value']}">{value}</div>
#         {caption_html}
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def render_primary_summary(level_label: str, total: int, daily: int, days: int, term_label: str, multiplier: float) -> None:
#     style = CARD_STYLES["primary"]
#     card_html = f"""
#     <div style="{style['container'].replace('text-align:center;', 'text-align:left;')}">
#         <div style="{style['title']}">{level_label} 기준 예상 일비 총액</div>
#         <div style="{style['value']}">$ {total:,}</div>
#         <div style="{style['caption']}">
#             <span style='font-size:0.95rem;opacity:0.8;'>계산식</span><br/>
#             $ {daily:,} × {days}일 일정 × {term_label} (×{multiplier:.2f})
#         </div>
#     </div>
#     """
#     st.markdown(card_html, unsafe_allow_html=True)


# def _normalize_employee_sections(sections: Any) -> Dict[str, bool]:
#     normalized = dict(EMPLOYEE_SECTION_DEFAULTS)
#     if isinstance(sections, dict):
#         for key in normalized:
#             normalized[key] = bool(sections.get(key, normalized[key]))
#     return normalized

# def _normalize_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Ensure UI settings include expected keys with correct types."""
#     normalized = dict(DEFAULT_UI_SETTINGS)
#     raw_visibility = settings.get("show_employee_tab", DEFAULT_UI_SETTINGS["show_employee_tab"])
#     normalized["show_employee_tab"] = bool(raw_visibility)
#     normalized["employee_sections"] = _normalize_employee_sections(settings.get("employee_sections"))
#     return normalized

# def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """Persist UI settings to disk and update cache."""
#     normalized = _normalize_ui_settings(settings)
#     with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)
#     global _UI_SETTINGS_CACHE
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return normalized

# def load_ui_settings(force: bool = False) -> Dict[str, Any]:
#     """Load UI settings, defaulting gracefully when missing or malformed."""
#     global _UI_SETTINGS_CACHE
#     if _UI_SETTINGS_CACHE and not force:
#         return dict(_UI_SETTINGS_CACHE)
#     if not os.path.exists(UI_SETTINGS_FILE):
#         normalized = save_ui_settings(DEFAULT_UI_SETTINGS)
#         return dict(normalized)
#     try:
#         with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, dict):
#             raise ValueError("UI settings must be a JSON object")
#     except Exception:
#         data = dict(DEFAULT_UI_SETTINGS)
#     normalized = _normalize_ui_settings(data)
#     _UI_SETTINGS_CACHE = dict(normalized)
#     return dict(normalized)

# JOB_LEVEL_RATIOS = {
#     "L3": 0.60, "L4": 0.60, "L5": 0.80, "L6)": 1.00,
#     "L7": 1.00, "L8": 1.20, "L9": 1.50, "L10": 1.50,
# }

# TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
# TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]
# DEFAULT_TRIP_LENGTH = ["Short-term"]
# LONG_TERM_THRESHOLD_DAYS = 30
# SHORT_TERM_MULTIPLIER = 1.0
# LONG_TERM_MULTIPLIER = 1.05
# TRIP_TERM_LABELS = {"Short-term": "숏텀", "Long-term": "롱텀"}


# def classify_trip_duration(days: int) -> Tuple[str, float]:
#     """Return trip term classification and multiplier based on duration in days."""
#     if days >= LONG_TERM_THRESHOLD_DAYS:
#         return "Long-term", LONG_TERM_MULTIPLIER
#     return "Short-term", SHORT_TERM_MULTIPLIER

# DEFAULT_TARGET_CITY_ENTRIES: List[Dict[str, Any]] = [
#     {"region": "North America", "city": "Nassau", "country": "Bahamas"},
#     {"region": "North America", "city": "Los Angeles", "country": "USA", "neighborhood": "Downtown & Convention Center", "hotel_cluster": "JW Marriott / Ritz-Carlton L.A. LIVE"},
#     {"region": "North America", "city": "Las Vegas", "country": "USA", "neighborhood": "The Strip (Paradise)", "hotel_cluster": "MGM Grand & Mandalay Bay"},
#     {"region": "North America", "city": "Seattle", "country": "USA"},
#     {"region": "North America", "city": "Florida", "country": "USA"},
#     {"region": "North America", "city": "San Francisco", "country": "USA", "neighborhood": "SoMa & Financial District", "hotel_cluster": "Hilton Union Square / Marriott Marquis"},
#     {"region": "North America", "city": "Toronto", "country": "Canada"},
#     {"region": "Europe", "city": "Valletta", "country": "Malta"},
#     {"region": "Europe", "city": "London", "country": "United Kingdom", "neighborhood": "City & Canary Wharf", "hotel_cluster": "Hilton Bankside / Novotel Canary Wharf"},
#     {"region": "Europe", "city": "Dublin", "country": "Ireland"},
#     {"region": "Europe", "city": "Lisbon", "country": "Portugal"},
#     {"region": "Europe", "city": "Karlovy Vary", "country": "Czech Republic"},
#     {"region": "Europe", "city": "Amsterdam", "country": "Netherlands"},
#     {"region": "Europe", "city": "San Remo", "country": "Italy"},
#     {"region": "Europe", "city": "Barcelona", "country": "Spain", "neighborhood": "Eixample & Fira Gran Via", "hotel_cluster": "AC Hotel Barcelona / Hyatt Regency Tower"},
#     {"region": "Europe", "city": "Nicosia", "country": "Cyprus"},
#     {"region": "Europe", "city": "Paris", "country": "France"},
#     {"region": "Europe", "city": "Provence", "country": "France"},
#     {"region": "Asia", "city": "Taipei", "country": "Taiwan", "un_dsa_substitute": {"city": "Kuala Lumpur", "country": "Malaysia"}},
#     {"region": "Asia", "city": "Tokyo", "country": "Japan", "neighborhood": "Shinjuku & Roppongi", "hotel_cluster": "Hilton Tokyo / ANA InterContinental"},
#     {"region": "Asia", "city": "Manila", "country": "Philippines"},
#     {"region": "Asia", "city": "Seoul", "country": "Korea, Republic of", "neighborhood": "Gangnam Business District", "hotel_cluster": "Grand InterContinental / Josun Palace"},
#     {"region": "Asia", "city": "Busan", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Jeju Island", "country": "Korea, Republic of"},
#     {"region": "Asia", "city": "Incheon", "country": "Korea, Republic of"},
#     {"region": "Others", "city": "Sydney", "country": "Australia"},
#     {"region": "Others", "city": "Rosario", "country": "Argentina"},
#     {"region": "Others", "city": "Marrakech", "country": "Morocco"},
#     {"region": "Others", "city": "Rio de Janeiro", "country": "Brazil"},
# ]


# def normalize_target_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """대상 도시 항목에 기본값을 채워 넣는다."""
#     entry = dict(entry)
#     entry.setdefault("region", "Others")
#     entry.setdefault("neighborhood", "")
#     entry.setdefault("hotel_cluster", "")
#     entry["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return entry


# def load_target_city_entries() -> List[Dict[str, Any]]:
#     if not os.path.exists(TARGET_CONFIG_FILE):
#         save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
#     try:
#         with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if not isinstance(data, list):
#             raise ValueError("Invalid target city config format")
#     except Exception:
#         data = DEFAULT_TARGET_CITY_ENTRIES
#     return [normalize_target_entry(item) for item in data]


# def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     normalized = [normalize_target_entry(item) for item in entries]
#     with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
#         json.dump(normalized, handle, ensure_ascii=False, indent=2)


# TARGET_CITIES_ENTRIES = load_target_city_entries()


# def get_target_city_entries() -> List[Dict[str, Any]]:
#     if "target_cities_entries" in st.session_state:
#         return st.session_state["target_cities_entries"]
#     return TARGET_CITIES_ENTRIES


# def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
#     st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
#     save_target_city_entries(st.session_state["target_cities_entries"])


# def get_target_cities_grouped(entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
#     entries = entries or get_target_city_entries()
#     grouped: Dict[str, List[Dict[str, Any]]] = {}
#     for entry in entries:
#         grouped.setdefault(entry.get("region", "Others"), []).append(entry)
#     return grouped


# def get_all_target_cities(entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
#     entries = entries or get_target_city_entries()
#     return [normalize_target_entry(entry) for entry in entries]

# # 도시 이름 별칭 매핑
# CITY_ALIASES = {
#     "jeju island": "cheju island", "busan": "pusan", "incheon": "incheon", "marrakech": "marrakesh",
#     "san remo": "san remo", "karlovy vary": "karlovy vary", "lisbon": "lisbon", "valletta": "malta island",
#     "kuala lumpur": "kuala lumpur"
# }

# # --- 도시 메타데이터 및 시즌 설정 ---

# SEASON_BANDS = [
#     {"months": (12, 1, 2), "label": "Peak-Holiday", "factor": 1.06},
#     {"months": (3, 4, 5), "label": "Spring-Shoulder", "factor": 1.02},
#     {"months": (6, 7, 8), "label": "Summer-Peak", "factor": 1.05},
#     {"months": (9, 10, 11), "label": "Autumn-Business", "factor": 1.03},
# ]

# CITY_SEASON_OVERRIDES: Dict[tuple, List[Dict[str, Any]]] = {
#     ("las vegas", "usa"): [
#         {"months": (1, 2), "label": "Winter Convention Peak", "factor": 1.07},
#         {"months": (6, 7, 8), "label": "Summer Off-Peak", "factor": 0.96},
#     ],
#     ("seoul", "korea, republic of"): [
#         {"months": (4, 5, 10), "label": "Cherry Blossom & Fall Peak", "factor": 1.05},
#         {"months": (1, 2), "label": "Winter Off-Peak", "factor": 0.97},
#     ],
#     ("barcelona", "spain"): [
#         {"months": (6, 7, 8), "label": "Summer Tourism Peak", "factor": 1.08},
#     ],
# }


# def get_city_context(city: str, country: str) -> Dict[str, Optional[str]]:
#     key = (city.lower(), country.lower())
#     for entry in get_target_city_entries():
#         if entry["city"].lower() == key[0] and entry["country"].lower() == key[1]:
#             return {
#                 "neighborhood": entry.get("neighborhood"),
#                 "hotel_cluster": entry.get("hotel_cluster"),
#             }
#     return {"neighborhood": None, "hotel_cluster": None}


# def get_current_season_info(city: str, country: str) -> Dict[str, Any]:
#     """해당 월과 도시 설정에 따라 계절 라벨과 계수를 반환한다."""
#     month = datetime.now().month
#     city_key = (city.lower(), country.lower())
#     overrides = CITY_SEASON_OVERRIDES.get(city_key, [])
#     for override in overrides:
#         if month in override["months"]:
#             return {
#                 "label": override["label"],
#                 "factor": override["factor"],
#                 "source": "city_override",
#             }

#     for band in SEASON_BANDS:
#         if month in band["months"]:
#             return {
#                 "label": band["label"],
#                 "factor": band["factor"],
#                 "source": "global_profile",
#             }

#     return {"label": "Standard", "factor": 1.0, "source": "default"}


# def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
#     """이상치를 제거하고 평균값을 계산해 투명하게 제공한다."""
#     if not totals:
#         return {"used_values": [], "removed_values": [], "mean": None}

#     sorted_totals = sorted(totals)
#     if len(sorted_totals) >= 4:
#         try:
#             q1, _, q3 = quantiles(sorted_totals, n=4, method="inclusive")
#             iqr = q3 - q1
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#             filtered = [v for v in sorted_totals if lower_bound <= v <= upper_bound]
#         except (ValueError, StatisticsError):  # type: ignore[name-defined]
#             filtered = sorted_totals
#     else:
#         filtered = sorted_totals

#     if not filtered:
#         filtered = sorted_totals

#     removed_values: List[int] = []
#     filtered_counter = Counter(filtered)
#     for value in sorted_totals:
#         if filtered_counter[value]:
#             filtered_counter[value] -= 1
#         else:
#             removed_values.append(value)

#     computed_mean = mean(filtered) if filtered else None
#     return {
#         "used_values": filtered,
#         "removed_values": removed_values,
#         "mean_raw": computed_mean,
#         "mean": round(computed_mean) if computed_mean is not None else None,
#     }

# # --- 핵심 로직 함수 ---

# def parse_pdf_to_text(uploaded_file):
#     uploaded_file.seek(0)
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     full_text = ""
#     for page_num in range(4, len(doc)):
#         full_text += doc[page_num].get_text("text") + "\n\n"
#     return full_text

# def get_history_files():
#     if not os.path.exists(DATA_DIR):
#         return []
#     files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
#     return sorted(files, reverse=True)

# def save_report_data(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)


# def _sanitize_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(data, dict):
#         return data
#     cities = data.get("cities")
#     if isinstance(cities, list):
#         for city in cities:
#             if isinstance(city, dict):
#                 city["trip_lengths"] = DEFAULT_TRIP_LENGTH.copy()
#     return data


# def load_report_data(filename):
#     filepath = os.path.join(DATA_DIR, filename)
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 return _sanitize_report_data(data)
#             except json.JSONDecodeError: return None
#     return None

# def build_tsv_conversion_prompt():
#     return """
# [Task]
# Convert noisy UN-DSA PDF text snippets into a clean TSV (Tab-Separated Values) table.
# [Guidelines]
# 1. Identify the country (Country) and the area/city (Area) entries inside the extracted text.
# 2. If a country header (for example "USA (US Dollar)") appears once and multiple areas follow, repeat the same country name for every subsequent row until a new country header is encountered.
# 3. Keep only four columns: `Country`, `Area`, `First 60 Days US$`, `Room as % of DSA`. Discard every other column.
# [Output Format]
# Return only the TSV content (one header row plus data rows) with tab separators, no explanations.
# Country	Area	First 60 Days US$	Room as % of DSA
# USA (US Dollar)	Washington D.C.	403	57
# """


# def call_openai_for_tsv_conversion(pdf_chunk, api_key):
#     client = openai.OpenAI(api_key=api_key)
#     system_prompt = build_tsv_conversion_prompt()
#     user_prompt = f"Here is a chunk of text extracted from a UN-DSA PDF. Convert it into TSV following the instructions.\n\n---\n\n{pdf_chunk}"
#     try:
#         response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
#         tsv_content = response.choices[0].message.content
#         if "```" in tsv_content:
#             tsv_content = tsv_content.split('```')[1].strip()
#             if tsv_content.startswith('tsv'): tsv_content = tsv_content[3:].strip()
#         return tsv_content
#     except Exception as e:
#         st.error(f"OpenAI API request failed: {e}")
#         return None

# def process_tsv_data(tsv_content):
#     try:
#         df = pd.read_csv(io.StringIO(tsv_content), sep='\t', on_bad_lines='skip', header=0)
#         df['Country'] = df['Country'].ffill()
#         df.rename(columns={'First 60 Days US$': 'TotalDSA', 'Room as % of DSA': 'RoomPct'}, inplace=True)
#         df = df[['Country', 'Area', 'TotalDSA', 'RoomPct']]
#         df['TotalDSA'] = pd.to_numeric(df['TotalDSA'], errors='coerce')
#         df['RoomPct'] = pd.to_numeric(df['RoomPct'], errors='coerce')
#         df.dropna(subset=['TotalDSA', 'RoomPct', 'Country', 'Area'], inplace=True)
#         df = df.astype({'TotalDSA': int, 'RoomPct': int})
#     except Exception as e:
#         st.error(f"TSV processing error: {e}")
#         return None

#     all_target_cities = get_all_target_cities()
#     final_cities_data = []
#     for target in all_target_cities:
#         city_data = {
#             "city": target["city"],
#             "country_display": target["country"],
#             "notes": "",
#             "neighborhood": target.get("neighborhood"),
#             "hotel_cluster": target.get("hotel_cluster"),
#             "trip_lengths": DEFAULT_TRIP_LENGTH.copy(),
#         }
#         found_row = None
#         search_target = target
#         is_substitute = "un_dsa_substitute" in target
#         if is_substitute: search_target = target["un_dsa_substitute"]
        
#         country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
#         if not country_df.empty:
#             target_city_lower = search_target["city"].lower()
#             target_alias = CITY_ALIASES.get(target_city_lower, target_city_lower)
#             exact_match = country_df[country_df['Area'].str.lower().str.contains(target_alias, na=False)]
#             non_special_rate = exact_match[~exact_match['Area'].str.contains(r'\(', na=False)]
#             if not non_special_rate.empty:
#                 found_row = non_special_rate.iloc[0]
#                 city_data["notes"] = "Exact city match"
#             elif not exact_match.empty:
#                 found_row = exact_match.iloc[0]
#                 city_data["notes"] = "Exact city match (special rate possible)"
#             if found_row is None:
#                 elsewhere_match = country_df[country_df['Area'].str.lower().str.contains('elsewhere|all areas', na=False, regex=True)]
#                 if not elsewhere_match.empty:
#                     found_row = elsewhere_match.iloc[0]
#                     city_data["notes"] = "Applied 'Elsewhere' or 'All Areas' rate"
        
#         if is_substitute and found_row is not None:
#             city_data["notes"] = f"UN-DSA substitute city: {search_target['city']}"
#         if found_row is not None:
#             total_dsa, room_pct = found_row['TotalDSA'], found_row['RoomPct']
#             if 0 < total_dsa and 0 <= room_pct <= 100:
#                 per_diem = round(total_dsa * (1 - room_pct / 100))
#                 city_data["un"] = {"source_row": {"Country": found_row['Country'], "Area": found_row['Area']}, "total_dsa": int(total_dsa), "room_pct": int(room_pct), "per_diem_excl_lodging": per_diem, "status": "ok"}
#             else: city_data["un"] = {"status": "not_found"}
#         else:
#             city_data["un"] = {"status": "not_found"}
#             if not is_substitute: city_data["notes"] = "Could not find matching city in UN-DSA table"
#         city_data["season_context"] = get_current_season_info(city_data["city"], city_data["country_display"])
#         final_cities_data.append(city_data)
#     return {"as_of": datetime.now().strftime("%Y-%m-%d"), "currency": "USD", "cities": final_cities_data}

# def get_market_data_from_ai(
#     city: str,
#     country: str,
#     api_key: str,
#     source_name: str = "",
#     context: Optional[Dict[str, Optional[str]]] = None,
#     season_context: Optional[Dict[str, Any]] = None,
#     menu_samples: Optional[List[Dict[str, Any]]] = None,
# ) -> Dict[str, Any]:
#     """AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
#     client = openai.OpenAI(api_key=api_key)
#     context = context or {}
#     season_context = season_context or {}
#     menu_samples = menu_samples or []

#     request_id = random.randint(10000, 99999)
#     called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

#     def _build_location_block() -> str:
#         lines: List[str] = []
#         if context.get("neighborhood"):
#             lines.append(f"- Primary neighborhood of stay: {context['neighborhood']}")
#         if context.get("hotel_cluster"):
#             lines.append(f"- Typical hotel cluster: {context['hotel_cluster']}")
#         return "\n".join(lines) if lines else "- No specific neighborhood context provided; rely on city-wide business areas."

#     def _build_menu_block() -> str:
#         if not menu_samples:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         snippets = []
#         for sample in menu_samples[:5]:
#             vendor = sample.get("vendor") or sample.get("name") or "Venue"
#             category = sample.get("category") or "General"
#             price = sample.get("price")
#             currency = sample.get("currency", "USD")
#             last_updated = sample.get("last_updated")
#             if price is None:
#                 continue
#             tail = f" (last updated {last_updated})" if last_updated else ""
#             snippets.append(f"- {vendor} ({category}): {currency} {price}{tail}")
#         if not snippets:
#             return "- No direct venue menu data available; use standard mid-range venues."
#         return "Menu price signals:\n" + "\n".join(snippets)

#     location_block = _build_location_block()
#     menu_block = _build_menu_block()
#     season_label = season_context.get("label", "Standard")
#     season_factor = season_context.get("factor", 1.0)
#     season_source = season_context.get("source", "global_profile")

#     prompt = f"""
# You are a corporate travel cost analyst. Request ID: {request_id}.
# Location context:
# {location_block}
# Season context: {season_label} (target multiplier {season_factor}) - source: {season_source}.
# {menu_block}

# For the city of {city}, {country}, provide a realistic, estimated daily cost of living for a business traveler in USD.
# Your response MUST be a JSON object with the following structure and nothing else. Do not add any explanation.

# IMPORTANT: If precise local data for {city} is unavailable, provide a reasonable estimate based on the national or regional average for {country}. It is crucial to provide a numerical estimate rather than returning null for all values.
# Interview insights to respect: breakfast is a simple meal with coffee, lunch is usually at a franchise or the hotel restaurant, dinner is at a local or franchise restaurant with tips included, daily transport is typically one 8km taxi ride mainly for evening meals, and miscellaneous costs cover water, drinks, snacks, toiletries, over-the-counter medicine, and laundry or hair grooming services (hotel laundry for short stays).

# {{
#   "food": {{
#     "description": "Average cost covering a simple breakfast with coffee, a franchise or hotel lunch, and a local or franchise dinner with tips included.",
#     "value": <integer>
#   }},
#   "transport": {{
#     "description": "Estimated cost for one 8km taxi ride used mainly for the evening meal commute, including tip.",
#     "value": <integer>
#   }},
#   "misc": {{
#     "description": "Estimated daily spend on essentials (water, drinks, snacks, toiletries), over-the-counter medication, and laundry or hair grooming services (hotel laundry for short stays).",
#     "value": <integer>
#   }}
# }}
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             temperature=0.4,
#         )
#         raw_content = response.choices[0].message.content
#         data = json.loads(raw_content)

#         food = data.get("food", {}).get("value")
#         transport = data.get("transport", {}).get("value")
#         misc = data.get("misc", {}).get("value")

#         food_val = food if isinstance(food, int) else 0
#         transport_val = transport if isinstance(transport, int) else 0
#         misc_val = misc if isinstance(misc, int) else 0

#         meta = {
#             "source_name": source_name,
#             "request_id": request_id,
#             "prompt": prompt.strip(),
#             "response_raw": raw_content,
#             "called_at": called_at,
#             "season_context": season_context,
#             "location_context": context,
#             "menu_samples_used": menu_samples[:5],
#         }

#         if food_val == 0 and transport_val == 0 and misc_val == 0:
#             return {
#                 "status": "na",
#                 "notes": f"{source_name}: AI가 유효한 값을 찾지 못했습니다.",
#                 "meta": meta,
#             }

#         total = food_val + transport_val + misc_val
#         notes = f"총액 ${total} (Food ${food_val}, Transport ${transport_val}, Misc ${misc_val})"
#         return {
#             "food": food_val,
#             "transport": transport_val,
#             "misc": misc_val,
#             "total": total,
#             "status": "ok",
#             "notes": notes,
#             "meta": meta,
#         }

#     except Exception as e:
#         return {
#             "status": "na",
#             "notes": f"{source_name} AI data extraction failed: {e}",
#             "meta": {
#                 "source_name": source_name,
#                 "request_id": request_id,
#                 "prompt": prompt.strip(),
#                 "called_at": called_at,
#                 "season_context": season_context,
#                 "location_context": context,
#                 "menu_samples_used": menu_samples[:5],
#                 "error": str(e),
#             },
#         }


# def generate_markdown_report(report_data):
#     md = f"# Business Travel Daily Allowance Report\n\n"
#     md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"
#     weights_cfg = load_weight_config()
#     md += f"**Weight mix:** UN {weights_cfg.get('un_weight', 0.5):.0%} / AI {weights_cfg.get('ai_weight', 0.5):.0%}\n\n"

#     valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
#     if valid_allowances:
#         md += "## 1. Summary\n\n"
#         md += (
#             f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
#             f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
#         )

#     md += "## 2. City Details\n\n"
#     table_data = []
#     all_reference_links: Set[str] = set()
#     all_target_cities = get_all_target_cities()
#     report_cities_map = {(c.get('city', '').lower(), c.get('country_display', '').lower()): c for c in report_data.get('cities', [])}
#     for target in all_target_cities:
#         city_data = report_cities_map.get((target['city'].lower(), target['country'].lower()))
#         if city_data:
#             un_data = city_data.get('un', {})
#             ai_summary = city_data.get('ai_summary', {})
#             season_context = city_data.get('season_context', {})

#             un_val = f"$ {un_data.get('per_diem_excl_lodging')}" if un_data.get('status') == 'ok' else "N/A"
#             final_val = f"$ {city_data.get('final_allowance')}" if city_data.get('final_allowance') is not None else "N/A"
#             delta = f"{city_data.get('delta_vs_un_pct')}%" if city_data.get('delta_vs_un_pct') != 'N/A' else 'N/A'
#             ai_season_avg = ai_summary.get('season_adjusted_mean_rounded')
#             ai_runs_used = ai_summary.get('successful_runs', 0)
#             ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#             removed_totals = ai_summary.get('removed_totals') or []
#             reference_links = city_data.get('reference_links') or ai_summary.get('reference_links') or []
#             for link in reference_links:
#                 if isinstance(link, str) and link.strip():
#                     all_reference_links.add(link.strip())

#             row = {
#                 'City': city_data.get('city', 'N/A'),
#                 'Country': city_data.get('country_display', 'N/A'),
#                 'UN-DSA (1 day)': un_val,
#                 'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
#                 'AI runs used': f"{ai_runs_used}/{ai_attempts}",
#                 'Season label': season_context.get('label', 'Standard'),
#                 'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 market_data = city_data.get(f"market_data_{j}", {})
#                 md_val = f"$ {market_data.get('total')}" if market_data.get('status') == 'ok' else 'N/A'
#                 row[f"AI run {j}"] = md_val

#             row.update({
#                 'Final allowance': final_val,
#                 'Delta vs UN (%)': delta,
#                 'Trip types': ', '.join(city_data.get('trip_lengths', [])) if city_data.get('trip_lengths') else '-',
#                 'Notes': city_data.get('notes', ''),
#             })
#             table_data.append(row)

#     df = pd.DataFrame(table_data)
#     md += df.to_markdown(index=False)
#     md += "\n\n*AI provenance, prompts, and menu references are stored with each run and visible in the app detail panels.*\n\n"

#     md += (
#         "---\n"
#         "## 3. Methodology\n\n"
#         "1. **Baseline (UN-DSA)**\n"
#         "   - Extract 'Per Diem Excl. Lodging' from the official UN PDF tables.\n"
#         "   - Normalize the data as TSV to align city/country names.\n\n"
#         "2. **Market data (AI)**\n"
#         "   - Query OpenAI GPT-4o, GPT-4o-mini ten times per city with local context, hotel clusters, and season tags.\n"
#         "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
#         "3. **Post-processing**\n"
#         "   - Remove outliers via the IQR rule and compute averages.\n"
#         "   - Apply season factors and blend with UN-DSA using configured weights.\n"
#         "   - Multiply by grade ratios to produce per-level allowances.\n\n"
#         "---\n"
#         "## 4. Sources\n\n"
#         "- UN-DSA Circular (International Civil Service Commission)\n"
#         "- Mercer Cost of Living (2025 edition)\n"
#         "- Numbeo Cost of Living Index (2025 snapshot)\n"
#         "- Expatistan Cost of Living Guide\n"
#     )

#     return md




# # --- 스트림릿 UI 구성 ---
# st.set_page_config(layout="wide")
# st.title("AICP: 출장 일비 계산 & 조회 시스템 (v14.0 - 고급 분석 모델)")

# if 'latest_analysis_result' not in st.session_state:
#     st.session_state.latest_analysis_result = None
# if 'target_cities_entries' not in st.session_state:
#     st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]
# if 'weight_config' not in st.session_state:
#     st.session_state.weight_config = load_weight_config()
# else:
#     st.session_state.weight_config = _normalize_weight_config(st.session_state.weight_config)

# ui_settings = load_ui_settings()
# stored_employee_tab_visible = bool(ui_settings.get("show_employee_tab", True))
# if "employee_tab_visibility" not in st.session_state:
#     st.session_state.employee_tab_visibility = stored_employee_tab_visible
# employee_tab_visible = bool(st.session_state.get("employee_tab_visibility", stored_employee_tab_visible))
# section_visibility_default = _normalize_employee_sections(ui_settings.get("employee_sections"))
# if "employee_sections_visibility" not in st.session_state:
#     st.session_state.employee_sections_visibility = section_visibility_default
# else:
#     st.session_state.employee_sections_visibility = _normalize_employee_sections(st.session_state.employee_sections_visibility)
# employee_sections_visibility = st.session_state.employee_sections_visibility




# if employee_tab_visible:
#     employee_tab, admin_tab = st.tabs(["일비 조회 (직원용)", "보고서 분석 (관리자)"])
# else:
#     (admin_tab,) = st.tabs(["보고서 분석 (관리자)"])
#     employee_tab = None

# if employee_tab is not None:
#     with employee_tab:
#         st.header("도시별 출장 일비 조회")
#         history_files = get_history_files()
#         if not history_files:
#             st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
#         else:
#             if "selected_report_file" not in st.session_state:
#                 st.session_state["selected_report_file"] = history_files[0]
#             if st.session_state["selected_report_file"] not in history_files:
#                 st.session_state["selected_report_file"] = history_files[0]
#             selected_file = st.session_state["selected_report_file"]
#             report_data = load_report_data(selected_file)
#             if report_data and 'cities' in report_data and report_data['cities']:
#                 cities_df = pd.DataFrame(report_data['cities'])
#                 target_entries = get_target_city_entries()
#                 countries = sorted({entry['country'] for entry in target_entries})

                
#                 col_country, col_city = st.columns(2)
#                 with col_country:
#                     selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
#                     sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
#                 filtered_cities_all = sorted({
#                     entry['city'] for entry in target_entries if entry['country'] == sel_country
#                 })
#                 with col_city:
#                     if filtered_cities_all:
#                         sel_city = st.selectbox("도시:", filtered_cities_all, key=f"city_{selected_file}")
#                     else:
#                         sel_city = None
#                         st.warning("선택한 국가에 등록된 도시가 없습니다.")

#                 col_start, col_end, col_level = st.columns([1, 1, 1])
#                 with col_start:
#                     trip_start = st.date_input(
#                         "출장 시작일",
#                         value=datetime.today().date(),
#                         key=f"trip_start_{selected_file}",
#                     )
#                 with col_end:
#                     trip_end = st.date_input(
#                         "출장 종료일",
#                         value=datetime.today().date() + timedelta(days=4),
#                         key=f"trip_end_{selected_file}",
#                     )
#                 with col_level:
#                     sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")

#                 if isinstance(trip_start, datetime):
#                     trip_start = trip_start.date()
#                 if isinstance(trip_end, datetime):
#                     trip_end = trip_end.date()

#                 trip_valid = trip_end >= trip_start
#                 if not trip_valid:
#                     st.error("종료일은 시작일 이후여야 합니다.")
#                     trip_days = None
#                     trip_term = "Short-term"
#                     trip_multiplier = SHORT_TERM_MULTIPLIER
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                 else:
#                     trip_days = (trip_end - trip_start).days + 1
#                     trip_term, trip_multiplier = classify_trip_duration(trip_days)
#                     trip_term_label = TRIP_TERM_LABELS.get(trip_term, trip_term)
#                     st.caption(f"자동 분류된 출장 유형: {trip_term_label} · {trip_days}일 일정")

#                 if sel_city:
#                     filtered_trip_cities = []
#                     for entry in target_entries:
#                         if entry['country'] != sel_country or entry['city'] != sel_city:
#                             continue
#                         if trip_valid and trip_term not in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS):
#                             continue
#                         filtered_trip_cities.append(entry['city'])
#                     if trip_valid and not filtered_trip_cities:
#                         st.warning("이 기간에 해당하는 도시 데이터가 없습니다. 출장 유형을 '숏텀'으로 조정하거나 도시 설정을 확인하세요.")
#                         sel_city = None

#                 if trip_valid and sel_city and sel_level:
#                     city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
#                     final_allowance = city_data.get('final_allowance')
#                     st.subheader(f"{sel_country} - {sel_city} 결과")
#                     if final_allowance:
#                         level_ratio = JOB_LEVEL_RATIOS[sel_level]
#                         adjusted_daily_allowance = round(final_allowance * trip_multiplier)
#                         level_daily_allowance = round(adjusted_daily_allowance * level_ratio)
#                         trip_total_allowance = level_daily_allowance * trip_days
#                         st.markdown(
#                             f"""
#                             <div style='margin-top:0.8rem;padding:1.8rem;border-radius:18px;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:#fff;box-shadow:0 12px 28px rgba(30,60,114,0.35);text-align:center;'>
#                                 <div style='font-size:2.4rem;font-weight:800;letter-spacing:0.02em;'>
#                                     {sel_level.split(' ')[0]} 기준 예상 일비 총액 $ {trip_total_allowance:,} = $ {level_daily_allowance:,} × {trip_days}일 일정 × {trip_term_label}(×{trip_multiplier:.2f})
#                                 </div>
#                             </div>
#                             """,
#                             unsafe_allow_html=True,
#                         )
#                     else:
#                         st.metric(f"{sel_level.split(' ')[0]} 일일 권장 일비", "금액 없음")

#                     #st.markdown("---")
#                     #st.write("**세부 산출 근거 (일비 기준)**")

#                     menu_samples = city_data.get('menu_samples') or []

#                     detail_cards_visible = any([
#                         employee_sections_visibility["show_un_basis"],
#                         employee_sections_visibility["show_ai_estimate"],
#                         employee_sections_visibility["show_weighted_result"],
#                         employee_sections_visibility["show_ai_market_detail"],
#                     ])
#                     extra_content_visible = (
#                         employee_sections_visibility["show_provenance"]
#                         or (employee_sections_visibility["show_menu_samples"] and menu_samples)
#                     )

#                     if detail_cards_visible or extra_content_visible:
#                         st.markdown("---")
#                         st.write("**세부 산출 근거 (일비 기준)**")
#                         un_data = city_data.get('un', {})
#                         ai_summary = city_data.get('ai_summary', {})
#                         season_context = city_data.get('season_context', {})

#                         ai_avg = ai_summary.get('season_adjusted_mean_rounded')
#                         ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
#                         ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
#                         removed_totals = ai_summary.get('removed_totals') or []
#                         season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
#                         season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

#                         ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
#                         if removed_totals:
#                             ai_notes_parts.append(f"제외값 {removed_totals}")
#                         if season_label:
#                             ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
#                         ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"

#                         weights_cfg = get_weight_config()

#                         un_base = None
#                         un_display = None
#                         if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
#                             un_base = un_data['per_diem_excl_lodging']
#                             un_display = round(un_base * trip_multiplier)

#                         ai_display = round(ai_avg * trip_multiplier) if ai_avg is not None else None
#                         weighted_display = round(final_allowance * trip_multiplier) if final_allowance is not None else None

#                         first_row_keys = []
#                         if employee_sections_visibility["show_un_basis"]:
#                             first_row_keys.append("un")
#                         if employee_sections_visibility["show_ai_estimate"]:
#                             first_row_keys.append("ai")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             first_row_keys.append("weighted")

#                         if first_row_keys:
#                             first_row_cols = st.columns(len(first_row_keys))
#                             for key, col in zip(first_row_keys, first_row_cols):
#                                 with col:
#                                     if key == "un":
#                                         st.info("UN-DSA 기준")
#                                         if un_display is not None:
#                                             st.metric("일비", f"$ {un_display:,}")
#                                             if trip_term == "Long-term":
#                                                 st.caption(f"숏텀 기준 $ {un_base:,} → 롱텀 $ {un_display:,}")
#                                             else:
#                                                 st.caption(f"숏텀 기준 $ {un_base:,}")
#                                         else:
#                                             st.metric("일비", "N/A")
#                                             st.caption(city_data.get("notes", ""))
#                                     elif key == "ai":
#                                         st.info("AI 시장 추정 (시즌 보정)")
#                                         if ai_display is not None:
#                                             st.metric("일비", f"$ {ai_display:,}")
#                                             caption_parts = [ai_notes]
#                                             caption_parts.append(f"숏텀 기준 $ {ai_avg:,}")
#                                             if trip_term == "Long-term":
#                                                 caption_parts.append(f"롱텀 적용 $ {ai_display:,}")
#                                             st.caption(" | ".join([part for part in caption_parts if part]))
#                                         else:
#                                             st.metric("일비", "N/A")
#                                             st.caption(ai_notes)
#                                     else:
#                                         st.info("가중 평균 결과")
#                                         if weighted_display is not None:
#                                             st.metric("일비", f"$ {weighted_display:,}")
#                                             caption_parts = [
#                                                 f"Blend of UN-DSA ({weights_cfg['un_weight']:.0%}) and AI estimate ({weights_cfg['ai_weight']:.0%})",
#                                                 f"숏텀 기준 $ {final_allowance:,}",
#                                             ]
#                                             if trip_term == "Long-term":
#                                                 caption_parts.append(f"롱텀 적용 $ {weighted_display:,}")
#                                             st.caption(" | ".join(caption_parts))
#                                         else:
#                                             st.metric("일비", "N/A")
#                                             st.caption(city_data.get("notes", ""))

#                         second_row_keys = []
#                         if employee_sections_visibility["show_ai_market_detail"]:
#                             second_row_keys.append("ai_market")
#                         if employee_sections_visibility["show_weighted_result"]:
#                             second_row_keys.append("weighted_detail")

#                         if second_row_keys:
#                             second_row_cols = st.columns(len(second_row_keys))
#                             for key, col in zip(second_row_keys, second_row_cols):
#                                 with col:
#                                     if key == "ai_market":
#                                         st.info("AI Market Estimate (Season-Adjusted)")
#                                         if ai_display is not None:
#                                             st.metric("Daily Allowance", f"$ {ai_display:,}")
#                                             st.caption(ai_notes)
#                                         else:
#                                             st.metric("Daily Allowance", "N/A")
#                                             st.caption(ai_notes)
#                                     else:
#                                         st.info("Weighted Combined Result")
#                                         if weighted_display is not None:
#                                             st.metric("Daily Allowance", f"$ {weighted_display:,}")
#                                             st.caption(f"Blend of UN-DSA ({weights_cfg['un_weight']:.0%}) and AI estimate ({weights_cfg['ai_weight']:.0%})")
#                                         else:
#                                             st.metric("Daily Allowance", "N/A")

#                         if employee_sections_visibility["show_provenance"]:
#                             with st.expander("AI provenance & prompts"):
#                                 provenance_payload = {
#                                     "season_context": season_context,
#                                     "ai_summary": ai_summary,
#                                     "ai_runs": city_data.get('ai_provenance', []),
#                                     "reference_links": build_reference_link_lines(menu_samples, max_items=8),
#                                     "weights": weights_cfg,
#                                 }
#                                 st.json(provenance_payload)

#                         if employee_sections_visibility["show_menu_samples"] and menu_samples:
#                             with st.expander("Reference menu samples"):
#                                 link_lines = build_reference_link_lines(menu_samples, max_items=8)
#                                 if link_lines:
#                                     st.markdown("**Direct links**")
#                                     for link_line in link_lines:
#                                         st.markdown(f"- {link_line}")
#                                     st.markdown("---")
#                                 st.table(pd.DataFrame(menu_samples))
#                     else:
#                         st.info("관리자가 세부 산출 근거를 숨겼습니다.")

# ACCESS_CODE_KEY = "admin_access_code_valid"
# ACCESS_CODE_VALUE = "VUCA0207"

# with admin_tab:
#     if not st.session_state.get(ACCESS_CODE_KEY, False):
#         with st.form("admin_access_form"):
#             input_code = st.text_input("Access Code", type="password")
#             submitted = st.form_submit_button("Enter")
#         if submitted:
#             if input_code == ACCESS_CODE_VALUE:
#                 st.session_state[ACCESS_CODE_KEY] = True
#                 st.success("Access granted.")
#             else:
#                 st.error("Access Code가 올바르지 않습니다.")
#         st.stop()

#     visibility_toggle = st.toggle("직원용 탭 노출", value=employee_tab_visible, key="employee_tab_visibility")
#     if visibility_toggle != stored_employee_tab_visible:
#         updated_settings = dict(ui_settings)
#         updated_settings["show_employee_tab"] = visibility_toggle
#         updated_settings["employee_sections"] = employee_sections_visibility
#         save_ui_settings(updated_settings)
#         ui_settings = updated_settings
#         st.success("직원용 탭 노출 상태가 업데이트되었습니다.")

#     st.subheader("직원 화면 노출 설정")
#     section_toggle_values: Dict[str, bool] = {}
#     for section_key, label in EMPLOYEE_SECTION_LABELS:
#         current_value = employee_sections_visibility.get(section_key, EMPLOYEE_SECTION_DEFAULTS[section_key])
#         section_toggle_values[section_key] = st.toggle(
#             label,
#             value=current_value,
#             key=f"employee_section_toggle_{section_key}",
#         )
#     if section_toggle_values != employee_sections_visibility:
#         updated_settings = dict(ui_settings)
#         updated_settings["employee_sections"] = section_toggle_values
#         save_ui_settings(updated_settings)
#         ui_settings["employee_sections"] = section_toggle_values
#         st.session_state.employee_sections_visibility = section_toggle_values
#         employee_sections_visibility = section_toggle_values
#         st.success("직원 화면 노출 설정이 업데이트되었습니다.")

#     st.subheader("보고서 버전 관리")
#     history_files = get_history_files()
#     if history_files:
#         if "selected_report_file" not in st.session_state:
#             st.session_state["selected_report_file"] = history_files[0]
#         if st.session_state["selected_report_file"] not in history_files:
#             st.session_state["selected_report_file"] = history_files[0]
#         default_index = history_files.index(st.session_state["selected_report_file"])
#         selected_file = st.selectbox("보고서 버전을 선택해 주세요.", history_files, index=default_index, key="admin_report_file_select")
#         st.session_state["selected_report_file"] = selected_file
#     else:
#         st.info("선택하신 보고서 버전이 없습니다.")

#     st.header("목표 도시 관리")
#     entries_df = pd.DataFrame(get_target_city_entries())
#     if not entries_df.empty:
#         entries_display = entries_df.copy()
#         entries_display["trip_lengths"] = DEFAULT_TRIP_LENGTH[0]
#         st.dataframe(entries_display[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], width="stretch")
#     else:
#         st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

#     st.subheader("비중 설정")
#     current_weights = get_weight_config()
#     st.caption(f"Current weights -> UN {current_weights.get('un_weight', 0.5):.0%} / AI {current_weights.get('ai_weight', 0.5):.0%}")
#     with st.form("weight_config_form"):
#         un_weight_input = st.slider("UN-DSA weight", min_value=0.0, max_value=1.0, value=float(current_weights.get("un_weight", 0.5)), step=0.05, format="%.2f")
#         ai_weight_preview = max(0.0, 1.0 - un_weight_input)
#         st.write(f"AI market estimate weight: **{ai_weight_preview:.2f}**")
#         st.caption("Weights are normalised to sum to 1.0 when saved.")
#         weight_submit = st.form_submit_button("Save weights")
#     if weight_submit:
#         updated = update_weight_config(un_weight_input, ai_weight_preview)
#         st.success(f"Weights saved (UN {updated['un_weight']:.2f} / AI {updated['ai_weight']:.2f})")

#     existing_regions = sorted({entry["region"] for entry in get_target_city_entries()})
#     st.subheader("신규 도시 추가")
#     with st.form("add_target_city_form", clear_on_submit=True):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             region_options = existing_regions + ["기타 (직접 입력)"]
#             region_choice = st.selectbox("지역", region_options, key="add_region_choice")
#             new_region = ""
#             if region_choice == "기타 (직접 입력)":
#                 new_region = st.text_input("새 지역 이름", key="add_region_text")
#         with col_b:
#             trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=DEFAULT_TRIP_LENGTH, key="add_trip_lengths")

#         col_c, col_d = st.columns(2)
#         with col_c:
#             city_name = st.text_input("도시", key="add_city")
#             neighborhood = st.text_input("세부 지역 (선택)", key="add_neighborhood")
#         with col_d:
#             country_name = st.text_input("국가", key="add_country")
#             hotel_cluster = st.text_input("추천 호텔 클러스터 (선택)", key="add_hotel_cluster")

#         with st.expander("UN-DSA 대체 도시 (선택)"):
#             substitute_city = st.text_input("대체 도시", key="add_sub_city")
#             substitute_country = st.text_input("대체 국가", key="add_sub_country")

#         add_submitted = st.form_submit_button("추가")

#     if add_submitted:
#         region_value = new_region.strip() if region_choice == "기타 (직접 입력)" else region_choice
#         if not region_value or not city_name.strip() or not country_name.strip():
#             st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#         else:
#             current_entries = get_target_city_entries()
#             canonical_key = (region_value.lower(), country_name.strip().lower(), city_name.strip().lower())
#             duplicate_exists = any(
#                 (entry.get("region", "").lower(), entry.get("country", "").lower(), entry.get("city", "").lower()) == canonical_key
#                 for entry in current_entries
#             )
#             if duplicate_exists:
#                 st.warning("동일한 항목이 이미 등록되어 있습니다.")
#             else:
#                 new_entry = {
#                     "region": region_value,
#                     "country": country_name.strip(),
#                     "city": city_name.strip(),
#                     "neighborhood": neighborhood.strip(),
#                     "hotel_cluster": hotel_cluster.strip(),
#                     "trip_lengths": trip_lengths_selected or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if substitute_city.strip() and substitute_country.strip():
#                     new_entry["un_dsa_substitute"] = {
#                         "city": substitute_city.strip(),
#                         "country": substitute_country.strip(),
#                     }
#                 current_entries.append(new_entry)
#                 set_target_city_entries(current_entries)
#                 st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
#                 st.rerun()
#     st.subheader("기존 도시 편집/삭제")
#     current_entries = get_target_city_entries()
    
#     if current_entries:
#         # 1. 드롭다운 옵션 구성
#         options = {
#             f"{entry['region']} | {entry['country']} | {entry['city']}": idx
#             for idx, entry in enumerate(current_entries)
#         }
#         sorted_labels = list(options.keys())

#         # 2. on_change 콜백 함수 정의 (위젯보다 먼저)
#         def _sync_edit_form_from_selection():
#             # 드롭다운에서 현재 선택된 값을 가져옴
#             selected_idx = options[st.session_state.edit_city_selector]
#             selected_entry = current_entries[selected_idx]
            
#             # session_state의 값을 선택된 도시의 데이터로 강제 업데이트
#             st.session_state.edit_region = selected_entry.get("region", "")
#             st.session_state.edit_city = selected_entry.get("city", "")
#             st.session_state.edit_neighborhood = selected_entry.get("neighborhood", "")
#             st.session_state.edit_country = selected_entry.get("country", "")
#             st.session_state.edit_hotel = selected_entry.get("hotel_cluster", "")
            
#             # 출장 기간 (trip_lengths) 설정
#             existing_trip_lengths = [t for t in selected_entry.get("trip_lengths", []) if t in TRIP_LENGTH_OPTIONS]
#             st.session_state.edit_trip_lengths = existing_trip_lengths or DEFAULT_TRIP_LENGTH.copy()
            
#             # UN-DSA 대체 도시 설정
#             sub_data = selected_entry.get("un_dsa_substitute") or {}
#             st.session_state.edit_sub_city = sub_data.get("city", "")
#             st.session_state.edit_sub_country = sub_data.get("country", "")

#         # 3. 드롭다운(Selectbox)에 on_change 콜백 연결
#         selected_label = st.selectbox(
#             "편집할 도시를 선택하세요", 
#             sorted_labels, 
#             key="edit_city_selector",
#             on_change=_sync_edit_form_from_selection  # <-- [수정] 콜백 함수 연결
#         )

#         # 4. 페이지 첫 로드 시 폼을 채우기 위한 초기화
#         if "edit_region" not in st.session_state and sorted_labels:
#             # 첫 로드 시, selectbox의 기본값(첫 번째 항목)에 맞춰 폼을 채움
#             _sync_edit_form_from_selection()

#         # 5. 폼 내부 위젯에서 'value=' 제거하고 'key='만 사용
#         with st.form("edit_target_city_form"):
#             col_e, col_f = st.columns(2)
#             with col_e:
#                 # [수정] value=... 제거
#                 region_edit = st.text_input("지역", key="edit_region")
#                 city_edit = st.text_input("도시", key="edit_city")
#                 neighborhood_edit = st.text_input("세부 지역 (선택)", key="edit_neighborhood")
#             with col_f:
#                 # [수정] value=... 제거
#                 country_edit = st.text_input("국가", key="edit_country")
#                 hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", key="edit_hotel")

#             # [수정] default=... 대신 key=... 사용
#             trip_lengths_edit = st.multiselect(
#                 "출장 기간",
#                 TRIP_LENGTH_OPTIONS,
#                 key="edit_trip_lengths", # 'default' 대신 'key'로 상태 관리
#             )

#             with st.expander("UN-DSA 대체 도시 (선택)"):
#                 # [수정] value=... 제거
#                 sub_city_edit = st.text_input("대체 도시", key="edit_sub_city")
#                 sub_country_edit = st.text_input("대체 국가", key="edit_sub_country")

#             col_btn1, col_btn2 = st.columns(2)
#             with col_btn1:
#                 update_btn = st.form_submit_button("변경사항 저장")
#             with col_btn2:
#                 delete_btn = st.form_submit_button("삭제", type="secondary")

#         # 6. 저장/삭제 로직은 session_state에서 값을 읽어오도록 수정
#         if update_btn:
#             # [수정] 위젯 변수(region_edit) 대신 st.session_state에서 직접 값을 읽음
#             if (not st.session_state.edit_region.strip() or 
#                 not st.session_state.edit_city.strip() or 
#                 not st.session_state.edit_country.strip()):
#                 st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
#             else:
#                 current_entries[options[selected_label]] = {
#                     "region": st.session_state.edit_region.strip(),
#                     "country": st.session_state.edit_country.strip(),
#                     "city": st.session_state.edit_city.strip(),
#                     "neighborhood": st.session_state.edit_neighborhood.strip(),
#                     "hotel_cluster": st.session_state.edit_hotel.strip(),
#                     "trip_lengths": st.session_state.edit_trip_lengths or DEFAULT_TRIP_LENGTH.copy(),
#                 }
#                 if st.session_state.edit_sub_city.strip() and st.session_state.edit_sub_country.strip():
#                     current_entries[options[selected_label]]["un_dsa_substitute"] = {
#                         "city": st.session_state.edit_sub_city.strip(),
#                         "country": st.session_state.edit_sub_country.strip(),
#                     }
#                 else:
#                     current_entries[options[selected_label]].pop("un_dsa_substitute", None)

#                 set_target_city_entries(current_entries)
#                 st.success("수정을 완료했습니다.")
#                 st.rerun()  # <-- [확인] 이미 올바르게 수정되어 있음
        
#         if delete_btn:
#             del current_entries[options[selected_label]]
#             set_target_city_entries(current_entries)
#             st.warning("선택한 항목을 삭제했습니다.")
#             st.rerun() # <-- [확인] 이미 올바르게 수정되어 있음
#     else:
#         st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

#     st.divider()
#     st.subheader("UN-DSA (PDF) 분석")
#     st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요.")
#     uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")
#     if uploaded_file and st.button("AI 분석 실행", type="primary"):
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
#         else:
#             st.session_state.latest_analysis_result = None
#             with st.spinner("PDF를 처리하는 중입니다..."):
#                 progress_bar = st.progress(0, text="PDF 텍스트 추출 중...")
#                 full_text = parse_pdf_to_text(uploaded_file)

#                 CHUNK_SIZE = 15000
#                 text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
#                 all_tsv_lines = []

#                 analysis_failed = False
#                 for i, chunk in enumerate(text_chunks):
#                     progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
#                     chunk_tsv = call_openai_for_tsv_conversion(chunk, openai_api_key)
#                     if chunk_tsv:
#                         lines = chunk_tsv.strip().split('\n')
#                         if not all_tsv_lines:
#                             all_tsv_lines.extend(lines)
#                         else:
#                             all_tsv_lines.extend(lines[1:])
#                     else:
#                         analysis_failed = True
#                         break

#                 if not analysis_failed:
#                     processed_data = process_tsv_data("\n".join(all_tsv_lines))
#                     if processed_data:
#                         total_cities = len(processed_data["cities"])
#                         for i, city_data in enumerate(processed_data["cities"]):
#                             city_name, country_name = city_data["city"], city_data["country_display"]
#                             progress_text = f"AI 추정치 계산 중... ({i+1}/{total_cities}) {city_name}"
#                             progress_bar.progress((i + 1) / max(total_cities, 1), text=progress_text)

#                             city_context = {
#                                 "neighborhood": city_data.get("neighborhood"),
#                                 "hotel_cluster": city_data.get("hotel_cluster"),
#                             }
#                             season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
#                             menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
#                             city_data["menu_samples"] = menu_samples
#                             city_data["reference_links"] = build_reference_link_lines(menu_samples, max_items=8)
#                             ai_totals_source: List[int] = []
#                             ai_meta_runs: List[Dict[str, Any]] = []

#                             for j in range(1, NUM_AI_CALLS + 1):
#                                 source_name = f"  {j}"
#                                 market_result = get_market_data_from_ai(
#                                     city_name,
#                                     country_name,
#                                     openai_api_key,
#                                     source_name,
#                                     context=city_context,
#                                     season_context=season_context,
#                                     menu_samples=menu_samples,
#                                 )
#                                 city_data[f"market_data_{j}"] = market_result
#                                 if market_result.get("status") == 'ok' and market_result.get("total") is not None:
#                                     ai_totals_source.append(market_result["total"])
#                                 if "meta" in market_result:
#                                     ai_meta_runs.append(market_result["meta"])
#                                 if j < NUM_AI_CALLS:
#                                     time.sleep(1)

#                             city_data["ai_provenance"] = ai_meta_runs

#                             final_allowance = None
#                             un_per_diem = city_data.get("un", {}).get("per_diem_excl_lodging")

#                             ai_stats = aggregate_ai_totals(ai_totals_source)
#                             season_factor = (season_context or {}).get("factor", 1.0)
#                             ai_base_mean = ai_stats.get("mean_raw")
#                             ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None
#                             weights_cfg = get_weight_config()

#                             city_data["ai_summary"] = {
#                                 "raw_totals": ai_totals_source,
#                                 "used_totals": ai_stats.get("used_values", []),
#                                 "removed_totals": ai_stats.get("removed_values", []),
#                                 "mean_base": ai_base_mean,
#                                 "mean_base_rounded": ai_stats.get("mean"),
#                                 "season_factor": season_factor,
#                                 "season_label": (season_context or {}).get("label"),
#                                 "season_adjusted_mean_raw": ai_season_adjusted,
#                                 "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
#                                 "successful_runs": len(ai_stats.get("used_values", [])),
#                                 "attempted_runs": NUM_AI_CALLS,
#                                 "reference_links": city_data.get("reference_links", []),
#                                 "weighted_average_components": {
#                                     "un_per_diem": un_per_diem,
#                                     "ai_season_adjusted": ai_season_adjusted,
#                                     "weights": {"UN": weights_cfg["un_weight"], "AI": weights_cfg["ai_weight"]},
#                                 },
#                             }

#                             if un_per_diem and ai_season_adjusted is not None:
#                                 weighted_average = (un_per_diem * weights_cfg["un_weight"]) + (ai_season_adjusted * weights_cfg["ai_weight"])
#                                 final_allowance = round(weighted_average)
#                             elif un_per_diem:
#                                 final_allowance = round(un_per_diem)
#                             elif ai_season_adjusted is not None:
#                                 final_allowance = round(ai_season_adjusted)

#                             city_data["final_allowance"] = final_allowance

#                             if final_allowance and un_per_diem and un_per_diem > 0:
#                                 city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
#                             else:
#                                 city_data["delta_vs_un_pct"] = "N/A"

#                         save_report_data(processed_data)
#                         st.session_state.latest_analysis_result = processed_data
#                         st.success("AI analysis completed.")

#     if st.session_state.latest_analysis_result:
#         st.markdown("---")
#         st.subheader("Latest Analysis Summary")
#         df_data = []
#         for city in st.session_state.latest_analysis_result['cities']:
#             row = {
#                 'City': city.get('city', 'N/A'),
#                 'Country': city.get('country_display', 'N/A'),
#                 'UN-DSA': city.get('un', {}).get('per_diem_excl_lodging'),
#             }
#             for j in range(1, NUM_AI_CALLS + 1):
#                 row[f"AI {j}"] = city.get(f'market_data_{j}', {}).get('total')

#             row.update({
#                 'Final Allowance': city.get('final_allowance'),
#                 'Delta (%)': city.get('delta_vs_un_pct'),
#                 'Trip Lengths': DEFAULT_TRIP_LENGTH[0],
#                 'Notes': city.get('notes', ''),
#             })
#             df_data.append(row)

#         st.dataframe(pd.DataFrame(df_data))
#         with st.expander("View generated markdown report"):
#             st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))
