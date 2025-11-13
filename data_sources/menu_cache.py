# data_sources/menu_cache.py

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- [중요] 파일 경로 설정 ---
# 이 파일(menu_cache.py)의 위치를 기준으로 한 단계 상위 폴더(프로젝트 루트)를 찾습니다.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 폴더 안에 있는 'analysis_history' 폴더를 데이터 디렉터리로 지정합니다.
_DATA_DIR = os.path.join(os.path.dirname(_MODULE_DIR), "analysis_history")
# 최종 캐시 파일 경로
MENU_CACHE_FILE = os.path.join(_DATA_DIR, "menu_cache.json")

# --- 내부 헬퍼 함수 ---

def _get_current_timestamp() -> str:
    """ YYYY-MM-DD 형식의 현재 날짜 문자열을 반환합니다. """
    return datetime.now().strftime("%Y-%m-%d")

def load_all_cache() -> List[Dict[str, Any]]:
    """
    디스크(JSON)에서 *전체* 메뉴 캐시 목록을 로드합니다.
    (관리자 탭의 '데이터 캐시 관리' UI에서 사용됩니다.)
    """
    if not os.path.exists(MENU_CACHE_FILE):
        return []  # 파일이 없으면 빈 리스트 반환
    
    try:
        with open(MENU_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Warning: menu_cache.json is not a list. Resetting.")
            return []  # 형식이 리스트가 아니면 빈 리스트 반환
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading menu cache: {e}")
        return []  # 파일 읽기/파싱 오류 시 빈 리스트 반환

# --- 메인 앱(app.py)에서 호출할 함수 ---

def load_cached_menu_prices(
    city: str, 
    country: str, 
    neighborhood: Optional[str]
) -> List[Dict[str, Any]]:
    """
    특정 위치(도시/국가/세부지역)에 대한 메뉴 가격 샘플을 로드합니다.
    
    - '세부지역(neighborhood)'이 일치하는 데이터를 최우선으로 찾습니다.
    - 일치하는 세부지역 데이터가 없으면, 세부지역이 지정되지 않은 '도시 전체' 데이터를 찾습니다.
    """
    all_samples = load_all_cache()
    if not all_samples:
        return []

    city_lower = city.lower()
    country_lower = country.lower()
    neighborhood_lower = neighborhood.lower().strip() if neighborhood else None

    neighborhood_matches = []
    city_matches = []

    for sample in all_samples:
        if not isinstance(sample, dict):
            continue # 샘플이 딕셔너리 형태가 아니면 건너뜀

        sample_city = sample.get("city", "").lower()
        sample_country = sample.get("country", "").lower()
        
        # 도시 또는 국가가 일치하지 않으면 무시
        if sample_city != city_lower or sample_country != country_lower:
            continue

        sample_neighborhood = sample.get("neighborhood", "").lower().strip()

        # 1순위: 세부지역(neighborhood)이 정확히 일치하는 경우
        if neighborhood_lower and sample_neighborhood == neighborhood_lower:
            neighborhood_matches.append(sample)
        # 2순위: 세부지역이 지정되지 않은 '도시 전체' 데이터인 경우
        elif not sample_neighborhood:
            city_matches.append(sample)

    # 세부지역 일치 데이터를 우선 반환하고, 없으면 도시 전체 데이터를 반환
    return neighborhood_matches if neighborhood_matches else city_matches


def save_cached_menu_prices(all_samples: List[Dict[str, Any]]) -> bool:
    """
    *전체* 메뉴 캐시 목록을 JSON 파일에 덮어씁니다.
    (관리자 탭의 '데이터 캐시 관리' UI에서 사용됩니다.)
    """
    try:
        # 데이터 디렉터리(analysis_history)가 없으면 생성
        data_dir = os.path.dirname(MENU_CACHE_FILE)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        with open(MENU_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=4)
        return True
    except (IOError, TypeError) as e:
        # streamlit 앱이 아니므로 print로 로깅
        print(f"Fatal Error: Could not save menu cache file to {MENU_CACHE_FILE}. Error: {e}")
        return False

def add_menu_cache_entry(new_entry: Dict[str, Any]) -> bool:
    """
    새로운 캐시 항목 1개를 기존 목록에 추가하고 저장합니다.
    (관리자 탭의 '데이터 캐시 관리' UI에서 사용됩니다.)
    """
    if not isinstance(new_entry, dict) or not new_entry.get("city") or not new_entry.get("country"):
        print("Error: New cache entry is invalid or missing city/country.")
        return False
        
    all_samples = load_all_cache()
    
    # 기본값 및 타임스탬프 추가
    new_entry.setdefault("neighborhood", "")
    new_entry.setdefault("vendor", "N/A")
    new_entry.setdefault("category", "N/A")
    new_entry.setdefault("price", 0)
    new_entry.setdefault("currency", "USD")
    new_entry.setdefault("url", "")
    new_entry["last_updated"] = _get_current_timestamp()
    
    all_samples.append(new_entry)
    
    return save_cached_menu_prices(all_samples)

# """
# Utilities for loading cached menu and price references to contextualise AI prompts.

# The initial implementation provides a lightweight in-memory shim so that the main
# application can be wired for future integrations (e.g., scraping or vendor exports).
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import json

# _CACHE_ROOT = Path(__file__).resolve().parent / "menu_cache"
# _CACHE_ROOT.mkdir(exist_ok=True)


# def _cache_file(city: str, country: str) -> Path:
#     safe_city = city.lower().replace(" ", "_")
#     safe_country = country.lower().replace(" ", "_")
#     return _CACHE_ROOT / f"{safe_country}__{safe_city}.json"


# def load_cached_menu_prices(
#     city: str,
#     country: str,
#     neighborhood: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Return cached menu/price samples for the given city.

#     The function falls back to an empty list when no cache exists. Each entry is
#     expected to be a JSON-serialisable dictionary with keys such as:
#         - vendor / name
#         - category
#         - price
#         - currency
#         - last_updated
#         - notes
#     """
#     cache_file = _cache_file(city, country)
#     if not cache_file.exists():
#         return []

#     try:
#         with cache_file.open("r", encoding="utf-8") as handle:
#             data = json.load(handle)
#         if isinstance(data, list):
#             return data
#     except json.JSONDecodeError:
#         pass

#     return []


# def save_menu_prices(
#     city: str,
#     country: str,
#     samples: List[Dict[str, Any]],
# ) -> None:
#     """
#     Persist menu samples to the local cache.

#     This helper is not currently invoked by the Streamlit app, but is provided
#     so that operational scripts or future ingestion jobs can populate the cache.
#     """
#     cache_file = _cache_file(city, country)
#     with cache_file.open("w", encoding="utf-8") as handle:
#         json.dump(samples, handle, ensure_ascii=False, indent=2)
