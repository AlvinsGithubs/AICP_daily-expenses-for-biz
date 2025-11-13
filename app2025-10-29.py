
# 2025-10-20-14 AI 기반 출장비 계산 도구 (고급 분석 모델 적용)
# --- 설치 안내 ---
# 1. 아래 명령으로 필요한 패키지를 설치하세요.
#    pip install streamlit pandas PyMuPDF tabulate openai python-dotenv
#
# 2. .env 파일에 OPENAI_API_KEY 값을 설정하세요.

import streamlit as st
import pandas as pd
import json
import os
import re
import fitz  # PyMuPDF 라이브러리
import openai
from dotenv import load_dotenv
import io
from datetime import datetime
import time
import random
from collections import Counter
from statistics import StatisticsError, mean, quantiles
from typing import Any, Dict, List, Optional

from data_sources.menu_cache import load_cached_menu_prices

# --- 초기 환경 설정 ---

# .env 파일에서 환경 변수 로드
load_dotenv()

# AI 호출 횟수 제한 값
NUM_AI_CALLS = 10

# --- 가중 평균 계산을 위한 비중 (합계 1.0 유지) ---
UN_WEIGHT = 0.5  # UN-DSA 값에 적용할 가중치 (50%)
AI_WEIGHT = 0.5  # AI 산출값에 적용할 가중치 (50%)


# 분석 결과를 저장할 디렉터리 경로
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "analysis_history")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 직급별 일비 비율
JOB_LEVEL_RATIOS = {
    "L3 (60%)": 0.60, "L4 (60%)": 0.60, "L5 (80%)": 0.80, "L6 (100%)": 1.00,
    "L7 (100%)": 1.00, "L8 (120%)": 1.20, "L9 (150%)": 1.50, "L10 (150%)": 1.50,
}

TARGET_CONFIG_FILE = os.path.join(DATA_DIR, "target_cities_config.json")
TRIP_LENGTH_OPTIONS = ["Short-term", "Long-term"]

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
    entry.setdefault("trip_lengths", TRIP_LENGTH_OPTIONS.copy())
    if not entry.get("trip_lengths"):
        entry["trip_lengths"] = TRIP_LENGTH_OPTIONS.copy()
    return entry


def load_target_city_entries() -> List[Dict[str, Any]]:
    if not os.path.exists(TARGET_CONFIG_FILE):
        save_target_city_entries(DEFAULT_TARGET_CITY_ENTRIES)
    try:
        with open(TARGET_CONFIG_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("Invalid target city config format")
    except Exception:
        data = DEFAULT_TARGET_CITY_ENTRIES
    return [normalize_target_entry(item) for item in data]


def save_target_city_entries(entries: List[Dict[str, Any]]) -> None:
    normalized = [normalize_target_entry(item) for item in entries]
    with open(TARGET_CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)


TARGET_CITIES_ENTRIES = load_target_city_entries()


def get_target_city_entries() -> List[Dict[str, Any]]:
    if "target_cities_entries" in st.session_state:
        return st.session_state["target_cities_entries"]
    return TARGET_CITIES_ENTRIES


def set_target_city_entries(entries: List[Dict[str, Any]]) -> None:
    st.session_state["target_cities_entries"] = [normalize_target_entry(item) for item in entries]
    save_target_city_entries(st.session_state["target_cities_entries"])


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


def aggregate_ai_totals(totals: List[int]) -> Dict[str, Any]:
    """이상치를 제거하고 평균값을 계산해 투명하게 제공한다."""
    if not totals:
        return {"used_values": [], "removed_values": [], "mean": None}

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
    return {
        "used_values": filtered,
        "removed_values": removed_values,
        "mean_raw": computed_mean,
        "mean": round(computed_mean) if computed_mean is not None else None,
    }

# --- 핵심 로직 함수 ---

def parse_pdf_to_text(uploaded_file):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(4, len(doc)):
        full_text += doc[page_num].get_text("text") + "\n\n"
    return full_text

def get_history_files():
    if not os.path.exists(DATA_DIR):
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")]
    return sorted(files, reverse=True)

def save_report_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"report_{timestamp}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_report_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return None
    return None

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
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
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
            "trip_lengths": target.get("trip_lengths", TRIP_LENGTH_OPTIONS.copy()),
        }
        found_row = None
        search_target = target
        is_substitute = "un_dsa_substitute" in target
        if is_substitute: search_target = target["un_dsa_substitute"]
        
        country_df = df[df['Country'].str.contains(search_target['country'], case=False, na=False)]
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

def get_market_data_from_ai(
    city: str,
    country: str,
    api_key: str,
    source_name: str = "",
    context: Optional[Dict[str, Optional[str]]] = None,
    season_context: Optional[Dict[str, Any]] = None,
    menu_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """AI 모델을 호출해 일일 체류비 데이터를 JSON 형식으로 받아온다."""
    client = openai.OpenAI(api_key=api_key)
    context = context or {}
    season_context = season_context or {}
    menu_samples = menu_samples or []

    request_id = random.randint(10000, 99999)
    called_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert cost-of-living data analyst. You provide data only in the requested JSON format."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
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


def generate_markdown_report(report_data):
    md = f"# Business Travel Daily Allowance Report\n\n"
    md += f"**As of:** {report_data.get('as_of', 'N/A')}\n\n"

    valid_allowances = [c['final_allowance'] for c in report_data['cities'] if c.get('final_allowance') is not None]
    if valid_allowances:
        md += "## 1. Summary\n\n"
        md += (
            f"- Recommended range: ${min(valid_allowances)} ~ ${max(valid_allowances)}\n"
            f"- Average recommended allowance: ${round(sum(valid_allowances) / len(valid_allowances))}\n\n"
        )

    md += "## 2. City Details\n\n"
    table_data = []
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

            row = {
                'City': city_data.get('city', 'N/A'),
                'Country': city_data.get('country_display', 'N/A'),
                'UN-DSA (1 day)': un_val,
                'AI (season adjusted)': f"$ {ai_season_avg}" if ai_season_avg is not None else 'N/A',
                'AI runs used': f"{ai_runs_used}/{ai_attempts}",
                'Season label': season_context.get('label', 'Standard'),
                'Removed outliers': ", ".join(map(str, removed_totals)) if removed_totals else '-',
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
        "   - Query OpenAI GPT-4o ten times per city with local context, hotel clusters, and season tags.\n"
        "   - Store prompts, request IDs, season info, and menu samples with the responses.\n\n"
        "3. **Post-processing**\n"
        "   - Remove outliers via the IQR rule and compute averages.\n"
        "   - Apply season factors and blend with UN-DSA at a 50:50 weight.\n"
        "   - Multiply by grade ratios to produce per-level allowances.\n\n"
        "---\n"
        "## 4. Sources\n\n"
        "- UN-DSA Circular (United Nations - International Civil Service Commission)\n"
        "- OpenAI GPT-4o derived market estimates\n"
    )

    return md




# --- 스트림릿 UI 구성 ---
st.set_page_config(layout="wide")
st.title("AICP: 출장 일비 계산 & 조회 시스템 (v14.0 - 고급 분석 모델)")

if 'latest_analysis_result' not in st.session_state:
    st.session_state.latest_analysis_result = None
if 'target_cities_entries' not in st.session_state:
    st.session_state.target_cities_entries = [normalize_target_entry(entry) for entry in TARGET_CITIES_ENTRIES]

tab1, tab2 = st.tabs(["일비 조회 (직원용)", "보고서 분석 (관리자)"])

with tab1:
    st.header("도시별 출장 일비 조회")
    history_files = get_history_files()
    if not history_files:
        st.info("먼저 '보고서 분석' 탭에서 PDF를 분석해 주세요.")
    else:
        selected_file = st.selectbox("조회할 보고서 버전을 선택하세요.", history_files)
        report_data = load_report_data(selected_file)
        if report_data and 'cities' in report_data and report_data['cities']:
            cities_df = pd.DataFrame(report_data['cities'])
            target_entries = get_target_city_entries()
            countries = sorted({entry['country'] for entry in target_entries})

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                selectable_countries = [c for c in countries if c in cities_df['country_display'].unique()]
                sel_country = st.selectbox("국가:", selectable_countries, key=f"country_{selected_file}")
            with c2:
                trip_filter = st.multiselect("출장 유형", TRIP_LENGTH_OPTIONS, default=TRIP_LENGTH_OPTIONS, key=f"trip_{selected_file}")
            with c3:
                filtered_cities = []
                for entry in target_entries:
                    if entry['country'] != sel_country:
                        continue
                    if trip_filter and not any(t in entry.get('trip_lengths', TRIP_LENGTH_OPTIONS) for t in trip_filter):
                        continue
                    filtered_cities.append(entry['city'])
                filtered_cities = sorted(set(filtered_cities))
                if filtered_cities:
                    sel_city = st.selectbox("도시:", filtered_cities, key=f"city_{selected_file}")
                else:
                    sel_city = None
                    st.warning("선택한 조건에 해당하는 도시가 없습니다.")
            with c4:
                sel_level = st.selectbox("직급:", list(JOB_LEVEL_RATIOS.keys()), key=f"l_{selected_file}")
            
            if sel_city and sel_level:
                city_data = cities_df[cities_df['city'] == sel_city].iloc[0].to_dict()
                final_allowance = city_data.get('final_allowance')
                st.subheader(f"{sel_country} - {sel_city} 결과")
                if final_allowance:
                    level_ratio = JOB_LEVEL_RATIOS[sel_level]
                    level_allowance = round(final_allowance * level_ratio)
                    st.metric(f"{sel_level.split(' ')[0]} 권장 일당", f"$ {level_allowance}")
                    st.caption(f"기준 일당 ${final_allowance} × 직급 비율 {int(level_ratio*100)}%")
                else:
                    st.metric(f"{sel_level.split(' ')[0]} 권장 일당", "데이터 없음")

                st.markdown("---")
                st.write("**세부 산출 근거 (일당 기준)**")
                un_data = city_data.get('un', {})
                ai_summary = city_data.get('ai_summary', {})
                season_context = city_data.get('season_context', {})

                ai_avg = ai_summary.get('season_adjusted_mean_rounded')
                ai_runs = ai_summary.get('successful_runs', len(ai_summary.get('used_totals', [])))
                ai_attempts = ai_summary.get('attempted_runs', NUM_AI_CALLS)
                removed_totals = ai_summary.get('removed_totals') or []
                season_label = season_context.get('label') or ai_summary.get('season_label', 'Standard')
                season_factor = season_context.get('factor', ai_summary.get('season_factor', 1.0))

                ai_notes_parts = [f"성공 {ai_runs}/{ai_attempts}회"]
                if removed_totals:
                    ai_notes_parts.append(f"제외값 {removed_totals}")
                if season_label:
                    ai_notes_parts.append(f"시즌 {season_label} ×{season_factor}")
                ai_notes = " | ".join(ai_notes_parts) if ai_notes_parts else "AI 데이터 없음"

                col_un, col_ai, col_final = st.columns(3)
                with col_un:
                    st.info("UN-DSA 기준")
                    if un_data.get('status') == 'ok' and isinstance(un_data.get('per_diem_excl_lodging'), (int, float)):
                        st.metric("일당", f"$ {un_data['per_diem_excl_lodging']}")
                        st.caption(f"총액 ${un_data.get('total_dsa')} | 객실 비중 {un_data.get('room_pct')}%")
                    else:
                        st.metric("일당", "N/A")
                        st.caption(city_data.get("notes", ""))
                with col_ai:
                    st.info("AI 시장 추정 (시즌 보정)")
                    if ai_avg is not None:
                        st.metric("일당", f"$ {ai_avg}")
                        st.caption(ai_notes)
                    else:
                        st.metric("일당", "N/A")
                        st.caption(ai_notes)
                with col_final:
                    st.info("가중 평균 결과")
                    if final_allowance is not None:
                        st.metric("일당", f"$ {final_allowance}")
                        st.caption("UN-DSA와 AI 추정치를 50:50으로 결합")
                    else:
                        st.metric("일당", "N/A")

                        st.caption(city_data.get("notes", ""))

                with col_ai:
                    st.info("AI Market Estimate (Season-Adjusted)")
                    if ai_avg is not None:
                        st.metric("Daily Allowance", f"$ {ai_avg}")
                        st.caption(ai_notes)
                    else:
                        st.metric("Daily Allowance", "N/A")
                        st.caption(ai_notes)

                with col_final:
                    st.info("Weighted Combined Result")
                    if final_allowance is not None:
                        st.metric("Daily Allowance", f"$ {final_allowance}")
                        st.caption("50:50 blend of UN-DSA baseline and AI estimate")
                    else:
                        st.metric("Daily Allowance", "N/A")

                with st.expander("AI provenance & prompts"):
                    provenance_payload = {
                        "season_context": season_context,
                        "ai_summary": ai_summary,
                        "ai_runs": city_data.get('ai_provenance', []),
                    }
                    st.json(provenance_payload)

                menu_samples = city_data.get('menu_samples') or []
                if menu_samples:
                    with st.expander("Reference menu samples"):
                        st.table(pd.DataFrame(menu_samples))

with tab2:
    st.header("목표 도시 관리")
    entries_df = pd.DataFrame(get_target_city_entries())
    if not entries_df.empty:
        st.dataframe(entries_df[["region", "country", "city", "neighborhood", "hotel_cluster", "trip_lengths"]], use_container_width=True)
    else:
        st.info("등록된 목표 도시가 없습니다. 아래에서 새 항목을 추가해 주세요.")

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
            trip_lengths_selected = st.multiselect("출장 기간", TRIP_LENGTH_OPTIONS, default=TRIP_LENGTH_OPTIONS, key="add_trip_lengths")

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
                    "trip_lengths": trip_lengths_selected or TRIP_LENGTH_OPTIONS.copy(),
                }
                if substitute_city.strip() and substitute_country.strip():
                    new_entry["un_dsa_substitute"] = {
                        "city": substitute_city.strip(),
                        "country": substitute_country.strip(),
                    }
                current_entries.append(new_entry)
                set_target_city_entries(current_entries)
                st.success(f"{region_value} - {city_name.strip()} 항목을 추가했습니다.")
                st.experimental_rerun()

    st.subheader("기존 도시 편집/삭제")
    current_entries = get_target_city_entries()
    if current_entries:
        options = {
            f"{entry['region']} | {entry['country']} | {entry['city']}": idx
            for idx, entry in enumerate(current_entries)
        }
        selected_label = st.selectbox("편집할 도시를 선택하세요", list(options.keys()), key="edit_city_selector")
        selected_entry = dict(current_entries[options[selected_label]])

        with st.form("edit_target_city_form"):
            col_e, col_f = st.columns(2)
            with col_e:
                region_edit = st.text_input("지역", value=selected_entry.get("region", ""), key="edit_region")
                city_edit = st.text_input("도시", value=selected_entry.get("city", ""), key="edit_city")
                neighborhood_edit = st.text_input("세부 지역 (선택)", value=selected_entry.get("neighborhood", ""), key="edit_neighborhood")
            with col_f:
                country_edit = st.text_input("국가", value=selected_entry.get("country", ""), key="edit_country")
                hotel_cluster_edit = st.text_input("추천 호텔 클러스터 (선택)", value=selected_entry.get("hotel_cluster", ""), key="edit_hotel")

            trip_lengths_edit = st.multiselect(
                "출장 기간",
                TRIP_LENGTH_OPTIONS,
                default=[t for t in selected_entry.get("trip_lengths", TRIP_LENGTH_OPTIONS) if t in TRIP_LENGTH_OPTIONS] or TRIP_LENGTH_OPTIONS,
                key="edit_trip_lengths",
            )

            with st.expander("UN-DSA 대체 도시 (선택)"):
                sub_data = selected_entry.get("un_dsa_substitute") or {}
                sub_city_edit = st.text_input("대체 도시", value=sub_data.get("city", ""), key="edit_sub_city")
                sub_country_edit = st.text_input("대체 국가", value=sub_data.get("country", ""), key="edit_sub_country")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                update_btn = st.form_submit_button("변경사항 저장")
            with col_btn2:
                delete_btn = st.form_submit_button("삭제", type="secondary")

        if update_btn:
            if not region_edit.strip() or not city_edit.strip() or not country_edit.strip():
                st.error("지역, 국가, 도시는 필수로 입력해 주세요.")
            else:
                current_entries[options[selected_label]] = {
                    "region": region_edit.strip(),
                    "country": country_edit.strip(),
                    "city": city_edit.strip(),
                    "neighborhood": neighborhood_edit.strip(),
                    "hotel_cluster": hotel_cluster_edit.strip(),
                    "trip_lengths": trip_lengths_edit or TRIP_LENGTH_OPTIONS.copy(),
                }
                if sub_city_edit.strip() and sub_country_edit.strip():
                    current_entries[options[selected_label]]["un_dsa_substitute"] = {
                        "city": sub_city_edit.strip(),
                        "country": sub_country_edit.strip(),
                    }
                else:
                    current_entries[options[selected_label]].pop("un_dsa_substitute", None)

                set_target_city_entries(current_entries)
                st.success("수정을 완료했습니다.")
                st.experimental_rerun()
        if delete_btn:
            del current_entries[options[selected_label]]
            set_target_city_entries(current_entries)
            st.warning("선택한 항목을 삭제했습니다.")
            st.experimental_rerun()
    else:
        st.info("등록된 목표 도시가 없어 편집할 항목이 없습니다.")

    st.divider()
    st.subheader("UN-DSA (PDF) 분석")
    st.warning(f"AI 호출이 {NUM_AI_CALLS}회 실행되므로 시간과 비용에 유의해 주세요.")
    uploaded_file = st.file_uploader("UN-DSA PDF 파일을 업로드하세요.", type="pdf")
    if uploaded_file and st.button("AI 분석 실행", type="primary"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
        else:
            st.session_state.latest_analysis_result = None
            with st.spinner("PDF를 처리하는 중입니다..."):
                progress_bar = st.progress(0, text="PDF 텍스트 추출 중...")
                full_text = parse_pdf_to_text(uploaded_file)

                CHUNK_SIZE = 15000
                text_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
                all_tsv_lines = []

                analysis_failed = False
                for i, chunk in enumerate(text_chunks):
                    progress_bar.progress(i / (len(text_chunks) + 1), text=f"AI PDF->TSV 변환 중... ({i+1}/{len(text_chunks)})")
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

                if not analysis_failed:
                    processed_data = process_tsv_data("\n".join(all_tsv_lines))
                    if processed_data:
                        total_cities = len(processed_data["cities"])
                        for i, city_data in enumerate(processed_data["cities"]):
                            city_name, country_name = city_data["city"], city_data["country_display"]
                            progress_text = f"AI 추정치 계산 중... ({i+1}/{total_cities}) {city_name}"
                            progress_bar.progress((i + 1) / max(total_cities, 1), text=progress_text)

                            city_context = {
                                "neighborhood": city_data.get("neighborhood"),
                                "hotel_cluster": city_data.get("hotel_cluster"),
                            }
                            season_context = city_data.get("season_context") or get_current_season_info(city_name, country_name)
                            menu_samples = load_cached_menu_prices(city_name, country_name, city_context.get("neighborhood"))
                            city_data["menu_samples"] = menu_samples
                            ai_totals_source: List[int] = []
                            ai_meta_runs: List[Dict[str, Any]] = []

                            for j in range(1, NUM_AI_CALLS + 1):
                                source_name = f"  {j}"
                                market_result = get_market_data_from_ai(
                                    city_name,
                                    country_name,
                                    openai_api_key,
                                    source_name,
                                    context=city_context,
                                    season_context=season_context,
                                    menu_samples=menu_samples,
                                )
                                city_data[f"market_data_{j}"] = market_result
                                if market_result.get("status") == 'ok' and market_result.get("total") is not None:
                                    ai_totals_source.append(market_result["total"])
                                if "meta" in market_result:
                                    ai_meta_runs.append(market_result["meta"])
                                if j < NUM_AI_CALLS:
                                    time.sleep(1)

                            city_data["ai_provenance"] = ai_meta_runs

                            final_allowance = None
                            un_per_diem = city_data.get("un", {}).get("per_diem_excl_lodging")

                            ai_stats = aggregate_ai_totals(ai_totals_source)
                            season_factor = (season_context or {}).get("factor", 1.0)
                            ai_base_mean = ai_stats.get("mean_raw")
                            ai_season_adjusted = ai_base_mean * season_factor if ai_base_mean is not None else None

                            city_data["ai_summary"] = {
                                "raw_totals": ai_totals_source,
                                "used_totals": ai_stats.get("used_values", []),
                                "removed_totals": ai_stats.get("removed_values", []),
                                "mean_base": ai_base_mean,
                                "mean_base_rounded": ai_stats.get("mean"),
                                "season_factor": season_factor,
                                "season_label": (season_context or {}).get("label"),
                                "season_adjusted_mean_raw": ai_season_adjusted,
                                "season_adjusted_mean_rounded": round(ai_season_adjusted) if ai_season_adjusted is not None else None,
                                "successful_runs": len(ai_stats.get("used_values", [])),
                                "attempted_runs": NUM_AI_CALLS,
                                "weighted_average_components": {
                                    "un_per_diem": un_per_diem,
                                    "ai_season_adjusted": ai_season_adjusted,
                                    "weights": {"UN": UN_WEIGHT, "AI": AI_WEIGHT},
                                },
                            }

                            if un_per_diem and ai_season_adjusted is not None:
                                weighted_average = (un_per_diem * UN_WEIGHT) + (ai_season_adjusted * AI_WEIGHT)
                                final_allowance = round(weighted_average)
                            elif un_per_diem:
                                final_allowance = round(un_per_diem)
                            elif ai_season_adjusted is not None:
                                final_allowance = round(ai_season_adjusted)

                            city_data["final_allowance"] = final_allowance

                            if final_allowance and un_per_diem and un_per_diem > 0:
                                city_data["delta_vs_un_pct"] = round(((final_allowance - un_per_diem) / un_per_diem) * 100)
                            else:
                                city_data["delta_vs_un_pct"] = "N/A"

                        save_report_data(processed_data)
                        st.session_state.latest_analysis_result = processed_data
                        st.success("AI analysis completed.")

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

            row.update({
                'Final Allowance': city.get('final_allowance'),
                'Delta (%)': city.get('delta_vs_un_pct'),
                'Trip Lengths': ", ".join(city.get('trip_lengths', [])) if city.get('trip_lengths') else '-',
                'Notes': city.get('notes', ''),
            })
            df_data.append(row)

        st.dataframe(pd.DataFrame(df_data))
        with st.expander("View generated markdown report"):
            st.markdown(generate_markdown_report(st.session_state.latest_analysis_result))
