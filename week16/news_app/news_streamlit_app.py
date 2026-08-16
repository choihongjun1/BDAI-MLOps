"""
뉴스 헤드라인 카테고리 분류 - Streamlit UI

FastAPI 서버(news_api.py)가 실행 중인 상태에서 사용합니다.
실행 예시:
    streamlit run news_streamlit_app.py
"""

import os

import requests
import streamlit as st

# 로컬 실행 시 기본값을 사용하고, Docker Compose에서는 환경변수로 API 주소를 전달합니다.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

CATEGORY_EMOJI = {
    "POLITICS": "🏛️",
    "WORLD NEWS": "🌍",
    "ENTERTAINMENT": "🎬",
    "TRAVEL": "✈️",
    "STYLE & BEAUTY": "💄",
    "PARENTING": "👨‍👩‍👧",
    "FOOD & DRINK": "🍽️",
    "SPORTS": "⚽",
    "BUSINESS": "💼",
    "TECH": "💻",
    "SCIENCE": "🔬",
    "HEALTH": "🏥",
    "CRIME": "🚔",
    "EDUCATION": "📚",
    "ENVIRONMENT": "🌿",
    "MEDIA": "📰",
    "ARTS": "🎨",
    "COMEDY": "😂",
    "RELIGION": "⛪",
    "MONEY": "💰",
    "WELLNESS": "🧘",
    "QUEER VOICES": "🏳️‍🌈",
    "WOMEN": "♀️",
    "IMPACT": "💥",
    "DIVORCE": "💔",
    "WEIRD NEWS": "🤪",
    "FIFTY": "5️⃣0️⃣",
    "GOOD NEWS": "😊",
    "LATINO VOICES": "🌎",
    "ARTS & CULTURE": "🎭",
    "HOME & LIVING": "🏠",
    "COLLEGE": "🎓",
    "WEDDINGS": "💍",
    "BLACK VOICES": "✊",
}


def get_category_emoji(category: str) -> str:
    for key, emoji in CATEGORY_EMOJI.items():
        if key in category.upper():
            return emoji
    return "📌"


def check_api_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "detail": "FastAPI 서버에 연결할 수 없습니다."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def predict_category(headline: str) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json={"headline": headline},
            timeout=10,
        )
        if resp.status_code == 200:
            return {"success": True, "data": resp.json()}
        else:
            detail = resp.json().get("detail", "알 수 없는 오류")
            return {"success": False, "error": f"[{resp.status_code}] {detail}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 페이지 설정 ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="뉴스 카테고리 분류기",
    page_icon="📰",
    layout="centered",
)

# ── 헤더 ────────────────────────────────────────────────────────────────────

st.title("📰 뉴스 헤드라인 카테고리 분류기")
st.caption("TF-IDF + LinearSVC 모델 | FastAPI 백엔드 연동")
st.divider()

# ── 서버 상태 표시 ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 서버 상태")
    if st.button("🔄 상태 새로고침", use_container_width=True):
        st.rerun()

    health = check_api_health()
    if health.get("status") == "ok":
        st.success("✅ 서버 정상 (모델 로드 완료)")
    else:
        st.error(f"❌ 서버 오류\n\n{health.get('detail', '')}")
        st.info("💡 FastAPI 서버를 먼저 실행해주세요:\n```\nuvicorn news_api:app --reload\n```")

    st.divider()
    st.header("📌 예시 헤드라인")
    examples = [
        "Biden says U.S. forces would defend Taiwan if China invaded",
        "Apple unveils new iPhone with AI-powered features",
        "Scientists discover new exoplanet that may support life",
        "Manchester United beats Arsenal in thrilling final",
        "Stock markets rally as inflation fears ease",
        "New study links processed food to higher cancer risk",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["headline_input"] = ex

# ── 입력 폼 ─────────────────────────────────────────────────────────────────

st.subheader("🔍 헤드라인 입력")

headline_value = st.session_state.get("headline_input", "")

with st.form("predict_form", clear_on_submit=False):
    headline = st.text_area(
        label="뉴스 헤드라인을 영어로 입력하세요",
        value=headline_value,
        height=100,
        placeholder="예: Scientists discover new treatment for Alzheimer's disease",
    )
    submitted = st.form_submit_button("🚀 카테고리 예측", use_container_width=True, type="primary")

# ── 예측 결과 ────────────────────────────────────────────────────────────────

if submitted:
    if not headline.strip():
        st.warning("⚠️ 헤드라인을 입력해주세요.")
    elif len(headline.strip()) < 3:
        st.warning("⚠️ 헤드라인은 최소 3자 이상 입력해주세요.")
    else:
        with st.spinner("예측 중..."):
            result = predict_category(headline.strip())

        if result["success"]:
            data = result["data"]
            category = data["predicted_category"]
            emoji = get_category_emoji(category)

            st.divider()
            st.subheader("✅ 예측 결과")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="예측 카테고리", value=f"{emoji} {category}")
            with col2:
                st.info(f"**입력 헤드라인:**\n\n{data['headline']}")

            # 세션 기록 저장
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, {
                "headline": data["headline"],
                "category": f"{emoji} {category}",
            })

        else:
            st.error(f"❌ 예측 실패: {result['error']}")

# ── 예측 기록 ────────────────────────────────────────────────────────────────

if st.session_state.get("history"):
    st.divider()
    st.subheader("📋 예측 기록")

    col_clear = st.columns([4, 1])[1]
    with col_clear:
        if st.button("🗑️ 기록 초기화"):
            st.session_state["history"] = []
            st.rerun()

    for i, record in enumerate(st.session_state["history"][:10]):
        with st.container():
            cols = st.columns([3, 1])
            with cols[0]:
                st.write(f"**{record['headline']}**")
            with cols[1]:
                st.write(record["category"])
            if i < len(st.session_state["history"]) - 1:
                st.divider()
