"""
RouteRescue AI — Bus-Only Smart Transit Platform
Main Application: app.py

A modern travel-tech platform with:
1. Cinematic Hero Welcome Screen & Authentication Gate
2. User Authentication (Login, Signup, Demo Access)
3. Bus Transit Command Center Dashboard
4. Scheduled Bus Search with Smart Alternatives (TGSRTC Static GTFS)
5. Bus Transfer Planner (Validated 1-Transfer Connections)
6. Nearby Bus Stops Locator (GPS Coordinates & Name Search)
7. Favourite Routes & Search History (Private Per-User Storage)
8. Evidence-Only Passenger Guidance & Timetable Advisor
"""

import datetime
import hashlib
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _load_env_file(filepath: str = ".env") -> None:
    try:
        env_path = Path(__file__).resolve().parent / filepath
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
    except Exception:
        pass

_load_env_file()
import streamlit as st
import pandas as pd

# Local engines and authentication manager
from route_engine import RouteEngine
from rag_engine import TransitRAGEngine
from data_sources import get_tgsrtc_gtfs_status
import auth_manager
import base64
from tgsrtc_gtfs import TgsrtcGtfsError, TgsrtcGtfsFeed
from knowledge_base import (
    KnowledgeBaseError,
    ingest_local_document,
    list_knowledge_sources,
    update_source_retrieval_time,
)


def _encode_image(path: str) -> str:
    """Return base64-encoded image content for inline CSS background."""
    try:
        with open(path, "rb") as img_f:
            return base64.b64encode(img_f.read()).decode("utf-8")
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RouteRescue AI — Bus Transit Command Center",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "welcome"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "last_bus_search" not in st.session_state:
    st.session_state["last_bus_search"] = None

if "last_transfer_search" not in st.session_state:
    st.session_state["last_transfer_search"] = None

if "passenger_advisor_messages" not in st.session_state:
    st.session_state["passenger_advisor_messages"] = []

if "selected_origin" not in st.session_state:
    st.session_state["selected_origin"] = None

if "selected_destination" not in st.session_state:
    st.session_state["selected_destination"] = None

# Support direct view navigation via URL query parameter for authenticated sessions
if "view" in st.query_params and st.session_state.get("authenticated"):
    _requested_view = st.query_params["view"]
    if _requested_view in [
        "dashboard",
        "bus_tracker",
        "transfers",
        "nearby_stops",
        "favourites_history",
        "ai_assistant",
    ]:
        st.session_state["view"] = _requested_view


# -----------------------------------------------------------------------------
# Passenger-Friendly Formatting Helpers
# -----------------------------------------------------------------------------
def format_passenger_time(val: Any, ref_date: Any = None) -> str:
    """Format ISO timestamp or time string as passenger-friendly '3:58 PM', handling midnight."""
    if not val:
        return ""
    try:
        s = str(val).strip()
        if "T" in s:
            dt = datetime.datetime.fromisoformat(s)
            formatted = dt.strftime("%I:%M %p").lstrip("0")
            if ref_date is not None:
                try:
                    if isinstance(ref_date, str):
                        if "T" in ref_date:
                            ref_d = datetime.datetime.fromisoformat(ref_date).date()
                        else:
                            ref_d = datetime.date.fromisoformat(ref_date)
                    elif isinstance(ref_date, (datetime.datetime, datetime.date)):
                        ref_d = ref_date.date() if isinstance(ref_date, datetime.datetime) else ref_date
                    else:
                        ref_d = None
                    if ref_d and dt.date() > ref_d:
                        return f"{formatted} (+1 day)"
                except Exception:
                    pass
            return formatted
        if ":" in s:
            parts = s.split(":")
            h = int(parts[0])
            m = int(parts[1])
            is_next = h >= 24
            t = datetime.time(h % 24, m)
            formatted = t.strftime("%I:%M %p").lstrip("0")
            if is_next:
                return f"{formatted} (+1 day)"
            return formatted
        return s
    except Exception:
        return str(val)


def format_passenger_date(val: Any) -> str:
    """Format date string or date object as passenger-friendly '16 Oct 2026'."""
    if not val:
        return ""
    try:
        if isinstance(val, str):
            d = datetime.date.fromisoformat(val)
        elif isinstance(val, (datetime.date, datetime.datetime)):
            d = val
        else:
            return str(val)
        return d.strftime("%d %b %Y").lstrip("0")
    except Exception:
        return str(val)


def format_passenger_duration(minutes: Any) -> str:
    """Format duration minutes as '5 hr 40 min' or '45 min'."""
    try:
        m = int(round(float(minutes)))
        if m >= 60:
            hrs = m // 60
            mins = m % 60
            return f"{hrs} hr {mins} min" if mins > 0 else f"{hrs} hr"
        return f"{m} min"
    except Exception:
        return f"{minutes} min"


# -----------------------------------------------------------------------------
# Passenger Advisor: deterministic fallback for known app questions
# -----------------------------------------------------------------------------
def get_passenger_advisor_fallback(question: str) -> Optional[Dict[str, Any]]:
    """Return deterministic guidance for a limited set of known app questions."""
    if not isinstance(question, str) or not question.strip():
        return None
    q = " ".join(question.casefold().split())

    def _contains(*terms: str) -> bool:
        return any(term in q for term in terms)

    def _result(topic: str, answer: str) -> Dict[str, Any]:
        return {
            "has_evidence": False,
            "is_fallback": True,
            "confidence_level": "Deterministic app guidance",
            "explanation": (
                f"Matched the known app question topic: {topic}. This is fixed "
                "guidance about how the application behaves, not retrieved "
                "source evidence and not a model-generated answer."
            ),
            "answer": answer,
            "answer_mode": (
                "Deterministic app guidance (fallback); no source retrieval and "
                "no language model were used for this answer."
            ),
            "results": [],
            "supporting_sources": [],
        }

    if _contains("train", "railway", "railways", "trains"):
        return _result(
            "bus-only transit platform scope",
            "RouteRescue AI is dedicated exclusively to bus transit and integrates "
            "with verified TGSRTC GTFS static timetables. Train schedules and railway "
            "tracking are not supported. For official Indian Railways timetables, "
            "use official channels such as NTES (enquiry.indianrail.gov.in).",
        )
    if _contains(
        "delay", "delays", "disruption", "cancellation", "cancellations",
        "cancelled", "live status", "live tracking", "fare", "fares",
    ):
        return _result(
            "whether live delays or fares are available",
            "No. No verified live status feed is connected to this application. "
            "All bus results are records from an archived static TGSRTC timetable "
            "snapshot (MobilityDatabase data provided by TGSRTC); they do not show "
            "current vehicle positions, delays, or cancellations. The validated feed "
            "contains no fare tables, so fares are not displayed and the Cheapest "
            "ranking stays unavailable. The app does not simulate disruptions or invent fare values.",
        )
    if _contains(
        "coverage", "cover", "dates", "timezone", "time zone", "valid range",
    ):
        return _result(
            "what timetable coverage means",
            "Timetable coverage means the range of service dates for which the "
            "validated TGSRTC static GTFS archive contains schedules. Only "
            "dates inside that range can be searched; the Bus page shows the "
            "exact coverage range loaded from the feed's feed_info, and the "
            "Command Center data-sources panel lists it too. Coverage describes "
            "the archived timetable only — it does not confirm that a bus "
            "operates on a given day.",
        )
    if _contains(
        "no bus", "no result", "no direct", "why are no", "nothing found",
        "empty search", "not available for my",
    ):
        return _result(
            "why a route may return no results",
            "A route search returns no journeys for verified reasons: no direct "
            "trip in the timetable connects the two stops in that direction on "
            "the selected service date; direct service exists only in the "
            "opposite direction; scheduled trips exist but all departed before "
            "the requested departure time; or the selected date is outside the "
            "timetable's coverage range. When no direct bus exists, use the "
            "Bus Transfer Planner to find valid 1-transfer connections across verified transit hubs.",
        )
    if _contains(
        "how do i find", "how to search", "how to find", "search scheduled",
        "find scheduled", "plan a journey", "between two stops",
    ):
        return _result(
            "how to search scheduled buses",
            "To search scheduled buses, open the Bus Route Finder from the "
            "Command Center, choose your boarding stop and destination stop "
            "from the curated Hyderabad stop list, pick a journey date inside "
            "the displayed timetable coverage and a departure time, then press "
            "'Search scheduled journeys'. The result set displays direct trips "
            "with up to 3 options, ranked by scheduled duration. If no direct bus exists, "
            "use the Bus Transfer Planner to find 1-transfer connections.",
        )
    return None


def answer_passenger_question(
    rag: TransitRAGEngine, question: str
) -> Dict[str, Any]:
    """Retrieve trusted evidence or fall back to labelled deterministic guidance."""
    if not isinstance(question, str) or not question.strip():
        return {
            "has_evidence": False,
            "confidence_level": "No evidence retrieved",
            "explanation": "Please enter a specific question about scheduled services or timetable limits.",
            "answer": "Please enter a question to consult the passenger advisor.",
            "answer_mode": "No answer generated; no question was entered.",
            "results": [],
            "supporting_sources": [],
        }

    evidence_result: Optional[Dict[str, Any]] = None
    if rag.is_indexed:
        try:
            evidence_result = rag.search_with_evidence(question, top_k=3)
        except Exception:
            evidence_result = None

    if (
        evidence_result is not None
        and evidence_result.get("has_evidence")
        and evidence_result.get("results")
    ):
        primary_match = evidence_result["results"][0]
        return {
            "has_evidence": True,
            "confidence_level": evidence_result.get("confidence_level", "Retrieved evidence"),
            "explanation": evidence_result.get("explanation", ""),
            "answer": primary_match.get("content") or primary_match.get("passage_text", ""),
            "answer_mode": evidence_result.get("answer_mode") or (
                "Verbatim passage extracted from validated source material. "
                "No generative language model was used."
            ),
            "results": evidence_result["results"],
            "supporting_sources": [
                {
                    "title": primary_match.get("title", ""),
                    "source_ref": primary_match.get("source_ref", ""),
                    "source_title": primary_match.get("source_title", "TGSRTC GTFS"),
                    "source_url": primary_match.get("source_url", ""),
                }
            ],
        }

    fallback = get_passenger_advisor_fallback(question)
    if fallback is not None:
        return fallback

    return {
        "has_evidence": False,
        "is_fallback": False,
        "confidence_level": "Unverified",
        "explanation": (
            "No indexed source passage met the verification threshold, and the "
            "question did not match any of the known application topics. "
            "RouteRescue AI does not generate answers without verifiable source evidence."
        ),
        "answer": (
            "The passenger advisor cannot verify an answer to that question from "
            "the available source evidence or application guidance."
        ),
        "answer_mode": (
            "No answer generated; no source retrieval and no language model were used."
        ),
        "results": [],
        "supporting_sources": [],
    }


# -----------------------------------------------------------------------------
# Cached Engines
# -----------------------------------------------------------------------------
@st.cache_resource
def get_route_engine() -> RouteEngine:
    return RouteEngine()

@st.cache_resource
def get_rag_engine() -> TransitRAGEngine:
    knowledge_dir = Path(__file__).resolve().parent / "data" / "passenger_knowledge"
    try:
        feed = get_tgsrtc_gtfs_feed()
    except TgsrtcGtfsError as feed_error:
        return TransitRAGEngine(
            knowledge_dir=knowledge_dir,
            index_error=(
                "The validated TGSRTC feed is unavailable for passenger-advisor "
                f"evidence: {feed_error}"
            )
        )
    return TransitRAGEngine(gtfs_feed=feed, knowledge_dir=knowledge_dir)

@st.cache_resource
def get_tgsrtc_gtfs_feed() -> TgsrtcGtfsFeed:
    project_root = Path(__file__).resolve().parent
    return TgsrtcGtfsFeed(project_root / "data" / "tgsrtc_gtfs")


route_engine = get_route_engine()
rag_engine = get_rag_engine()


# -----------------------------------------------------------------------------
# Custom Styling (Cinematic Dark Navy & Electric Cyan Transit Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: #090d16;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }
    .brand-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    /* Cinematic Hero Container */
    .hero-container {
        background: linear-gradient(145deg, #090d16 0%, #0f172a 60%, #082f49 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 45px 40px;
        color: #ffffff;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-headline {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.15;
        margin-bottom: 12px;
        background: linear-gradient(90deg, #ffffff 30%, #7dd3fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tagline {
        font-size: 1.25rem;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 14px;
        letter-spacing: -0.2px;
    }
    .hero-description {
        font-size: 0.96rem;
        color: #cbd5e1;
        max-width: 680px;
        line-height: 1.6;
        margin-bottom: 24px;
    }

    /* Dashboard Cards */
    .command-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease-in-out;
    }
    .command-card:hover {
        border-color: #0284c7;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.12);
    }
    .card-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .card-scope {
        font-size: 0.82rem;
        color: #38bdf8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    .card-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* Feature Pills */
    .feature-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .status-ontime { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .status-scheduled { background: #1e293b; color: #cbd5e1; border: 1px solid #475569; }
    .status-transfer { background: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; }

    /* Results Card */
    .result-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        color: #f8fafc;
    }

    /* Timeline Transfer Box */
    .transfer-leg-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navigation Helpers
# -----------------------------------------------------------------------------
def go_to(view_name: str):
    st.session_state["view"] = view_name
    st.rerun()

def perform_logout():
    st.session_state["authenticated"] = False
    st.session_state["user_profile"] = None
    st.session_state["demo_mode"] = False
    st.session_state["last_bus_search"] = None
    st.session_state["last_transfer_search"] = None
    st.session_state["selected_origin"] = None
    st.session_state["selected_destination"] = None
    st.session_state["passenger_advisor_messages"] = []
    st.session_state["view"] = "welcome"
    st.rerun()

def perform_demo_login():
    st.session_state["authenticated"] = True
    st.session_state["user_profile"] = auth_manager.get_demo_user()
    st.session_state["demo_mode"] = True
    st.session_state["last_bus_search"] = None
    st.session_state["last_transfer_search"] = None
    st.session_state["passenger_advisor_messages"] = []
    st.session_state["view"] = "dashboard"
    st.rerun()


# -----------------------------------------------------------------------------
# Top Header Navigation (shown on interior views)
# -----------------------------------------------------------------------------
if st.session_state["view"] not in ["welcome", "login", "signup"]:
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px;">
            <span class="brand-logo">🚌 RouteRescue AI</span>
            <span style="font-size: 0.8rem; color: #38bdf8; background: #0f172a; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.25);">
                Bus Transit Platform
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_nav2:
        user_name = st.session_state["user_profile"]["name"] if st.session_state["user_profile"] else "Commuter"
        st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.85rem;'>👤 <b>{user_name}</b></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.session_state["view"] != "dashboard":
                if st.button("Command Center", use_container_width=True, key="hdr_dash"):
                    go_to("dashboard")
        with btn_c2:
            if st.button("Log Out", use_container_width=True, key="hdr_logout"):
                perform_logout()

    if st.session_state["demo_mode"]:
        st.warning(
            "Demo Mode active: Explore RouteRescue AI's bus features with verified TGSRTC static GTFS schedule data."
        )

    # Quick Navigation Bar
    nav_cols = st.columns(6)
    nav_items = [
        ("dashboard", "🎛️ Overview"),
        ("bus_tracker", "🚌 Bus Search"),
        ("transfers", "🔀 Transfers"),
        ("nearby_stops", "📍 Nearby Stops"),
        ("favourites_history", "⭐ Favourites"),
        ("ai_assistant", "🤖 Advisor"),
    ]
    for idx, (v_key, v_title) in enumerate(nav_items):
        with nav_cols[idx]:
            is_active = st.session_state["view"] == v_key
            btn_style = "primary" if is_active else "secondary"
            if st.button(v_title, key=f"quick_nav_{v_key}", type=btn_style, use_container_width=True):
                if not is_active:
                    go_to(v_key)
    st.divider()


# =============================================================================
# VIEW 1: Welcome Screen (First Screen — Auth Gate)
# =============================================================================
if st.session_state["view"] == "welcome":
    st.markdown("""
    <style>
        .auth-hero {
            text-align: center;
            margin-bottom: 24px;
            padding: 30px 24px;
            background: linear-gradient(160deg, #090d16 0%, #0f172a 60%, #0c1c2e 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-top: 3px solid #38bdf8;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }
        .auth-brand {
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ffffff 15%, #38bdf8 65%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1.2px;
            display: block;
            margin-bottom: 8px;
            line-height: 1.15;
        }
        .auth-tagline {
            font-size: 1.12rem;
            color: #38bdf8;
            font-weight: 600;
            margin-bottom: 10px;
            letter-spacing: -0.2px;
        }
        .auth-desc {
            font-size: 0.90rem;
            color: #94a3b8;
            max-width: 460px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .proto-notice {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.8rem;
            color: #38bdf8;
            margin-bottom: 18px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    hero_img_path = os.path.join("assets", "hero_transit_network.jpg")
    if os.path.exists(hero_img_path):
        encoded = _encode_image(hero_img_path)
        if encoded:
            st.markdown(
                f"""<div style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;
                    background:url('data:image/jpeg;base64,{encoded}') center/cover no-repeat;
                    opacity:0.09;pointer-events:none;"></div>""",
                unsafe_allow_html=True,
            )

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown("""
        <div class="auth-hero">
            <span class="auth-brand">RouteRescue AI</span>
            <div class="auth-tagline">Smart Bus Transit Platform</div>
            <div class="auth-desc">
                Search scheduled TGSRTC bus trips from the verified GTFS timetable.
                Plan smart alternatives, find nearby stops, and map 1-transfer connections.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="proto-notice">
            🚍 <b>Bus-Only Smart Transit</b>: Direct bus search, transfer planning, nearby stops, and route bookmarks.
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Enter Transit Platform")
            if st.button("🚀 Explore Demo Access (1-Click)", type="primary", use_container_width=True):
                perform_demo_login()

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Log In", use_container_width=True):
                    go_to("login")
            with col_b:
                if st.button("Sign Up", use_container_width=True):
                    go_to("signup")


# =============================================================================
# VIEW 2: Sign Up Screen
# =============================================================================
elif st.session_state["view"] == "signup":
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown("### 📝 Create RouteRescue Account")
            st.caption("Sign up for personalized saved routes and search history.")

            name = st.text_input("Full Name:", placeholder="e.g. Ramesh Kumar")
            email = st.text_input("Email Address:", placeholder="e.g. commuter@example.com")
            pwd = st.text_input("Password (min 6 characters):", type="password", placeholder="Enter password")
            confirm_pwd = st.text_input("Confirm Password:", type="password", placeholder="Re-enter password")

            if st.button("Create Account", type="primary", use_container_width=True):
                success, msg = auth_manager.register_user(name, email, pwd, confirm_pwd)
                if success:
                    st.success(msg)
                    st.info("Please log in with your new credentials.")
                    go_to("login")
                else:
                    st.error(msg)

            st.markdown("---")
            if st.button("Already have an account? Log In", use_container_width=True):
                go_to("login")

    if st.button("← Back to Welcome Page"):
        go_to("welcome")


# =============================================================================
# VIEW 3: Login Screen
# =============================================================================
elif st.session_state["view"] == "login":
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown("### 🔐 Commuter Login")
            st.caption("Access your saved bus routes and personal transit dashboard.")

            email = st.text_input("Email Address:", placeholder="e.g. demo@routerescue.ai")
            pwd = st.text_input("Password:", type="password", placeholder="Enter password")

            if st.button("Log In", type="primary", use_container_width=True):
                success, user_info, msg = auth_manager.authenticate_user(email, pwd)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["user_profile"] = user_info
                    st.session_state["demo_mode"] = bool(user_info.get("is_demo"))
                    st.session_state["last_bus_search"] = None
                    st.session_state["last_transfer_search"] = None
                    st.success(msg)
                    go_to("dashboard")
                else:
                    st.error(msg)

            st.markdown("---")
            if st.button("Don't have an account? Sign Up", use_container_width=True):
                go_to("signup")

    if st.button("← Back to Welcome Page"):
        go_to("welcome")


# =============================================================================
# VIEW 4: Travel Command Center (Main Dashboard)
# =============================================================================
elif st.session_state["view"] == "dashboard":
    st.markdown("### 🎛️ Bus Transit Command Center")
    st.caption(
        "Find scheduled buses, plan 1-transfer connections, locate nearby stops, "
        "and manage your favourite routes."
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            """
            <div class="command-card">
                <div class="card-icon">🚌</div>
                <div class="card-title">Bus Search & Alternatives</div>
                <div class="card-scope">TGSRTC · Hyderabad & Telangana</div>
                <div class="card-desc">
                    Search scheduled direct bus trips between verified stops. View the best matching bus first with up to 3 smart alternative options.
                </div>
                <div>
                    <span class="feature-pill">Direct Services</span>
                    <span class="feature-pill">Verified Timetable</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Bus Route Finder →", key="dash_btn_bus", type="primary", use_container_width=True):
            go_to("bus_tracker")

    with col2:
        st.markdown(
            """
            <div class="command-card">
                <div class="card-icon">🔀</div>
                <div class="card-title">Bus Transfer Planner</div>
                <div class="card-scope">TGSRTC · Multi-Leg Connections</div>
                <div class="card-desc">
                    No direct bus? Find scheduled 1-transfer connections across verified transit hubs with validated layover buffers.
                </div>
                <div>
                    <span class="feature-pill">1-Transfer Routes</span>
                    <span class="feature-pill">Timetable Validated</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Transfer Planner →", key="dash_btn_transfers", use_container_width=True):
            go_to("transfers")

    st.write("")
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown(
            """
            <div class="command-card">
                <div class="card-icon">📍</div>
                <div class="card-title">Nearby Bus Stops</div>
                <div class="card-scope">GPS & Stop Name Lookup</div>
                <div class="card-desc">
                    Locate verified bus stops within 5 km of your coordinates or search by stop name to explore available bus routes.
                </div>
                <div>
                    <span class="feature-pill">5,000+ Stops</span>
                    <span class="feature-pill">Route Information</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Nearby Stops →", key="dash_btn_nearby", use_container_width=True):
            go_to("nearby_stops")

    with col4:
        st.markdown(
            """
            <div class="command-card">
                <div class="card-icon">⭐</div>
                <div class="card-title">Favourites & Search History</div>
                <div class="card-scope">Commuter Bookmarks</div>
                <div class="card-desc">
                    Quickly re-run frequently traveled routes and review your recent transit searches in a private, dedicated dashboard.
                </div>
                <div>
                    <span class="feature-pill">1-Click Re-Search</span>
                    <span class="feature-pill">Private Bookmarks</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Favourites & History →", key="dash_btn_favs", use_container_width=True):
            go_to("favourites_history")

    st.write("")
    with st.expander("ℹ️ Data sources and technical details", expanded=False):
        st.markdown(
            "**TGSRTC City Bus:** Verified archived static GTFS timetable (MobilityDatabase `mdb-3361`). "
            "Contains data provided by TGSRTC. Coverage 2026-02-01 to 2031-02-01. "
            "Static schedule only — no live vehicle tracking or fare tables.  \n\n"
            "Live vehicle positions, cancellations, and real-time delays are not connected. "
            "TGSRTC does not endorse RouteRescue AI."
        )
        st.caption(
            "Contains data provided by TGSRTC. TGSRTC does not endorse RouteRescue AI. "
            "All bus schedules reflect an archived static snapshot; no live status feeds are connected."
        )

    st.write("")
    st.divider()

    t_rag = st.container()
    with t_rag:
        st.markdown("#### Passenger Advisor")
        st.caption(
            "Ask about interpreting this app's scheduled bus results and the limits of "
            "the TGSRTC timetable. Answers quote retrieved source passages when they "
            "are available; a small set of known app questions can also be answered "
            "by clearly labelled deterministic app guidance."
        )
        st.info(
            "Evidence-only mode: no language-model adapter is configured. The advisor "
            "does not generate answers or use API credentials."
        )
        if rag_engine.legacy_document_count:
            st.warning(
                f"{rag_engine.legacy_document_count} records in data/travel_guides.csv "
                "are illustrative or lack verifiable provenance and are excluded from "
                "retrieval and answers."
            )
        st.caption(
            "Indexed evidence: validated TGSRTC static GTFS metadata and the local "
            "GTFS search implementation. This does not provide live operation, delay, "
            "cancellation, or current-fare information."
        )
        st.caption(
            "Retrieval uses local TF-IDF lexical similarity, not semantic embeddings. "
            "A relevance score is not factual verification; answers quote one passage "
            "and cite that passage's stored provenance."
        )

        question_examples = [
            ("Interpret scheduled results", "How is a scheduled bus journey duration calculated?"),
            ("Check timetable coverage", "What dates and timezone does the TGSRTC timetable cover?"),
            ("Understand transfers", "Does this planner build transfers between separate trips?"),
            ("Missed connection guidance", "What should I do if I miss a connection?"),
        ]
        example_columns = st.columns(2)
        for example_index, (example_label, example_question) in enumerate(question_examples):
            with example_columns[example_index % 2]:
                if st.button(
                    example_label,
                    key=f"dash_advisory_example_{example_index}",
                    use_container_width=True,
                ):
                    st.session_state["passenger_advisor_pending_question"] = example_question

        pending_question = st.session_state.pop(
            "passenger_advisor_pending_question", None
        )
        rag_question = st.chat_input(
            "Ask about scheduled results or timetable limitations",
            key="dash_passenger_advisor_chat_input",
        )
        submitted_question = pending_question or rag_question
        if not rag_engine.is_indexed:
            st.warning(
                "Trusted passenger-advisor evidence is unavailable, so answers "
                "cannot cite retrieved passages. Questions still work: known app "
                "questions receive clearly labelled deterministic guidance, and "
                "anything else is reported as unverifiable. "
                f"{rag_engine.index_error or 'The source index could not be loaded.'}"
            )
        if submitted_question:
            answer_result = answer_passenger_question(
                rag_engine,
                submitted_question,
            )
            st.session_state["passenger_advisor_messages"].append({
                "role": "user",
                "content": submitted_question,
            })
            st.session_state["passenger_advisor_messages"].append({
                "role": "assistant",
                "content": answer_result["answer"],
                "answer_mode": answer_result["answer_mode"],
                "confidence_level": answer_result["confidence_level"],
                "explanation": answer_result["explanation"],
                "results": answer_result["results"],
                "has_evidence": answer_result["has_evidence"],
                "is_fallback": answer_result.get("is_fallback", False),
            })

        for message in st.session_state["passenger_advisor_messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                    continue
                if message["has_evidence"]:
                    st.markdown("**Retrieved evidence (extractive; not model-generated):**")
                    st.write(message["content"])
                elif message.get("is_fallback"):
                    st.info(
                        "**Deterministic app guidance (fallback — not retrieved "
                        "source evidence):**\n\n" + message["content"]
                    )
                else:
                    st.warning(message["content"])
                    if message["confidence_level"] == "Error":
                        st.error(message["explanation"])
                st.caption(message["answer_mode"])
                if message["has_evidence"]:
                    st.markdown("**Sources**")
                    for document in message["results"][:1]:
                        with st.container(border=True):
                            st.markdown(f"**{document['source_title']} — {document['title']}**")
                            if document.get("source_url"):
                                st.markdown(f"[Open source URL]({document['source_url']})")
                            st.caption(
                                f"Publisher: {document.get('publisher') or 'not provided'} · "
                                f"Source category: {document.get('source_class') or 'not provided'} · "
                                f"Document type: {document.get('document_type') or 'not provided'}"
                            )
                            st.caption(
                                f"Document reference: {document.get('original_filename') or document['source_ref']} · "
                                f"Publication date: {document.get('published_date') or 'not provided'} · "
                                f"Source retrieved: {document.get('retrieved_at') or 'not provided'}"
                            )


# =============================================================================
# VIEW 5: Bus Search & Smart Alternatives (bus_tracker)
# =============================================================================
elif st.session_state["view"] == "bus_tracker":
    target_mode = "Bus"
    page_title = "🚌 Bus Route Finder — Hyderabad Terminals & Stops"
    page_scope = "Hyderabad-area bus stops and terminals ONLY (MGBS, JBS, LB Nagar, Uppal, Kukatpally, Miyapur, etc.)"

    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.markdown(f"### {page_title}")
        st.caption(f"Strict Scope: **{page_scope}**")
    with col_hdr2:
        if st.button("← Return to Command Center", use_container_width=True, key="btn_bus_back"):
            go_to("dashboard")

    gtfs_feed = None
    gtfs_feed_error = None
    gtfs_status = get_tgsrtc_gtfs_status()
    try:
        gtfs_feed = get_tgsrtc_gtfs_feed()
    except TgsrtcGtfsError as feed_error:
        gtfs_feed_error = str(feed_error)

    if gtfs_feed is not None:
        st.markdown("#### TGSRTC static GTFS source")
        st.success(
            f"Verified TGSRTC timetable loaded · "
            f"{gtfs_feed.route_count:,} routes · {gtfs_feed.stop_count:,} stops · "
            f"coverage {gtfs_feed.feed_start_date.isoformat()} to {gtfs_feed.feed_end_date.isoformat()}"
        )
        with st.container(border=True):
            st.caption(
                f"Contains data provided by TGSRTC · "
                f"{gtfs_status.attribution} · TGSRTC does not endorse RouteRescue AI."
            )
            st.caption(
                f"Source: {gtfs_feed.source_metadata.get('source_url', 'MobilityDatabase TGSRTC GTFS snapshot')} · "
                f"Archive retrieved (UTC): {gtfs_feed.source_metadata.get('retrieved_at_utc', 'not recorded')} · "
                f"locally validated (UTC): {gtfs_feed.validated_at_utc}"
            )
            st.caption(
                f"{gtfs_feed.agency['agency_name']} · feed version {gtfs_feed.feed_info['feed_version']} · "
                f"timezone {gtfs_feed.agency['agency_timezone']} · "
                f"{gtfs_feed.trip_count:,} trips · {gtfs_feed.stop_times_count:,} stop-time records"
            )
            if not {"pickup_type", "drop_off_type"}.issubset(gtfs_feed.stop_time_columns):
                st.caption(
                    "This feed does not include pickup/drop-off restriction fields; "
                    "the planner cannot apply restrictions that are absent from the source."
                )
            st.caption(
                "Scheduled times only — no live operation, delays, or cancellations. "
                "No fare tables. Searches return direct trips only; use Transfer Planner for connections."
            )
    elif gtfs_status.status == "missing":
        st.info(f"{gtfs_status.message} Save the official download as `{gtfs_status.feed_path}`.")
    else:
        st.error(f"The archive is present but failed GTFS validation: {gtfs_feed_error or gtfs_status.message}")

    use_tgsrtc_schedule = gtfs_feed is not None
    if use_tgsrtc_schedule:
        mode_stops = gtfs_feed.get_hyderabad_stops()
    else:
        mode_stops = []

    stop_labels = [s["stop_name"] for s in mode_stops]
    stop_id_map = {s["stop_name"]: s["stop_id"] for s in mode_stops}
    if not mode_stops:
        st.info(
            "No verified Hyderabad stops are available. Bus search is disabled until the "
            "TGSRTC GTFS feed passes validation and contains recognized stops."
        )

    # Determine default indices if navigated from Favourites or Nearby Stops
    pre_orig = st.session_state.pop("selected_origin", None)
    pre_dest = st.session_state.pop("selected_destination", None)

    orig_index = 0
    if pre_orig and pre_orig in stop_labels:
        orig_index = stop_labels.index(pre_orig)

    dest_index = max(0, min(1, len(stop_labels) - 1))
    if pre_dest and pre_dest in stop_labels:
        dest_index = stop_labels.index(pre_dest)

    # Search Form
    with st.container(border=True):
        st.markdown("#### Plan a journey")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            orig_choice = st.selectbox(
                "Origin stop",
                options=stop_labels or ["No validated bus stops available"],
                index=orig_index,
                help="Choose the stop where this journey begins (Get on at).",
            )
        with col_f2:
            dest_choice = st.selectbox(
                "Destination stop",
                options=stop_labels or ["No validated bus stops available"],
                index=dest_index,
                help="Choose the stop where this journey ends (Get off at).",
            )

        col_f3, col_f4 = st.columns(2)
        with col_f3:
            travel_date = st.date_input(
                "Journey date",
                value=datetime.date.today(),
                help=(
                    f"Scheduled service coverage: {gtfs_feed.feed_start_date.isoformat()} "
                    f"to {gtfs_feed.feed_end_date.isoformat()}."
                    if use_tgsrtc_schedule
                    else None
                ),
            )
        with col_f4:
            dep_time_value = st.time_input(
                "Departure time (not before)",
                value=datetime.time(8, 0),
                help="Only scheduled trips departing at or after this local time are searched.",
            )
            dep_time_choice = dep_time_value.strftime("%H:%M")

        svc_filter = ""
        priority_options = ["Fastest"]
        pref_choice = st.radio(
            "Rank journeys by",
            options=priority_options,
            horizontal=True,
            help=(
                "Fastest uses timetable-derived elapsed time. Fewest transfers is unavailable "
                "because this feed search returns direct trips only and does not construct transfers."
                if use_tgsrtc_schedule
                else None
            ),
        )
        st.caption(
            "Cheapest is unavailable without comparable fare data. Fewest transfers is unavailable "
            "because searches return direct trips only."
        )
        pref_key = "fastest"

        apply_search = st.button(
            "Search scheduled journeys",
            type="primary",
            use_container_width=True,
            disabled=not mode_stops,
        )

    # Perform Search
    search_key = "last_bus_search"
    if apply_search:
        orig_id = stop_id_map.get(orig_choice)
        dest_id = stop_id_map.get(dest_choice)

        if orig_id == dest_id:
            st.warning("⚠️ Origin and Destination cannot be identical. Please select different stops.")
            st.session_state[search_key] = None
        else:
            with st.spinner("Searching the TGSRTC static schedule..."):
                try:
                    if use_tgsrtc_schedule:
                        if not (
                            gtfs_feed.feed_start_date
                            <= travel_date
                            <= gtfs_feed.feed_end_date
                        ):
                            raise ValueError(
                                "UNSUPPORTED_DATE: the requested travel date is outside "
                                f"the feed coverage "
                                f"({gtfs_feed.feed_start_date.isoformat()} to "
                                f"{gtfs_feed.feed_end_date.isoformat()})."
                            )
                        results = gtfs_feed.search_journeys(
                            origin_stop_id=orig_id,
                            destination_stop_id=dest_id,
                            travel_date=travel_date,
                            earliest_departure=datetime.datetime.strptime(
                                dep_time_choice, "%H:%M"
                            ).time(),
                            service_filter=svc_filter,
                        )
                        result_source_id = "tgsrtc_static_gtfs"
                    else:
                        raise RuntimeError(
                            "The validated TGSRTC feed is unavailable; prototype records "
                            "are not used for bus searches."
                        )

                    # Record in search history for logged-in user
                    if st.session_state.get("user_profile") and st.session_state["user_profile"].get("email"):
                        auth_manager.add_user_search_history(
                            email=st.session_state["user_profile"]["email"],
                            origin_name=orig_choice,
                            destination_name=dest_choice,
                            travel_date=str(travel_date),
                            departure_time=dep_time_choice,
                            results_count=len(results),
                        )

                    st.session_state[search_key] = {
                        "orig_label": orig_choice,
                        "dest_label": dest_choice,
                        "orig_id": orig_id,
                        "dest_id": dest_id,
                        "date": str(travel_date),
                        "dep_time": dep_time_choice,
                        "service_filter": svc_filter,
                        "results": results,
                        "search_error": None,
                        "widen_used": False,
                        "source_id": result_source_id,
                    }
                except (ValueError, RuntimeError) as search_err:
                    st.session_state[search_key] = {
                        "orig_label": orig_choice,
                        "dest_label": dest_choice,
                        "orig_id": orig_id,
                        "dest_id": dest_id,
                        "date": str(travel_date),
                        "dep_time": dep_time_choice,
                        "service_filter": svc_filter,
                        "results": [],
                        "search_error": str(search_err),
                        "widen_used": False,
                        "source_id": "tgsrtc_static_gtfs",
                    }
                except Exception as unexpected_err:
                    import traceback
                    st.error(
                        f"Unexpected search failure: {unexpected_err}. "
                        "Please check server logs for details."
                    )
                    print("[RouteRescue Search ERROR]", traceback.format_exc())
                    st.session_state[search_key] = None

    # Render Journey Status Results
    search_data = st.session_state.get(search_key)
    if search_data is not None and search_data.get("source_id") != "tgsrtc_static_gtfs":
        search_data = None
    if gtfs_feed is None:
        search_data = None

    if search_data is not None and (
        search_data.get("orig_label") != orig_choice
        or search_data.get("dest_label") != dest_choice
        or search_data.get("date") != str(travel_date)
        or search_data.get("dep_time") != dep_time_choice
        or search_data.get("service_filter", "") != svc_filter
    ):
        st.info("Search inputs changed. Apply / Check Journey to calculate candidates for these inputs.")
        search_data = None

    if search_data is not None:
        raw_results = search_data.get("results", [])
        search_error = search_data.get("search_error")
        widen_used = search_data.get("widen_used", False)

        st.write("")
        st.markdown("### 🚌 Scheduled Bus Journeys (Static GTFS)")
        st.caption(
            f"Routing between **{search_data['orig_label']}** and **{search_data['dest_label']}** "
            f"on **{format_passenger_date(search_data['date'])}** — departing at or after **{format_passenger_time(search_data['dep_time'])}**"
        )
        st.caption(
            "Candidate set: date-active trips in the downloaded TGSRTC static timetable. "
            "Ranking is recalculated from this same set when the priority changes."
        )

        if search_error:
            if "UNSUPPORTED_DATE" in search_error:
                st.info(search_error.replace("UNSUPPORTED_DATE: ", ""))
            elif "UNVERIFIED_LOCAL_DATA" in search_error:
                st.warning("No verified operator data source is configured.")
            else:
                st.error(f"Search error: {search_error}")

        # Filter departures departing strictly at or after requested time
        requested_time = None
        try:
            requested_time = datetime.datetime.strptime(search_data["dep_time"], "%H:%M").time()
        except Exception:
            pass

        results = []
        for r in raw_results:
            dep_str = r.get("scheduled_departure_at") or r.get("departure_time")
            if dep_str and requested_time is not None:
                try:
                    dep_time_obj = datetime.datetime.fromisoformat(dep_str).time()
                    if dep_time_obj < requested_time:
                        continue
                except Exception:
                    pass
            results.append(r)

        results = route_engine.rank_journeys(results, pref_key)

        if not results and not search_error:
            st.info(
                f"**No scheduled TGSRTC trip connects these stops on this service date and time.**\n\n"
                "No matching trip in the loaded static TGSRTC timetable connects these stops "
                "on this service date after the selected time. No live operating status is available."
            )

            if gtfs_feed is not None:
                orig_id_val = stop_id_map.get(orig_choice, "")
                dest_id_val = stop_id_map.get(dest_choice, "")
                if orig_id_val and dest_id_val:
                    try:
                        diag = gtfs_feed.diagnose_empty_search(
                            origin_stop_id=orig_id_val,
                            destination_stop_id=dest_id_val,
                            travel_date=datetime.date.fromisoformat(search_data["date"]),
                            earliest_departure=datetime.datetime.strptime(search_data["dep_time"], "%H:%M").time(),
                        )
                    except Exception:
                        diag = None
                    if diag:
                        with st.container(border=True):
                            st.markdown("#### 🔍 Timetable Investigation & Directional Analysis")
                            _diag_reason_labels = {
                                "outside_date_range": "Diagnosed reason: the selected date is outside the timetable's coverage.",
                                "reverse_service_exists": "Diagnosed reason: opposite-direction service only.",
                                "earlier_service_only": "Diagnosed reason: earlier departures only.",
                                "no_direct_service": "Diagnosed reason: no direct service in this direction.",
                            }
                            _diag_reason_label = _diag_reason_labels.get(diag.get("reason"))
                            if _diag_reason_label:
                                st.warning(_diag_reason_label)
                            st.write(diag["message"])
                            if diag.get("reverse_trips_count", 0) > 0:
                                st.caption(
                                    f"💡 **Reverse Corridor Notice:** The TGSRTC feed confirms direct service in the opposite direction "
                                    f"({dest_choice} → {orig_choice}: {diag['reverse_trips_count']} trips). "
                                    f"Buses traveling from {orig_choice} toward {dest_choice} typically board at different directional stop IDs."
                                )
                                if st.button(
                                    f"🔄 Search Reverse Corridor ({dest_choice} → {orig_choice})",
                                    key="btn_reverse_search_bus",
                                    use_container_width=True,
                                ):
                                    try:
                                        rev_results = gtfs_feed.search_journeys(
                                            origin_stop_id=dest_id_val,
                                            destination_stop_id=orig_id_val,
                                            travel_date=datetime.date.fromisoformat(search_data["date"]),
                                            earliest_departure=datetime.datetime.strptime(search_data["dep_time"], "%H:%M").time(),
                                            service_filter=svc_filter,
                                        )
                                        st.session_state[search_key] = {
                                            "orig_label": dest_choice,
                                            "dest_label": orig_choice,
                                            "orig_id": dest_id_val,
                                            "dest_id": orig_id_val,
                                            "date": search_data["date"],
                                            "dep_time": search_data["dep_time"],
                                            "service_filter": svc_filter,
                                            "results": rev_results,
                                            "search_error": None,
                                            "widen_used": False,
                                            "source_id": "tgsrtc_static_gtfs",
                                        }
                                        st.rerun()
                                    except Exception as rev_err:
                                        st.error(f"Reverse search failed: {rev_err}")

                    # One-Click Transfer Planner action
                    st.write("")
                    st.markdown("#### 🔀 Try Bus Transfer Planner")
                    st.caption("Search for a 1-transfer journey connecting these stops via a transit interchange hub.")
                    if st.button("Open Transfer Planner for This Route →", type="primary", use_container_width=True, key="btn_go_transfers"):
                        st.session_state["selected_origin"] = orig_choice
                        st.session_state["selected_destination"] = dest_choice
                        go_to("transfers")

        elif results:
            if len(results) == 1:
                st.info(
                    "Only one schedule-backed candidate was found. Different priorities "
                    "cannot produce distinct journey options."
                )

            # Smart Bus Alternatives: Display best matching direct bus first
            primary = results[0]
            st.markdown("### 🚌 Best matching direct bus")

            dep_fmt = format_passenger_time(primary['scheduled_departure_at'])
            arr_fmt = format_passenger_time(primary['scheduled_arrival_at'])
            dur_fmt = format_passenger_duration(primary.get('elapsed_duration_min', primary.get('effective_duration_min', 0)))

            st.markdown(
                f"""
                <div class="result-box">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                        <div>
                            <span class="status-badge status-ontime" style="margin-bottom: 6px;">Best Match</span>
                            <h3 style="margin: 0; color: #f8fafc;">{primary['title']}</h3>
                            <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">Service #{primary.get('service_number', 'N/A')}</div>
                        </div>
                        <span class="status-badge status-scheduled">Static Timetable</span>
                    </div>
                    <p style="margin: 4px 0;"><b>From:</b> {primary['origin_name']} &nbsp;➔&nbsp; <b>To:</b> {primary['destination_name']}</p>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;margin:14px 0;">
                        <div>
                            <small style="color:#64748b;text-transform:uppercase;font-size:0.75rem;">DEPARTURE</small>
                            <h3 style="margin:2px 0 0 0;color:#f8fafc;">{dep_fmt}</h3>
                        </div>
                        <div>
                            <small style="color:#64748b;text-transform:uppercase;font-size:0.75rem;">ARRIVAL</small>
                            <h3 style="margin:2px 0 0 0;color:#f8fafc;">{arr_fmt}</h3>
                        </div>
                        <div>
                            <small style="color:#64748b;text-transform:uppercase;font-size:0.75rem;">SCHEDULED DURATION</small>
                            <h3 style="margin:2px 0 0 0;color:#38bdf8;">{dur_fmt}</h3>
                        </div>
                    </div>
                    <p style="font-size:0.82rem;color:#94a3b8;margin:0;">Scheduled timetable; live vehicle positions and operational status are not provided.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Save to favourites button
            col_fav1, _ = st.columns([2, 3])
            with col_fav1:
                if st.button("⭐ Save Route to Favourites", key=f"fav_primary_{primary['journey_id']}", use_container_width=True):
                    if st.session_state.get("user_profile") and st.session_state["user_profile"].get("email"):
                        ok, fav_msg = auth_manager.add_user_favourite(
                            email=st.session_state["user_profile"]["email"],
                            origin_name=primary["origin_name"],
                            destination_name=primary["destination_name"],
                            origin_id=search_data.get("orig_id"),
                            destination_id=search_data.get("dest_id"),
                            notes=f"Route: {primary['title']}",
                        )
                        if ok:
                            st.success(fav_msg)
                        else:
                            st.info(fav_msg)
                    else:
                        st.info("Log in to save routes to your personal favourites.")



            # Display no more than 3 options total (best match + up to 2 alternatives)
            alternatives = results[1:3]
            if alternatives:
                with st.expander("🚌 Show up to 3 alternative buses"):
                    for b_idx, bus in enumerate(alternatives, start=1):
                        b_dep = format_passenger_time(bus['scheduled_departure_at'])
                        b_arr = format_passenger_time(bus['scheduled_arrival_at'])
                        b_dur = format_passenger_duration(bus.get('elapsed_duration_min', bus.get('effective_duration_min', 0)))
                        st.markdown(
                            f"""
                            <div class="result-box">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <div>
                                        <b style="font-size:1.1rem;color:#f8fafc;">Option {b_idx + 1}: {bus['title']}</b>
                                        <div style="font-size:0.82rem;color:#38bdf8;">Service #{bus.get('service_number', 'N/A')}</div>
                                    </div>
                                    <span class="status-badge status-scheduled">Scheduled</span>
                                </div>
                                <p style="margin:6px 0;"><b>From:</b> {bus['origin_name']} &nbsp;➔&nbsp; <b>To:</b> {bus['destination_name']}</p>
                                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;background:rgba(255,255,255,0.02);padding:10px;border-radius:8px;margin-top:8px;">
                                    <div><small style="color:#64748b;">DEPARTURE</small><div style="font-weight:700;">{b_dep}</div></div>
                                    <div><small style="color:#64748b;">ARRIVAL</small><div style="font-weight:700;">{b_arr}</div></div>
                                    <div><small style="color:#64748b;">DURATION</small><div style="font-weight:700;color:#38bdf8;">{b_dur}</div></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No alternative direct buses depart on this schedule after the selected time.")

            st.caption(
                f"Service date: {format_passenger_date(search_data['date'])} · "
                "Contains data provided by TGSRTC. Static timetable only."
            )


# =============================================================================
# VIEW 6: Bus Transfer Planner (transfers)
# =============================================================================
elif st.session_state["view"] == "transfers":
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.markdown("### 🔀 Bus Transfer Planner")
        st.caption("Plan verified 1-transfer connections across TGSRTC bus routes when no direct bus operates.")
    with col_hdr2:
        if st.button("← Return to Command Center", use_container_width=True, key="btn_transfer_back"):
            go_to("dashboard")

    gtfs_feed = None
    try:
        gtfs_feed = get_tgsrtc_gtfs_feed()
    except Exception as e:
        st.error(f"GTFS timetable feed could not be loaded: {e}")

    if gtfs_feed is not None:
        hyderabad_stops = gtfs_feed.get_hyderabad_stops()
        stop_labels = [s["stop_name"] for s in hyderabad_stops]
        stop_id_map = {s["stop_name"]: s["stop_id"] for s in hyderabad_stops}

        pre_orig = st.session_state.pop("selected_origin", None)
        pre_dest = st.session_state.pop("selected_destination", None)

        orig_idx = 0
        if pre_orig and pre_orig in stop_labels:
            orig_idx = stop_labels.index(pre_orig)

        dest_idx = max(0, min(1, len(stop_labels) - 1))
        if pre_dest and pre_dest in stop_labels:
            dest_idx = stop_labels.index(pre_dest)

        with st.container(border=True):
            st.markdown("#### Find a journey when one bus cannot take you all the way")
            st.caption("A transfer means you get off the first bus at an interchange stop and board a second bus. Choose your start and destination; RouteRescue checks the timetable and connection wait.")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                t_orig = st.selectbox("Origin stop (Boarding)", options=stop_labels, index=orig_idx, key="xfer_orig")
            with col_t2:
                t_dest = st.selectbox("Destination stop (Arrival)", options=stop_labels, index=dest_idx, key="xfer_dest")

            col_t3, col_t4, col_t5 = st.columns(3)
            with col_t3:
                t_date = st.date_input("Journey date", value=datetime.date.today(), key="xfer_date")
            with col_t4:
                t_time = st.time_input("Departure time (not before)", value=datetime.time(8, 0), key="xfer_time")
            with col_t5:
                min_buffer = st.number_input("Min transfer layover (minutes)", min_value=3, max_value=45, value=5, step=1, key="xfer_min_buf")

            search_xfer_clicked = st.button("Search 1-Transfer Journeys", type="primary", use_container_width=True, key="btn_search_xfer")

        if search_xfer_clicked:
            orig_sid = stop_id_map.get(t_orig)
            dest_sid = stop_id_map.get(t_dest)

            if orig_sid == dest_sid:
                st.warning("⚠️ Origin and Destination stops cannot be identical.")
            else:
                with st.spinner("Searching timetable for valid 1-transfer connections..."):
                    try:
                        transfer_results = gtfs_feed.search_transfer_journeys(
                            origin_stop_id=orig_sid,
                            destination_stop_id=dest_sid,
                            travel_date=t_date,
                            earliest_departure=t_time,
                            min_transfer_buffer_min=int(min_buffer),
                            max_layover_min=120,
                            max_options=3,
                        )
                        st.session_state["last_transfer_search"] = {
                            "orig": t_orig,
                            "dest": t_dest,
                            "date": str(t_date),
                            "time": t_time.strftime("%H:%M"),
                            "results": transfer_results,
                        }
                    except Exception as err:
                        st.error(f"Transfer calculation failed: {err}")
                        st.session_state["last_transfer_search"] = None

        last_xfer = st.session_state.get("last_transfer_search")
        if last_xfer is not None:
            results = last_xfer.get("results", [])
            st.write("")
            st.markdown("### 🔀 Validated 1-Transfer Journey Options")
            st.caption(
                f"Connection corridor: **{last_xfer['orig']}** ➔ **{last_xfer['dest']}** on **{format_passenger_date(last_xfer['date'])}** "
                f"after **{format_passenger_time(last_xfer['time'])}**"
            )

            if not results:
                st.info(
                    f"No scheduled 1-transfer connection connects **{last_xfer['orig']}** and **{last_xfer['dest']}** "
                    f"in the TGSRTC timetable for {format_passenger_date(last_xfer['date'])} departing at or after {format_passenger_time(last_xfer['time'])}.\n\n"
                    "RouteRescue AI avoids impossible connections and does not fabricate missing data."
                )
            else:
                for idx, journey in enumerate(results, start=1):
                    leg1 = journey["legs"][0]
                    leg2 = journey["legs"][1]
                    transfer_stop = journey["transfer_station"]
                    layover_m = int(round(journey["layover_minutes"]))
                    total_dur = format_passenger_duration(journey["elapsed_duration_min"])

                    st.html(
                        textwrap.dedent(f"""
                        <div class="result-box">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                <div>
                                    <span class="status-badge status-transfer" style="margin-bottom:6px;">Option {idx} · 1 Transfer</span>
                                    <h3 style="margin:0;color:#f8fafc;">{journey['title']}</h3>
                                </div>
                                <span class="status-badge status-scheduled">Verified Connection</span>
                            </div>

                            <div class="transfer-leg-box">
                                <div style="display:flex;justify-content:space-between;">
                                    <b>🚌 Leg 1: {leg1['route_name']} (Service #{leg1['service_number']})</b>
                                    <span style="color:#38bdf8;">{format_passenger_duration(leg1['effective_duration_min'])}</span>
                                </div>
                                <div style="font-size:0.9rem;color:#cbd5e1;margin-top:6px;">
                                    Board at: <b>{leg1['origin_name']}</b> at <span style="color:#f8fafc;font-weight:700;">{format_passenger_time(leg1['departure_time'])}</span><br>
                                    Arrive at: <b>{leg1['destination_name']}</b> at <span style="color:#f8fafc;font-weight:700;">{format_passenger_time(leg1['arrival_time'])}</span>
                                </div>
                            </div>

                            <div style="background:rgba(56,189,248,0.1);border:1px dashed #0284c7;border-radius:8px;padding:8px 14px;margin:8px 0;display:flex;align-items:center;gap:10px;">
                                <span style="font-size:1.2rem;">⏱️</span>
                                <div>
                                    <span style="color:#38bdf8;font-weight:700;">Transfer at {transfer_stop}</span>
                                    <span style="color:#94a3b8;font-size:0.85rem;"> — {layover_m} minutes scheduled connection window</span>
                                </div>
                            </div>

                            <div class="transfer-leg-box">
                                <div style="display:flex;justify-content:space-between;">
                                    <b>🚌 Leg 2: {leg2['route_name']} (Service #{leg2['service_number']})</b>
                                    <span style="color:#38bdf8;">{format_passenger_duration(leg2['effective_duration_min'])}</span>
                                </div>
                                <div style="font-size:0.9rem;color:#cbd5e1;margin-top:6px;">
                                    Board at: <b>{leg2['origin_name']}</b> at <span style="color:#f8fafc;font-weight:700;">{format_passenger_time(leg2['departure_time'])}</span><br>
                                    Arrive at: <b>{leg2['destination_name']}</b> at <span style="color:#f8fafc;font-weight:700;">{format_passenger_time(leg2['arrival_time'])}</span>
                                </div>
                            </div>

                            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);">
                                <div><b>Total Elapsed Journey:</b> <span style="color:#38bdf8;font-weight:700;">{total_dur}</span></div>
                                <div style="font-size:0.8rem;color:#94a3b8;">Departure: {format_passenger_time(journey['departure_time'])} ➔ Arrival: {format_passenger_time(journey['arrival_time'])}</div>
                            </div>
                        </div>
                        """).strip()
                    )

                    # Save to favourites button
                    if st.button(f"⭐ Save Transfer Route #{idx}", key=f"fav_xfer_{idx}"):
                        if st.session_state.get("user_profile") and st.session_state["user_profile"].get("email"):
                            ok, fav_msg = auth_manager.add_user_favourite(
                                email=st.session_state["user_profile"]["email"],
                                origin_name=journey["origin_name"],
                                destination_name=journey["destination_name"],
                                notes=f"1-Transfer via {transfer_stop} ({journey['service_number']})",
                            )
                            if ok:
                                st.success(fav_msg)
                            else:
                                st.info(fav_msg)
                        else:
                            st.info("Log in to save routes to your favourites.")

                    with st.expander(f"Intermediate stops ({len(leg1['stops'])} stops on Leg 1, {len(leg2['stops'])} stops on Leg 2)", expanded=False):
                        st.markdown(f"**Leg 1 Stops ({leg1['origin_name']} ➔ {transfer_stop}):**")
                        st.dataframe(pd.DataFrame([
                            {"Seq": s["stop_sequence"], "Stop": s["stop_name"], "Departure": format_passenger_time(s["departure_at"])}
                            for s in leg1["stops"]
                        ]), hide_index=True, use_container_width=True)
                        st.markdown(f"**Leg 2 Stops ({transfer_stop} ➔ {leg2['destination_name']}):**")
                        st.dataframe(pd.DataFrame([
                            {"Seq": s["stop_sequence"], "Stop": s["stop_name"], "Departure": format_passenger_time(s["departure_at"])}
                            for s in leg2["stops"]
                        ]), hide_index=True, use_container_width=True)

                st.caption(
                    "All transfer connections are validated using actual scheduled TGSRTC timetable records. "
                    "Operating delays, cancellations, and live bus positions are not available."
                )


# =============================================================================
# VIEW 7: Nearby Bus Stops (nearby_stops)
# =============================================================================
elif st.session_state["view"] == "nearby_stops":
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.markdown("### 📍 Nearby Bus Stops")
        st.caption("Find verified TGSRTC bus stops near familiar landmarks or search by stop name.")
    with col_hdr2:
        if st.button("← Return to Command Center", use_container_width=True, key="btn_nearby_back"):
            go_to("dashboard")

    gtfs_feed = None
    try:
        gtfs_feed = get_tgsrtc_gtfs_feed()
    except Exception as e:
        st.error(f"GTFS feed could not be loaded: {e}")

    if gtfs_feed is not None:
        tab_coords, tab_name = st.tabs(["📍 Find Near a Place", "🔍 Search by Stop Name"])

        with tab_coords:
            st.markdown("##### Find bus stops near a familiar place")
            st.caption("Choose a landmark you know. You do not need to enter latitude or longitude.")
            landmark_presets = {
                "Select preset location...": (None, None),
                "Charminar (Old City)": (17.3616, 78.4747),
                "MGBS — Mahatma Gandhi Bus Station": (17.3783, 78.4842),
                "Secunderabad Railway Station": (17.4399, 78.4983),
                "JBS — Jubilee Bus Station": (17.4476, 78.4976),
                "Ameerpet Metro Interchange": (17.4375, 78.4483),
                "Hitech City (Cyber Towers)": (17.4504, 78.3808),
                "Gachibowli Stadium / Junction": (17.4401, 78.3489),
                "Kukatpally / KPHB": (17.4849, 78.4138),
                "Mehdipatnam Bus Depot": (17.3916, 78.4400),
                "LB Nagar Ring Road": (17.3457, 78.5522),
                "Uppal X Roads": (17.4018, 78.5602),
            }

            landmark_presets.pop("Select preset location...", None)
            preset_choice = st.selectbox("Choose a nearby landmark:", options=list(landmark_presets.keys()), key="preset_lm")
            u_lat, u_lon = landmark_presets[preset_choice]
            radius_km = st.slider("How far should we look?", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="u_rad", format="%.1f km")

            if st.button("Find Nearby Stops", type="primary", use_container_width=True, key="btn_find_nearby"):
                try:
                    nearby_stops = gtfs_feed.find_nearby_stops(u_lat, u_lon, max_distance_km=radius_km, limit=12)
                    st.session_state["last_nearby_results"] = nearby_stops
                except Exception as err:
                    st.error(f"Failed to find nearby stops: {err}")

        with tab_name:
            st.markdown("##### Search by stop name or landmark")
            search_query = st.text_input("Enter stop name or landmark:", placeholder="e.g. Charminar, Kukatpally, Mehdipatnam, Koti", key="txt_stop_query")
            if st.button("Search Stops by Name", key="btn_search_by_name", use_container_width=True) and search_query.strip():
                try:
                    found_stops = gtfs_feed.search_stops_by_name(search_query.strip(), limit=15)
                    st.session_state["last_nearby_results"] = found_stops
                except Exception as err:
                    st.error(f"Search failed: {err}")

        # Render Stop Cards
        displayed_stops = st.session_state.get("last_nearby_results")
        if displayed_stops is not None:
            st.write("")
            st.markdown(f"#### 📍 Bus Stops Found ({len(displayed_stops)})")
            if not displayed_stops:
                st.info("No bus stops found matching your search parameters.")
            else:
                for s_item in displayed_stops:
                    s_name = s_item["stop_name"]
                    s_id = s_item["stop_id"]
                    dist_text = f"📍 {s_item['distance_km']} km away" if "distance_km" in s_item else "📍 Verified GTFS Stop"
                    routes = s_item.get("routes", [])

                    with st.container(border=True):
                        col_s1, col_s2 = st.columns([3, 1])
                        with col_s1:
                            st.markdown(f"### {s_name}")
                            st.caption(f"{dist_text} · Stop reference: `{s_id}`")
                            if routes:
                                r_names = [r["route_name"] for r in routes[:8]]
                                st.markdown(f"**Passing routes ({len(routes)}):** " + ", ".join(f"`{rn}`" for rn in r_names))
                                if len(routes) > 8:
                                    st.caption(f"+ {len(routes) - 8} more routes pass through this stop.")
                            else:
                                st.caption("No passing route metadata found in current feed index.")
                        with col_s2:
                            if st.button("Plan from this stop ➔", key=f"btn_plan_from_{s_id}", use_container_width=True):
                                st.session_state["selected_origin"] = s_name
                                go_to("bus_tracker")

            st.caption(
                "Stop coordinates from TGSRTC GTFS data. Live bus arrival positions and "
                "real-time vehicle tracking are not available."
            )


# =============================================================================
# VIEW 8: Favourite Routes and Search History (favourites_history)
# =============================================================================
elif st.session_state["view"] == "favourites_history":
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.markdown("### ⭐ Favourite Routes & Search History")
        st.caption("Manage your private saved bus routes and view recent journey searches.")
    with col_hdr2:
        if st.button("← Return to Command Center", use_container_width=True, key="btn_fav_back"):
            go_to("dashboard")

    user_email = (
        st.session_state["user_profile"]["email"]
        if st.session_state.get("user_profile") and st.session_state["user_profile"].get("email")
        else "demo@routerescue.ai"
    )

    tab_favs, tab_hist = st.tabs(["⭐ Saved Favourite Routes", "🕒 Recent Search History"])

    with tab_favs:
        st.markdown("##### ⭐ Your Saved Routes")
        user_favs = auth_manager.get_user_favourites(user_email)
        if not user_favs:
            st.info(
                "You have no saved favourite routes yet. "
                "Search for a journey in Bus Search or Transfer Planner and click '⭐ Save to Favourites' to bookmark it here!"
            )
        else:
            for fav in user_favs:
                f_id = fav["id"]
                with st.container(border=True):
                    col_f1, col_f2 = st.columns([3, 1])
                    with col_f1:
                        st.markdown(f"#### 🚌 {fav['origin_name']} ➔ {fav['destination_name']}")
                        if fav.get("notes"):
                            st.caption(f"Note: {fav['notes']}")
                        st.caption(f"Saved on: {format_passenger_date(fav.get('created_at', ''))}")
                    with col_f2:
                        if st.button("Search This Route ➔", key=f"btn_search_fav_{f_id}", type="primary", use_container_width=True):
                            st.session_state["selected_origin"] = fav["origin_name"]
                            st.session_state["selected_destination"] = fav["destination_name"]
                            go_to("bus_tracker")
                        if st.button("Remove", key=f"btn_remove_fav_{f_id}", use_container_width=True):
                            auth_manager.remove_user_favourite(user_email, f_id)
                            st.rerun()

        st.write("")
        st.markdown("##### ➕ Bookmark a Route")
        gtfs_feed = None
        try:
            gtfs_feed = get_tgsrtc_gtfs_feed()
        except Exception:
            pass

        if gtfs_feed is not None:
            stop_labels = [s["stop_name"] for s in gtfs_feed.get_hyderabad_stops()]
            with st.container(border=True):
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    new_orig = st.selectbox("Origin Stop", options=stop_labels, index=0, key="new_fav_orig")
                with col_m2:
                    new_dest = st.selectbox("Destination Stop", options=stop_labels, index=min(1, len(stop_labels) - 1), key="new_fav_dest")
                fav_note = st.text_input("Optional label/note (e.g. Daily Office Commute):", key="new_fav_note")

                if st.button("Add to Favourites", key="btn_add_manual_fav"):
                    if new_orig == new_dest:
                        st.warning("Origin and destination must be different.")
                    else:
                        ok, msg = auth_manager.add_user_favourite(
                            user_email, new_orig, new_dest, notes=fav_note
                        )
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.warning(msg)

    with tab_hist:
        st.markdown("##### 🕒 Recent Searches")
        history = auth_manager.get_user_search_history(user_email, limit=15)
        if not history:
            st.info("No recent searches found. Use Bus Search to plan scheduled journeys.")
        else:
            col_clr1, _ = st.columns([1, 4])
            with col_clr1:
                if st.button("Clear History", key="btn_clear_history"):
                    auth_manager.clear_user_search_history(user_email)
                    st.rerun()

            for h in history:
                with st.container(border=True):
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"**{h['origin_name']} ➔ {h['destination_name']}**")
                        st.caption(
                            f"Date: {format_passenger_date(h['travel_date'])} · "
                            f"Departure: after {format_passenger_time(h['departure_time'])} · "
                            f"{h.get('results_count', 0)} option(s) found"
                        )
                    with col_h2:
                        if st.button("Re-run Search ➔", key=f"btn_rerun_{h['id']}", use_container_width=True):
                            st.session_state["selected_origin"] = h["origin_name"]
                            st.session_state["selected_destination"] = h["destination_name"]
                            go_to("bus_tracker")


# =============================================================================
# VIEW 9: Dedicated AI Assistant Page
# =============================================================================
elif st.session_state["view"] == "ai_assistant":
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.markdown("### 🤖 RouteRescue AI — Passenger Advisor")
        st.caption("Evidence-only passenger guidance, grounded in the verified TGSRTC timetable.")
    with col_hdr2:
        if st.button("← Return to Command Center", use_container_width=True, key="btn_ai_back"):
            go_to("dashboard")

    st.info(
        "Evidence-only mode: no language-model adapter is configured. The advisor "
        "does not generate answers or use API credentials."
    )
    if rag_engine.legacy_document_count:
        st.warning(
            f"{rag_engine.legacy_document_count} records in data/travel_guides.csv "
            "are illustrative or lack verifiable provenance and are excluded from "
            "retrieval and answers."
        )
    st.caption(
        "Indexed evidence: validated TGSRTC static GTFS metadata and the local "
        "GTFS search implementation. This does not provide live operation, delay, "
        "cancellation, or current-fare information."
    )
    st.caption(
        "Retrieval uses local TF-IDF lexical similarity, not semantic embeddings. "
        "A relevance score is not factual verification; answers quote one passage "
        "and cite that passage's stored provenance."
    )

    st.markdown("##### 💡 Example passenger questions")
    _ai_question_examples = [
        ("🚌 Find buses between stops", "How do I find scheduled buses between two stops?"),
        ("❓ Why are no buses available?", "Why are no buses available for my selected route?"),
        ("⚡ What is the fastest scheduled bus?", "What is the fastest scheduled bus and how is duration calculated?"),
        ("🔀 How do bus transfers work?", "Does this planner build transfers between separate trips?"),
        ("⚠️ Verified disruptions status", "Are any verified service disruptions or live delays available?"),
        ("📅 Timetable coverage & timezone", "What dates and timezone does the TGSRTC timetable cover?"),
    ]
    _ai_cols = st.columns(3)
    for _ex_idx, (_ex_label, _ex_question) in enumerate(_ai_question_examples):
        with _ai_cols[_ex_idx % 3]:
            if st.button(
                _ex_label,
                key=f"ai_page_example_{_ex_idx}",
                use_container_width=True,
            ):
                st.session_state["passenger_advisor_pending_question"] = _ex_question
                st.rerun()

    with st.container(border=True):
        st.markdown("##### 💬 Ask RouteRescue AI")
        _ai_col_in, _ai_col_btn = st.columns([4, 1])
        with _ai_col_in:
            _ai_text_input = st.text_input(
                "Ask a question about your journey, scheduled trips, or transport data:",
                placeholder="e.g. How are bus journeys calculated? or Does this planner build transfers?",
                key="ai_page_question_input",
                label_visibility="collapsed",
            )
        with _ai_col_btn:
            _ai_ask_clicked = st.button(
                "Ask AI",
                key="btn_ai_page_ask",
                type="primary",
                use_container_width=True,
            )

    _ai_pending = st.session_state.pop("passenger_advisor_pending_question", None)
    _ai_submitted = (
        _ai_pending
        or (_ai_text_input if _ai_ask_clicked and _ai_text_input.strip() else None)
    )

    if not rag_engine.is_indexed:
        st.warning(
            "Trusted passenger-advisor evidence is unavailable, so answers "
            "cannot cite retrieved passages. Questions still work: known app "
            "questions receive clearly labelled deterministic guidance, and "
            "anything else is reported as unverifiable. "
            f"{rag_engine.index_error or 'The source index could not be loaded.'}"
        )
    if _ai_submitted:
        _ai_result = answer_passenger_question(rag_engine, _ai_submitted)
        st.session_state["passenger_advisor_messages"].append({
            "role": "user", "content": _ai_submitted,
        })
        st.session_state["passenger_advisor_messages"].append({
            "role": "assistant",
            "content": _ai_result["answer"],
            "answer_mode": _ai_result["answer_mode"],
            "confidence_level": _ai_result["confidence_level"],
            "explanation": _ai_result["explanation"],
            "results": _ai_result["results"],
            "has_evidence": _ai_result["has_evidence"],
            "is_fallback": _ai_result.get("is_fallback", False),
        })

    for _ai_msg in st.session_state["passenger_advisor_messages"]:
        with st.chat_message(_ai_msg["role"]):
            if _ai_msg["role"] == "user":
                st.write(_ai_msg["content"])
                continue
            if _ai_msg["has_evidence"]:
                st.markdown("**Retrieved evidence (extractive; not model-generated):**")
                st.write(_ai_msg["content"])
            elif _ai_msg.get("is_fallback"):
                st.info(
                    "**Deterministic app guidance (fallback — not retrieved "
                    "source evidence):**\n\n" + _ai_msg["content"]
                )
            else:
                st.warning(_ai_msg["content"])
                if _ai_msg["confidence_level"] == "Error":
                    st.error(_ai_msg["explanation"])
            st.caption(_ai_msg["answer_mode"])
            if _ai_msg["has_evidence"]:
                st.markdown("**Sources**")
                for _ai_doc in _ai_msg["results"][:1]:
                    with st.container(border=True):
                        st.markdown(f"**{_ai_doc['source_title']} — {_ai_doc['title']}**")
                        if _ai_doc.get("source_url"):
                            st.markdown(f"[Open source URL]({_ai_doc['source_url']})")
                        st.caption(
                            f"Publisher: {_ai_doc.get('publisher') or 'not provided'} · "
                            f"Source category: {_ai_doc.get('source_class') or 'not provided'} · "
                            f"Document type: {_ai_doc.get('document_type') or 'not provided'}"
                        )
                        st.caption(
                            f"Document reference: {_ai_doc.get('original_filename') or _ai_doc['source_ref']} · "
                            f"Publication date: {_ai_doc.get('published_date') or 'not provided'} · "
                            f"Source retrieved: {_ai_doc.get('retrieved_at') or 'not provided'}"
                        )
