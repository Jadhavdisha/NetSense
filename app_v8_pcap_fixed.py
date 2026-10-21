
import tempfile
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.context_engine import calculate_priority
from src.scheduler import compare_schedulers
from src.feature_extraction import extract_packet_features, create_flow_features


# ============================================================
# NETSENSE — INTELLIGENT NETWORK MANAGEMENT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="NetSense | Intelligent Network Management",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "traffic_dataset.csv"
MODEL_PATH = ROOT / "models" / "traffic_classifier.pkl"


# -----------------------------
# Global styling
# -----------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            --bg: #050810;
            --panel: #0a101d;
            --panel2: #0d1524;
            --line: rgba(148,163,184,.12);
            --text: #edf5ff;
            --muted: #a8b7ca;
            --cyan: #38bdf8;
            --cyan2: #22d3ee;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
            --purple: #a78bfa;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                linear-gradient(rgba(56,189,248,.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56,189,248,.018) 1px, transparent 1px),
                radial-gradient(circle at 12% 0%, rgba(34,211,238,.10), transparent 28%),
                radial-gradient(circle at 92% 12%, rgba(167,139,250,.08), transparent 25%),
                #050810;
            background-size: 42px 42px, 42px 42px, auto, auto, auto;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#070c16 0%,#060a12 100%);
            border-right: 1px solid rgba(56,189,248,.10);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.3rem;
        }

        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1550px;
        }

        /* ---------- Brand ---------- */
        .brand {
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -1.8px;
            color: #f7fbff;
            line-height: 1;
            margin-bottom: 4px;
        }

        .brand span { color: var(--cyan); }

        .subtitle {
            color: #9fb0c5;
            font-size: .82rem;
            letter-spacing: .01em;
            margin-bottom: 1.1rem;
        }

        .eyebrow {
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: .67rem;
            font-weight: 600;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }

        /* ---------- Top command bar ---------- */
        .topbar {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:16px;
            margin-bottom:18px;
            padding:10px 14px;
            border:1px solid var(--line);
            border-radius:12px;
            background:rgba(8,14,25,.76);
            backdrop-filter: blur(14px);
        }

        .topbar-left {
            display:flex;
            align-items:center;
            gap:10px;
            color:#d3deea;
            font-size:.76rem;
        }

        .live-dot {
            width:8px;height:8px;border-radius:50%;
            background:var(--green);
            box-shadow:0 0 0 4px rgba(52,211,153,.08),0 0 16px rgba(52,211,153,.65);
        }

        .mono {
            font-family:'JetBrains Mono',monospace;
            color:#b7c6d8;
            font-size:.72rem;
        }

        .topbar-right {
            color:#9aabc0;
            font-family:'JetBrains Mono',monospace;
            font-size:.68rem;
        }

        /* ---------- Section headers ---------- */
        .section-title {
            font-size: 1.22rem;
            font-weight: 750;
            color: var(--text);
            letter-spacing: -.35px;
            margin: .8rem 0 .55rem 0;
        }

        .section-caption {
            color: #a8b7ca;
            font-size: .78rem;
            margin-top: -.3rem;
            margin-bottom: .9rem;
        }

        /* ---------- KPI cards ---------- */
        .metric-card {
            position:relative;
            overflow:hidden;
            background:linear-gradient(145deg,rgba(13,22,38,.97),rgba(7,12,22,.97));
            border:1px solid rgba(148,163,184,.12);
            border-radius:16px;
            padding:17px 18px;
            min-height:116px;
            box-shadow:0 14px 35px rgba(0,0,0,.20);
            transition:transform .18s ease,border-color .18s ease;
        }

        .metric-card:hover {
            transform:translateY(-2px);
            border-color:rgba(56,189,248,.30);
        }

        .metric-card:after {
            content:"";
            position:absolute;
            right:-34px;
            top:-45px;
            width:105px;height:105px;
            border-radius:50%;
            background:radial-gradient(circle,rgba(56,189,248,.13),transparent 68%);
        }

        .metric-label {
            color:#73839a;
            font-size:.67rem;
            text-transform:uppercase;
            letter-spacing:.13em;
            font-weight:700;
        }

        .metric-value {
            color:#f5f9ff;
            font-size:1.78rem;
            font-weight:800;
            letter-spacing:-.7px;
            margin-top:8px;
        }

        .metric-note {
            color:#53657d;
            font-size:.69rem;
            margin-top:4px;
        }

        /* ---------- Pipeline ---------- */
        .pipeline {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:8px;
            padding:17px;
            background:rgba(7,13,24,.78);
            border:1px solid rgba(56,189,248,.12);
            border-radius:16px;
            box-shadow:inset 0 1px rgba(255,255,255,.02);
            overflow-x:auto;
        }

        .pipe-node {
            min-width:148px;
            text-align:center;
            padding:14px 10px;
            border-radius:12px;
            background:linear-gradient(180deg,#0e1829,#0a1220);
            border:1px solid rgba(56,189,248,.14);
        }

        .pipe-node:hover {
            border-color:rgba(56,189,248,.42);
            box-shadow:0 0 25px rgba(56,189,248,.07);
        }

        .pipe-icon { font-size:1.35rem;margin-bottom:5px; }
        .pipe-title { color:#e9f2ff;font-size:.76rem;font-weight:750; }
        .pipe-sub { color:#62748c;font-size:.62rem;margin-top:3px; }
        .arrow { color:#38bdf8;font-size:1.25rem;font-weight:800; }

        /* ---------- Status / insight ---------- */
        .status {
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:7px 11px;
            border-radius:999px;
            font-family:'JetBrains Mono',monospace;
            font-size:.64rem;
            font-weight:700;
            letter-spacing:.04em;
            background:rgba(52,211,153,.06);
            color:#4ade80;
            border:1px solid rgba(52,211,153,.17);
        }

        .status-dot {
            width:7px;height:7px;border-radius:50%;
            background:#4ade80;
            box-shadow:0 0 10px #4ade80;
        }

        /* ---------- Flow Decision Trace ---------- */
        .trace-step {
            display:flex;
            align-items:flex-start;
            gap:16px;
            padding:14px 18px;
            margin:0;
            border:1px solid rgba(56,189,248,.16);
            border-radius:12px;
            background:linear-gradient(135deg, rgba(13,25,43,.96), rgba(8,15,27,.98));
            box-shadow:0 8px 24px rgba(0,0,0,.18);
        }

        .trace-number {
            flex:0 0 38px;
            width:38px;
            height:38px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:10px;
            background:rgba(56,189,248,.10);
            border:1px solid rgba(56,189,248,.22);
            color:#38bdf8 !important;
            font-family:'JetBrains Mono',monospace;
            font-size:.72rem;
            font-weight:800;
        }

        .trace-title {
            color:#f1f7ff !important;
            font-size:.72rem;
            font-weight:800;
            letter-spacing:.12em;
            margin-bottom:5px;
        }

        .trace-value {
            color:#c7d7e8 !important;
            font-size:.82rem;
            line-height:1.5;
        }

        .trace-arrow {
            color:#38bdf8 !important;
            text-align:center;
            font-size:1.15rem;
            font-weight:800;
            line-height:1.25;
            margin:3px 0;
        }

        .trace-container {
            padding:18px;
            border:1px solid rgba(56,189,248,.12);
            border-radius:16px;
            background:rgba(7,13,24,.72);
        }

        .insight {
            padding:14px 16px;
            border-radius:13px;
            background:linear-gradient(90deg,rgba(56,189,248,.065),rgba(167,139,250,.035));
            border:1px solid rgba(56,189,248,.14);
            color:#c5d3e2;
            font-size:.79rem;
            line-height:1.6;
        }

        .high { color:#fb7185;font-weight:800; }
        .medium { color:#fbbf24;font-weight:800; }
        .low { color:#4ade80;font-weight:800; }

        /* ---------- Streamlit controls ---------- */
        div[data-testid="stDataFrame"] {
            border:1px solid rgba(148,163,184,.10);
            border-radius:13px;
            overflow:hidden;
        }

        div[data-testid="stFileUploader"] {
            background:rgba(9,16,28,.7);
            border:1px dashed rgba(56,189,248,.20);
            border-radius:14px;
            padding:4px;
        }

        button[kind="secondary"] {
            border-color:rgba(56,189,248,.17);
        }

        /* ---------- Sidebar navigation ---------- */
        [data-testid="stSidebar"] .stRadio label {
            padding:5px 3px;
            border-radius:8px;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background:rgba(56,189,248,.055);
        }

        [data-testid="stSidebar"] .stSlider {
            margin-bottom:8px;
        }


        /* ---------- Scheduler Command Center ---------- */
        .lab-hero {
            position:relative;
            overflow:hidden;
            padding:20px 22px;
            border-radius:18px;
            background:
                radial-gradient(circle at 92% 10%, rgba(56,189,248,.11), transparent 30%),
                linear-gradient(135deg, rgba(12,22,38,.98), rgba(7,12,22,.96));
            border:1px solid rgba(56,189,248,.16);
            margin:8px 0 16px;
        }

        .lab-hero-title {
            font-size:1.42rem;
            font-weight:800;
            color:#f1f7ff;
            letter-spacing:-.45px;
        }

        .lab-hero-sub {
            color:#73859d;
            font-size:.76rem;
            margin-top:5px;
            max-width:780px;
            line-height:1.55;
        }

        .lab-chip {
            display:inline-flex;
            align-items:center;
            gap:7px;
            margin-top:13px;
            padding:6px 9px;
            border-radius:999px;
            border:1px solid rgba(56,189,248,.14);
            background:rgba(56,189,248,.045);
            color:#8ba4bd;
            font-family:'JetBrains Mono',monospace;
            font-size:.62rem;
        }

        .lab-stat {
            padding:15px 16px;
            border:1px solid rgba(148,163,184,.11);
            border-radius:14px;
            background:linear-gradient(145deg,rgba(13,22,38,.94),rgba(8,13,23,.96));
            min-height:92px;
        }

        .lab-stat-label {
            color:#697b92;
            font-size:.61rem;
            text-transform:uppercase;
            letter-spacing:.12em;
            font-weight:700;
        }

        .lab-stat-value {
            color:#f4f8ff;
            font-size:1.45rem;
            font-weight:800;
            margin-top:7px;
            letter-spacing:-.4px;
        }

        .lab-stat-note {
            color:#52657d;
            font-size:.65rem;
            margin-top:3px;
        }

        .decision-banner {
            padding:16px 18px;
            border-radius:14px;
            border:1px solid rgba(52,211,153,.15);
            background:linear-gradient(90deg,rgba(52,211,153,.065),rgba(56,189,248,.035));
            margin:12px 0 18px;
        }

        .decision-big {
            color:#4ade80;
            font-size:1.1rem;
            font-weight:800;
            letter-spacing:.02em;
        }

        .decision-small {
            color:#718399;
            font-size:.71rem;
            margin-top:4px;
            line-height:1.5;
        }

        .queue-title {
            display:flex;
            justify-content:space-between;
            align-items:center;
            color:#dce9f8;
            font-size:.84rem;
            font-weight:750;
            margin:4px 0 8px;
        }

        .queue-title span {
            color:#566980;
            font-family:'JetBrains Mono',monospace;
            font-size:.62rem;
            font-weight:500;
        }

        .queue {
            display:flex;
            gap:7px;
            flex-wrap:wrap;
            padding:12px;
            border-radius:14px;
            background:#070d18;
            border:1px solid rgba(148,163,184,.09);
            min-height:72px;
        }

        .queue-flow {
            padding:8px 9px;
            border-radius:9px;
            background:linear-gradient(145deg,#101b2d,#0b1422);
            border:1px solid rgba(56,189,248,.12);
            min-width:86px;
            text-align:center;
        }

        .queue-flow.high { border-color:rgba(251,113,133,.28); }
        .queue-flow.medium { border-color:rgba(251,191,36,.22); }
        .queue-flow.low { border-color:rgba(52,211,153,.20); }

        .queue-id {
            color:#e6eef9;
            font-family:'JetBrains Mono',monospace;
            font-size:.63rem;
            font-weight:700;
        }

        .queue-class {
            color:#71859e;
            font-size:.57rem;
            margin-top:3px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .queue-score {
            color:#a7b9ce;
            font-family:'JetBrains Mono',monospace;
            font-size:.56rem;
            margin-top:4px;
        }

        .queue-arrow {
            color:#334961;
            display:flex;
            align-items:center;
            font-size:1rem;
        }

        .compare-panel {
            padding:16px;
            border-radius:15px;
            background:rgba(8,14,25,.72);
            border:1px solid rgba(148,163,184,.10);
        }

        .compare-title {
            color:#dce9f7;
            font-size:.8rem;
            font-weight:750;
            margin-bottom:4px;
        }

        .compare-sub {
            color:#596d85;
            font-size:.64rem;
            margin-bottom:12px;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align:center;
            color:#3f4d61;
            font-family:'JetBrains Mono',monospace;
            font-size:.62rem;
            letter-spacing:.05em;
            padding:24px 0 5px;
        }

        /* ---------- Hide Streamlit chrome ---------- */
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {background:transparent !important;}

        /* ---------- Scrollbar ---------- */
        ::-webkit-scrollbar { width:7px;height:7px; }
        ::-webkit-scrollbar-track { background:#060a12; }
        ::-webkit-scrollbar-thumb { background:#1b2a3e;border-radius:10px; }
        ::-webkit-scrollbar-thumb:hover { background:#29415e; }
        /* ---------- GLOBAL READABILITY OVERRIDES ---------- */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
        .stApp div[data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stMetricLabel"] {
            color: #d6e1ee;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #f4f8ff !important;
        }

        .stApp p, .stApp li,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"] p {
            color: #b8c7d8 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #c5d2e1;
        }
        [data-testid="stSidebar"] .brand,
        [data-testid="stSidebar"] .brand span { color: #f7fbff !important; }
        [data-testid="stSidebar"] .brand span { color: #38bdf8 !important; }

        /* Section descriptions / helper text */
        .section-caption, .section-subtitle, .subtitle {
            color: #aebed0 !important;
        }

        /* Tables and dataframe text */
        .stDataFrame, .stTable {
            color: #d9e4ef !important;
        }
        .stDataFrame [role="columnheader"],
        .stDataFrame [role="gridcell"] {
            color: #d9e4ef !important;
        }

        /* Inputs */
        .stSelectbox label, .stSlider label, .stNumberInput label,
        .stTextInput label, .stFileUploader label {
            color: #cbd8e6 !important;
        }
        .stSelectbox [data-baseweb="select"] * {
            color: #172033 !important;
        }

        /* Decision trace — intentionally high contrast */
        .trace-container {
            padding: 18px !important;
            border: 1px solid rgba(56,189,248,.25) !important;
            border-radius: 16px !important;
            background: rgba(6,12,22,.94) !important;
        }
        .trace-step {
            background: linear-gradient(135deg, #101d30, #0b1422) !important;
            border: 1px solid rgba(56,189,248,.28) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,.30) !important;
            padding: 16px 18px !important;
        }
        .trace-number {
            color: #06111d !important;
            background: #38bdf8 !important;
            border-color: #38bdf8 !important;
            font-weight: 900 !important;
        }
        .trace-title {
            color: #f8fbff !important;
            font-size: .82rem !important;
            font-weight: 800 !important;
            letter-spacing: .10em !important;
        }
        .trace-value {
            color: #dce9f7 !important;
            font-size: .92rem !important;
            font-weight: 600 !important;
        }
        .trace-arrow {
            color: #38bdf8 !important;
            font-size: 1.25rem !important;
            text-shadow: 0 0 10px rgba(56,189,248,.45);
        }

        /* Cards / important values */
        .metric-card, .insight-card, .decision-card, .status-card {
            color: #dce8f5 !important;
        }
        .metric-value, .metric-label, .metric-sub {
            color: #f1f7ff !important;
        }

        /* Streamlit alerts */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {
            color: #dbe7f3 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    obj = joblib.load(MODEL_PATH)

    # Current classifier.py saves a dictionary.
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        accuracy = obj.get("accuracy", None)
        features = obj.get("features", None)
        return model, accuracy, features

    return obj, None, None


def format_bytes(value):
    """Format a byte count for compact dashboard display."""
    value = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    size = abs(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        value /= 1024.0
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{value:,.0f} {units[unit_index]}"
    return f"{value:,.2f} {units[unit_index]}"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_class(score):
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def apply_context(df, network_load, queue_condition):
    out = df.copy()

    scores = []
    levels = []

    for traffic_class in out["traffic_class"]:
        result = calculate_priority(
            traffic_class,
            network_load,
            queue_condition,
        )

        if isinstance(result, dict):
            score = result.get("priority_score", result.get("score"))
            level = result.get("priority_level", result.get("level"))

            if score is None:
                for key, value in result.items():
                    if "score" in str(key).lower():
                        score = value
                        break

            if level is None:
                for key, value in result.items():
                    if "level" in str(key).lower() and isinstance(value, str):
                        level = value
                        break

            if score is None or level is None:
                raise ValueError(f"Unexpected calculate_priority result: {result}")

        elif isinstance(result, (tuple, list)):
            score, level = result[0], result[1]

        else:
            raise ValueError(
                f"Unexpected calculate_priority return type: {type(result)}"
            )

        scores.append(float(score))
        levels.append(str(level))

    out["priority_score"] = scores
    out["priority_level"] = levels
    return out


def scheduler_metrics(df):
    fifo, context, comparison = compare_schedulers(df.copy())
    return fifo, context, comparison


def comparison_value(comparison, metric, policy):
    """Read a scheduler metric from either dict- or DataFrame-shaped output."""
    
    metric_aliases = {
        "High Priority Waiting": "High Priority Waiting Time",
        "Low Priority Waiting": "Low Priority Waiting Time",
        "Average Waiting": "Average Waiting Time",
        "Maximum Waiting": "Maximum Waiting Time",
    }

    metric = metric_aliases.get(metric, metric)
    
    if isinstance(comparison, pd.DataFrame):
        if "Metric" not in comparison.columns:
            raise KeyError(
                f"Scheduler comparison DataFrame has no 'Metric' column. "
                f"Columns: {list(comparison.columns)}"
            )

        # Match metric names robustly (case/whitespace independent).
        metric_key = str(metric).strip().lower()
        rows = comparison[
            comparison["Metric"].astype(str).str.strip().str.lower() == metric_key
        ]
        if rows.empty:
            raise KeyError(
                f"Could not find metric '{metric}' for policy '{policy}'. "
                f"Available metrics: {comparison['Metric'].tolist()}"
            )

        # Allow harmless naming variations in the policy column.
        policy_col = None
        wanted = str(policy).strip().lower()
        for col in comparison.columns:
            if str(col).strip().lower() == wanted:
                policy_col = col
                break
        if policy_col is None:
            raise KeyError(
                f"Could not find policy '{policy}'. "
                f"Available columns/keys: {list(comparison.columns)}"
            )

        return float(rows.iloc[0][policy_col])

    if isinstance(comparison, dict):
        metric_data = comparison.get(metric)
        if isinstance(metric_data, dict):
            if policy in metric_data:
                return float(metric_data[policy])
            # Case-insensitive policy lookup.
            wanted = str(policy).strip().lower()
            for key, value in metric_data.items():
                if str(key).strip().lower() == wanted:
                    return float(value)

        # Also support a dict that is already keyed by policy.
        if policy in comparison and isinstance(comparison[policy], dict):
            policy_data = comparison[policy]
            if metric in policy_data:
                return float(policy_data[metric])

    raise KeyError(
        f"Could not find metric '{metric}' for policy '{policy}'. "
        f"Available comparison type: {type(comparison).__name__}"
    )


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b7c4d8"),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa9bd"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.08)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.08)",
    )
    return fig


def gauge(title, value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"size": 26}},
            title={"text": title, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#718096"}},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#0f1725",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#101a28"},
                    {"range": [40, 70], "color": "#121d2d"},
                    {"range": [70, 100], "color": "#172336"},
                ],
            },
        )
    )
    return chart_layout(fig, 220)


# -----------------------------
# Data / model availability
# -----------------------------
df = load_dataset()
model_info = load_model()

if df.empty:
    st.error(
        "traffic_dataset.csv was not found. Run the preprocessing pipeline first."
    )
    st.stop()

model = model_info[0] if model_info else None
model_accuracy = model_info[1] if model_info else None
model_features = model_info[2] if model_info else None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:1.45rem;font-weight:800;color:#f5f9ff;">
            Net<span style="color:#38bdf8;">Sense</span>
        </div>
        <div style="font-size:0.72rem;color:#65748b;margin-bottom:18px;">
            Intelligent Network Management
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Control Center")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Traffic Intelligence",
            "Context Engine",
            "Scheduler Lab",
            "Flow Inspector",
            "PCAP Analyzer",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    sample_size = st.slider(
        "Flows in analysis",
        min_value=25,
        max_value=min(500, len(df)),
        value=min(100, len(df)),
        step=25,
    )

    network_load = st.slider(
        "Network Load",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05,
        format="%.2f",
    )

    queue_condition = st.slider(
        "Queue Condition",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        format="%.2f",
    )

    st.divider()

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            PIPELINE ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("PCAP → Flow → ML → Context → Scheduler")


# -----------------------------
# Sample + context
# -----------------------------
sample = df.sample(
    n=min(sample_size, len(df)),
    random_state=42,
).reset_index(drop=True)

sample["flow_id"] = np.arange(1, len(sample) + 1)
sample = apply_context(sample, network_load, queue_condition)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="brand">Net<span>Sense</span></div>
    <div class="subtitle">
        Multi-Modal Traffic Classification Pipeline for Context-Aware Scheduling
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-left">
            <span class="live-dot"></span>
            <b style="color:#dce9f7;">NETWORK OPERATIONS</b>
            <span>·</span>
            <span>Context engine active</span>
        </div>
        <div class="topbar-right">
            LOAD {network_load:.2f} &nbsp;|&nbsp; QUEUE {queue_condition:.2f}
            &nbsp;|&nbsp; FLOWS {len(sample):,}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    st.markdown(
        '<div class="section-title">Network Operations Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="eyebrow">NETSENSE / REAL-TIME DECISION SURFACE</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Flows",
            f"{len(df):,}",
            "Current processed dataset",
        )

    with c2:
        accuracy_text = f"{model_accuracy * 100:.2f}%" if model_accuracy else "N/A"
        metric_card(
            "ML Accuracy",
            accuracy_text,
            "Random Forest test accuracy",
        )

    with c3:
        metric_card(
            "Network Load",
            f"{network_load * 100:.0f}%",
            "Interactive context input",
        )

    with c4:
        metric_card(
            "Queue Condition",
            f"{queue_condition * 100:.0f}%",
            "Interactive context input",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pipeline">
            <div class="pipe-node">
                <div class="pipe-icon">📡</div>
                <div class="pipe-title">PCAP / Traffic</div>
                <div class="pipe-sub">Raw packets</div>
            </div>
            <div class="arrow">→</div>
            <div class="pipe-node">
                <div class="pipe-icon">🔬</div>
                <div class="pipe-title">Flow Features</div>
                <div class="pipe-sub">Packet statistics</div>
            </div>
            <div class="arrow">→</div>
            <div class="pipe-node">
                <div class="pipe-icon">🧠</div>
                <div class="pipe-title">Classification</div>
                <div class="pipe-sub">Random Forest</div>
            </div>
            <div class="arrow">→</div>
            <div class="pipe-node">
                <div class="pipe-icon">⚡</div>
                <div class="pipe-title">Context Engine</div>
                <div class="pipe-sub">Load + queue + urgency</div>
            </div>
            <div class="arrow">→</div>
            <div class="pipe-node">
                <div class="pipe-icon">🚦</div>
                <div class="pipe-title">Scheduler</div>
                <div class="pipe-sub">Priority-aware ordering</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            '<div class="section-title">Traffic Distribution</div>',
            unsafe_allow_html=True,
        )

        counts = df["traffic_class"].value_counts().reset_index()
        counts.columns = ["traffic_class", "flows"]

        fig = px.pie(
            counts,
            names="traffic_class",
            values="flows",
            hole=0.62,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent",
        )
        fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>{len(df):,}</b><br>flows",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18, color="#edf5ff"),
                )
            ]
        )
        st.plotly_chart(chart_layout(fig, 390), width='stretch')

    with right:
        st.markdown(
            '<div class="section-title">Current Priority State</div>',
            unsafe_allow_html=True,
        )

        priority_counts = sample["priority_level"].value_counts().reindex(
            ["HIGH", "MEDIUM", "LOW"],
            fill_value=0,
        )

        fig = px.bar(
            x=priority_counts.index,
            y=priority_counts.values,
            labels={"x": "Priority", "y": "Flows"},
            text=priority_counts.values,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(chart_layout(fig, 390), width='stretch')

    high_pct = (sample["priority_level"] == "HIGH").mean() * 100

    st.markdown(
        f"""
        <div class="insight">
            <b>Context insight:</b>
            Under the current network state
            (<b>{network_load:.2f}</b> load,
            <b>{queue_condition:.2f}</b> queue condition),
            <b>{high_pct:.1f}%</b> of the analyzed flows are classified as
            <span class="high">HIGH priority</span>.
            Adjust the controls in the sidebar to observe how scheduling priorities change.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TRAFFIC INTELLIGENCE
# ============================================================
elif page == "Traffic Intelligence":

    st.markdown(
        '<div class="section-title">Traffic Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Flow-level characteristics extracted from the selected PCAP dataset.</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        metric_card("Flows", f"{len(sample):,}", "Current analysis sample")
    with b:
        metric_card(
            "Total Bytes",
            f"{sample['total_bytes'].sum() / 1e6:.2f} MB",
            "Across selected flows",
        )
    with c:
        metric_card(
            "Avg Packet Size",
            f"{sample['average_packet_size'].mean():.1f}",
            "Bytes / packet",
        )
    with d:
        metric_card(
            "Avg Duration",
            f"{sample['duration'].mean():.3f}s",
            "Per flow",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # ML CLASSIFICATION SNAPSHOT
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="eyebrow">NETSENSE / MACHINE INTELLIGENCE</div>
        <div class="section-title" style="margin-top:0;">Random Forest Classification</div>
        <div class="section-caption">
            The flow feature vector is classified into the four traffic categories used by the current prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_count = int(df["traffic_class"].nunique())
    feature_count = len(model_features) if model_features is not None else 8
    accuracy_display = f"{model_accuracy * 100:.2f}%" if model_accuracy is not None else "N/A"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Model", "Random Forest", "Supervised flow classifier")
    with m2:
        metric_card("Test Accuracy", accuracy_display, "Current flow-level experiment")
    with m3:
        metric_card("Traffic Classes", str(class_count), "Web · Gaming · Video · Data")
    with m4:
        metric_card("Input Features", str(feature_count), "Flow-level feature vector")

    st.markdown("<br>", unsafe_allow_html=True)

    ml_left, ml_right = st.columns([1.05, 1])

    with ml_left:
        st.markdown(
            '<div class="section-title">Feature Importance</div>',
            unsafe_allow_html=True,
        )

        if model is not None and hasattr(model, "feature_importances_"):
            feature_names = model_features if model_features is not None else [
                "protocol",
                "packet_count",
                "total_bytes",
                "duration",
                "average_packet_size",
                "packet_rate",
                "byte_rate",
                "average_iat",
            ]
            importances = np.asarray(model.feature_importances_, dtype=float)
            n = min(len(feature_names), len(importances))

            importance_df = pd.DataFrame({
                "feature": feature_names[:n],
                "importance": importances[:n],
            }).sort_values("importance", ascending=True)

            fig = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                text=importance_df["importance"].map(lambda x: f"{x:.3f}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_xaxes(title="Relative importance", range=[0, max(0.22, float(importance_df["importance"].max()) * 1.18)])
            fig.update_yaxes(title="")
            st.plotly_chart(chart_layout(fig, 390), width='stretch')
        else:
            st.info("Feature-importance values are unavailable for the loaded model.")

    with ml_right:
        st.markdown(
            '<div class="section-title">Training Class Distribution</div>',
            unsafe_allow_html=True,
        )

        class_counts = (
            df["traffic_class"]
            .value_counts()
            .rename_axis("traffic_class")
            .reset_index(name="flows")
        )

        fig = px.bar(
            class_counts,
            x="traffic_class",
            y="flows",
            text="flows",
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(tickangle=-15, title="")
        fig.update_yaxes(title="Flows")
        st.plotly_chart(chart_layout(fig, 390), width='stretch')

        st.markdown(
            f"""
            <div class="insight">
                <b>Classifier scope:</b> the reported <b>{accuracy_display}</b> accuracy belongs to the current
                flow-level train/test experiment on the selected four-PCAP dataset. It should not be interpreted
                as universal accuracy for arbitrary networks.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    l, r = st.columns(2)

    with l:
        fig = px.box(
            sample,
            x="traffic_class",
            y="packet_count",
            title="Packet Count by Traffic Class",
        )
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(chart_layout(fig), width='stretch')

    with r:
        fig = px.scatter(
            sample,
            x="packet_rate",
            y="byte_rate",
            size="total_bytes",
            color="traffic_class",
            hover_data=["flow_id", "duration"],
            title="Flow Behaviour Map",
        )
        st.plotly_chart(chart_layout(fig), width='stretch')

    st.markdown(
        '<div class="section-title">Flow Feature Matrix</div>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "flow_id",
        "traffic_class",
        "protocol",
        "packet_count",
        "total_bytes",
        "duration",
        "average_packet_size",
        "packet_rate",
        "byte_rate",
        "average_iat",
    ]

    st.dataframe(
        sample[display_cols].round(4),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# CONTEXT ENGINE
# ============================================================
elif page == "Context Engine":

    st.markdown(
        '<div class="section-title">Context-Aware Decision Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Priority is calculated from urgency, delay sensitivity, network load and queue condition.</div>',
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        st.plotly_chart(gauge("Network Load", network_load), width='stretch')

    with g2:
        st.plotly_chart(gauge("Queue Condition", queue_condition), width='stretch')

    st.markdown(
        """
        <div class="insight">
            <b>Priority model:</b>
            0.35 × Urgency + 0.25 × Delay Sensitivity
            + 0.20 × Network Load + 0.20 × Queue Condition
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    context_summary = (
        sample.groupby("traffic_class")["priority_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        context_summary,
        x="traffic_class",
        y="priority_score",
        text=context_summary["priority_score"].round(3),
        title="Average Priority Score by Traffic Class",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_traces(textposition="outside")
    st.plotly_chart(chart_layout(fig), width='stretch')

    priority_table = (
        sample.groupby("traffic_class")
        .agg(
            flows=("flow_id", "count"),
            avg_priority=("priority_score", "mean"),
            max_priority=("priority_score", "max"),
        )
        .reset_index()
        .sort_values("avg_priority", ascending=False)
    )

    st.dataframe(
        priority_table.round(3),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# SCHEDULER LAB
# ============================================================
elif page == "Scheduler Lab":

    fifo, context, comparison = scheduler_metrics(sample)

    avg_fifo = comparison_value(comparison, "Average Waiting Time", "FIFO")
    avg_context = comparison_value(comparison, "Average Waiting Time", "Context-Aware")
    high_fifo = comparison_value(comparison, "High Priority Waiting Time", "FIFO")
    high_context = comparison_value(comparison, "High Priority Waiting Time", "Context-Aware")
    low_fifo = comparison_value(comparison, "Low Priority Waiting Time", "FIFO")
    low_context = comparison_value(comparison, "Low Priority Waiting Time", "Context-Aware")
    throughput_fifo = comparison_value(comparison, "Throughput", "FIFO")
    throughput_context = comparison_value(comparison, "Throughput", "Context-Aware")

    high_reduction = (
        ((high_fifo - high_context) / high_fifo) * 100
        if high_fifo > 0 else 0
    )

    st.markdown(
        """
        <div class="eyebrow">NETSENSE / DECISION LAB</div>
        <div class="lab-hero">
            <div class="lab-hero-title">Context-Aware Scheduler Lab</div>
            <div class="lab-hero-sub">
                Watch the same network flows move through two scheduling policies.
                FIFO preserves arrival order; the proposed scheduler promotes
                flows using context-derived priority.
            </div>
            <div class="lab-chip">● LIVE SIMULATION &nbsp;|&nbsp; SAME FLOW SET &nbsp;|&nbsp; PRIORITY-DRIVEN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Decision headline
    st.markdown(
        f"""
        <div class="decision-banner">
            <div class="decision-big">
                ↑ {high_reduction:.1f}% lower high-priority waiting time
            </div>
            <div class="decision-small">
                Observed on the current {len(sample):,}-flow simulation.
                The scheduler intentionally trades some lower-priority waiting
                time for faster service of urgent, delay-sensitive traffic.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(
            f"""
            <div class="lab-stat">
                <div class="lab-stat-label">High Priority Wait</div>
                <div class="lab-stat-value">{high_context*1000:.1f} ms</div>
                <div class="lab-stat-note">Context-aware</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="lab-stat">
                <div class="lab-stat-label">Baseline Wait</div>
                <div class="lab-stat-value">{high_fifo*1000:.1f} ms</div>
                <div class="lab-stat-note">FIFO baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="lab-stat">
                <div class="lab-stat-label">Average Wait</div>
                <div class="lab-stat-value">{avg_context*1000:.1f} ms</div>
                <div class="lab-stat-note">Context-aware</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="lab-stat">
                <div class="lab-stat-label">Throughput</div>
                <div class="lab-stat-value">{throughput_context/1e6:.2f} MB/s</div>
                <div class="lab-stat-note">Simulated service rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""
            <div class="lab-stat">
                <div class="lab-stat-label">Flows Evaluated</div>
                <div class="lab-stat-value">{len(sample):,}</div>
                <div class="lab-stat-note">Same set for both policies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Queue transformation
    st.markdown(
        '<div class="section-title">Queue Transformation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">The scheduler does not create traffic. It changes service order based on context.</div>',
        unsafe_allow_html=True,
    )

    def queue_html(frame, title, n=8):
        rows = frame.head(n)
        cards = []

        for _, r in rows.iterrows():
            lvl = str(r["priority_level"]).lower()
            cls = str(r["traffic_class"])
            short_cls = {
                "Video/Communication": "VIDEO",
                "Data Transfer": "DATA",
            }.get(cls, cls.upper())

            cards.append(
                textwrap.dedent(
                    f"""
                    <div class="queue-flow {lvl}">
                        <div class="queue-id">F-{int(r['flow_id']):03d}</div>
                        <div class="queue-class">{short_cls}</div>
                        <div class="queue-score">P {float(r['priority_score']):.3f}</div>
                    </div>
                    """
                ).strip()
            )

        return textwrap.dedent(
            f"""
            <div class="queue-title">
                {title}
                <span>FIRST {min(n, len(frame))} FLOWS</span>
            </div>
            <div class="queue">
                {'<div class="queue-arrow">→</div>'.join(cards)}
            </div>
            """
        ).strip()

    q1, q2 = st.columns(2)

    with q1:
        st.markdown(
            queue_html(fifo, "FIFO / arrival order"),
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            queue_html(context, "CONTEXT-AWARE / priority order"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance comparison
    st.markdown(
        '<div class="section-title">Scheduling Performance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Lower waiting time is better. Throughput is preserved by the current simulated service model.</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)

    with p1:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="FIFO",
                x=["High Priority", "Average", "Low Priority"],
                y=[high_fifo * 1000, avg_fifo * 1000, low_fifo * 1000],
                text=[f"{high_fifo*1000:.1f}", f"{avg_fifo*1000:.1f}", f"{low_fifo*1000:.1f}"],
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Context-Aware",
                x=["High Priority", "Average", "Low Priority"],
                y=[high_context * 1000, avg_context * 1000, low_context * 1000],
                text=[f"{high_context*1000:.1f}", f"{avg_context*1000:.1f}", f"{low_context*1000:.1f}"],
                textposition="outside",
            )
        )

        fig.update_layout(
            barmode="group",
            title="Waiting Time by Traffic Importance",
            yaxis_title="Waiting Time (ms)",
        )
        st.plotly_chart(chart_layout(fig, 390), width="stretch")

    with p2:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="FIFO",
                x=["Throughput"],
                y=[throughput_fifo / 1e6],
                text=[f"{throughput_fifo/1e6:.2f}"],
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Context-Aware",
                x=["Throughput"],
                y=[throughput_context / 1e6],
                text=[f"{throughput_context/1e6:.2f}"],
                textposition="outside",
            )
        )

        fig.update_layout(
            barmode="group",
            title="Service Throughput",
            yaxis_title="MB/s",
        )
        st.plotly_chart(chart_layout(fig, 390), width="stretch")

    # Technical explanation
    st.markdown(
        """
        <div class="insight">
            <b style="color:#dce9f7;">Scheduling policy:</b>
            flows are ordered by descending context-derived priority score.
            Gaming and Video/Communication receive higher urgency and delay-sensitivity
            weights, while Web and Data Transfer are less time-critical.
            <br><br>
            <b style="color:#dce9f7;">Interpretation:</b>
            a higher average or low-priority waiting time is not automatically a failure.
            The objective is to protect urgent traffic under congestion rather than
            minimize every flow's waiting time equally.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Compact metric matrix
    comparison_rows = []
    for metric in [
        "Average Waiting Time",
        "Maximum Waiting Time",
        "High Priority Waiting",
        "Low Priority Waiting",
        "Throughput",
    ]:
        fifo_val = comparison_value(comparison, metric, "FIFO")
        context_val = comparison_value(comparison, metric, "Context-Aware")

        if metric == "Throughput":
            fifo_display = f"{fifo_val/1e6:.2f} MB/s"
            context_display = f"{context_val/1e6:.2f} MB/s"
        else:
            fifo_display = f"{fifo_val*1000:.2f} ms"
            context_display = f"{context_val*1000:.2f} ms"

        comparison_rows.append(
            {
                "Metric": metric,
                "FIFO": fifo_display,
                "Context-Aware": context_display,
            }
        )

    st.dataframe(
        pd.DataFrame(comparison_rows),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        <div class="section-caption" style="margin-top:8px;">
            Prototype evaluation uses the same sampled flows and a simulated 1 MB/s service capacity.
            Results are comparative simulation outputs, not measurements from a production router.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# # FLOW INSPECTOR
# ============================================================
# ============================================================
# FLOW INSPECTOR
# ============================================================
elif page == "Flow Inspector":

    st.markdown(
        '<div class="section-title">Flow Decision Inspector</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Trace one network flow from extracted features → classification → '
        'context evaluation → priority → scheduling.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Use the globally prepared context-enriched sample.
    # Do NOT use flow_df here — flow_df only exists inside PCAP Analyzer.
    flow_options = sample["flow_id"].tolist()

    selected_id = st.selectbox(
        "Select Flow",
        flow_options,
        format_func=lambda x: f"Flow {x}",
    )

    flow = sample.loc[
        sample["flow_id"] == selected_id
    ].iloc[0]

    # Selected-flow values used by the inspector panels below.
    traffic_class = str(flow["traffic_class"])
    packet_count = float(flow["packet_count"])
    total_bytes = float(flow["total_bytes"])
    duration = float(flow["duration"])
    avg_packet_size = float(flow["average_packet_size"])
    packet_rate = float(flow["packet_rate"])
    byte_rate = float(flow["byte_rate"])
    average_iat = float(flow["average_iat"])
    priority_score = float(flow["priority_score"])
    priority_level = str(flow["priority_level"])
    selected_flow = int(flow["flow_id"]) - 1

    context_values = {
        "Gaming": (0.95, 0.90),
        "Video/Communication": (0.80, 0.85),
        "Web": (0.50, 0.45),
        "Data Transfer": (0.30, 0.25),
    }
    urgency, delay_sensitivity = context_values.get(
        traffic_class, (0.50, 0.50)
    )

    # --------------------------------------------------------
    # Flow summary
    # --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Traffic Class",
            flow["traffic_class"],
            "ML classification",
        )

    with c2:
        metric_card(
            "Priority Score",
            f"{flow['priority_score']:.3f}",
            "Context engine output",
        )

    with c3:
        metric_card(
            "Priority Level",
            flow["priority_level"],
            "Scheduling category",
        )

    with c4:
        metric_card(
            "Flow Size",
            f"{flow['total_bytes'] / 1000:.1f} KB",
            "Total bytes",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Features + decision trace
    # --------------------------------------------------------
    f1, f2 = st.columns(2)

    with f1:

        st.markdown("### Flow Features")

        feature_names = [
            "protocol",
            "packet_count",
            "total_bytes",
            "duration",
            "average_packet_size",
            "packet_rate",
            "byte_rate",
            "average_iat",
        ]

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": [flow[x] for x in feature_names],
        })

        st.dataframe(
            feature_df,
            width="stretch",
            hide_index=True,
        )

    with f2:
        st.markdown("### Flow Classification")
        st.markdown(
            f"""<div class=\"insight-card\"><div class=\"insight-label\">TRAFFIC CLASS</div>"""
            f"""<div class=\"insight-value\">{traffic_class}</div>"""
            f"""<div class=\"insight-text\">Selected flow classification from the traffic model.</div></div>""",
            unsafe_allow_html=True,
        )


    f3, f4 = st.columns(2)
    with f3:
        st.metric(
            "Duration",
            f"{duration:.4f} s",
        )

    with f4:
        st.metric(
            "Average Packet",
            f"{avg_packet_size:.2f} B",
        )

    f5, f6, f7, f8 = st.columns(4)

    with f5:
        st.metric(
            "Packet Rate",
            f"{packet_rate:,.1f} pkt/s",
        )

    with f6:
        st.metric(
            "Byte Rate",
            f"{byte_rate:,.2f} B/s",
        )

    with f7:
        st.metric(
            "Average IAT",
            f"{average_iat:.6f} s",
        )

    with f8:
        st.metric(
            "Total Bytes",
            format_bytes(total_bytes),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # MAIN DECISION COLUMNS
    # ---------------------------------------------------------
    left, right = st.columns([1, 1])

    with left:

        st.markdown(
            """
            <div class="section-title">Context Evaluation</div>
            <div class="section-subtitle">
                Context values used to calculate scheduling priority.
            </div>
            """,
            unsafe_allow_html=True,
        )

        context_data = pd.DataFrame(
            {
                "Context Factor": [
                    "Urgency",
                    "Delay Sensitivity",
                    "Network Load",
                    "Queue Condition",
                ],
                "Value": [
                    urgency,
                    delay_sensitivity,
                    network_load,
                    queue_condition,
                ],
            }
        )

        st.dataframe(
            context_data.style.format(
                {"Value": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">PRIORITY CALCULATION</div>
                <div class="insight-value">{priority_score:.3f}</div>
                <div class="insight-text">
                    0.35 × urgency +
                    0.25 × delay sensitivity +
                    0.20 × network load +
                    0.20 × queue condition
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="section-title">Scheduling Decision</div>
            <div class="section-subtitle">
                Final action produced by the context-aware scheduler.
            </div>
            """,
            unsafe_allow_html=True,
        )

        level_upper = str(priority_level).upper()

        if level_upper == "HIGH":
            decision = "SERVE FIRST"
            reason = (
                "This flow receives elevated priority because its "
                "traffic context indicates high scheduling urgency."
            )
        elif level_upper == "MEDIUM":
            decision = "NORMAL SERVICE"
            reason = (
                "This flow remains eligible for service while "
                "higher-priority traffic is handled first."
            )
        else:
            decision = "DEFER"
            reason = (
                "This flow can tolerate additional waiting compared "
                "with higher-priority traffic."
            )

        st.markdown(
            f"""
            <div class="decision-card">
                <div class="decision-kicker">SCHEDULER ACTION</div>
                <div class="decision-title">{decision}</div>
                <div class="decision-score">
                    Priority {priority_score:.3f}
                    · {level_upper}
                </div>
                <div class="decision-reason">
                    {reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">TRAFFIC CLASS</div>
                <div class="insight-value">{traffic_class}</div>
                <div class="insight-text">
                    Classified flow context is propagated into the
                    scheduling decision.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # DECISION TRACE
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">Decision Trace</div>
        <div class="section-subtitle">
            End-to-end reasoning path for the selected flow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use native Streamlit markdown for the trace so the HTML source is never
    # exposed as a code block in the UI.
    trace_steps = [
        ("01", "FLOW INPUT", f"{format_bytes(total_bytes)} · {packet_count:,.0f} packets"),
        ("02", "RANDOM FOREST", f"Traffic class → {traffic_class}"),
        ("03", "CONTEXT ENGINE", f"Urgency {urgency:.2f} · Delay Sensitivity {delay_sensitivity:.2f}"),
        ("04", "PRIORITY SCORE", f"{priority_score:.3f} → {level_upper}"),
        ("05", "SCHEDULER DECISION", decision),
    ]

    for i, (number, title, value) in enumerate(trace_steps):
        if i > 0:
            st.markdown("<div class='trace-arrow'>↓</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class=\"trace-step\">
                <div class=\"trace-number\">{number}</div>
                <div>
                    <div class=\"trace-title\">{title}</div>
                    <div class=\"trace-value\">{value}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # EXPLANATION
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        f"Flow #{selected_flow + 1:03d} is classified as "
        f"**{traffic_class}**, producing a priority score of "
        f"**{priority_score:.3f} ({level_upper})** under the current "
        f"network load and queue conditions."
    )

# ============================================================
# PCAP ANALYZER
# ============================================================
elif page == "PCAP Analyzer":

    st.markdown(
        '<div class="section-title">PCAP Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Upload a real packet capture and trace it through the same flow extraction → Random Forest → context → priority pipeline.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload PCAP / PCAPNG",
        type=["pcap", "pcapng"],
        help="Use a packet capture containing IP traffic. NetSense will aggregate packets into bidirectional flows.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="insight">
                <b>Live analysis pipeline</b><br>
                PCAP upload → packet parsing → bidirectional flow extraction →
                Random Forest classification → context-aware priority.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("### 01 · Extract")
            st.caption("Convert raw packets into flow-level behavioural features.")
        with p2:
            st.markdown("### 02 · Classify")
            st.caption("Predict Web, Video/Communication, Gaming, or Data Transfer.")
        with p3:
            st.markdown("### 03 · Prioritize")
            st.caption("Combine traffic context with current network conditions.")

    else:
        if model is None:
            st.error("Trained model not found in models/traffic_classifier.pkl.")
            st.stop()

        with st.spinner("Parsing packets → building bidirectional flows → classifying traffic..."):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name

            try:
                packet_df = extract_packet_features(temp_path)
                flow_df = create_flow_features(packet_df)

                feature_cols = model_features or [
                    "protocol",
                    "packet_count",
                    "total_bytes",
                    "duration",
                    "average_packet_size",
                    "packet_rate",
                    "byte_rate",
                    "average_iat",
                ]

                X_new = flow_df[feature_cols].fillna(0)
                predictions = model.predict(X_new)

                result = flow_df.copy()
                result["predicted_class"] = predictions
                result["flow_id"] = np.arange(1, len(result) + 1)

                scores = []
                levels = []
                for cls in result["predicted_class"]:
                    priority_result = calculate_priority(
                        cls,
                        network_load,
                        queue_condition,
                    )

                    # Context engine versions may return either a dictionary
                    # or a (score, level) tuple. Normalize both forms here.
                    if isinstance(priority_result, dict):
                        score = priority_result.get(
                            "priority_score",
                            priority_result.get("score", 0.0),
                        )
                        level = priority_result.get(
                            "priority_level",
                            priority_result.get("level", "LOW"),
                        )
                    elif isinstance(priority_result, (tuple, list)):
                        score = priority_result[0]
                        level = priority_result[1]
                    else:
                        score = float(priority_result)
                        level = (
                            "HIGH" if score >= 0.70
                            else "MEDIUM" if score >= 0.40
                            else "LOW"
                        )

                    scores.append(float(score))
                    levels.append(str(level))

                result["priority_score"] = scores
                result["priority_level"] = levels

            except Exception as exc:
                st.error(f"PCAP processing failed: {exc}")
                st.stop()

        st.success(
            f"Analysis complete · {len(packet_df):,} packets · {len(result):,} flows"
        )

        # --------------------------------------------------------
        # Analysis summary
        # --------------------------------------------------------
        high_count = int((result["priority_level"] == "HIGH").sum())
        medium_count = int((result["priority_level"] == "MEDIUM").sum())
        low_count = int((result["priority_level"] == "LOW").sum())
        avg_priority = float(result["priority_score"].mean()) if len(result) else 0.0
        total_bytes = float(result["total_bytes"].sum()) if len(result) else 0.0

        a, b, c, d = st.columns(4)
        with a:
            metric_card("Packets", f"{len(packet_df):,}", uploaded.name)
        with b:
            metric_card("Flows", f"{len(result):,}", "Bidirectional extraction")
        with c:
            metric_card("High Priority", f"{high_count:,}", "Current context state")
        with d:
            metric_card("Avg Priority", f"{avg_priority:.3f}", "Across extracted flows")

        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # Visual intelligence
        # --------------------------------------------------------
        left, right = st.columns(2)

        with left:
            counts = result["predicted_class"].value_counts().reset_index()
            counts.columns = ["traffic_class", "flows"]

            fig = px.pie(
                counts,
                names="traffic_class",
                values="flows",
                hole=0.60,
                title="Traffic Classification",
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(chart_layout(fig, 390), width="stretch")

        with right:
            priority_counts = pd.DataFrame({
                "Priority": ["HIGH", "MEDIUM", "LOW"],
                "Flows": [high_count, medium_count, low_count],
            })

            fig2 = px.bar(
                priority_counts,
                x="Priority",
                y="Flows",
                text="Flows",
                title="Scheduling Priority Distribution",
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig2, 390), width="stretch")

        # --------------------------------------------------------
        # Context state
        # --------------------------------------------------------
        st.markdown("### Context State")
        st.caption("The same network load and queue conditions are applied to every extracted flow.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Network Load", f"{network_load:.2f}")
        with c2:
            st.metric("Queue Condition", f"{queue_condition:.2f}")
        with c3:
            st.metric("Captured Data", format_bytes(total_bytes))
        with c4:
            st.metric("Priority Mean", f"{avg_priority:.3f}")

        # --------------------------------------------------------
        # Results explorer
        # --------------------------------------------------------
        st.markdown("### Flow Results")
        st.caption("Filter the extracted flows to inspect the decisions produced by NetSense.")

        f1, f2 = st.columns(2)
        with f1:
            class_options = ["All"] + sorted(result["predicted_class"].dropna().unique().tolist())
            selected_class = st.selectbox("Traffic Class", class_options)
        with f2:
            priority_options = ["All", "HIGH", "MEDIUM", "LOW"]
            selected_priority = st.selectbox("Priority Level", priority_options)

        filtered = result.copy()
        if selected_class != "All":
            filtered = filtered[filtered["predicted_class"] == selected_class]
        if selected_priority != "All":
            filtered = filtered[filtered["priority_level"] == selected_priority]

        st.caption(f"Showing {len(filtered):,} of {len(result):,} extracted flows")

        show_cols = [
            "flow_id",
            "predicted_class",
            "packet_count",
            "total_bytes",
            "duration",
            "packet_rate",
            "byte_rate",
            "priority_score",
            "priority_level",
        ]

        st.dataframe(
            filtered[show_cols].round(4),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Analysis CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="netsense_pcap_analysis.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight">
                <b>Analysis interpretation:</b>
                The uploaded capture produced <b>{len(result):,} flows</b>.
                The classifier identified the traffic behaviour of each flow, while the
                context engine assigned priority using load <b>{network_load:.2f}</b>
                and queue condition <b>{queue_condition:.2f}</b>.
                High-priority flows: <b>{high_count:,}</b> ·
                Medium: <b>{medium_count:,}</b> ·
                Low: <b>{low_count:,}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABOUT
# ============================================================
elif page == "About":

    st.markdown(
        '<div class="section-title">About NetSense</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Multi-Modal Traffic Classification Pipeline as a Front End for Context-Aware Scheduling

        **NetSense** demonstrates how network traffic can be transformed from raw packet
        observations into context-aware scheduling decisions.

        #### System pipeline

        `PCAP / Traffic → Flow Features → Random Forest → Context Engine → Priority → Scheduler`

        #### Traffic classes

        - Web
        - Video / Communication
        - Gaming
        - Data Transfer

        #### Context model

        The prototype combines:

        - Traffic urgency
        - Delay sensitivity
        - Network load
        - Queue condition

        into a normalized priority score.

        #### Scheduler

        The proposed scheduler orders flows by context-derived priority and is evaluated
        against a FIFO baseline.

        #### Current evaluation scope

        The current prototype is trained and evaluated using the selected four-PCAP dataset:

        `Gmail.pcap · Skype.pcap · WorldOfWarcraft.pcap · BitTorrent.pcap`

        The reported ML accuracy is the measured result for the current flow-level
        train/test experiment and should not be interpreted as universal network
        classification accuracy.
        """,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="insight">
            <b>Engineering principle:</b>
            NetSense separates <b>what the traffic is</b> from
            <b>what the network should do about it</b>.
            The classifier identifies traffic behaviour, while the context engine
            converts network conditions into scheduling priorities.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        NetSense · Intelligent Network Management · Prototype Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
