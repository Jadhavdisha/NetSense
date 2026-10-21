
import tempfile
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(0, 212, 255, 0.08), transparent 25%),
                radial-gradient(circle at 90% 20%, rgba(99, 102, 241, 0.08), transparent 25%),
                #070b14;
        }

        [data-testid="stSidebar"] {
            background: #090e19;
            border-right: 1px solid rgba(255,255,255,0.07);
        }

        [data-testid="stSidebar"] * {
            color: #d9e2f0;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        .brand {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -1px;
            color: #f4f8ff;
            margin-bottom: 0;
        }

        .brand span {
            color: #38bdf8;
        }

        .subtitle {
            color: #8290a8;
            font-size: 0.88rem;
            margin-top: -3px;
            margin-bottom: 1.2rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #eef5ff;
            margin: 0.8rem 0 0.7rem 0;
        }

        .section-caption {
            color: #7e8ba2;
            font-size: 0.82rem;
            margin-top: -0.45rem;
            margin-bottom: 0.8rem;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(17,26,43,0.96), rgba(10,16,28,0.96));
            border: 1px solid rgba(148,163,184,0.13);
            border-radius: 15px;
            padding: 17px 18px;
            min-height: 115px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }

        .metric-label {
            color: #8492a8;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }

        .metric-value {
            color: #f5f9ff;
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 8px;
        }

        .metric-note {
            color: #64748b;
            font-size: 0.72rem;
            margin-top: 4px;
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            background: rgba(34,197,94,0.09);
            color: #4ade80;
            border: 1px solid rgba(34,197,94,0.2);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #4ade80;
            box-shadow: 0 0 10px #4ade80;
        }

        .pipeline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 20px;
            background: rgba(10,16,28,0.75);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 15px;
            overflow-x: auto;
        }

        .pipe-node {
            min-width: 145px;
            text-align: center;
            padding: 15px 10px;
            border-radius: 12px;
            background: #101827;
            border: 1px solid rgba(56,189,248,0.16);
        }

        .pipe-icon {
            font-size: 1.45rem;
            margin-bottom: 5px;
        }

        .pipe-title {
            color: #edf5ff;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .pipe-sub {
            color: #718096;
            font-size: 0.66rem;
            margin-top: 3px;
        }

        .arrow {
            color: #38bdf8;
            font-size: 1.35rem;
            font-weight: 800;
        }

        .insight {
            padding: 14px 16px;
            border-radius: 12px;
            background: rgba(56,189,248,0.06);
            border: 1px solid rgba(56,189,248,0.15);
            color: #b9c9dc;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        .high {
            color: #fb7185;
            font-weight: 800;
        }

        .medium {
            color: #fbbf24;
            font-weight: 800;
        }

        .low {
            color: #4ade80;
            font-weight: 800;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .footer {
            text-align: center;
            color: #536176;
            font-size: 0.72rem;
            padding: 22px 0 5px 0;
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

        # Handle dictionary returned by context_engine.py
        if isinstance(result, dict):

            # Try the expected key names
            score = result.get("priority_score", result.get("score"))
            level = result.get("priority_level", result.get("level"))

            # If the exact names differ, detect them automatically
            if score is None:
                for key, value in result.items():
                    if "score" in str(key).lower():
                        score = value
                        break

            if level is None:
                for key, value in result.items():
                    if "level" in str(key).lower() or "priority" in str(key).lower():
                        if isinstance(value, str):
                            level = value
                            break

            if score is None or level is None:
                raise ValueError(
                    f"Unexpected calculate_priority result: {result}"
                )

        # Handle tuple/list return values as a fallback
        elif isinstance(result, (tuple, list)):
            score = result[0]
            level = result[1]

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


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    st.markdown(
        '<div class="section-title">Network Operations Overview</div>',
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
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

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
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

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

    l, r = st.columns(2)

    with l:
        fig = px.box(
            sample,
            x="traffic_class",
            y="packet_count",
            title="Packet Count by Traffic Class",
        )
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(chart_layout(fig), use_container_width=True)

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
        st.plotly_chart(chart_layout(fig), use_container_width=True)

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
        use_container_width=True,
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
        st.plotly_chart(gauge("Network Load", network_load), use_container_width=True)

    with g2:
        st.plotly_chart(gauge("Queue Condition", queue_condition), use_container_width=True)

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
    st.plotly_chart(chart_layout(fig), use_container_width=True)

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
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SCHEDULER LAB
# ============================================================
elif page == "Scheduler Lab":

    st.markdown(
        '<div class="section-title">Context-Aware Scheduler Lab</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Compare conventional FIFO ordering against priority-aware scheduling on the same flows.</div>',
        unsafe_allow_html=True,
    )

    fifo, context, comparison = scheduler_metrics(sample)

    # Metrics returned as dictionaries in the current scheduler implementation.
    if isinstance(comparison, dict):
        metrics_df = pd.DataFrame(comparison).T.reset_index()
        metrics_df = metrics_df.rename(columns={"index": "Metric"})
    else:
        metrics_df = pd.DataFrame()

    if not metrics_df.empty:
        st.dataframe(
            metrics_df.round(5),
            use_container_width=True,
            hide_index=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        high_fifo = (
            comparison["High Priority Waiting"]
            if isinstance(comparison, dict)
            else None
        )

        st.markdown(
            '<div class="section-title">High-Priority Waiting Time</div>',
            unsafe_allow_html=True,
        )

        if isinstance(comparison, dict):
            values = [
                comparison["High Priority Waiting"]["FIFO"],
                comparison["High Priority Waiting"]["Context-Aware"],
            ]
            labels = ["FIFO", "Context-Aware"]

            fig = px.bar(
                x=labels,
                y=values,
                text=[f"{v:.4f}" for v in values],
                labels={"x": "", "y": "Waiting Time"},
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(chart_layout(fig, 320), use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-title">Average Waiting Time</div>',
            unsafe_allow_html=True,
        )

        if isinstance(comparison, dict):
            values = [
                comparison["Average Waiting Time"]["FIFO"],
                comparison["Average Waiting Time"]["Context-Aware"],
            ]

            fig = px.bar(
                x=["FIFO", "Context-Aware"],
                y=values,
                text=[f"{v:.4f}" for v in values],
                labels={"x": "", "y": "Waiting Time"},
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(chart_layout(fig, 320), use_container_width=True)

    st.markdown(
        """
        <div class="insight">
            The proposed scheduler is <b>not designed to minimize every flow's waiting time</b>.
            Its purpose is to move urgent and delay-sensitive traffic ahead of less critical traffic.
            Therefore, low-priority flows may wait longer while high-priority waiting time is reduced.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Scheduling Order</div>',
        unsafe_allow_html=True,
    )

    order_col1, order_col2 = st.columns(2)

    with order_col1:
        st.markdown("**FIFO — baseline**")
        fifo_show = fifo[
            ["flow_id", "traffic_class", "priority_score", "total_bytes"]
        ].head(15)
        st.dataframe(
            fifo_show.round(3),
            use_container_width=True,
            hide_index=True,
        )

    with order_col2:
        st.markdown("**Context-Aware — proposed**")
        context_show = context[
            ["flow_id", "traffic_class", "priority_score", "total_bytes"]
        ].head(15)
        st.dataframe(
            context_show.round(3),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FLOW INSPECTOR
# ============================================================
elif page == "Flow Inspector":

    st.markdown(
        '<div class="section-title">Flow Decision Inspector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Trace one flow through the complete NetSense decision pipeline.</div>',
        unsafe_allow_html=True,
    )

    selected_id = st.selectbox(
        "Select Flow",
        sample["flow_id"].tolist(),
    )

    flow = sample.loc[sample["flow_id"] == selected_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Traffic Class", flow["traffic_class"], "Classification result")
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

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("### Flow Features")
        feature_names = [
            "protocol",
            "packet_count",
            "duration",
            "average_packet_size",
            "packet_rate",
            "byte_rate",
            "average_iat",
        ]

        feature_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Value": [flow[x] for x in feature_names],
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
        )

    with f2:
        st.markdown("### Decision Trace")

        st.markdown(
            f"""
            <div class="insight">
                <b>01 — Traffic:</b> {flow['traffic_class']}<br><br>
                <b>02 — Network Load:</b> {network_load:.2f}<br><br>
                <b>03 — Queue Condition:</b> {queue_condition:.2f}<br><br>
                <b>04 — Priority Score:</b> {flow['priority_score']:.3f}<br><br>
                <b>05 — Scheduling Level:</b>
                <span class="{flow['priority_level'].lower()}">
                    {flow['priority_level']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
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
        '<div class="section-caption">Upload a PCAP and pass it through the same feature extraction and ML pipeline.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload PCAP / PCAPNG",
        type=["pcap", "pcapng"],
    )

    if uploaded is None:
        st.info(
            "Upload a PCAP file to run flow extraction and traffic classification. "
            "The current trained model supports the four classes in the selected training dataset."
        )
    else:
        if model is None:
            st.error("Trained model not found in models/traffic_classifier.pkl.")
            st.stop()

        with st.spinner("Extracting packets and building bidirectional flows..."):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:
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
                    score = priority_result[0]
                    level = priority_result[1]

                scores.append(score)
                levels.append(level)

            except Exception as exc:
                st.error(f"PCAP processing failed: {exc}")
                st.stop()

        st.success(
            f"Processed {len(packet_df):,} packets into {len(result):,} flows."
        )

        a, b, c = st.columns(3)

        with a:
            metric_card(
                "Packets",
                f"{len(packet_df):,}",
                uploaded.name,
            )
        with b:
            metric_card(
                "Flows",
                f"{len(result):,}",
                "Bidirectional flow extraction",
            )
        with c:
            metric_card(
                "High Priority",
                f"{(result['priority_level'] == 'HIGH').sum():,}",
                "Current context state",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        counts = result["predicted_class"].value_counts().reset_index()
        counts.columns = ["traffic_class", "flows"]

        fig = px.pie(
            counts,
            names="traffic_class",
            values="flows",
            hole=0.58,
            title="Predicted Traffic Classes",
        )
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

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
            result[show_cols].round(4),
            use_container_width=True,
            hide_index=True,
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
