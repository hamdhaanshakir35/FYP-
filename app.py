# ============================================================
#  NBA PLAYER CLUSTERING DASHBOARD — STREAMLIT APP
#  2019-2020 Season | Play-Type Analysis
#  Run with: streamlit run streamlit_app.py
# ============================================================
#
#  REQUIREMENTS:
#    pip install streamlit pandas numpy plotly scikit-learn
#
#  USAGE:
#    1. Make sure final_clustered_output.csv is in the same folder.
#    2. Run:  streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Player Clustering | 2019-20",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean Professional Theme
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── General App ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #1a1d2e;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid #2d3155;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #6366f1;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
    margin: 18px 0 12px 0;
}

/* ── Insight Cards ── */
.insight-card {
    background: #1e2235;
    border: 1px solid #2d3155;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.insight-card h4 {
    color: #a5b4fc;
    margin: 0 0 6px 0;
    font-size: 0.9rem;
}
.insight-card p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.82rem;
    line-height: 1.4;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1d2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: white !important;
}

/* ── Metric Overrides ── */
[data-testid="metric-container"] {
    background: #1e2235;
    border: 1px solid #2d3155;
    border-radius: 10px;
    padding: 12px;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6366f1;
    margin: 16px 0 6px 0;
}

/* ── Table styling ── */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PLOTLY THEME  (shared across all charts)
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#141824",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    title_font=dict(size=15, color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e2235", zeroline=False, linecolor="#2d3155"),
    yaxis=dict(gridcolor="#1e2235", zeroline=False, linecolor="#2d3155"),
    legend=dict(bgcolor="#1e2235", bordercolor="#2d3155", borderwidth=1),
    margin=dict(l=50, r=30, t=55, b=45),
)

CLUSTER_PALETTE = [
    "#6366f1", "#10b981", "#f59e0b", "#ef4444",
    "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4",
    "#84cc16", "#f97316",
]


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # Use pathlib so Windows backslash paths never cause escape-sequence bugs
    from pathlib import Path
    csv_path = Path(__file__).parent / "final_clustered_output.csv"
    df = pd.read_csv('final_clustered_output.csv')
    
    # Ensure cluster columns exist and are int / string
    for col in ["cluster_kmeans", "cluster_dbscan", "cluster_kmodes"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


try:
    df = load_data()
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False


# ──────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ──────────────────────────────────────────────────────────────

FREQ_FEATURES = [
    "iso_freq", "tra_freq", "prh_freq", "prr_freq",
    "postup_freq", "spotup_freq", "handoff_freq", "cut_freq",
]
PPP_FEATURES = [
    "iso_pts", "tra_pts", "prh_pts", "prr_pts",
    "postup_pts", "spotup_pts", "handoff_pts", "cut_pts",
]
ENGINEERED = [
    "on_ball_creation", "off_ball_finishing",
    "transition_impact", "iso_impact", "prh_impact",
    "spotup_impact", "overall_efficiency", "play_diversity",
]

RADAR_FEATURES = [
    "iso_freq", "tra_freq", "prh_freq", "spotup_freq", "cut_freq",
    "on_ball_creation", "off_ball_finishing", "overall_efficiency",
]
RADAR_LABELS = [
    "Isolation", "Transition", "P&R Handler", "Spot-Up",
    "Cut/Finish", "On-Ball\nCreation", "Off-Ball\nFinish", "Efficiency",
]

MODEL_CLUSTER_COL = {
    "KMeans": "cluster_kmeans",
    "DBSCAN": "cluster_dbscan",
    "KModes": "cluster_kmodes",
}

PLAY_TYPE_LABELS = {
    "iso_freq"    : "Isolation",
    "tra_freq"    : "Transition",
    "prh_freq"    : "P&R Handler",
    "prr_freq"    : "P&R Rollman",
    "postup_freq" : "Post-Up",
    "spotup_freq" : "Spot-Up",
    "handoff_freq": "Hand-Off",
    "cut_freq"    : "Cut",
}


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_cluster_col(model_name):
    return MODEL_CLUSTER_COL.get(model_name, "cluster_kmeans")


def compute_metrics(df_in, cluster_col):
    """Return silhouette and davies-bouldin metrics."""
    num_cols = FREQ_FEATURES + PPP_FEATURES
    available = [c for c in num_cols if c in df_in.columns]
    labels = df_in[cluster_col].astype(int)
    valid_mask = labels != -1
    X = df_in.loc[valid_mask, available].values
    y = labels[valid_mask].values
    n_clusters = len(set(y))
    if n_clusters < 2 or len(X) < 10:
        return None, None
    try:
        sil = silhouette_score(X, y)
        dbi = davies_bouldin_score(X, y)
        return round(sil, 4), round(dbi, 4)
    except Exception:
        return None, None


def make_pca_scatter(df_in, cluster_col, title=""):
    """Return a Plotly scatter figure for PCA space."""
    fig = px.scatter(
        df_in,
        x="pca1",
        y="pca2",
        color=cluster_col,
        hover_name="player",
        hover_data={
            "team": True,
            "pca1": False,
            "pca2": False,
            cluster_col: True,
        },
        color_discrete_sequence=CLUSTER_PALETTE,
        title=title or f"PCA Projection — {cluster_col}",
        labels={"pca1": "Principal Component 1", "pca2": "Principal Component 2"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="#0f1117")))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def make_radar(df_in, players):
    """Return a Plotly radar chart comparing selected players."""
    available = [f for f in RADAR_FEATURES if f in df_in.columns]
    labels_avail = [RADAR_LABELS[RADAR_FEATURES.index(f)] for f in available]

    fig = go.Figure()
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#3b82f6"]

    for i, player in enumerate(players):
        row = df_in[df_in["player"] == player]
        if row.empty:
            continue
        vals = row[available].values[0].tolist()
        # Normalise to [0,1] for radar
        maxes = df_in[available].max().values
        vals_norm = [v / m if m > 0 else 0 for v, m in zip(vals, maxes)]
        vals_norm.append(vals_norm[0])  # close the polygon
        theta = labels_avail + [labels_avail[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_norm,
            theta=theta,
            name=player,
            fill="toself",
            line=dict(color=palette[i % len(palette)], width=2),
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#141824",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#2d3155",
                color="#64748b",
            ),
            angularaxis=dict(gridcolor="#2d3155", color="#94a3b8"),
        ),
        showlegend=True,
        title="Player Style Radar Comparison (Normalised)",
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def cluster_bar_chart(df_in, cluster_col, feature_cols, title):
    """Bar chart: mean feature values per cluster."""
    df_valid = df_in[df_in[cluster_col] != "-1"].copy()
    means = df_valid.groupby(cluster_col)[feature_cols].mean().reset_index()
    means_melt = means.melt(id_vars=cluster_col, var_name="Feature", value_name="Mean")

    fig = px.bar(
        means_melt,
        x="Feature",
        y="Mean",
        color=cluster_col,
        barmode="group",
        title=title,
        color_discrete_sequence=CLUSTER_PALETTE,
        labels={cluster_col: "Cluster"},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(tickangle=-35)
    return fig


# ──────────────────────────────────────────────────────────────
# ERROR STATE — CSV not found
# ──────────────────────────────────────────────────────────────

if not DATA_LOADED:
    st.error("❌ **final_clustered_output.csv not found.**")
    st.markdown("""
    **Steps to fix:**
    1. Run the Google Colab notebook (`nba_clustering_colab.py`) to generate `final_clustered_output.csv`.
    2. Place the CSV file in the **same directory** as `streamlit_app.py`.
    3. Refresh this page.
    """)
    st.stop()


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 18px 0;'>
        <span style='font-size:2.5rem;'>🏀</span><br>
        <span style='font-size:1.1rem; font-weight:800; color:#e2e8f0;'>NBA Clustering</span><br>
        <span style='font-size:0.7rem; color:#64748b; letter-spacing:1px;'>2019 – 2020 SEASON</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model Selector ──
    st.markdown("<div class='sidebar-section'>📊 Clustering Model</div>", unsafe_allow_html=True)
    selected_model = st.radio(
        "Select model",
        options=["KMeans", "DBSCAN", "KModes"],
        label_visibility="collapsed",
    )
    cluster_col = get_cluster_col(selected_model)

    st.divider()

    # ── Team Filter ──
    st.markdown("<div class='sidebar-section'>🏟️ Team Filter</div>", unsafe_allow_html=True)
    all_teams = sorted(df["team"].dropna().unique().tolist())
    selected_teams = st.multiselect(
        "Select team(s)",
        options=all_teams,
        default=all_teams,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Cluster Filter ──
    st.markdown("<div class='sidebar-section'>🎯 Cluster Filter</div>", unsafe_allow_html=True)
    all_clusters = sorted(df[cluster_col].unique().tolist(), key=lambda x: int(x))
    selected_clusters = st.multiselect(
        "Select cluster(s)",
        options=all_clusters,
        default=[c for c in all_clusters if c != "-1"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Only real players toggle ──
    real_only = st.toggle("Show original players only", value=True,
                          help="Hides augmented (Gaussian noise + synthetic) players.")

    st.divider()
    st.caption("Final Year Project · NBA Clustering · 2019-20")


# ──────────────────────────────────────────────────────────────
# FILTER DATA
# ──────────────────────────────────────────────────────────────

df_filtered = df[
    df["team"].isin(selected_teams) &
    df[cluster_col].isin(selected_clusters)
].copy()

if real_only:
    df_filtered = df_filtered[
        ~df_filtered["player"].str.contains("_aug|Synth", na=False)
    ]


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div style='margin-bottom:18px;'>
    <h1 style='margin:0; color:#e2e8f0; font-size:2rem; font-weight:800;'>
        🏀 NBA Player Clustering Dashboard
    </h1>
    <p style='margin:4px 0 0 0; color:#64748b; font-size:0.9rem;'>
        2019–2020 Season · Play-Type Unsupervised Learning · Final Year Project
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────

sil_score, dbi_score = compute_metrics(df_filtered, cluster_col)
n_players   = len(df_filtered)
n_clusters  = df_filtered[cluster_col][df_filtered[cluster_col] != "-1"].nunique()
n_teams     = df_filtered["team"].nunique()
avg_eff     = df_filtered["overall_efficiency"].mean() if "overall_efficiency" in df_filtered.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)

for col, val, label, sub in [
    (col1, n_players,      "Players",          "in filtered view"),
    (col2, n_clusters,     "Clusters",          selected_model),
    (col3, n_teams,        "Teams",             "represented"),
    (col4, f"{sil_score:.3f}" if sil_score else "—", "Silhouette", "Higher = better"),
    (col5, f"{dbi_score:.3f}" if dbi_score else "—", "Davies-Bouldin", "Lower = better"),
]:
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────

tab_overview, tab_insights, tab_comparison, tab_players = st.tabs([
    "📈 Overview",
    "🔍 Cluster Insights",
    "⚖️ Model Comparison",
    "👤 Player Analysis",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

with tab_overview:

    st.markdown("<div class='section-header'>PCA Cluster Projection</div>", unsafe_allow_html=True)

    col_pca, col_dist = st.columns([2, 1])

    with col_pca:
        if "pca1" in df_filtered.columns and "pca2" in df_filtered.columns:
            pca_fig = make_pca_scatter(
                df_filtered, cluster_col,
                title=f"{selected_model} Clusters — PCA 2-D Space"
            )
            st.plotly_chart(pca_fig, use_container_width=True)
        else:
            st.warning("PCA columns (pca1, pca2) not found in dataset.")

    with col_dist:
        # Cluster distribution donut
        cluster_counts = df_filtered[df_filtered[cluster_col] != "-1"][cluster_col].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        donut = px.pie(
            cluster_counts,
            names="Cluster",
            values="Count",
            hole=0.55,
            title="Cluster Distribution",
            color_discrete_sequence=CLUSTER_PALETTE,
        )
        donut.update_traces(textinfo="label+percent", pull=0.02)
        donut.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")})
        st.plotly_chart(donut, use_container_width=True)

        # Small table of cluster sizes
        st.dataframe(
            cluster_counts.set_index("Cluster").rename(columns={"Count": "Players"}),
            use_container_width=True,
        )

    # ── Play-type frequency heatmap ──
    st.markdown("<div class='section-header'>Play-Type Frequency by Cluster</div>", unsafe_allow_html=True)

    available_freq = [f for f in FREQ_FEATURES if f in df_filtered.columns]
    if available_freq and n_clusters >= 1:
        df_valid = df_filtered[df_filtered[cluster_col] != "-1"]
        heatmap_data = df_valid.groupby(cluster_col)[available_freq].mean().round(3)
        heatmap_data.columns = [PLAY_TYPE_LABELS.get(c, c) for c in heatmap_data.columns]

        heat_fig = px.imshow(
            heatmap_data,
            text_auto=".3f",
            color_continuous_scale="Viridis",
            aspect="auto",
            title=f"Mean Play-Type Frequency per Cluster ({selected_model})",
            labels={"x": "Play Type", "y": "Cluster", "color": "Freq"},
        )
        heat_fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")})
        heat_fig.update_coloraxes(colorbar_bgcolor="#1e2235", colorbar_tickcolor="#94a3b8")
        st.plotly_chart(heat_fig, use_container_width=True)

    # ── PPP efficiency heatmap ──
    st.markdown("<div class='section-header'>Points-Per-Possession Efficiency by Cluster</div>", unsafe_allow_html=True)

    available_ppp = [f for f in PPP_FEATURES if f in df_filtered.columns]
    if available_ppp and n_clusters >= 1:
        df_valid = df_filtered[df_filtered[cluster_col] != "-1"]
        ppp_heatmap = df_valid.groupby(cluster_col)[available_ppp].mean().round(3)
        ppp_heatmap.columns = [PLAY_TYPE_LABELS.get(c.replace("_pts","_freq"), c) for c in ppp_heatmap.columns]

        ppp_fig = px.imshow(
            ppp_heatmap,
            text_auto=".3f",
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title=f"Mean PPP per Cluster ({selected_model})",
            labels={"x": "Play Type", "y": "Cluster", "color": "PPP"},
        )
        ppp_fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")})
        st.plotly_chart(ppp_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER INSIGHTS
# ══════════════════════════════════════════════════════════════

with tab_insights:

    st.markdown("<div class='section-header'>Engineered Feature Analysis per Cluster</div>", unsafe_allow_html=True)

    available_eng = [f for f in ENGINEERED if f in df_filtered.columns]

    if available_eng:
        bar_fig = cluster_bar_chart(
            df_filtered, cluster_col,
            available_eng,
            f"Engineered Feature Averages by Cluster ({selected_model})",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ── Cluster summary table ──
    st.markdown("<div class='section-header'>Cluster Summary Statistics</div>", unsafe_allow_html=True)

    df_valid = df_filtered[df_filtered[cluster_col] != "-1"]
    if len(df_valid) > 0:
        summary_cols = available_freq + available_eng
        cluster_summary = (
            df_valid.groupby(cluster_col)[summary_cols]
            .mean()
            .round(3)
        )

        # Rename columns for readability
        col_rename = {
            "iso_freq": "ISO Freq", "tra_freq": "Trans Freq",
            "prh_freq": "PRH Freq", "spotup_freq": "SpotUp Freq",
            "cut_freq": "Cut Freq", "on_ball_creation": "OnBall",
            "off_ball_finishing": "OffBall", "transition_impact": "Trans Imp",
            "iso_impact": "ISO Imp", "overall_efficiency": "Efficiency",
            "play_diversity": "Diversity",
        }
        cluster_summary = cluster_summary.rename(columns=col_rename)
        st.dataframe(
            cluster_summary.style.background_gradient(cmap="Blues", axis=0),
            use_container_width=True,
        )

    # ── Parallel coordinates ──
    st.markdown("<div class='section-header'>Parallel Coordinates — Multi-Feature View</div>", unsafe_allow_html=True)

    para_features = [f for f in ["on_ball_creation", "off_ball_finishing",
                                   "transition_impact", "iso_impact",
                                   "overall_efficiency", "play_diversity"]
                      if f in df_filtered.columns]

    if para_features and len(df_valid) > 0:
        df_para = df_valid[para_features + [cluster_col]].copy()
        df_para[cluster_col] = df_para[cluster_col].astype(int)

        para_fig = px.parallel_coordinates(
            df_para,
            color=cluster_col,
            dimensions=para_features,
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f"Parallel Coordinates — Cluster Profiles ({selected_model})",
        )
        para_fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                                  if k not in ("xaxis", "yaxis")})
        st.plotly_chart(para_fig, use_container_width=True)

    # ── Top players per cluster ──
    st.markdown("<div class='section-header'>Top Players per Cluster (by Overall Efficiency)</div>", unsafe_allow_html=True)

    if "overall_efficiency" in df_filtered.columns:
        real_df = df_filtered[~df_filtered["player"].str.contains("_aug|Synth", na=False)]
        for cluster_id in sorted(real_df[cluster_col].unique(), key=lambda x: int(x) if x != "-1" else 999):
            if cluster_id == "-1":
                continue
            cluster_players = (
                real_df[real_df[cluster_col] == cluster_id]
                .sort_values("overall_efficiency", ascending=False)
                [["player", "team", "overall_efficiency", "on_ball_creation", "off_ball_finishing"]]
                .head(6)
                .reset_index(drop=True)
            )
            with st.expander(f"Cluster {cluster_id} — Top Players", expanded=(cluster_id == "0")):
                st.dataframe(cluster_players, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════

with tab_comparison:

    st.markdown("<div class='section-header'>All-Model Evaluation Metrics</div>", unsafe_allow_html=True)

    # Compute metrics for all three models on the full (unfiltered) dataset
    all_metrics = []
    for model_name, col_name in MODEL_CLUSTER_COL.items():
        if col_name not in df.columns:
            continue
        df_eval = df.copy()
        df_eval[col_name] = df_eval[col_name].astype(str)
        real_only_eval = df_eval[~df_eval["player"].str.contains("_aug|Synth", na=False)]
        real_only_eval = real_only_eval.copy()
        real_only_eval[col_name] = real_only_eval[col_name].astype(int)
        sil, dbi = compute_metrics(real_only_eval, col_name)
        n_cl = real_only_eval[col_name][real_only_eval[col_name] != -1].nunique()
        all_metrics.append({
            "Model": model_name,
            "N Clusters": n_cl,
            "Silhouette ↑": round(sil, 4) if sil else None,
            "Davies-Bouldin ↓": round(dbi, 4) if dbi else None,
        })

    metrics_df = pd.DataFrame(all_metrics)

    # ── Metric KPIs ──
    m_cols = st.columns(len(all_metrics))
    for col, row in zip(m_cols, all_metrics):
        with col:
            st.metric(
                label=f"🏅 {row['Model']}",
                value=f"Sil: {row['Silhouette ↑'] or '—'}",
                delta=f"DBI: {row['Davies-Bouldin ↓'] or '—'}",
                delta_color="inverse",
            )

    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # ── Silhouette comparison ──
    st.markdown("<div class='section-header'>Silhouette Score Comparison</div>", unsafe_allow_html=True)

    sil_fig = go.Figure()
    for i, row in metrics_df.iterrows():
        sil_fig.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row["Silhouette ↑"] or 0],
            name=row["Model"],
            marker_color=CLUSTER_PALETTE[i],
            text=[f"{row['Silhouette ↑']:.4f}" if row["Silhouette ↑"] else "N/A"],
            textposition="outside",
            width=0.4,
        ))
    sil_fig.update_layout(
        title="Silhouette Score by Model (Higher is Better)",
        yaxis_title="Silhouette Score",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(sil_fig, use_container_width=True)

    # ── Davies-Bouldin comparison ──
    st.markdown("<div class='section-header'>Davies-Bouldin Index Comparison</div>", unsafe_allow_html=True)

    dbi_fig = go.Figure()
    for i, row in metrics_df.iterrows():
        dbi_fig.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row["Davies-Bouldin ↓"] or 0],
            name=row["Model"],
            marker_color=CLUSTER_PALETTE[i],
            text=[f"{row['Davies-Bouldin ↓']:.4f}" if row["Davies-Bouldin ↓"] else "N/A"],
            textposition="outside",
            width=0.4,
        ))
    dbi_fig.update_layout(
        title="Davies-Bouldin Index by Model (Lower is Better)",
        yaxis_title="Davies-Bouldin Index",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(dbi_fig, use_container_width=True)

    # ── Side-by-side PCA plots ──
    st.markdown("<div class='section-header'>PCA Cluster Plots — All Models Side-by-Side</div>", unsafe_allow_html=True)

    comp_cols = st.columns(3)
    for col_ui, (model_name, c_col) in zip(comp_cols, MODEL_CLUSTER_COL.items()):
        if c_col not in df.columns:
            continue
        with col_ui:
            mini_df = df[~df["player"].str.contains("_aug|Synth", na=False)].copy()
            mini_df[c_col] = mini_df[c_col].astype(str)
            mini_fig = make_pca_scatter(mini_df, c_col, title=model_name)
            mini_fig.update_layout(height=380, margin=dict(l=20, r=10, t=45, b=30))
            mini_fig.update_traces(marker=dict(size=6))
            st.plotly_chart(mini_fig, use_container_width=True)

    # ── Insight Cards ──
    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)

    with ic1:
        st.markdown("""
        <div class='insight-card'>
            <h4>🔵 KMeans</h4>
            <p>Partitions every player into exactly K balanced clusters.
            Best for identifying broad, well-separated archetypes.
            Uses Euclidean distance on scaled features.</p>
        </div>
        """, unsafe_allow_html=True)

    with ic2:
        st.markdown("""
        <div class='insight-card'>
            <h4>🟢 DBSCAN</h4>
            <p>Density-based: discovers clusters of arbitrary shape and
            marks atypical players as noise (-1).
            Excellent for spotting outlier / unique play-styles.</p>
        </div>
        """, unsafe_allow_html=True)

    with ic3:
        st.markdown("""
        <div class='insight-card'>
            <h4>🟡 KModes</h4>
            <p>Designed for categorical data — features are discretised
            (Low / Medium / High) before clustering.
            Captures qualitative role differences rather than precise numbers.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PLAYER ANALYSIS
# ══════════════════════════════════════════════════════════════

with tab_players:

    st.markdown("<div class='section-header'>Player Comparison — Radar Chart</div>", unsafe_allow_html=True)

    # Filter to only real (non-augmented) players in filtered set
    real_players_df = df_filtered[
        ~df_filtered["player"].str.contains("_aug|Synth", na=False)
    ].copy()

    real_player_list = sorted(real_players_df["player"].unique().tolist())

    if len(real_player_list) < 2:
        st.warning("Not enough players in the filtered dataset. Adjust filters.")
    else:
        default_players = real_player_list[:min(3, len(real_player_list))]
        selected_players = st.multiselect(
            "Select players to compare (up to 5):",
            options=real_player_list,
            default=default_players,
            max_selections=5,
        )

        if selected_players:
            radar_fig = make_radar(real_players_df, selected_players)
            st.plotly_chart(radar_fig, use_container_width=True)

            # ── Player stat table ──
            st.markdown("<div class='section-header'>Selected Player Statistics</div>", unsafe_allow_html=True)
            display_cols = (
                ["player", "team"] +
                [c for c in FREQ_FEATURES + ["on_ball_creation", "off_ball_finishing",
                                               "overall_efficiency", "play_diversity"]
                 if c in df_filtered.columns] +
                [cluster_col]
            )
            player_stats = real_players_df[real_players_df["player"].isin(selected_players)][display_cols]
            st.dataframe(player_stats.set_index("player").round(3), use_container_width=True)

    # ── Bar chart: single player vs cluster average ──
    st.markdown("<div class='section-header'>Player vs. Their Cluster Average</div>", unsafe_allow_html=True)

    compare_player = st.selectbox(
        "Select player for cluster comparison:",
        options=real_player_list,
        index=0,
    )

    if compare_player:
        player_row = real_players_df[real_players_df["player"] == compare_player]
        if not player_row.empty:
            player_cluster = player_row[cluster_col].values[0]
            cluster_peers  = real_players_df[real_players_df[cluster_col] == player_cluster]

            compare_feats = [f for f in FREQ_FEATURES + ["on_ball_creation", "off_ball_finishing"]
                             if f in df_filtered.columns]
            feat_labels   = [PLAY_TYPE_LABELS.get(f, f.replace("_", " ").title()) for f in compare_feats]

            player_vals  = player_row[compare_feats].values[0].tolist()
            cluster_vals = cluster_peers[compare_feats].mean().values.tolist()

            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Bar(
                name=compare_player,
                x=feat_labels,
                y=player_vals,
                marker_color="#6366f1",
                opacity=0.9,
            ))
            cmp_fig.add_trace(go.Bar(
                name=f"Cluster {player_cluster} Avg",
                x=feat_labels,
                y=cluster_vals,
                marker_color="#10b981",
                opacity=0.7,
            ))
            cmp_fig.update_layout(
                title=f"{compare_player} vs. Cluster {player_cluster} Average ({selected_model})",
                barmode="group",
                xaxis_tickangle=-35,
                yaxis_title="Value",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Player info card
            st.markdown(f"""
            <div class='insight-card'>
                <h4>📋 Player Profile — {compare_player}</h4>
                <p>
                    <b>Team:</b> {player_row['team'].values[0]} &nbsp;|&nbsp;
                    <b>{selected_model} Cluster:</b> {player_cluster}<br>
                    <b>Overall Efficiency:</b>
                    {player_row['overall_efficiency'].values[0]:.3f} PPP &nbsp;|&nbsp;
                    <b>On-Ball Creation:</b>
                    {player_row['on_ball_creation'].values[0]:.3f} &nbsp;|&nbsp;
                    <b>Off-Ball Finishing:</b>
                    {player_row['off_ball_finishing'].values[0]:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Raw data explorer ──
    st.markdown("<div class='section-header'>Raw Data Explorer</div>", unsafe_allow_html=True)

    with st.expander("View filtered player data table", expanded=False):
        show_cols = ["player", "team", cluster_col] + [
            c for c in df_filtered.columns
            if c not in ["player", "team", "pca1", "pca2",
                          "cluster_kmeans", "cluster_dbscan", "cluster_kmodes", "archetype"]
        ]
        show_df = real_players_df[[c for c in show_cols if c in real_players_df.columns]]
        st.dataframe(show_df.round(3), use_container_width=True, height=350)

        csv_bytes = show_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_bytes,
            file_name="nba_filtered_players.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.78rem; padding:8px 0;'>
    🏀 NBA Player Clustering Dashboard &nbsp;·&nbsp; 2019-20 Season &nbsp;·&nbsp;
    Built with Streamlit &amp; Plotly &nbsp;·&nbsp; Final Year Project
</div>
""", unsafe_allow_html=True)
