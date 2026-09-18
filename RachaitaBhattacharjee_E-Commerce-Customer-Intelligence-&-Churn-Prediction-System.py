"""
E-Commerce Customer Intelligence & Churn Prediction System
Author: Rachaita Bhattacharjee
Dataset: E-Commerce Customer Behavior Dataset (Kaggle)
https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# DESIGN SYSTEM — embedded CSS
# ══════════════════════════════════════════════════════════
ACCENT       = "#2563eb"   # blue-600
ACCENT_LIGHT = "#dbeafe"   # blue-100
SUCCESS      = "#16a34a"   # green-600
SUCCESS_BG   = "#dcfce7"   # green-100
WARN         = "#d97706"   # amber-600
WARN_BG      = "#fef3c7"   # amber-100
DANGER       = "#dc2626"   # red-600
DANGER_BG    = "#fee2e2"   # red-100
TEXT_MAIN    = "#0f172a"   # slate-900
TEXT_MUTED   = "#64748b"   # slate-500
SURFACE      = "#ffffff"
PAGE_BG      = "#f1f5f9"   # slate-100
BORDER       = "#e2e8f0"   # slate-200

CHART_COLORS = ["#2563eb","#16a34a","#d97706","#9333ea","#0891b2","#be185d","#15803d","#b45309"]
PALETTE      = CHART_COLORS

def inject_css():
    st.markdown(f"""
<style>
/* ── Reset & base ─────────────────────────────── */
[data-testid="stAppViewContainer"] {{
    background: {PAGE_BG};
}}
[data-testid="stSidebar"] {{
    background: {TEXT_MAIN} !important;
    border-right: none;
}}
[data-testid="stSidebar"] * {{
    color: #cbd5e1 !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {{
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #94a3b8 !important;
    font-size: 0.78rem;
}}
/* sidebar radio — nav links */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
    font-weight: 500;
    padding: 6px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] [aria-checked="true"] + div label {{
    color: #ffffff !important;
    font-weight: 700;
}}
/* ── Top page header ──────────────────────────── */
.page-header {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    padding: 18px 28px 14px 28px;

    /* Push page header below the Streamlit top bar */
    margin: 48px -1rem 1.5rem -1rem;

    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}}
.page-header-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    line-height: 1.2;
    margin: 0;
}}
.page-header-sub {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    margin: 2px 0 0 0;
}}
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {SUCCESS_BG};
    color: {SUCCESS};
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
}}
.status-badge-offline {{
    background: #f1f5f9;
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}
.status-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {SUCCESS};
    flex-shrink: 0;
}}
.status-dot-offline {{
    background: #94a3b8;
}}
/* ── White card ────────────────────────────────── */
.card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
/* ── KPI card ──────────────────────────────────── */
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px 16px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: {ACCENT};
    border-radius: 12px 12px 0 0;
}}
.kpi-card.kpi-danger::before  {{ background: {DANGER}; }}
.kpi-card.kpi-success::before {{ background: {SUCCESS}; }}
.kpi-card.kpi-warn::before    {{ background: {WARN}; }}
.kpi-icon {{
    font-size: 1.5rem;
    line-height: 1;
    margin-bottom: 8px;
    display: block;
}}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
}}
.kpi-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {TEXT_MAIN};
    line-height: 1.1;
}}
.kpi-delta {{
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}
/* ── Section header ────────────────────────────── */
.section-header {{
    font-size: 1.0rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    letter-spacing: -0.01em;
    padding-bottom: 8px;
    border-bottom: 2px solid {ACCENT_LIGHT};
    margin: 8px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
/* ── Page title ─────────────────────────────────── */
.pg-title {{
    font-size: 1.45rem;
    font-weight: 800;
    color: {TEXT_MAIN};
    margin-bottom: 2px;
    letter-spacing: -0.02em;
}}
.pg-subtitle {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-bottom: 20px;
}}
/* ── Insight cards ──────────────────────────────── */
.insight-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 16px 18px;
    margin-bottom: 14px;
}}
.insight-card.warn  {{ border-left-color: {WARN}; }}
.insight-card.danger {{ border-left-color: {DANGER}; }}
.insight-card.success {{ border-left-color: {SUCCESS}; }}
.insight-type {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {ACCENT};
    margin-bottom: 4px;
}}
.insight-card.warn  .insight-type  {{ color: {WARN}; }}
.insight-card.danger .insight-type {{ color: {DANGER}; }}
.insight-card.success .insight-type {{ color: {SUCCESS}; }}
.insight-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    margin-bottom: 6px;
}}
.insight-body {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    line-height: 1.55;
}}
.insight-rec {{
    font-size: 0.80rem;
    color: {TEXT_MAIN};
    background: {PAGE_BG};
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
    border: 1px solid {BORDER};
}}
/* ── Model metric card ──────────────────────────── */
.model-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.model-card-name {{
    font-size: 1.0rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid {BORDER};
}}
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #f8fafc;
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-lbl {{ font-size: 0.78rem; color: {TEXT_MUTED}; font-weight: 500; }}
.metric-val {{ font-size: 0.90rem; font-weight: 700; color: {TEXT_MAIN}; }}
.metric-bar-wrap {{
    height: 5px; background: {BORDER}; border-radius: 3px;
    margin-top: 3px; overflow: hidden;
}}
.metric-bar {{
    height: 100%; border-radius: 3px; background: {ACCENT};
    transition: width 0.4s ease;
}}
/* ── Prediction result card ─────────────────────── */
.pred-card {{
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    border: 2px solid {BORDER};
    background: {SURFACE};
}}
.pred-card.low    {{ border-color: {SUCCESS}; background: {SUCCESS_BG}; }}
.pred-card.medium {{ border-color: {WARN};    background: {WARN_BG}; }}
.pred-card.high   {{ border-color: {DANGER};  background: {DANGER_BG}; }}
.pred-risk-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}}
.pred-risk-label.low    {{ color: {SUCCESS}; }}
.pred-risk-label.medium {{ color: {WARN}; }}
.pred-risk-label.high   {{ color: {DANGER}; }}
.pred-prob {{
    font-size: 3.6rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 4px;
}}
.pred-prob.low    {{ color: {SUCCESS}; }}
.pred-prob.medium {{ color: {WARN}; }}
.pred-prob.high   {{ color: {DANGER}; }}
.pred-verdict {{
    font-size: 1.0rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-top: 10px;
}}
/* ── Disclaimer box ─────────────────────────────── */
.disclaimer {{
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #92400e;
    margin-top: 12px;
    line-height: 1.5;
}}
/* ── Segment badge ──────────────────────────────── */
.seg-badge {{
    display: inline-block;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}}
/* ── Tag chips ──────────────────────────────────── */
.tag {{
    display: inline-block;
    background: {ACCENT_LIGHT};
    color: {ACCENT};
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}}
/* ── Remove default streamlit padding on main block ── */
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 100% !important;
}}
/* ── Tab styling ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    background: #ffffff !important;
    padding: 6px 8px !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 0 !important;
}}

.stTabs [data-baseweb="tab-panel"] {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px 16px !important;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 9px 16px !important;
    color: #475569 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    transition: color 0.15s, border-color 0.15s !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: #1e40af !important;
    background: #f0f9ff !important;
}}

.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] > div,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] button {{
    color: #475569 !important;
    font-weight: 600 !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: #eff6ff !important;
    border-bottom: 2px solid #2563eb !important;
    border-radius: 6px 6px 0 0 !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] *,
.stTabs [data-baseweb="tab"][aria-selected="true"] > div,
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
.stTabs [data-baseweb="tab"][aria-selected="true"] button {{
    color: #2563eb !important;
    font-weight: 700 !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
/* ── Spinner / computation message ───────────────── */
[data-testid="stSpinner"] {{
    color: #334155 !important;
}}

[data-testid="stSpinner"] p {{
    color: #334155 !important;
}}

[data-testid="stSpinner"] span {{
    color: #334155 !important;
}}
/* ── Info/status message ─────────────────────────── */
[data-testid="stAlert"] {{
    color: #334155 !important;
}}

[data-testid="stAlert"] p {{
    color: #334155 !important;
}}

[data-testid="stAlert"] span {{
    color: #334155 !important;
}}
/* ── Chart card title/subtitle ──────────────────── */
.chart-card-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
}}

.chart-card-subtitle {{
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 10px;
    line-height: 1.4;
}}

/* ── Tab section title ──────────────────────────── */
.tab-section-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 4px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}}
/* ── Sidebar nav radio hide circle ──────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {{
    display: none;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding: 9px 12px !important;
    border-radius: 8px !important;
    background: transparent !important;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(255,255,255,0.07) !important;
}}
[data-testid="stSidebar"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {{
    background: rgba(37,99,235,0.25) !important;
    color: #ffffff !important;
}}
/* ── Sidebar status panel ───────────────────────── */
.sb-status {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
}}
.sb-status-label {{
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 8px;
}}
.sb-status-row {{
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 4px;
}}
.sb-dot-on  {{ width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0; }}
.sb-dot-off {{ width:7px;height:7px;border-radius:50%;background:#475569;flex-shrink:0; }}
.sb-status-text {{ font-size:0.78rem;color:#e2e8f0; }}
.sb-stat {{ font-size:0.72rem;color:#94a3b8;padding-left:14px;margin-bottom:1px; }}
/* ── Filter section ─────────────────────────────── */
.sb-filter-header {{
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin: 14px 0 6px 0;
}}
/* ── Dataframe overrides ─────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}
/* ── Streamlit form labels ─────────────────────── */
[data-testid="stWidgetLabel"] {{
    color: #334155 !important;
    font-weight: 600 !important;
}}

[data-testid="stWidgetLabel"] p {{
    color: #334155 !important;
    font-weight: 600 !important;
}}

[data-testid="stNumberInput"] label {{
    color: #334155 !important;
}}

[data-testid="stSelectbox"] label {{
    color: #334155 !important;
}}

[data-testid="stSlider"] label {{
    color: #334155 !important;
}}

[data-testid="stNumberInput"] input {{
    color: #f8fafc !important;
}}

[data-testid="stSelectbox"] [data-baseweb="select"] {{
    color: #f8fafc !important;
}}
/* ── Button ──────────────────────────────────────── */
.stButton > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 22px;
    transition: background 0.2s;
}}
.stButton > button:hover {{
    background: #1d4ed8;
    color: white;
}}
/* ── Form submit button ───────────────────────────── */
[data-testid="stFormSubmitButton"] > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 12px;
    width: 100%;
    transition: background 0.2s;
}}
[data-testid="stFormSubmitButton"] > button:hover {{
    background: #1d4ed8;
}}
/* ── Streamlit header cleanup ───────────────────────── */
/* Hide Streamlit branding */
/* ── Streamlit header cleanup ───────────────────────── */
#MainMenu,
footer {{
    visibility: hidden !important;
}}

/* Keep Streamlit header visible so sidebar control works */
header {{
    visibility: visible !important;
}}
/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {SURFACE};
}}
[data-testid="stExpander"] {{
    border: 1px solid #dbe3ef !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
}}

[data-testid="stExpander"] details {{
    background: #ffffff !important;
}}

[data-testid="stExpander"] summary {{
    background: #ffffff !important;
    color: #0f172a !important;
    padding: 16px 18px !important;
    min-height: 54px !important;
    font-weight: 700 !important;
}}

[data-testid="stExpander"] summary:hover {{
    background: #f8fafc !important;
}}

[data-testid="stExpander"] summary p {{
    color: #0f172a !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}}

[data-testid="stExpander"] summary span {{
    color: #0f172a !important;
}}

[data-testid="stExpander"] summary svg {{
    color: #475569 !important;
    fill: #475569 !important;
    stroke: #475569 !important;
}}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    background: #ffffff !important;
    color: #334155 !important;
}}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] p {{
    color: #334155 !important;
}}
/* ══════════════════════════════════════════════════════
   FINAL TAB TEXT OVERRIDE
   ══════════════════════════════════════════════════════ */

.stTabs [role="tab"],
.stTabs [role="tab"] *,
.stTabs button,
.stTabs button *,
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] p {{
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

/* Selected tab */
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"] *,
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] * {{
    color: #2563eb !important;
    -webkit-text-fill-color: #2563eb !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

/* Tab hover */
.stTabs [role="tab"]:hover,
.stTabs [role="tab"]:hover *,
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover * {{
    color: #1e40af !important;
    -webkit-text-fill-color: #1e40af !important;
    opacity: 1 !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
REQUIRED_COLUMNS = [
    "Age", "Gender", "Country", "City", "Membership_Years",
    "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
    "Cart_Abandonment_Rate", "Wishlist_Items", "Total_Purchases",
    "Average_Order_Value", "Days_Since_Last_Purchase", "Discount_Usage_Rate",
    "Returns_Rate", "Email_Open_Rate", "Customer_Service_Calls",
    "Product_Reviews_Written", "Social_Media_Engagement_Score",
    "Mobile_App_Usage", "Payment_Method_Diversity", "Lifetime_Value",
    "Credit_Balance", "Churned", "Signup_Quarter"
]

TARGET     = "Churned"
MODEL_PATH = "churn_model_pipeline.joblib"
C_RETAIN   = "#16a34a"
C_CHURN    = "#dc2626"
C_WARN     = "#d97706"
C_ACCENT   = "#2563eb"


# ══════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════
# Axis colours used across all charts
_AX_TICK   = "#475569"   # slate-600  — readable tick labels
_AX_TITLE  = "#334155"   # slate-700  — axis titles
_GRID      = "#e2e8f0"   # slate-200  — subtle gridlines
_CHART_BG  = "#ffffff"   # pure white chart area

def _fig_style(fig, height=320):
    """Apply the standard chart theme.
    - Solid white backgrounds (no transparency).
    - Explicit, readable axis label and tick colours.
    - Clean gridlines, hidden modebar.
    """
    axis_style = dict(
        gridcolor=_GRID,
        linecolor=_GRID,
        tickfont=dict(size=11, color=_AX_TICK, family="Inter, system-ui, sans-serif"),
        title_font=dict(size=11, color=_AX_TITLE, family="Inter, system-ui, sans-serif"),
        zerolinecolor=_GRID,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0f172a"),
        margin=dict(l=4, r=4, t=38, b=8),
        title_font=dict(size=13, color="#0f172a", family="Inter, system-ui, sans-serif"),
        title_x=0,
        legend=dict(
            bgcolor="#ffffff",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(size=11, color="#334155"),
        ),
        xaxis=axis_style,
        yaxis=axis_style,
        modebar_remove=["zoom","pan","select","lasso2d","zoomIn2d","zoomOut2d",
                        "autoScale2d","resetScale2d","toImage"],
    )
    return fig


# ══════════════════════════════════════════════════════════
# DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file_source) -> pd.DataFrame:
    return pd.read_csv(file_source)


def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


# ══════════════════════════════════════════════════════════
# DATA CLEANING
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    df = df.copy()
    report["raw_rows"] = len(df)
    report["raw_cols"] = len(df.columns)

    dup_count = df.duplicated().sum()
    report["duplicates_removed"] = int(dup_count)
    df = df.drop_duplicates()

    age_outliers = ((df["Age"] < 5) | (df["Age"] > 100)).sum()
    report["age_outliers_capped"] = int(age_outliers)
    df["Age"] = df["Age"].clip(5, 100)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    report["invalid_target_rows"] = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    report["missing_values"] = df.isnull().sum().to_dict()
    report["total_missing"]  = int(df.isnull().sum().sum())
    report["clean_rows"]     = len(df)

    num_cols = [
        "Age","Membership_Years","Login_Frequency","Session_Duration_Avg",
        "Pages_Per_Session","Cart_Abandonment_Rate","Wishlist_Items",
        "Total_Purchases","Average_Order_Value","Days_Since_Last_Purchase",
        "Discount_Usage_Rate","Returns_Rate","Email_Open_Rate",
        "Customer_Service_Calls","Product_Reviews_Written",
        "Social_Media_Engagement_Score","Mobile_App_Usage",
        "Payment_Method_Diversity","Lifetime_Value","Credit_Balance"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, report


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Purchase_Value"] = df["Total_Purchases"] * df["Average_Order_Value"]

    ltv_q = df["Lifetime_Value"].quantile([0.25, 0.5, 0.75])
    df["Customer_Value_Category"] = pd.cut(
        df["Lifetime_Value"],
        bins=[-np.inf, ltv_q[0.25], ltv_q[0.5], ltv_q[0.75], np.inf],
        labels=["Low Value","Medium Value","High Value","Premium Value"]
    )
    df["Membership_Category"] = pd.cut(
        df["Membership_Years"],
        bins=[-np.inf, 1, 3, 6, np.inf],
        labels=["New (<1yr)","Developing (1-3yr)","Established (3-6yr)","Loyal (6+yr)"]
    )

    eng_feats = ["Login_Frequency","Session_Duration_Avg","Pages_Per_Session",
                 "Email_Open_Rate","Social_Media_Engagement_Score","Mobile_App_Usage"]
    eng_df = df[eng_feats].copy()
    for col in eng_feats:
        cmin, cmax = eng_df[col].min(), eng_df[col].max()
        if cmax > cmin:
            eng_df[col] = (eng_df[col] - cmin) / (cmax - cmin)
    df["Engagement_Score"] = eng_df.mean(axis=1) * 100

    df["Recency_Category"] = pd.cut(
        df["Days_Since_Last_Purchase"],
        bins=[-np.inf, 30, 60, 120, np.inf],
        labels=["Recent (<30d)","Moderate (30-60d)","Distant (60-120d)","Lapsed (120+d)"]
    )
    df["Purchase_Frequency_Category"] = pd.cut(
        df["Total_Purchases"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["Rare","Occasional","Regular","Frequent"]
    )
    return df


# ══════════════════════════════════════════════════════════
# KPI METRICS
# ══════════════════════════════════════════════════════════
def calculate_metrics(df: pd.DataFrame) -> dict:
    total   = len(df)
    churned = int(df[TARGET].sum())
    return {
        "total_customers":    total,
        "churned_customers":  churned,
        "retained_customers": total - churned,
        "churn_rate":         churned / total * 100 if total > 0 else 0,
        "avg_ltv":            df["Lifetime_Value"].mean(),
        "avg_aov":            df["Average_Order_Value"].mean(),
        "avg_purchases":      df["Total_Purchases"].mean(),
        "avg_membership":     df["Membership_Years"].mean(),
        "avg_engagement":     df["Engagement_Score"].mean() if "Engagement_Score" in df.columns else 0,
    }


# ══════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════
SEGMENT_FEATURES = [
    "Membership_Years","Login_Frequency","Session_Duration_Avg",
    "Pages_Per_Session","Total_Purchases","Average_Order_Value",
    "Days_Since_Last_Purchase","Discount_Usage_Rate","Returns_Rate",
    "Email_Open_Rate","Social_Media_Engagement_Score","Mobile_App_Usage",
    "Lifetime_Value"
]

@st.cache_data(show_spinner=False)
def create_segments(df: pd.DataFrame, n_clusters: int = 4):
    seg_df  = df[SEGMENT_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    seg_arr = imputer.fit_transform(seg_df)
    scaler  = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_arr)
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(seg_scaled)
    df_out = df.copy()
    df_out["Cluster"] = labels
    return df_out, km, scaler


@st.cache_data(show_spinner=False)
def compute_elbow_silhouette(df: pd.DataFrame, k_range=range(2, 9)):
    seg_df  = df[SEGMENT_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    seg_arr = imputer.fit_transform(seg_df)
    scaler  = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_arr)
    inertias, sil_scores = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(seg_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(seg_scaled, lbl))
    return list(k_range), inertias, sil_scores


def name_segments(df_seg: pd.DataFrame) -> dict:
    cluster_ids = sorted(df_seg["Cluster"].unique())
    stats = df_seg.groupby("Cluster").agg(
        engagement=("Engagement_Score","median"),
        ltv=("Lifetime_Value","median"),
        churn_rate=(TARGET,"mean"),
        recency=("Days_Since_Last_Purchase","median"),
    ).reindex(cluster_ids)
    names = {}
    for cid in cluster_ids:
        row      = stats.loc[cid]
        high_eng   = row["engagement"] > stats["engagement"].median()
        high_ltv   = row["ltv"]        > stats["ltv"].median()
        high_churn = row["churn_rate"] > stats["churn_rate"].median()
        recent     = row["recency"]    < stats["recency"].median()
        if high_eng and high_ltv and not high_churn:
            names[cid] = "Champions"
        elif high_ltv and not high_eng:
            names[cid] = "At-Risk Valuable"
        elif high_eng and not high_ltv:
            names[cid] = "Active Explorers"
        elif high_churn and not recent:
            names[cid] = "Churning Disengaged"
        else:
            names[cid] = f"Segment {cid + 1}"
    return names


# ══════════════════════════════════════════════════════════
# DATA LEAKAGE CHECK
# ══════════════════════════════════════════════════════════
def check_data_leakage(df: pd.DataFrame) -> dict:
    report = {}
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    report["high_correlation_features"] = corr[corr > 0.8].index.tolist()
    report["lifetime_value_flag"] = (
        "Lifetime_Value shows near-zero correlation with Churned (r≈−0.01), "
        "suggesting it may be calculated independently of churn events. "
        "It is retained in descriptive analytics but excluded from ML predictors "
        "to avoid any forward-looking information leakage."
    )
    report["purchase_value_note"] = (
        "Purchase_Value = Total_Purchases × Average_Order_Value. "
        "Both inputs are historical activity metrics, not post-event fields."
    )
    corr_tbl = corr.head(10).reset_index()
    corr_tbl.columns = ["Feature", "|Correlation with Churned|"]
    report["correlation_table"] = corr_tbl
    return report


# ══════════════════════════════════════════════════════════
# ML FEATURE LISTS
# ══════════════════════════════════════════════════════════
ML_NUMERICAL_FEATURES = [
    "Age","Membership_Years","Login_Frequency","Session_Duration_Avg",
    "Pages_Per_Session","Cart_Abandonment_Rate","Wishlist_Items",
    "Total_Purchases","Average_Order_Value","Days_Since_Last_Purchase",
    "Discount_Usage_Rate","Returns_Rate","Email_Open_Rate",
    "Customer_Service_Calls","Product_Reviews_Written",
    "Social_Media_Engagement_Score","Mobile_App_Usage",
    "Payment_Method_Diversity","Credit_Balance"
]
ML_CATEGORICAL_FEATURES = ["Gender","Country","Signup_Quarter"]
ML_ALL_FEATURES = ML_NUMERICAL_FEATURES + ML_CATEGORICAL_FEATURES


# ══════════════════════════════════════════════════════════
# ML PIPELINE
# ══════════════════════════════════════════════════════════
def prepare_ml_data(df: pd.DataFrame):
    a_num = [f for f in ML_NUMERICAL_FEATURES  if f in df.columns]
    a_cat = [f for f in ML_CATEGORICAL_FEATURES if f in df.columns]
    X = df[a_num + a_cat].copy()
    y = df[TARGET].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, a_num, a_cat


def build_preprocessor(num_features, cat_features) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])


@st.cache_resource(show_spinner=False)
def train_models(_X_train, _y_train, num_features: list, cat_features: list) -> dict:
    preprocessor = build_preprocessor(num_features, cat_features)
    clfs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", C=0.1),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }
    trained = {}
    for name, clf in clfs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(_X_train, _y_train)
        trained[name] = pipe
    return trained


def evaluate_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
        })
    return pd.DataFrame(rows)


def save_model(pipeline, path: str = MODEL_PATH):
    joblib.dump(pipeline, path)


def load_model(path: str = MODEL_PATH):
    return joblib.load(path) if os.path.exists(path) else None


# ══════════════════════════════════════════════════════════
# CHURN PREDICTION
# ══════════════════════════════════════════════════════════
def predict_churn(pipeline, input_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    prob = pipeline.predict_proba(input_df)[0][1]
    pred = int(prob >= threshold)
    if prob < 0.35:
        risk, risk_cls = "LOW RISK",    "low"
    elif prob < 0.65:
        risk, risk_cls = "MEDIUM RISK", "medium"
    else:
        risk, risk_cls = "HIGH RISK",   "high"
    return {"prediction": pred, "probability": prob, "risk_level": risk, "risk_cls": risk_cls}


# ══════════════════════════════════════════════════════════
# BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════
def generate_business_insights(df: pd.DataFrame) -> list[dict]:
    insights = []
    churn_rate  = df[TARGET].mean() * 100
    cart_churn  = df[df[TARGET]==1]["Cart_Abandonment_Rate"].mean()
    cart_retain = df[df[TARGET]==0]["Cart_Abandonment_Rate"].mean()
    cs_churn    = df[df[TARGET]==1]["Customer_Service_Calls"].mean()
    cs_retain   = df[df[TARGET]==0]["Customer_Service_Calls"].mean()
    login_churn  = df[df[TARGET]==1]["Login_Frequency"].mean()
    login_retain = df[df[TARGET]==0]["Login_Frequency"].mean()
    rec_churn   = df[df[TARGET]==1]["Days_Since_Last_Purchase"].median()
    rec_retain  = df[df[TARGET]==0]["Days_Since_Last_Purchase"].median()
    email_churn  = df[df[TARGET]==1]["Email_Open_Rate"].mean()
    email_retain = df[df[TARGET]==0]["Email_Open_Rate"].mean()
    disc_churn   = df[df[TARGET]==1]["Discount_Usage_Rate"].mean()
    disc_retain  = df[df[TARGET]==0]["Discount_Usage_Rate"].mean()
    mob_churn    = df[df[TARGET]==1]["Mobile_App_Usage"].mean()
    mob_retain   = df[df[TARGET]==0]["Mobile_App_Usage"].mean()

    insights = [
        {
            "icon": "📊", "type": "CHURN INSIGHT", "cls": "danger",
            "title": "Overall Churn Rate",
            "observation": f"{churn_rate:.1f}% of customers have churned — nearly 1 in 3 disengages.",
            "implication": "Early detection and proactive retention programmes are critical to protect revenue.",
            "recommendation": "Implement a churn early-warning system and deploy targeted re-engagement campaigns."
        },
        {
            "icon": "🛒", "type": "CHURN INSIGHT", "cls": "danger",
            "title": "Cart Abandonment & Churn",
            "observation": (
                f"Churned customers average {cart_churn:.1f}% cart abandonment "
                f"vs {cart_retain:.1f}% for retained customers (Δ {cart_churn-cart_retain:+.1f}%)."
            ),
            "implication": "High cart abandonment is strongly associated with eventual churn.",
            "recommendation": "Deploy automated cart-recovery emails at 1-hour and 24-hour windows; offer limited-time discounts for Cart_Abandonment_Rate > 60%."
        },
        {
            "icon": "📞", "type": "CHURN INSIGHT", "cls": "danger",
            "title": "Customer Service Calls",
            "observation": (
                f"Churned customers average {cs_churn:.1f} service calls "
                f"vs {cs_retain:.1f} for retained customers."
            ),
            "implication": "Repeated service contacts signal unresolved issues and are the strongest single predictor of churn.",
            "recommendation": "Flag customers with ≥5 service calls for proactive outreach; improve first-contact resolution rates."
        },
        {
            "icon": "🔑", "type": "RETENTION INSIGHT", "cls": "warn",
            "title": "Login Engagement Decline",
            "observation": (
                f"Churned customers log in {login_churn:.1f}× on average "
                f"vs {login_retain:.1f}× for retained customers."
            ),
            "implication": "Falling login frequency is an early-warning signal preceding churn.",
            "recommendation": "Trigger personalised re-engagement campaigns for customers with <5 logins/month or no login in 14+ days."
        },
        {
            "icon": "📅", "type": "RETENTION INSIGHT", "cls": "warn",
            "title": "Purchase Recency",
            "observation": (
                f"Median days since last purchase: {rec_churn:.0f}d (churned) "
                f"vs {rec_retain:.0f}d (retained)."
            ),
            "implication": "Prolonged inactivity is a reliable churn precursor.",
            "recommendation": "Create win-back campaigns for customers inactive for 60+ days; offer personalised incentives."
        },
        {
            "icon": "📧", "type": "RETENTION INSIGHT", "cls": "warn",
            "title": "Email Engagement Gap",
            "observation": (
                f"Email open rate: {email_churn:.1f}% (churned) vs {email_retain:.1f}% (retained)."
            ),
            "implication": "Low email engagement is associated with reduced brand connection and eventual churn.",
            "recommendation": "Improve personalisation and subject-line testing; segment email lists by Engagement_Score."
        },
        {
            "icon": "💰", "type": "VALUE INSIGHT", "cls": "",
            "title": "Discount Dependency",
            "observation": (
                f"Churned: {disc_churn:.1f}% avg discount usage vs retained: {disc_retain:.1f}%."
            ),
            "implication": "Customers over-reliant on discounts may churn when offers stop; under-incentivised customers may also disengage.",
            "recommendation": "Develop loyalty-based reward programmes that reduce discount dependency while maintaining perceived value."
        },
        {
            "icon": "📱", "type": "VALUE INSIGHT", "cls": "",
            "title": "Mobile App Engagement",
            "observation": (
                f"Retained customers use the mobile app {mob_retain:.1f}% of the time "
                f"vs {mob_churn:.1f}% for churned customers."
            ),
            "implication": "Mobile app engagement is strongly associated with retention.",
            "recommendation": "Invest in app-exclusive features, push notifications, and mobile-first promotions to drive adoption among at-risk customers."
        },
    ]
    return insights


# ══════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ══════════════════════════════════════════════════════════
def kpi_card(label: str, value, delta: str = "", variant: str = ""):
    """Render a single KPI card. variant: '', 'danger', 'success', 'warn'."""
    cls = f"kpi-card kpi-{variant}" if variant else "kpi-card"
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="{cls}">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  {delta_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def tab_section(title: str):
    """Small uppercase section divider used inside tab panels."""
    st.markdown(f'<div class="tab-section-title">{title}</div>', unsafe_allow_html=True)


def chart_card(title: str = "", subtitle: str = ""):
    """
    Returns a Streamlit container(border=True) context manager.
    Optionally renders a card title + subtitle above the chart.
    Usage:
        with chart_card("Title", "Subtitle"):
            st.plotly_chart(...)
    """
    if title:
        st.markdown(
            f'<div class="chart-card-title">{title}</div>'
            + (f'<div class="chart-card-subtitle">{subtitle}</div>' if subtitle else ""),
            unsafe_allow_html=True
        )
    return st.container(border=True)


def page_header(title: str, subtitle: str, connected: bool = True, n_rows: int = 0, n_cols: int = 0):
    if connected:
        badge = (
            f'<span class="status-badge">'
            f'  <span class="status-dot"></span>'
            f'  Connected &nbsp;·&nbsp; {n_rows:,} records · {n_cols} variables'
            f'</span>'
        )
    else:
        badge = (
            '<span class="status-badge status-badge-offline">'
            '  <span class="status-dot status-dot-offline"></span>'
            '  Not Connected'
            '</span>'
        )
    st.markdown(
        f'<div class="page-header">'
        f'  <div>'
        f'    <div class="page-header-title">{title}</div>'
        f'    <div class="page-header-sub">{subtitle}</div>'
        f'  </div>'
        f'  {badge}'
        f'</div>',
        unsafe_allow_html=True
    )


def insight_card(icon, itype, title, observation, implication, recommendation, cls=""):
    st.markdown(
        f'<div class="insight-card {cls}">'
        f'  <div class="insight-type">{icon} {itype}</div>'
        f'  <div class="insight-title">{title}</div>'
        f'  <div class="insight-body"><strong>Finding:</strong> {observation}<br>'
        f'  <strong>Interpretation:</strong> {implication}</div>'
        f'  <div class="insight-rec">💡 <strong>Recommendation:</strong> {recommendation}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def model_metric_card(name: str, metrics: dict):
    def bar(v): return (
        f'<div class="metric-bar-wrap">'
        f'  <div class="metric-bar" style="width:{v*100:.1f}%"></div>'
        f'</div>'
    )
    rows_html = ""
    for lbl, val in metrics.items():
        rows_html += (
            f'<div class="metric-row">'
            f'  <span class="metric-lbl">{lbl}</span>'
            f'  <span class="metric-val">{val:.4f}</span>'
            f'</div>'
            f'{bar(val)}'
        )
    st.markdown(
        f'<div class="model-card">'
        f'  <div class="model-card-name">🤖 {name}</div>'
        f'  {rows_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def prediction_result_card(prob: float, risk_cls: str, risk_label: str, prediction: int):
    verdict = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Retain"
    st.markdown(
        f'<div class="pred-card {risk_cls}">'
        f'  <div class="pred-risk-label {risk_cls}">CHURN RISK</div>'
        f'  <div class="pred-prob {risk_cls}">{prob*100:.1f}%</div>'
        f'  <div class="pred-risk-label {risk_cls}" style="font-size:1.0rem;letter-spacing:0.06em">'
        f'    {risk_label}'
        f'  </div>'
        f'  <div class="pred-verdict">{verdict}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════
def plot_churn_by_category(df, col, title, h=380):
    grp = df.groupby([col, TARGET]).size().reset_index(name="Count")
    grp["Status"] = grp[TARGET].map({0: "Retained", 1: "Churned"})
    total = grp.groupby(col)["Count"].transform("sum")
    grp["Pct"] = grp["Count"] / total * 100
    fig = px.bar(
        grp, x=col, y="Pct", color="Status", barmode="stack", title=title,
        color_discrete_map={"Retained": C_RETAIN, "Churned": C_CHURN},
        labels={"Pct": "% of Customers", col: col.replace("_"," ")},
        text_auto=".1f"
    )
    return _fig_style(fig, h)


def plot_boxplot(df, col, title, h=360):
    df2 = df[[col, TARGET]].dropna().copy()
    df2["Status"] = df2[TARGET].map({0: "Retained", 1: "Churned"})
    fig = px.box(
        df2, x="Status", y=col, color="Status", title=title,
        color_discrete_map={"Retained": C_RETAIN, "Churned": C_CHURN},
        labels={col: col.replace("_"," ")}
    )
    fig.update_layout(showlegend=False)
    return _fig_style(fig, h)


def plot_histogram(df, col, title, color=C_ACCENT, nbins=40, h=360):
    fig = px.histogram(
        df, x=col, nbins=nbins, title=title,
        color_discrete_sequence=[color],
        labels={col: col.replace("_"," ")}
    )
    fig.update_layout(bargap=0.05)
    return _fig_style(fig, h)


def plot_confusion_matrix(y_true, y_pred, model_name, h=380):
    cm  = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, title=f"Confusion Matrix — {model_name}",
        labels=dict(x="Predicted", y="Actual"),
        x=["Predicted Retained","Predicted Churned"],
        y=["Actual Retained","Actual Churned"],
        color_continuous_scale="Blues", text_auto=True
    )
    fig.update_layout(coloraxis_showscale=False)
    return _fig_style(fig, h)


def plot_roc_curves(trained_models, X_test, y_test, h=420):
    fig = go.Figure()
    colors = [C_ACCENT,"#16a34a","#d97706","#9333ea"]
    for (name, pipe), color in zip(trained_models.items(), colors):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
            line=dict(color=color, width=2.5)
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="#94a3b8", width=1.5),
        name="Random Classifier"
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.08)
    )
    return _fig_style(fig, h)


def plot_feature_importance(pipeline, feature_names: list, model_name: str, h=500):
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return None
    try:
        ohe      = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(ML_CATEGORICAL_FEATURES))
        all_names = [f for f in feature_names if f in ML_NUMERICAL_FEATURES] + cat_names
    except Exception:
        all_names = feature_names
    n = min(len(importances), len(all_names))
    imp_df = pd.DataFrame({"Feature": all_names[:n], "Importance": importances[:n]})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(18)
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        title=f"Feature Importance — {model_name}",
        color="Importance", color_continuous_scale=[[0,"#dbeafe"],[1,C_ACCENT]]
    )
    fig.update_layout(coloraxis_showscale=False)
    return _fig_style(fig, h)


def plot_correlation_heatmap(df, h=680):
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    fig = px.imshow(
        corr, title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", text_auto=".2f"
    )
    fig.update_layout(coloraxis_colorbar_title="r")
    return _fig_style(fig, h)


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar(df_loaded=False, df=None, show_uploader=True):
    """Render sidebar. Dataset uploader is rendered only once."""
    uploaded = None

    # ── Brand header ─────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="padding:18px 4px 4px 4px;">'
        '  <div style="font-size:1.1rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-0.02em;">'
        '    🛒 E-Commerce Intelligence'
        '  </div>'
        '  <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">'
        '    Customer Analytics Platform'
        '  </div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Navigation ───────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="margin:16px 0 6px 0;"></div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        '<div class="sb-filter-header">Navigation</div>',
        unsafe_allow_html=True
    )

    PAGES = [
        "🏠 Overview",
        "👥 Customer Analytics",
        "⚠️ Churn Analysis",
        "🎯 Segmentation",
        "🤖 Machine Learning",
        "🔮 Churn Prediction",
        "💡 Business Insights",
        "📋 Data Quality",
    ]

    page = st.sidebar.radio(
        "",
        PAGES,
        label_visibility="collapsed",
        key="main_navigation"
    )

    # ── Filters ───────────────────────────────────────────────
    filtered_df = df

    if df is not None and df_loaded:

        st.sidebar.markdown(
            '<div class="sb-filter-header">Filters</div>',
            unsafe_allow_html=True
        )

        countries = ["All"] + sorted(
            df["Country"].dropna().unique().tolist()
        )

        genders = ["All"] + sorted(
            df["Gender"].dropna().unique().tolist()
        )

        quarters = ["All"] + sorted(
            df["Signup_Quarter"].dropna().unique().tolist()
        )

        churn_opts = {
            "All": None,
            "Churned": 1,
            "Retained": 0
        }

        sel_country = st.sidebar.selectbox(
            "Country",
            countries,
            key="f_country"
        )

        sel_gender = st.sidebar.selectbox(
            "Gender",
            genders,
            key="f_gender"
        )

        sel_quarter = st.sidebar.selectbox(
            "Signup Quarter",
            quarters,
            key="f_quarter"
        )

        sel_churn = st.sidebar.selectbox(
            "Churn Status",
            list(churn_opts.keys()),
            key="f_churn"
        )

        vc_opts = ["All"]

        if "Customer_Value_Category" in df.columns:
            vc_opts += [
                str(v)
                for v in df["Customer_Value_Category"]
                .dropna()
                .unique()
                .tolist()
            ]

        sel_vc = st.sidebar.selectbox(
            "Customer Value",
            vc_opts,
            key="f_vc"
        )

        mc_opts = ["All"]

        if "Membership_Category" in df.columns:
            mc_opts += [
                str(v)
                for v in df["Membership_Category"]
                .dropna()
                .unique()
                .tolist()
            ]

        sel_mc = st.sidebar.selectbox(
            "Membership",
            mc_opts,
            key="f_mc"
        )

        if st.sidebar.button(
            "↺ Reset Filters",
            key="reset_filters"
        ):
            for key in [
                "f_country",
                "f_gender",
                "f_quarter",
                "f_churn",
                "f_vc",
                "f_mc"
            ]:
                st.session_state.pop(key, None)

            st.rerun()

        # Apply filters
        filtered_df = df.copy()

        if sel_country != "All":
            filtered_df = filtered_df[
                filtered_df["Country"] == sel_country
            ]

        if sel_gender != "All":
            filtered_df = filtered_df[
                filtered_df["Gender"] == sel_gender
            ]

        if sel_quarter != "All":
            filtered_df = filtered_df[
                filtered_df["Signup_Quarter"] == sel_quarter
            ]

        if churn_opts[sel_churn] is not None:
            filtered_df = filtered_df[
                filtered_df[TARGET] == churn_opts[sel_churn]
            ]

        if (
            sel_vc != "All"
            and "Customer_Value_Category" in df.columns
        ):
            filtered_df = filtered_df[
                filtered_df["Customer_Value_Category"].astype(str)
                == sel_vc
            ]

        if (
            sel_mc != "All"
            and "Membership_Category" in df.columns
        ):
            filtered_df = filtered_df[
                filtered_df["Membership_Category"].astype(str)
                == sel_mc
            ]

        st.sidebar.markdown(
            f'<div style="margin-top:10px;padding:8px 4px;'
            f'border-top:1px solid #1e293b;">'
            f'  <div class="sb-stat" '
            f'style="color:#94a3b8;font-size:0.74rem;">'
            f'    Showing '
            f'<strong style="color:#e2e8f0">'
            f'{len(filtered_df):,}</strong> of '
            f'<strong style="color:#e2e8f0">'
            f'{len(df):,}</strong> customers'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Footer ────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="margin-top:20px;padding-top:12px;'
        'border-top:1px solid #1e293b;">'
        '  <div style="font-size:0.68rem;color:#475569;line-height:1.6;">'
        '    Author: Rachaita Bhattacharjee<br>'
        '    <a href="https://www.kaggle.com/datasets/dhairyajeetsingh/'
        'ecommerce-customer-behavior-dataset" '
        'style="color:#3b82f6;" target="_blank">'
        '    Kaggle Dataset ↗</a>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True
    )

    return page, uploaded, filtered_df

# ══════════════════════════════════════════════════════════
# PAGE: OVERVIEW (EXECUTIVE DASHBOARD)
# ══════════════════════════════════════════════════════════
def render_executive_dashboard(df: pd.DataFrame, filtered_df: pd.DataFrame):
    page_header(
        "🏠 Executive Overview",
        "Customer Intelligence & Churn Prediction Dashboard",
        connected=True, n_rows=len(df), n_cols=len(df.columns)
    )

    if len(filtered_df) == 0:
        st.warning("No customers match the current filters.")
        return

    m = calculate_metrics(filtered_df)

    # ── Row 1: 4 KPI cards ──
    section_header("📌 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Customers", f"{m['total_customers']:,}", "Full filtered dataset")
    with k2:
        kpi_card("Churned Customers", f"{m['churned_customers']:,}",
                 f"{m['churn_rate']:.1f}% of total", variant="danger")
    with k3:
        kpi_card("Retained Customers", f"{m['retained_customers']:,}",
                 f"{100-m['churn_rate']:.1f}% retention rate", variant="success")
    with k4:
        kpi_card("Avg Lifetime Value", f"${m['avg_ltv']:,.0f}",
                 f"Avg Order ${m['avg_aov']:,.0f}")

    st.markdown('<div style="margin:6px 0 0 0"></div>', unsafe_allow_html=True)
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        kpi_card("Churn Rate", f"{m['churn_rate']:.1f}%",
                 "Target: <20% for healthy retention", variant="danger" if m['churn_rate'] > 25 else "warn")
    with k6:
        kpi_card("Avg Order Value", f"${m['avg_aov']:,.0f}", f"Avg purchases {m['avg_purchases']:.1f}")
    with k7:
        kpi_card("Avg Membership", f"{m['avg_membership']:.1f} yrs", "Years on platform")
    with k8:
        kpi_card("Avg Engagement", f"{m['avg_engagement']:.1f}",
                 "Composite score 0-100",
                 variant="success" if m['avg_engagement'] > 50 else "warn")

    # ── Row 2: Churn overview + Churn by country ──
    st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
    section_header("📊 Churn Overview")
    c1, c2 = st.columns(2)
    with c1:
        with chart_card("Customer Retention vs. Churn", "Share of retained vs. churned customers"):
            cts = filtered_df[TARGET].value_counts().reset_index()
            cts.columns = ["Churned","Count"]
            cts["Status"] = cts["Churned"].map({0:"Retained", 1:"Churned"})
            fig = px.pie(
                cts, values="Count", names="Status",
                color="Status",
                color_discrete_map={"Retained": C_RETAIN, "Churned": C_CHURN},
                hole=0.5
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(_fig_style(fig, 360), use_container_width=True)
    with c2:
        with chart_card("Churn Rate by Country", "Which geographies have the highest churn rate"):
            ct = filtered_df.groupby("Country")[TARGET].agg(["sum","count"]).reset_index()
            ct.columns = ["Country","Churned","Total"]
            ct["Churn_Rate"] = ct["Churned"]/ct["Total"]*100
            ct = ct.sort_values("Churn_Rate", ascending=False)
            fig2 = px.bar(
                ct, x="Country", y="Churn_Rate",
                color="Churn_Rate", color_continuous_scale=[[0,"#fef2f2"],[1,C_CHURN]],
                labels={"Churn_Rate":"Churn Rate (%)"}
            )
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_fig_style(fig2, 360), use_container_width=True)

    # ── Row 3: Customer value + Engagement vs churn ──
    section_header("💼 Customer Value & Engagement")
    c3, c4 = st.columns(2)
    with c3:
        if "Customer_Value_Category" in filtered_df.columns:
            with chart_card("Customers by Value Category", "Distribution across value tiers"):
                vc = filtered_df["Customer_Value_Category"].value_counts().reset_index()
                vc.columns = ["Category","Count"]
                fig3 = px.bar(
                    vc, x="Category", y="Count",
                    color="Category",
                    color_discrete_sequence=PALETTE,
                    labels={"Count":"Number of Customers"}
                )
                fig3.update_layout(showlegend=False)
                st.plotly_chart(_fig_style(fig3, 320), use_container_width=True)
    with c4:
        if "Engagement_Score" in filtered_df.columns:
            with chart_card("Engagement Score by Churn Status", "Higher engagement is associated with lower churn"):
                eng_col = filtered_df[TARGET].map({0:"Retained", 1:"Churned"})
                fig4 = px.histogram(
                    filtered_df, x="Engagement_Score",
                    color=eng_col, barmode="overlay", nbins=40,
                    color_discrete_map={"Retained": C_RETAIN, "Churned": C_CHURN},
                    labels={"Engagement_Score":"Engagement Score","color":"Status"}
                )
                fig4.update_layout(bargap=0.04, legend_title="Status")
                st.plotly_chart(_fig_style(fig4, 320), use_container_width=True)

    # ── Row 4: Top churn indicators ──
    section_header("🚨 Top Churn Risk Indicators")
    with chart_card("Feature Correlation with Churn (Top 10)", "Absolute Pearson correlation — positive values increase churn risk"):
        corr = filtered_df.select_dtypes(include=[np.number]).corr()[TARGET].drop(TARGET)
        corr_abs = corr.abs().sort_values(ascending=False).head(10)
        corr_df = pd.DataFrame({
            "Feature":       corr_abs.index,
            "Correlation":   corr.loc[corr_abs.index].values,
            "|Correlation|": corr_abs.values
        })
        corr_df["Direction"] = corr_df["Correlation"].apply(
            lambda x: "↑ Increases churn risk" if x > 0 else "↓ Decreases churn risk"
        )
        fig5 = px.bar(
            corr_df.sort_values("|Correlation|", ascending=True),
            x="|Correlation|", y="Feature", orientation="h",
            color="Correlation",
            color_continuous_scale=[[0, C_RETAIN],[0.5,"#f1f5f9"],[1, C_CHURN]],
            hover_data=["Direction"],
            labels={"|Correlation|":"Absolute Correlation Coefficient"}
        )
        fig5.update_layout(coloraxis_colorbar_title="r")
        st.plotly_chart(_fig_style(fig5, 420), use_container_width=True)

    # ── Row 5: Insight summary cards ──
    section_header("💡 Key Insights")
    ins_data = generate_business_insights(filtered_df)
    i1, i2, i3 = st.columns(3)
    with i1:
        d = ins_data[2]  # Customer service
        insight_card(d["icon"], d["type"], d["title"], d["observation"], d["implication"], d["recommendation"], d["cls"])
    with i2:
        d = ins_data[1]  # Cart abandonment
        insight_card(d["icon"], d["type"], d["title"], d["observation"], d["implication"], d["recommendation"], d["cls"])
    with i3:
        d = ins_data[3]  # Login engagement
        insight_card(d["icon"], d["type"], d["title"], d["observation"], d["implication"], d["recommendation"], d["cls"])


# ══════════════════════════════════════════════════════════
# PAGE: CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════
def render_customer_analytics(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("👥 Customer Analytics", "Explore demographics, engagement, and purchasing behaviour",
                connected=True, n_rows=n_total, n_cols=n_cols_total)

    if len(df) == 0:
        st.warning("No data to display with current filters.")
        return

    tabs = st.tabs([
        "🗺️ Demographics",
        "🤝 Membership",
        "💬 Engagement",
        "🛍️ Purchases",
        "💎 Customer Value",
    ])

    # ── Tab 0: Demographics ──────────────────────────────────
    with tabs[0]:
        tab_section("Customer Distribution")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Age Distribution</div>'
                            '<div class="chart-card-subtitle">Distribution of customer ages across the filtered dataset.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    plot_histogram(df, "Age", "", color=C_ACCENT, nbins=30, h=300),
                    use_container_width=True
                )
        with c2:
            with st.container(border=True):
                gc = df["Gender"].value_counts().reset_index()
                gc.columns = ["Gender", "Count"]
                fig = px.pie(gc, values="Count", names="Gender",
                             color_discrete_sequence=PALETTE)
                st.markdown('<div class="chart-card-title">Gender Distribution</div>'
                            '<div class="chart-card-subtitle">Proportion of customers by gender category.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(_fig_style(fig, 300), use_container_width=True)

        tab_section("Geographic Spread")
        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                cc = df["Country"].value_counts().reset_index()
                cc.columns = ["Country", "Count"]
                fig = px.bar(cc, x="Country", y="Count",
                             color="Country", color_discrete_sequence=PALETTE,
                             labels={"Count": "Customers"})
                fig.update_layout(showlegend=False)
                st.markdown('<div class="chart-card-title">Customers by Country</div>'
                            '<div class="chart-card-subtitle">Total customer count per country.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(_fig_style(fig, 300), use_container_width=True)
        with c4:
            with st.container(border=True):
                tc = df["City"].value_counts().head(15).reset_index()
                tc.columns = ["City", "Count"]
                fig = px.bar(tc, x="Count", y="City", orientation="h",
                             color="Count",
                             color_continuous_scale=[[0, "#dbeafe"], [1, C_ACCENT]],
                             labels={"Count": "Customers"})
                fig.update_layout(coloraxis_showscale=False)
                st.markdown('<div class="chart-card-title">Top 15 Cities</div>'
                            '<div class="chart-card-subtitle">Cities with the highest number of customers.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(_fig_style(fig, 400), use_container_width=True)

    # ── Tab 1: Membership ────────────────────────────────────
    with tabs[1]:
        tab_section("Membership Tenure")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Membership Years Distribution</div>'
                            '<div class="chart-card-subtitle">How long customers have been members.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    plot_histogram(df, "Membership_Years", "", h=300),
                    use_container_width=True
                )
        with c2:
            if "Membership_Category" in df.columns:
                with st.container(border=True):
                    mc = df["Membership_Category"].value_counts().reset_index()
                    mc.columns = ["Category", "Count"]
                    fig = px.bar(mc, x="Category", y="Count",
                                 color="Category", color_discrete_sequence=PALETTE,
                                 labels={"Count": "Customers"})
                    fig.update_layout(showlegend=False)
                    st.markdown('<div class="chart-card-title">Membership Category Distribution</div>'
                                '<div class="chart-card-subtitle">Customers grouped by tenure category.</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(_fig_style(fig, 300), use_container_width=True)

        tab_section("Churn by Membership")
        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Churn Rate by Signup Quarter</div>'
                            '<div class="chart-card-subtitle">Stacked share of churned vs retained customers per signup cohort.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    plot_churn_by_category(df, "Signup_Quarter", "", h=300),
                    use_container_width=True
                )
        with c4:
            if "Membership_Category" in df.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-card-title">Churn Rate by Membership Category</div>'
                                '<div class="chart-card-subtitle">Proportion of churned customers across membership tiers.</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(
                        plot_churn_by_category(df, "Membership_Category", "", h=300),
                        use_container_width=True
                    )

    # ── Tab 2: Engagement ────────────────────────────────────
    with tabs[2]:
        tab_section("Engagement Metrics vs. Churn Status")
        pairs = [
            ("Login_Frequency",            "Login Frequency",         "Average monthly login count per customer."),
            ("Session_Duration_Avg",        "Avg Session Duration",    "Mean session length in minutes."),
            ("Pages_Per_Session",           "Pages Per Session",       "Average pages viewed each visit."),
            ("Email_Open_Rate",             "Email Open Rate (%)",     "Percentage of marketing emails opened."),
            ("Social_Media_Engagement_Score","Social Media Score",     "Composite social media engagement index."),
            ("Mobile_App_Usage",            "Mobile App Usage",        "Percentage of sessions via the mobile app."),
            ("Product_Reviews_Written",     "Product Reviews Written", "Number of reviews submitted."),
        ]
        for i in range(0, len(pairs), 2):
            c1, c2 = st.columns(2)
            with c1:
                col, lbl, sub = pairs[i]
                if col in df.columns:
                    with st.container(border=True):
                        st.markdown(f'<div class="chart-card-title">{lbl}</div>'
                                    f'<div class="chart-card-subtitle">{sub}</div>',
                                    unsafe_allow_html=True)
                        st.plotly_chart(
                            plot_boxplot(df, col, "", h=300),
                            use_container_width=True
                        )
            with c2:
                if i + 1 < len(pairs):
                    col2, lbl2, sub2 = pairs[i + 1]
                    if col2 in df.columns:
                        with st.container(border=True):
                            st.markdown(f'<div class="chart-card-title">{lbl2}</div>'
                                        f'<div class="chart-card-subtitle">{sub2}</div>',
                                        unsafe_allow_html=True)
                            st.plotly_chart(
                                plot_boxplot(df, col2, "", h=300),
                                use_container_width=True
                            )

    # ── Tab 3: Purchases ─────────────────────────────────────
    with tabs[3]:
        tab_section("Volume & Value")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Total Purchases</div>'
                            '<div class="chart-card-subtitle">Number of completed purchases per customer.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Total_Purchases", "", h=300), use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Average Order Value</div>'
                            '<div class="chart-card-subtitle">Mean spend per order in USD.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Average_Order_Value", "", h=300), use_container_width=True)

        tab_section("Recency & Behaviour")
        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Days Since Last Purchase</div>'
                            '<div class="chart-card-subtitle">Recency metric — higher values indicate more lapsed customers.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Days_Since_Last_Purchase", "", h=300), use_container_width=True)
        with c4:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Cart Abandonment Rate</div>'
                            '<div class="chart-card-subtitle">Percentage of shopping carts left without purchase.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Cart_Abandonment_Rate", "", h=300), use_container_width=True)

        tab_section("Returns & Discounts")
        c5, c6 = st.columns(2)
        with c5:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Returns Rate</div>'
                            '<div class="chart-card-subtitle">Proportion of purchased items returned.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Returns_Rate", "", h=300), use_container_width=True)
        with c6:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Discount Usage Rate</div>'
                            '<div class="chart-card-subtitle">Percentage of orders placed using a discount code.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Discount_Usage_Rate", "", h=300), use_container_width=True)

        if "Purchase_Value" in df.columns:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Estimated Purchase Value</div>'
                            '<div class="chart-card-subtitle">Derived metric: Total Purchases × Average Order Value. '
                            'An approximation of total spend — not an exact recorded revenue figure.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Purchase_Value", "", h=300), use_container_width=True)

    # ── Tab 4: Customer Value ────────────────────────────────
    with tabs[4]:
        st.markdown(
            '<div class="tab-section-title">💎 Customer Value Analysis</div>'
            '<div style="font-size:0.82rem;color:#64748b;margin:-6px 0 16px 0;line-height:1.55;">'
            'Compare Lifetime Value and Credit Balance between churned and retained customers, '
            'and examine how churn rates vary across customer value tiers.'
            '</div>',
            unsafe_allow_html=True
        )

        # Row 1: LTV card  |  Credit Balance card
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Customer Lifetime Value</div>'
                            '<div class="chart-card-subtitle">Distribution of LTV ($) split by churn status.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Lifetime_Value", "", h=300), use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Credit Balance</div>'
                            '<div class="chart-card-subtitle">Customer credit balance ($) by churn status.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, "Credit_Balance", "", h=300), use_container_width=True)

        # Row 2: full-width churn-by-value-category
        if "Customer_Value_Category" in df.columns:
            with st.container(border=True):
                st.markdown('<div class="chart-card-title">Churn Rate by Customer Value Category</div>'
                            '<div class="chart-card-subtitle">Stacked proportion of churned vs retained customers '
                            'across LTV-based value tiers (Low → Premium). '
                            'Identifies which value segments are most at risk.</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    plot_churn_by_category(df, "Customer_Value_Category", "", h=320),
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════
# PAGE: CHURN ANALYSIS
# ══════════════════════════════════════════════════════════
def render_churn_analysis(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("⚠️ Churn Analysis",
                "Understand factors associated with customer churn",
                connected=True, n_rows=n_total, n_cols=n_cols_total)
    if len(df) == 0:
        st.warning("No data to display with current filters.")
        return

    st.markdown(
        '<div class="disclaimer" style="background:#f0f9ff;border-color:#bae6fd;color:#0c4a6e;">'
        '📌 <strong>Observational Analysis:</strong> Associations shown are correlational, not causal. '
        'We use the phrase "associated with churn" rather than "causes churn" throughout.'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Summary KPIs ──
    m = calculate_metrics(df)
    st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
    section_header("📊 Churn KPIs")
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Churn Rate",        f"{m['churn_rate']:.1f}%",   "Overall", variant="danger")
    with k2: kpi_card("Churned",           f"{m['churned_customers']:,}", "Customers lost", variant="danger")
    with k3: kpi_card("Retained",          f"{m['retained_customers']:,}", "Active customers", variant="success")
    with k4: kpi_card("Avg LTV (Churned)",
                      f"${df[df[TARGET]==1]['Lifetime_Value'].mean():,.0f}",
                      f"vs ${df[df[TARGET]==0]['Lifetime_Value'].mean():,.0f} retained")

    # ── Summary stats table ──
    section_header("📋 Churned vs. Retained — Key Metrics")
    cs = df.groupby(TARGET).agg(
        Count=(TARGET,"count"),
        Avg_Age=("Age","mean"),
        Avg_Login=("Login_Frequency","mean"),
        Avg_Cart_Abandon=("Cart_Abandonment_Rate","mean"),
        Avg_Service_Calls=("Customer_Service_Calls","mean"),
        Avg_Days_Recency=("Days_Since_Last_Purchase","mean"),
        Avg_Email_Open=("Email_Open_Rate","mean"),
        Avg_LTV=("Lifetime_Value","mean"),
    ).reset_index()
    cs[TARGET] = cs[TARGET].map({0:"Retained", 1:"Churned"})
    cs = cs.rename(columns={TARGET:"Status"})
    st.dataframe(cs.round(2).set_index("Status"), use_container_width=True)

    st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
    tabs = st.tabs(["🗺️ Demographics", "💬 Engagement", "🛍️ Purchase & Recency", "📞 Customer Service", "📈 Correlation"])

    with tabs[0]:
        tab_section("CHURN BY DEMOGRAPHICS")
        c1, c2 = st.columns(2)
        with c1:
            with chart_card("Churn Rate by Gender", "Comparison of churn rates across gender groups"):
                st.plotly_chart(plot_churn_by_category(df, "Gender", ""), use_container_width=True)
        with c2:
            with chart_card("Churn Rate by Country", "Geographic churn rate distribution"):
                st.plotly_chart(plot_churn_by_category(df, "Country", ""), use_container_width=True)

        tab_section("AGE & MEMBERSHIP")
        c3, c4 = st.columns(2)
        with c3:
            with chart_card("Age Distribution by Churn Status", "Boxplot comparing age spread for churned vs. retained"):
                st.plotly_chart(plot_boxplot(df, "Age", ""), use_container_width=True)
        with c4:
            if "Membership_Category" in df.columns:
                with chart_card("Churn by Membership Category", "Churn rate across New / Regular / Loyal tiers"):
                    st.plotly_chart(plot_churn_by_category(df, "Membership_Category", ""), use_container_width=True)

    with tabs[1]:
        tab_section("DIGITAL ENGAGEMENT")
        pairs = [
            ("Login_Frequency",              "Login Frequency",       "Sessions logged per month"),
            ("Session_Duration_Avg",          "Avg Session Duration",  "Average minutes per visit"),
            ("Pages_Per_Session",             "Pages Per Session",     "Pages browsed per visit"),
            ("Email_Open_Rate",               "Email Open Rate",       "Share of marketing emails opened"),
            ("Social_Media_Engagement_Score", "Social Media Score",    "Composite social interaction score"),
            ("Mobile_App_Usage",              "Mobile App Usage",      "Mobile sessions per month"),
        ]
        for i in range(0, len(pairs), 2):
            c1, c2 = st.columns(2)
            with c1:
                col, lbl, sub = pairs[i]
                if col in df.columns:
                    with chart_card(f"{lbl} by Churn Status", sub):
                        st.plotly_chart(plot_boxplot(df, col, ""), use_container_width=True)
            with c2:
                if i + 1 < len(pairs):
                    col2, lbl2, sub2 = pairs[i + 1]
                    if col2 in df.columns:
                        with chart_card(f"{lbl2} by Churn Status", sub2):
                            st.plotly_chart(plot_boxplot(df, col2, ""), use_container_width=True)

    with tabs[2]:
        tab_section("PURCHASE BEHAVIOUR")
        c1, c2 = st.columns(2)
        with c1:
            with chart_card("Total Purchases by Churn Status", "Number of orders placed — churned vs. retained"):
                st.plotly_chart(plot_boxplot(df, "Total_Purchases", ""), use_container_width=True)
        with c2:
            with chart_card("Days Since Last Purchase", "Recency — how long since the last order"):
                st.plotly_chart(plot_boxplot(df, "Days_Since_Last_Purchase", ""), use_container_width=True)

        tab_section("CART & RETURNS")
        c3, c4 = st.columns(2)
        with c3:
            with chart_card("Cart Abandonment Rate by Churn Status", "Proportion of sessions ending without purchase"):
                st.plotly_chart(plot_boxplot(df, "Cart_Abandonment_Rate", ""), use_container_width=True)
        with c4:
            with chart_card("Returns Rate by Churn Status", "Share of purchased items returned"):
                st.plotly_chart(plot_boxplot(df, "Returns_Rate", ""), use_container_width=True)

        if "Recency_Category" in df.columns:
            tab_section("RECENCY SEGMENTS")
            with chart_card("Churn Rate by Recency Category", "Customers grouped by days since last purchase"):
                st.plotly_chart(plot_churn_by_category(df, "Recency_Category", ""), use_container_width=True)

    with tabs[3]:
        tab_section("CUSTOMER SERVICE CALLS")
        c1, c2 = st.columns(2)
        with c1:
            with chart_card("Service Calls Distribution by Churn Status", "Churned customers tend to contact support more"):
                st.plotly_chart(plot_boxplot(df, "Customer_Service_Calls", ""), use_container_width=True)
        with c2:
            with chart_card("Churn Rate vs. Service Calls Count", "How churn rate rises with each additional support call"):
                grp = df.groupby("Customer_Service_Calls")[TARGET].mean().reset_index()
                grp.columns = ["Service Calls", "Churn Rate"]
                grp["Churn Rate"] *= 100
                fig = px.line(grp, x="Service Calls", y="Churn Rate",
                              markers=True, color_discrete_sequence=[C_CHURN],
                              labels={"Churn Rate": "Churn Rate (%)"})
                st.plotly_chart(_fig_style(fig, 300), use_container_width=True)

    with tabs[4]:
        tab_section("CORRELATION HEATMAP")
        with chart_card("Feature Correlation Heatmap", "Pearson correlations across all numerical features"):
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

        tab_section("CORRELATION WITH CHURN")
        corr = df.select_dtypes(include=[np.number]).corr()[TARGET].drop(TARGET)
        cdf  = corr.abs().sort_values(ascending=False).reset_index()
        cdf.columns = ["Feature", "|Correlation|"]
        cdf["Correlation"] = corr.loc[cdf["Feature"]].values
        cdf["Direction"] = cdf["Correlation"].apply(
            lambda x: "Positive (increases churn risk)" if x > 0 else "Negative (decreases churn risk)"
        )
        st.dataframe(cdf.round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════
def render_segmentation(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("🎯 Customer Segmentation",
                "K-Means clustering — group customers into meaningful behavioural segments",
                connected=True, n_rows=n_total, n_cols=n_cols_total)
    if len(df) < 50:
        st.warning("Not enough data for meaningful clustering.")
        return

    st.markdown(
        '<div class="card" style="font-size:0.83rem;color:#64748b;">'
        '<strong style="color:#0f172a">Clustering features:</strong> '
        + ", ".join([f'<span class="tag">{f}</span>' for f in SEGMENT_FEATURES])
        + '<br><br>⚠️ <code>Churned</code> is <strong>NOT</strong> used as a clustering input — '
        'it is shown post-hoc to validate segment quality.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Computing elbow & silhouette scores…"):
        k_range, inertias, sil_scores = compute_elbow_silhouette(df)

    section_header("📐 Optimal K Selection")
    c1, c2 = st.columns(2)
    with c1:
        with chart_card("Elbow Method — Inertia vs. K", "Look for the 'elbow' where inertia flattens"):
            fig_e = px.line(x=k_range, y=inertias, markers=True,
                            labels={"x": "K (Clusters)", "y": "Inertia"},
                            color_discrete_sequence=[C_ACCENT])
            st.plotly_chart(_fig_style(fig_e, 300), use_container_width=True)
    with c2:
        with chart_card("Silhouette Score vs. K", "Higher silhouette = better-defined clusters"):
            fig_s = px.line(x=k_range, y=sil_scores, markers=True,
                            labels={"x": "K (Clusters)", "y": "Silhouette Score"},
                            color_discrete_sequence=[C_WARN])
            st.plotly_chart(_fig_style(fig_s, 300), use_container_width=True)

    best_k = k_range[int(np.argmax(sil_scores))]
    n_clusters = st.slider("Select number of clusters (K):", 2, 8, best_k)

    with st.spinner("Running K-Means…"):
        df_seg, _, _ = create_segments(df, n_clusters)

    seg_names = name_segments(df_seg)
    df_seg["Segment"] = df_seg["Cluster"].map(seg_names)

    seg_sum = df_seg.groupby("Segment").agg(
        Customers=("Cluster", "count"),
        Avg_LTV=("Lifetime_Value", "mean"),
        Avg_Login=("Login_Frequency", "mean"),
        Avg_Purchases=("Total_Purchases", "mean"),
        Avg_Cart_Abandon=("Cart_Abandonment_Rate", "mean"),
        Avg_Service_Calls=("Customer_Service_Calls", "mean"),
        Churn_Rate=(TARGET, "mean"),
    ).reset_index()
    seg_sum["Churn_Rate"] = (seg_sum["Churn_Rate"] * 100).round(1)
    seg_sum[["Avg_LTV", "Avg_Login", "Avg_Purchases", "Avg_Cart_Abandon", "Avg_Service_Calls"]] = \
        seg_sum[["Avg_LTV", "Avg_Login", "Avg_Purchases", "Avg_Cart_Abandon", "Avg_Service_Calls"]].round(1)

    section_header("📋 Segment Profiles")
    st.dataframe(seg_sum.set_index("Segment"), use_container_width=True)

    section_header("📊 Segment Metrics")
    c3, c4 = st.columns(2)
    with c3:
        with chart_card("Churn Rate by Segment (%)", "Segments with the highest churn risk"):
            fig = px.bar(seg_sum, x="Segment", y="Churn_Rate",
                         color="Churn_Rate",
                         color_continuous_scale=[[0, "#fef2f2"], [1, C_CHURN]],
                         labels={"Churn_Rate": "Churn Rate (%)"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_fig_style(fig, 300), use_container_width=True)
    with c4:
        with chart_card("Average Lifetime Value by Segment ($)", "Economic value of each customer cluster"):
            fig = px.bar(seg_sum, x="Segment", y="Avg_LTV",
                         color="Avg_LTV",
                         color_continuous_scale=[[0, "#dbeafe"], [1, C_ACCENT]],
                         labels={"Avg_LTV": "Avg LTV ($)"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_fig_style(fig, 300), use_container_width=True)

    # ── PCA projection ──
    seg_df_vis = df[SEGMENT_FEATURES].copy()
    imp = SimpleImputer(strategy="median")
    seg_arr = imp.fit_transform(seg_df_vis)
    sc  = StandardScaler()
    seg_sc = sc.fit_transform(seg_arr)
    pca = PCA(n_components=2, random_state=42)
    pca_r = pca.fit_transform(seg_sc)
    pca_df = pd.DataFrame({
        "PC1": pca_r[:, 0], "PC2": pca_r[:, 1],
        "Segment": df_seg["Segment"].values,
        "Churned": df_seg[TARGET].map({0: "Retained", 1: "Churned"}).values
    })
    sample = pca_df.sample(min(5000, len(pca_df)), random_state=42)

    section_header("🔵 PCA — 2D Segment Projection")
    c5, c6 = st.columns(2)
    with c5:
        with chart_card("Customer Segments (PCA Projection)", f"PC1 + PC2 explain {sum(pca.explained_variance_ratio_)*100:.1f}% of variance"):
            fig_p = px.scatter(sample, x="PC1", y="PC2", color="Segment",
                               opacity=0.55, color_discrete_sequence=PALETTE,
                               labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)",
                                       "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)"})
            fig_p.update_traces(marker=dict(size=3))
            st.plotly_chart(_fig_style(fig_p, 380), use_container_width=True)
    with c6:
        with chart_card("Churn Status (PCA Projection)", "Spatial overlap shows churn is not purely cluster-driven"):
            fig_p2 = px.scatter(sample, x="PC1", y="PC2", color="Churned",
                                opacity=0.45,
                                color_discrete_map={"Retained": C_RETAIN, "Churned": C_CHURN})
            fig_p2.update_traces(marker=dict(size=3))
            st.plotly_chart(_fig_style(fig_p2, 380), use_container_width=True)

    # ── Engagement by segment ──
    section_header("💬 Engagement Patterns by Segment")
    with chart_card("Engagement Metrics Breakdown by Segment", "Grouped bars — each metric averaged per cluster"):
        seg_eng = df_seg.groupby("Segment")[
            ["Login_Frequency", "Email_Open_Rate", "Social_Media_Engagement_Score",
             "Mobile_App_Usage", "Pages_Per_Session"]
        ].mean().reset_index()
        seg_melt = seg_eng.melt(id_vars="Segment", var_name="Metric", value_name="Value")
        fig_e2 = px.bar(seg_melt, x="Segment", y="Value", color="Metric",
                        barmode="group", color_discrete_sequence=PALETTE)
        st.plotly_chart(_fig_style(fig_e2, 360), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: MACHINE LEARNING
# ══════════════════════════════════════════════════════════
def render_machine_learning(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("🤖 Machine Learning",
                "Binary churn classification — model training, evaluation, and feature analysis",
                connected=True, n_rows=n_total, n_cols=n_cols_total)
    if len(df) == 0:
        st.warning("No data available.")
        return

    leakage = check_data_leakage(df)

    with st.expander("🔒 Data Leakage Assessment", expanded=False):
        if leakage["high_correlation_features"]:
            st.warning(f"High-correlation features (|r|>0.8): {leakage['high_correlation_features']}")
        else:
            st.success("✅ No features with |r| > 0.8 to target detected.")
        st.markdown(f"**Lifetime_Value:** {leakage['lifetime_value_flag']}")
        st.markdown(f"**Purchase_Value:** {leakage['purchase_value_note']}")
        st.dataframe(leakage["correlation_table"].round(4), use_container_width=True)

    with st.expander("🔍 Selected ML Features", expanded=False):
        st.markdown('<strong>Numerical (19):</strong> '
                    + " ".join([f'<span class="tag">{f}</span>' for f in ML_NUMERICAL_FEATURES]),
                    unsafe_allow_html=True)
        st.markdown('<strong>Categorical (3):</strong> '
                    + " ".join([f'<span class="tag">{f}</span>' for f in ML_CATEGORICAL_FEATURES]),
                    unsafe_allow_html=True)
        st.info("`Lifetime_Value` excluded (leakage risk). `City` excluded (high cardinality). All derived features excluded.")

    a_num = [f for f in ML_NUMERICAL_FEATURES  if f in df.columns]
    a_cat = [f for f in ML_CATEGORICAL_FEATURES if f in df.columns]

    with st.spinner("Training models — please wait…"):
        X_tr, X_te, y_tr, y_te, a_num, a_cat = prepare_ml_data(df)
        trained = train_models(X_tr, y_tr, a_num, a_cat)

    eval_df = evaluate_models(trained, X_te, y_te)

    # ── Model performance cards ──
    section_header("📊 Model Performance Summary")
    st.info(
        "**Business note:** In churn prediction **Recall** (catching actual churners) "
        "often matters more than Accuracy. **ROC-AUC** is the primary threshold-independent metric."
    )
    model_cols = st.columns(len(trained))
    for i, (name, pipe) in enumerate(trained.items()):
        row = eval_df[eval_df["Model"] == name].iloc[0]
        with model_cols[i]:
            model_metric_card(name, {
                "Accuracy":  row["Accuracy"],
                "Precision": row["Precision"],
                "Recall":    row["Recall"],
                "F1 Score":  row["F1 Score"],
                "ROC-AUC":   row["ROC-AUC"],
            })

    # ── Full metrics table ──
    section_header("📋 Full Evaluation Table")
    with st.container(border=True):
        styled = eval_df.set_index("Model").style.format(
            {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
             "F1 Score": "{:.4f}", "ROC-AUC": "{:.4f}"}
        ).background_gradient(subset=["ROC-AUC", "F1 Score", "Recall"], cmap="Blues")
        st.dataframe(styled, use_container_width=True)

    # ── ROC Curves ──
    section_header("📈 ROC Curves")
    with chart_card("ROC Curves — All Models", "Area Under Curve (AUC) measures discrimination power across all thresholds"):
        st.plotly_chart(plot_roc_curves(trained, X_te, y_te), use_container_width=True)

    # ── Per-model: Confusion + Feature Importance ──
    section_header("🔎 Confusion Matrices & Feature Importance")
    model_tabs = st.tabs(list(trained.keys()))
    for i, (name, pipe) in enumerate(trained.items()):
        with model_tabs[i]:
            c1, c2 = st.columns(2)
            with c1:
                with chart_card("Confusion Matrix", "Actual vs. predicted classifications on the test set"):
                    y_pred = pipe.predict(X_te)
                    st.plotly_chart(plot_confusion_matrix(y_te, y_pred, name), use_container_width=True)
            with c2:
                fig_fi = plot_feature_importance(pipe, a_num + a_cat, name)
                if fig_fi:
                    with chart_card("Feature Importance", "Top predictors ranked by model-specific importance score"):
                        st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            with st.expander(f"Classification Report — {name}"):
                st.code(classification_report(y_te, pipe.predict(X_te), target_names=["Retained", "Churned"]))

    # ── Save best model ──
    best_name = eval_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    best_pipe  = trained[best_name]
    save_model(best_pipe)
    best_auc = eval_df[eval_df["Model"] == best_name]["ROC-AUC"].values[0]
    st.success(
        f"✅ Best model by ROC-AUC: **{best_name}** (AUC = {best_auc:.4f}) — saved to `{MODEL_PATH}`"
    )

    st.session_state["trained_models"]  = trained
    st.session_state["best_model_name"] = best_name
    st.session_state["ml_num_features"] = a_num
    st.session_state["ml_cat_features"] = a_cat
    st.session_state["ml_X_te"] = X_te
    st.session_state["ml_y_te"] = y_te


# ══════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTION
# ══════════════════════════════════════════════════════════
def render_prediction(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("🔮 Churn Prediction",
                "Enter customer profile to receive an estimated churn probability",
                connected=True, n_rows=n_total, n_cols=n_cols_total)

    trained = st.session_state.get("trained_models")
    if trained is None:
        saved = load_model()
        if saved is not None:
            trained = {"Saved Model": saved}
            a_num = [f for f in ML_NUMERICAL_FEATURES  if f in df.columns]
            a_cat = [f for f in ML_CATEGORICAL_FEATURES if f in df.columns]
            st.session_state["trained_models"]  = trained
            st.session_state["ml_num_features"] = a_num
            st.session_state["ml_cat_features"] = a_cat
        else:
            st.warning("⚠️ Models not yet trained. Navigate to **🤖 Machine Learning** first.")
            return

    a_num = st.session_state.get("ml_num_features", [f for f in ML_NUMERICAL_FEATURES  if f in df.columns])
    a_cat = st.session_state.get("ml_cat_features", [f for f in ML_CATEGORICAL_FEATURES if f in df.columns])

    left_col, right_col = st.columns([1.25, 0.75], gap="large")

    with left_col:
        st.markdown('<div class="pg-title">Customer Feature Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pg-subtitle">Only the final selected ML features are required. '
            'Values default to dataset medians.</div>',
            unsafe_allow_html=True
        )

        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            model_choice = st.selectbox("Model", list(trained.keys()))
        with ctrl2:
            threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                                  help="Probability ≥ threshold → Predicted Churned")

        with st.form("churn_pred_form"):
            num_inputs, cat_inputs = {}, {}
            # Render numerical features in groups of 4
            for batch_start in range(0, len(a_num), 4):
                batch = a_num[batch_start: batch_start+4]
                cols  = st.columns(len(batch))
                for col_widget, feat in zip(cols, batch):
                    with col_widget:
                        fmin = float(df[feat].min())  if feat in df.columns else 0.0
                        fmax = float(df[feat].max())  if feat in df.columns else 100.0
                        fmed = float(df[feat].median()) if feat in df.columns else (fmin+fmax)/2
                        num_inputs[feat] = st.number_input(
                            feat.replace("_"," "), min_value=fmin, max_value=fmax,
                            value=fmed, step=(fmax-fmin)/100, format="%.2f"
                        )

            # Categorical features
            if a_cat:
                cat_cols = st.columns(len(a_cat))
                for col_widget, feat in zip(cat_cols, a_cat):
                    with col_widget:
                        opts = sorted(df[feat].dropna().unique().tolist()) if feat in df.columns else []
                        cat_inputs[feat] = st.selectbox(feat.replace("_"," "), opts)

            submitted = st.form_submit_button("🔮 Predict Churn Probability", use_container_width=True)

    with right_col:
        if submitted:
            input_df = pd.DataFrame([{**num_inputs, **cat_inputs}])
            try:
                res      = predict_churn(trained[model_choice], input_df, threshold)
                prob     = res["probability"]
                risk_cls = res["risk_cls"]
                verdict  = "⚠️ Likely to Churn" if res["prediction"] == 1 else "✅ Likely to Retain"

                prediction_result_card(prob, risk_cls, res["risk_level"], res["prediction"])

                # Gauge
                bar_color = C_CHURN if risk_cls == "high" else (C_WARN if risk_cls == "medium" else C_RETAIN)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    title={"text":"Churn Probability", "font":{"size":13}},
                    gauge={
                        "axis": {"range":[0,100], "tickfont":{"size":10}},
                        "bar":  {"color": bar_color},
                        "steps": [
                            {"range":[0,35],  "color":"#dcfce7"},
                            {"range":[35,65], "color":"#fef3c7"},
                            {"range":[65,100],"color":"#fee2e2"},
                        ],
                        "threshold": {
                            "line":{"color":"#0f172a","width":2.5},
                            "thickness":0.8,
                            "value": threshold*100
                        }
                    }
                ))
                fig_g.update_layout(
                    height=250, margin=dict(l=20,r=20,t=40,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a")
                )
                st.plotly_chart(fig_g, use_container_width=True)

                st.markdown(
                    '<div class="disclaimer">'
                    '⚠️ <strong>Disclaimer:</strong> This probability is a model estimate based on historical '
                    'data. It is not a guarantee of future behaviour. Business decisions should incorporate '
                    'domain expertise alongside this score.'
                    '</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            # Placeholder prompt
            st.markdown(
                '<div class="card" style="text-align:center;padding:48px 24px;">'
                '  <div style="font-size:3rem;margin-bottom:12px">🔮</div>'
                '  <div style="font-size:1.05rem;font-weight:600;color:#0f172a">Awaiting Input</div>'
                '  <div style="font-size:0.82rem;color:#64748b;margin-top:6px">'
                '    Fill in the customer profile on the left and click Predict.'
                '  </div>'
                '</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════
def render_business_insights(df: pd.DataFrame, n_total: int, n_cols_total: int):
    page_header("💡 Business Insights",
                "Data-driven observations, interpretations, and retention recommendations",
                connected=True, n_rows=n_total, n_cols=n_cols_total)
    if len(df) == 0:
        st.warning("No data to display.")
        return

    st.markdown(
        '<div class="disclaimer" style="background:#f0fdf4;border-color:#bbf7d0;color:#14532d;">'
        '✅ All insights are derived from <strong>actual dataset statistics</strong>. '
        'Associations are observational. No fabricated findings are presented.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

    insights = generate_business_insights(df)
    # Row 1: 2 cols
    for i in range(0, len(insights), 2):
        c1, c2 = st.columns(2)
        with c1:
            d = insights[i]
            insight_card(d["icon"],d["type"],d["title"],d["observation"],d["implication"],d["recommendation"],d["cls"])
        with c2:
            if i+1 < len(insights):
                d = insights[i+1]
                insight_card(d["icon"],d["type"],d["title"],d["observation"],d["implication"],d["recommendation"],d["cls"])

    # ── Churn Risk Matrix ──
    section_header("🎯 Churn Risk Priority Matrix")
    df2 = df.copy()
    df2["High_Cart"]    = df2["Cart_Abandonment_Rate"] > df2["Cart_Abandonment_Rate"].median()
    df2["High_Service"] = df2["Customer_Service_Calls"] > df2["Customer_Service_Calls"].median()
    df2["Risk_Segment"] = "Low Risk"
    df2.loc[df2["High_Cart"] | df2["High_Service"], "Risk_Segment"] = "Medium Risk"
    df2.loc[df2["High_Cart"] & df2["High_Service"], "Risk_Segment"] = "High Risk"
    rs = df2.groupby("Risk_Segment").agg(
        Customers=("Risk_Segment","count"),
        Churn_Rate=(TARGET,"mean"),
        Avg_LTV=("Lifetime_Value","mean"),
    ).reset_index()
    rs["Churn_Rate"] = (rs["Churn_Rate"]*100).round(1)
    rs["Avg_LTV"]    = rs["Avg_LTV"].round(0)

    c1, c2 = st.columns(2)
    with c1:
        with chart_card("Churn Rate by Risk Segment (%)", "High-risk customers show significantly elevated churn"):
            fig = px.bar(rs, x="Risk_Segment", y="Churn_Rate",
                         color="Risk_Segment",
                         color_discrete_map={"Low Risk": C_RETAIN, "Medium Risk": C_WARN, "High Risk": C_CHURN},
                         labels={"Churn_Rate": "Churn Rate (%)"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(_fig_style(fig, 300), use_container_width=True)
    with c2:
        with chart_card("Average LTV by Risk Segment ($)", "At-risk customers may still hold significant lifetime value"):
            fig2 = px.bar(rs, x="Risk_Segment", y="Avg_LTV",
                          color="Risk_Segment",
                          color_discrete_map={"Low Risk": C_RETAIN, "Medium Risk": C_WARN, "High Risk": C_CHURN},
                          labels={"Avg_LTV": "Avg Lifetime Value ($)"})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(_fig_style(fig2, 300), use_container_width=True)
    st.dataframe(rs.set_index("Risk_Segment"), use_container_width=True)

    # ── Action Framework ──
    section_header("📋 Retention Action Framework")
    action_df = pd.DataFrame({
        "Priority": ["🔴 High","🔴 High","🟡 Medium","🟡 Medium","🟢 Low"],
        "At-Risk Pattern": [
            "High cart abandonment (>60%) + high service calls (>5)",
            "Days since last purchase > 90 days",
            "Low login frequency (<5/month)",
            "Low email open rate (<20%)",
            "High returns rate (>10%)",
        ],
        "Recommended Action": [
            "Immediate customer success outreach + personalised discount",
            "Win-back email campaign + exclusive returning-customer offer",
            "Personalised push/email re-engagement sequence",
            "Subject-line A/B testing + list segmentation by engagement",
            "Post-purchase follow-up survey + quality review trigger",
        ],
        "Expected Outcome": [
            "Highest churn risk — proactive retention most cost-effective",
            "Lapsed customers — recover before permanent disengagement",
            "Early disengagement signal — intervene before habit breaks",
            "Communication disconnect — improve channel effectiveness",
            "Product dissatisfaction — quality/fit improvement opportunity",
        ]
    })
    st.dataframe(action_df.set_index("Priority"), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: DATA QUALITY
# ══════════════════════════════════════════════════════════
def render_data_quality(raw_df, clean_df, quality_report, n_total, n_cols_total):
    page_header("📋 Data Quality",
                "Dataset validation, missing values, and preprocessing summary",
                connected=True, n_rows=n_total, n_cols=n_cols_total)

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Raw Rows",          f"{quality_report['raw_rows']:,}")
    with k2: kpi_card("Clean Rows",        f"{quality_report['clean_rows']:,}", variant="success")
    with k3: kpi_card("Duplicates Removed",str(quality_report["duplicates_removed"]),
                      variant="warn" if quality_report["duplicates_removed"] > 0 else "success")
    with k4: kpi_card("Missing Cells",     f"{quality_report['total_missing']:,}",
                      variant="warn" if quality_report["total_missing"] > 0 else "success")

    with st.expander("📊 Missing Values by Column", expanded=True):
        mv = pd.Series(quality_report["missing_values"]).sort_values(ascending=False)
        mv_df = mv.reset_index()
        mv_df.columns = ["Column","Missing Count"]
        mv_df["Missing %"] = (mv_df["Missing Count"] / quality_report["raw_rows"] * 100).round(2)
        mv_df = mv_df[mv_df["Missing Count"] > 0]
        if len(mv_df) > 0:
            fig = px.bar(mv_df, x="Column", y="Missing %",
                         title="Missing Value Rate by Column (%)",
                         color="Missing %",
                         color_continuous_scale=[[0,"#fef3c7"],[1,"#d97706"]])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_fig_style(fig, 360), use_container_width=True)
            st.dataframe(mv_df.set_index("Column"), use_container_width=True)
        else:
            st.success("No missing values found.")

    with st.expander("🗂️ Data Types & Sample"):
        st.dataframe(clean_df.dtypes.reset_index().rename(columns={"index":"Column",0:"Dtype"}), use_container_width=True)
        st.dataframe(clean_df.head(5), use_container_width=True)


# ══════════════════════════════════════════════════════════
# LANDING (no data)
# ══════════════════════════════════════════════════════════
def render_landing():
    page_header(
        "🛒 E-Commerce Customer Intelligence",
        "Customer Analytics & Churn Prediction System",
        connected=False
    )
    st.markdown(
        '<div class="card" style="max-width:620px;margin:40px auto;text-align:center;">'
        '  <div style="font-size:3.5rem;margin-bottom:16px">🛒</div>'
        '  <div style="font-size:1.3rem;font-weight:700;color:#0f172a;margin-bottom:8px">'
        '    Upload Your Dataset to Begin'
        '  </div>'
        '  <div style="font-size:0.88rem;color:#64748b;line-height:1.65;margin-bottom:20px">'
        '    Upload the <strong>E-Commerce Customer Behavior Dataset</strong> CSV using '
        '    the sidebar uploader, or place <code>ecommerce_customer_churn_dataset.csv</code>'
        '    in the same directory as this script.'
        '  </div>'
        '  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
        '              padding:12px 16px;font-size:0.80rem;color:#64748b;text-align:left;">'
        '    📂 <strong>Dataset source:</strong><br>'
        '    <a href="https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset"'
        '       style="color:#2563eb;" target="_blank">'
        '      kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset'
        '    </a>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    inject_css()

    LOCAL_CSV = "ecommerce_customer_churn_dataset.csv"

    # ==========================================================
    # STEP 1 — Render sidebar ONCE
    # ==========================================================
    page, uploaded_file, _ = render_sidebar(
        df_loaded=False,
        df=None,
        show_uploader=True
    )

    # ==========================================================
    # STEP 2 — Determine data source
    # ==========================================================
    data_source = None

    if uploaded_file is not None:
        data_source = uploaded_file

    elif os.path.exists(LOCAL_CSV):
        data_source = LOCAL_CSV

    # ==========================================================
    # STEP 3 — No dataset
    # ==========================================================
    if data_source is None:
        render_landing()
        return

    # ==========================================================
    # STEP 4 — Load dataset
    # ==========================================================
    try:
        raw_df = load_data(data_source)

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    # ==========================================================
    # STEP 5 — Validate columns
    # ==========================================================
    valid, missing_cols = validate_columns(raw_df)

    if not valid:
        st.error(
            f"❌ CSV is missing required columns: "
            f"**{', '.join(missing_cols)}**\n\n"
            "Please upload the correct "
            "E-Commerce Customer Behavior Dataset."
        )
        return

    # ==========================================================
    # STEP 6 — Clean data
    # ==========================================================
    clean_df, quality_report = clean_data(raw_df)

    if len(clean_df) == 0:
        st.error(
            "❌ Dataset is empty after cleaning. "
            "Please check the uploaded file."
        )
        return

    # ==========================================================
    # STEP 7 — Feature engineering
    # ==========================================================
    df = engineer_features(clean_df)

    # ==========================================================
    # STEP 8 — Apply filters WITHOUT rendering sidebar again
    # ==========================================================
    filtered_df = df.copy()

    # Retrieve existing filter values from session state
    sel_country = st.session_state.get("f_country", "All")
    sel_gender = st.session_state.get("f_gender", "All")
    sel_quarter = st.session_state.get("f_quarter", "All")
    sel_churn = st.session_state.get("f_churn", "All")
    sel_vc = st.session_state.get("f_vc", "All")
    sel_mc = st.session_state.get("f_mc", "All")

    churn_opts = {
        "All": None,
        "Churned": 1,
        "Retained": 0
    }

    if sel_country != "All":
        filtered_df = filtered_df[
            filtered_df["Country"] == sel_country
        ]

    if sel_gender != "All":
        filtered_df = filtered_df[
            filtered_df["Gender"] == sel_gender
        ]

    if sel_quarter != "All":
        filtered_df = filtered_df[
            filtered_df["Signup_Quarter"] == sel_quarter
        ]

    if churn_opts.get(sel_churn) is not None:
        filtered_df = filtered_df[
            filtered_df[TARGET] == churn_opts[sel_churn]
        ]

    if (
        sel_vc != "All"
        and "Customer_Value_Category" in filtered_df.columns
    ):
        filtered_df = filtered_df[
            filtered_df["Customer_Value_Category"].astype(str)
            == sel_vc
        ]

    if (
        sel_mc != "All"
        and "Membership_Category" in filtered_df.columns
    ):
        filtered_df = filtered_df[
            filtered_df["Membership_Category"].astype(str)
            == sel_mc
        ]

    # ==========================================================
    # STEP 9 — Dataset information
    # ==========================================================
    n_total = len(df)
    n_cols_total = len(df.columns)

    # ==========================================================
    # STEP 10 — Page routing
    # ==========================================================
    if page == "🏠 Overview":

        render_executive_dashboard(
            df,
            filtered_df
        )

    elif page == "👥 Customer Analytics":

        render_customer_analytics(
            filtered_df,
            n_total,
            n_cols_total
        )

    elif page == "⚠️ Churn Analysis":

        render_churn_analysis(
            filtered_df,
            n_total,
            n_cols_total
        )

    elif page == "🎯 Segmentation":

        render_segmentation(
            df,
            n_total,
            n_cols_total
        )

    elif page == "🤖 Machine Learning":

        render_machine_learning(
            df,
            n_total,
            n_cols_total
        )

    elif page == "🔮 Churn Prediction":

        render_prediction(
            df,
            n_total,
            n_cols_total
        )

    elif page == "💡 Business Insights":

        render_business_insights(
            filtered_df,
            n_total,
            n_cols_total
        )

    elif page == "📋 Data Quality":

        render_data_quality(
            raw_df,
            clean_df,
            quality_report,
            n_total,
            n_cols_total
        )


if __name__ == "__main__":
    main()