"""E-Ride client-facing operating model dashboard.

Run with:
    streamlit run eride_simulation_dashboard.py

The file is intentionally standalone. It does not require workbook uploads or
sidecar scripts when deployed to Streamlit Community Cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="E-Ride | Client Operating Model",
    page_icon="ER",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

dark_mode = st.sidebar.toggle("Dark mode", value=False, key="theme_dark")
theme = {
    "bg": "#08111f" if dark_mode else "#f5f7fb",
    "surface": "#101b2d" if dark_mode else "#ffffff",
    "surface2": "#17243a" if dark_mode else "#eef3f8",
    "text": "#f4f8fc" if dark_mode else "#142334",
    "muted": "#a9b8c8" if dark_mode else "#607386",
    "border": "#263851" if dark_mode else "#dce5ed",
    "accent": "#62c9ee" if dark_mode else "#159dcc",
    "accent_soft": "#12344a" if dark_mode else "#e3f6fc",
    "positive": "#42d3a2" if dark_mode else "#0b9f72",
    "positive_soft": "#102f2d" if dark_mode else "#e4f7f0",
    "warning": "#ffcb70" if dark_mode else "#a86200",
    "warning_soft": "#352b1d" if dark_mode else "#fff6e5",
    "negative": "#ff8392" if dark_mode else "#d8495a",
    "negative_soft": "#3d1c24" if dark_mode else "#fdecef",
    "button_text": "#071722",
}

st.markdown(
    f"""
<style>
    :root {{
      --bg:{theme['bg']}; --surface:{theme['surface']}; --surface2:{theme['surface2']};
      --text:{theme['text']}; --muted:{theme['muted']}; --border:{theme['border']};
      --accent:{theme['accent']}; --accent-soft:{theme['accent_soft']};
      --positive:{theme['positive']}; --positive-soft:{theme['positive_soft']};
      --warning:{theme['warning']}; --warning-soft:{theme['warning_soft']};
      --negative:{theme['negative']}; --negative-soft:{theme['negative_soft']};
    }}
    html, body, [class*="css"] {{
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .stApp {{ background:var(--bg); color:var(--text); }}
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] {{ color:var(--text); }}
    [data-testid="stCaptionContainer"], .small-note {{ color:var(--muted) !important; }}
    [data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
    [data-testid="stSidebar"] * {{ color:var(--text); }}
    header[data-testid="stHeader"], [data-testid="stToolbar"] {{
        color:var(--text) !important; background:var(--surface) !important;
        border-bottom:1px solid var(--border);
    }}
    header[data-testid="stHeader"] button, [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] a {{ color:var(--text) !important; background:transparent !important; }}
    header[data-testid="stHeader"] svg, [data-testid="stToolbar"] svg {{
        color:var(--text) !important; fill:currentColor !important;
    }}

    [data-baseweb="input"] > div, [data-baseweb="select"] > div {{
        background:var(--surface2) !important; border-color:var(--border) !important;
    }}
    [data-baseweb="input"] input, [data-baseweb="select"] span {{
        color:var(--text) !important; -webkit-text-fill-color:var(--text) !important;
    }}
    [data-testid="stNumberInput"] input {{
        pointer-events:auto !important; user-select:text !important; cursor:text !important;
        caret-color:var(--accent) !important;
    }}
    [data-testid="stNumberInput"] button, [data-baseweb="input"] button,
    button[aria-label="Open menu"], button[aria-label="Close"] {{
        color:var(--text) !important; background:var(--surface2) !important;
        border-color:var(--border) !important;
    }}
    [data-testid="stNumberInput"] button:hover, [data-baseweb="input"] button:hover {{
        color:var(--text) !important; background:var(--accent-soft) !important;
    }}
    [data-baseweb="popover"], [role="listbox"] {{
        background:var(--surface) !important; color:var(--text) !important;
    }}
    [role="option"] {{ color:var(--text) !important; background:var(--surface) !important; }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{
        color:var(--text) !important; background:var(--accent-soft) !important;
    }}

    .stButton > button, .stDownloadButton > button,
    button[data-testid="baseButton-secondary"], button[kind="secondary"] {{
        color:var(--text) !important; background:var(--surface) !important;
        border:1px solid var(--border) !important; box-shadow:none !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover, button[kind="secondary"]:hover {{
        color:var(--text) !important; background:var(--accent-soft) !important;
        border-color:var(--accent) !important;
    }}
    button[data-testid="baseButton-primary"], button[kind="primary"],
    button[aria-pressed="true"] {{
        color:{theme['button_text']} !important; background:var(--accent) !important;
        border:1px solid var(--accent) !important;
    }}
    button[data-testid="baseButton-primary"] *, button[kind="primary"] *,
    button[aria-pressed="true"] * {{
        color:{theme['button_text']} !important; -webkit-text-fill-color:{theme['button_text']} !important;
    }}
    button[aria-pressed="false"] {{
        color:var(--text) !important; background:var(--surface) !important; border-color:var(--border) !important;
    }}
    button:disabled, .stButton > button:disabled, .stDownloadButton > button:disabled {{
        color:var(--muted) !important; background:var(--surface2) !important;
        border-color:var(--border) !important; opacity:.72 !important;
    }}
    button svg {{ color:currentColor !important; }}

    [data-baseweb="radio"] label, [data-baseweb="checkbox"] label {{ color:var(--text) !important; }}
    [data-testid="stCheckbox"] [role="checkbox"] {{ border-color:var(--border) !important; }}
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {{
        background:var(--accent) !important; border-color:var(--accent) !important;
    }}
    button[role="switch"] {{ background:var(--surface2) !important; border:1px solid var(--border) !important; }}
    button[role="switch"][aria-checked="true"] {{ background:var(--accent) !important; border-color:var(--accent) !important; }}
    [data-testid="stSlider"] [role="slider"] {{
        background:var(--accent) !important; border-color:var(--accent) !important;
        box-shadow:0 0 0 2px var(--surface) !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background:var(--border); }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{ background:var(--accent) !important; }}
    button[aria-label^="Help for"] {{
        color:var(--text) !important; background:transparent !important; border:0 !important;
    }}
    button[aria-label^="Help for"] svg {{ color:var(--text) !important; }}

    .hero {{
        padding:1.45rem 1.6rem; border:1px solid var(--border); border-radius:18px;
        background:linear-gradient(135deg,var(--surface) 0%,var(--accent-soft) 145%);
        box-shadow:0 10px 28px #0000000c; margin-bottom:1rem; position:relative; overflow:hidden;
    }}
    .hero:after {{
        content:'E'; position:absolute; right:1.2rem; top:-1.2rem; font-size:9rem;
        color:var(--accent); opacity:.11; font-weight:900; transform:rotate(8deg);
    }}
    .eyebrow {{
        color:var(--accent); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;
    }}
    .hero h1 {{
        color:var(--text); margin:.28rem 0 .25rem; font-size:clamp(1.9rem,4vw,2.75rem); letter-spacing:-.045em;
    }}
    .hero p {{ color:var(--muted); max-width:850px; margin:0; font-size:.98rem; }}
    .client-badge {{
        display:inline-block; margin-left:.55rem; padding:.2rem .55rem; border-radius:999px;
        background:var(--accent-soft); color:var(--accent); font-size:.68rem; letter-spacing:.08em; vertical-align:middle;
    }}
    [data-testid="stMetric"] {{
        background:var(--surface); padding:1rem 1.05rem; border:1px solid var(--border);
        border-radius:14px; box-shadow:0 6px 18px #00000008; overflow:visible;
    }}
    [data-testid="stMetricLabel"] p {{ color:var(--muted) !important; }}
    [data-testid="stMetricValue"] {{
        color:var(--text); letter-spacing:-.035em; font-size:clamp(1.35rem,2.2vw,2rem) !important;
        line-height:1.12 !important; white-space:normal !important; overflow-wrap:anywhere !important;
    }}
    [data-testid="stMetricValue"] div {{
        white-space:normal !important; overflow-wrap:anywhere !important; text-overflow:clip !important;
    }}
    [data-testid="stMetricDelta"] {{ color:var(--muted); }}
    .callout {{
        color:var(--text); background:var(--positive-soft); border:1px solid var(--positive);
        border-radius:14px; padding:1rem 1.15rem; margin:.5rem 0 1rem;
    }}
    .warning {{ background:var(--warning-soft); border-color:var(--warning); }}
    .danger {{ background:var(--negative-soft); border-color:var(--negative); }}
    .source-card {{
        color:var(--text); background:var(--surface); border:1px solid var(--border);
        border-radius:13px; padding:.9rem 1rem; margin:.55rem 0;
    }}
    .source-card a {{ color:var(--accent); }}
    div[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:13px; overflow:hidden; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid var(--border); }}
    .stTabs [data-baseweb="tab"] {{
        color:var(--muted); background:transparent; border-radius:9px 9px 0 0; padding:.4rem .8rem;
    }}
    .stTabs [aria-selected="true"] {{ color:var(--accent) !important; background:var(--accent-soft); }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
        background-color:var(--accent) !important;
    }}
    [data-testid="stExpander"] {{
        background:var(--surface) !important; border-color:var(--border) !important; border-radius:12px;
    }}
    [data-testid="stExpander"] details, [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] summary > div {{
        color:var(--text) !important; background:var(--surface2) !important;
    }}
    [data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover > div {{
        background:var(--accent-soft) !important;
    }}
    [data-testid="stExpander"] summary *, [data-testid="stExpander"] summary svg {{
        color:var(--text) !important; fill:currentColor !important;
    }}
    .stAlert {{ background:var(--surface); color:var(--text); border-color:var(--border); }}
    code {{ color:var(--text) !important; background:var(--surface2) !important; }}
    footer {{ visibility:hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("### Quick recovery switches")
st.caption("Use these to decide which investment costs E-Ride must recover through monthly operations.")
q1, q2, q3 = st.columns(3)
with q1:
    include_fleet_capex = st.checkbox(
        "Recover fleet assets",
        value=True,
        help="Includes motorcycle and backup-battery amortization. Maintenance and insurance remain included either way.",
        key="include_fleet_capex",
    )
with q2:
    include_dev_recovery = st.checkbox(
        "Recover development cost",
        value=True,
        help="Includes the six-month build recovery and ongoing dev support retainer.",
        key="include_dev_recovery",
    )
with q3:
    include_cloud_cost = st.checkbox(
        "Recover AWS/cloud/API cost",
        value=True,
        help="Includes POC/cloud/API cost growth from the workbook-based model.",
        key="include_cloud_cost",
    )

st.markdown("### Scenario shortcut")
st.caption("The dashboard now opens on the profitable utilization case. Use these buttons to reset or stress-test the model.")
sc1, sc2 = st.columns([1, 1])
with sc1:
    if st.button("Reset to profitable scenario", type="primary"):
        st.session_state["users_m"] = 6000.0
        st.session_state["riders_m"] = 50.0
        st.session_state["rpu_m"] = 8.0
        st.session_state["rpd_m"] = 20.0
        st.session_state["days_m"] = 26.0
        st.session_state["completion_m"] = 90.0
with sc2:
    if st.button("Show conservative stress case"):
        st.session_state["users_m"] = 3750.0
        st.session_state["riders_m"] = 50.0
        st.session_state["rpu_m"] = 4.0
        st.session_state["rpd_m"] = 10.0
        st.session_state["days_m"] = 26.0
        st.session_state["completion_m"] = 90.0

excel_model_path = Path(__file__).with_name("eride_operating_model.xlsx")
if excel_model_path.exists():
    st.download_button(
        "Download Excel fallback model",
        data=excel_model_path.read_bytes(),
        file_name="eride_operating_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Use this spreadsheet if the dashboard is unavailable or if the client wants to inspect formulas.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Dist:
    mean: float
    sd: float
    low: float = 0.0
    high: float | None = None


def peso(x: float, decimals: int = 0) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"PHP {x:,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{100 * x:.{decimals}f}%"


def ninput(label, value, step=1.0, min_value=0.0, help=None, key=None, decimals=None):
    if decimals is None:
        decimals = 2 if abs(float(step)) < 1 else 0
    return st.number_input(
        label,
        min_value=float(min_value),
        value=float(value),
        step=float(step),
        help=help,
        key=key,
        format=f"%.{decimals}f",
    )


def mean_sd(label, mean, sd, step=1.0, min_value=0.0, help=None, key=""):
    m = ninput(label, mean, step, min_value, help, f"{key}_m")
    return Dist(m, 0.0, min_value)


def fare_from_inputs(distance_km, minutes, base, included_km_, per_km_, long_threshold, long_per_km_, included_min, per_min, discount):
    short_km = np.minimum(
        np.maximum(np.asarray(distance_km) - included_km_, 0),
        max(long_threshold - included_km_, 0),
    )
    long_km = np.maximum(np.asarray(distance_km) - long_threshold, 0)
    charged_minutes = np.maximum(np.asarray(minutes) - included_min, 0)
    before_discount = base + short_km * per_km_ + long_km * long_per_km_ + charged_minutes * per_min
    return np.maximum(before_discount * (1 - discount), 0)


def cloud_cost_from_workbook(active_users):
    """Workbook-based cloud/API scaling.

    The E-Ride Dev Costing Foundation workbook pilot sheet uses:
    - PHP 58/USD
    - 21 pilot days
    - 10,000 free monthly events per priced API
    - per-user/day API events inferred from 20-user pilot quantities:
      5 autocomplete, 2 place-detail, 1 geocoding, 2 routes
    - post-free-tier prices of USD 2.83/1k, USD 5/1k, USD 5/1k, USD 5/1k

    The developer-confirmed POC baseline is PHP 1,500 because current POC usage
    is expected to sit inside free tiers except for a small hosted-fallback budget.
    """
    users_ = np.asarray(active_users, dtype=float)
    baseline_php = float(st.session_state.get("poc_cloud", 1500.0))
    pilot_days = 21.0
    php_per_usd = 58.0
    free_events = 10_000.0
    usage_per_user_day = np.array([5.0, 2.0, 1.0, 2.0])
    usd_per_1000 = np.array([2.83, 5.0, 5.0, 5.0])
    events = np.expand_dims(users_, axis=-1) * pilot_days * usage_per_user_day
    overage = np.maximum(events - free_events, 0.0)
    variable_php = np.sum(overage / 1000.0 * usd_per_1000, axis=-1) * php_per_usd
    return baseline_php + variable_php


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## E-Ride operating model")
    st.caption("Client-ready inputs. Type directly in any number field; the +/- buttons are optional.")

    with st.expander("Demand and fleet", expanded=True):
        users = mean_sd("Active users", 6000, 600, 100, 1, "Monthly transacting users.", "users")
        riders = mean_sd("Number of riders / vehicles", 50, 3, 1, 1, "One salaried rider per active vehicle.", "riders")
        rides_user = mean_sd("Rides per user / month", 8, 1, 1, 1, "Actual ride count per active user per month.", "rpu")
        rides_rider = mean_sd("Rides per rider / day", 20.0, 2.0, 0.5, 0.1, "Capacity target per rider/vehicle.", "rpd")
        days = mean_sd("Operating days / month", 26, 1, 1, 1, "Typical operating days in the month.", "days")
        completion = mean_sd("Completion rate (%)", 90, 4, 1, 1, "Completed rides after requests/cancellations.", "completion")
        avg_km = mean_sd("Average paid trip distance (km)", 6.0, 1.5, 0.5, 0.5, "Distance used by fare and energy calculations.", "km")

    with st.expander("Fare and rider pay", expanded=True):
        st.caption("Angkas-style structure: base + distance + time, then E-Ride discount.")
        base_fare = mean_sd("Base fare", 50, 5, 1, 1, "Benchmark base for first 2 km.", "basefare")
        included_km = ninput("Distance included in base fare (km)", 2, 0.5, 0, key="included_km")
        per_km = mean_sd("Charge / km after base", 10, 1.5, 1, 0, "Benchmark through 7 km.", "perkm")
        long_km_threshold = ninput("Long-distance threshold (km)", 7, 0.5, 0, key="long_km")
        long_per_km = mean_sd("Charge / km after threshold", 15, 2, 1, 0, "Benchmark after 7 km.", "longperkm")
        avg_minutes = mean_sd("Average trip duration (minutes)", 25, 6, 1, 1, "Paid trip time.", "minutes")
        included_minutes = ninput("Minutes included in base fare", 10, 1, 0, key="included_minutes")
        per_minute = mean_sd("Charge / additional minute", 1, 0.25, 0.25, 0, "Editable planning allowance.", "perminute")
        eride_discount = ninput("E-Ride electric discount (%)", 10, 1, 0, "Lower fare enabled by electric energy savings.", "fare_discount") / 100
        company_share = ninput("Company share of fare (%)", 100, 1, 1, "100% for owned fleet and salaried riders.", "share") / 100
        user_fee = ninput("Monthly membership / user fee", 0, 5, 0, "Optional recurring user revenue.", "ufee")
        wage = mean_sd("Rider daily base wage", 695, 25, 5, 1, "NCR non-agriculture statutory minimum effective July 18, 2025.", "wage")
        burden = ninput("Employer burden / benefits (%)", 15, 1, 0, "Planning allowance for benefits and statutory overhead.", "burden") / 100
        incentive_threshold = ninput("Daily gross incentive threshold", 3000, 100, 0, "Commission only above this gross fare per rider/day.", "threshold")
        incentive_rate = ninput("Commission above threshold (%)", 20, 1, 0, key="inc_rate") / 100

    with st.expander("Fleet, energy and assets", expanded=True):
        st.caption("Fleet recovery is controlled by the quick switches above.")
        motorcycle_cost = mean_sd("E-motorcycle cost", 80000, 8000, 5000, 1, "Primary battery assumed included.", "bike")
        backup_battery = mean_sd("Backup battery cost / bike", 20000, 3000, 1000, 0, "One backup battery per motorcycle.", "battery")
        asset_life = ninput("Asset recovery period (months)", 48, 1, 1, key="life")
        maintenance = mean_sd("Maintenance / bike / month", 1500, 350, 100, 0, "Service, consumables, repairs.", "maint")
        insurance = mean_sd("Insurance and registration / bike / month", 500, 100, 50, 0, "Monthlyized provision.", "ins")
        kwh_km = mean_sd("Energy use (kWh / km)", 0.045, 0.008, 0.005, 0.005, "Replace with telemetry after pilot.", "kwh")
        power_rate = mean_sd("Electricity rate / kWh", 13.0, 1.5, 0.5, 1, "Blended charging tariff.", "power")

    with st.expander("Development and operations", expanded=True):
        st.caption("Development and cloud recovery are controlled by the quick switches above.")
        build_cost = mean_sd("Development build budget", 1_000_000, 100_000, 50_000, 0, "Client-facing build budget spread over six months.", "build")
        build_recovery = ninput("Development recovery period (months)", 6, 1, 1, "Use 6 months to match the implementation work plan.", "build_life")
        dev_retainer = mean_sd("Ongoing dev support retainer", 55_000, 10_000, 5_000, 0, "Post-build support and iteration tier.", "dev")
        poc_cloud_base = ninput("POC cloud baseline", 1500, 100, 0, "Developer-confirmed POC/free-tier baseline.", "poc_cloud")
        ops_fixed = mean_sd("Operations, support and admin", 120_000, 25_000, 5_000, 0, "Dispatch/admin/customer support excluding rider payroll.", "ops")
        support_user = mean_sd("Support cost / active user", 2.0, 0.5, 0.1, 0, "Variable support allowance.", "support")
        marketing = mean_sd("Marketing / active user", 5.0, 2.0, 0.5, 0, "Retention and acquisition allowance.", "mkt")

    with st.expander("Transaction and reserves"):
        digital_share = ninput("Digital payment share (%)", 35, 1, 0, key="digital") / 100
        payment_fee = ninput("Payment fee on digital fares (%)", 2.5, 0.1, 0, key="payfee") / 100
        regulatory_reserve = ninput("Tax / regulatory reserve on revenue (%)", 3.0, 0.5, 0, key="reserve") / 100


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

expected_fare = fare_from_inputs(
    avg_km.mean,
    avg_minutes.mean,
    base_fare.mean,
    included_km,
    per_km.mean,
    long_km_threshold,
    long_per_km.mean,
    included_minutes,
    per_minute.mean,
    eride_discount,
)


def deterministic_model(
    fare: float,
    completed_override: float | None = None,
    users_override: float | None = None,
    riders_override: float | None = None,
    rides_user_override: float | None = None,
    rides_rider_override: float | None = None,
):
    users_ = users.mean if users_override is None else users_override
    riders_ = riders.mean if riders_override is None else riders_override
    rides_user_ = rides_user.mean if rides_user_override is None else rides_user_override
    rides_rider_ = rides_rider.mean if rides_rider_override is None else rides_rider_override
    days_ = days.mean
    completion_ = completion.mean / 100
    requested_demand = users_ * rides_user_
    requested_capacity = riders_ * rides_rider_ * days_
    requested_served = min(requested_demand, requested_capacity)
    completed = requested_served * completion_ if completed_override is None else completed_override

    revenue = completed * fare * company_share + users_ * user_fee
    rider_salary = riders_ * wage.mean * days_ * (1 + burden)

    gross_per_rider_day = (completed * fare) / max(riders_ * days_, 1)
    incentive = riders_ * days_ * max(gross_per_rider_day - incentive_threshold, 0) * incentive_rate

    energy = completed * avg_km.mean * kwh_km.mean * power_rate.mean
    acquisition = (
        riders_ * (motorcycle_cost.mean + backup_battery.mean) / asset_life if include_fleet_capex else 0
    )
    fleet_ops = riders_ * (maintenance.mean + insurance.mean)
    development = build_cost.mean / build_recovery + dev_retainer.mean if include_dev_recovery else 0.0
    technology = float(cloud_cost_from_workbook(users_)) if include_cloud_cost else 0.0
    operations = ops_fixed.mean + users_ * (support_user.mean + marketing.mean)
    payment = revenue * digital_share * payment_fee
    reserve = revenue * regulatory_reserve

    total_cost = (
        rider_salary
        + incentive
        + energy
        + acquisition
        + fleet_ops
        + development
        + technology
        + operations
        + payment
        + reserve
    )
    profit = revenue - total_cost
    return {
        "users": users_,
        "riders": riders_,
        "demand": requested_demand,
        "capacity": requested_capacity,
        "requested_served": requested_served,
        "trips": completed,
        "completed_per_day": completed / max(days_, 1),
        "completed_per_rider_day": completed / max(riders_ * days_, 1),
        "fare": fare,
        "revenue": revenue,
        "rider_salary": rider_salary,
        "incentive": incentive,
        "energy": energy,
        "fleet_acquisition": acquisition,
        "fleet_ops": fleet_ops,
        "development": development,
        "technology": technology,
        "operations": operations,
        "payment": payment,
        "reserve": reserve,
        "total_cost": total_cost,
        "profit": profit,
        "cost_user": total_cost / max(users_, 1),
        "revenue_user": revenue / max(users_, 1),
        "profit_user": profit / max(users_, 1),
        "cost_ride": total_cost / max(completed, 1),
        "profit_ride": profit / max(completed, 1),
        "utilization": requested_served / max(requested_capacity, 1),
        "gross_per_rider_day": gross_per_rider_day,
    }


def profit_at_fare(test_fare: float) -> float:
    return deterministic_model(test_fare)["profit"]


lo, hi = 0.0, 1000.0
for _ in range(80):
    mid = (lo + hi) / 2
    if profit_at_fare(mid) >= 0:
        hi = mid
    else:
        lo = mid
break_even_fare = hi if profit_at_fare(hi) >= 0 else math.nan


def profit_at_completed(completed_rides: float) -> float:
    return deterministic_model(expected_fare, completed_override=completed_rides)["profit"]


lo_rides, hi_rides = 0.0, max(1000.0, users.mean * rides_user.mean * 4, riders.mean * rides_rider.mean * days.mean * 4)
while profit_at_completed(hi_rides) < 0 and hi_rides < 10_000_000:
    hi_rides *= 2

if profit_at_completed(hi_rides) >= 0:
    for _ in range(80):
        mid = (lo_rides + hi_rides) / 2
        if profit_at_completed(mid) >= 0:
            hi_rides = mid
        else:
            lo_rides = mid
    break_even_completed_rides = hi_rides
else:
    break_even_completed_rides = math.nan

scenario = deterministic_model(expected_fare)
profitable_case = deterministic_model(
    expected_fare,
    users_override=6000.0,
    riders_override=50.0,
    rides_user_override=8.0,
    rides_rider_override=20.0,
)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Client operating model <span class="client-badge">E-Ride</span></div>
  <h1>How many rides does E-Ride need to stay profitable?</h1>
  <p>
    This dashboard turns the project documents into a client-ready decision model:
    fares, rider payroll, vehicle economics, development cost, cloud cost, and operating overhead
    are all editable so the owner can test when the business becomes sustainable.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if company_share < 0.5:
    st.markdown(
        '<div class="callout warning"><b>Model warning:</b> this setting keeps salaried riders and vehicles while retaining a low fare share. That hybrid is usually structurally hard to profit from.</div>',
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Client summary", "Cost breakdown", "Cloud tiers", "Assumptions"])


with tabs[0]:
    st.markdown("### Executive answer")
    c1, c2 = st.columns(2)
    c1.metric("Monthly profit / loss", peso(scenario["profit"]), f"{peso(scenario['profit_user'], 2)} / user")
    c2.metric("Required revenue / user", peso(scenario["cost_user"], 2), f"current {peso(scenario['revenue_user'], 2)}")
    c3, c4 = st.columns(2)
    c3.metric("Break-even fare", peso(break_even_fare, 2), f"current fare {peso(scenario['fare'], 2)}")
    c4.metric("Needed completed rides/day", f"{break_even_completed_rides / max(days.mean, 1):,.0f}" if np.isfinite(break_even_completed_rides) else "N/A", "at current fare")

    if scenario["profit"] >= 0:
        st.markdown(
            f'<div class="callout"><b>Client message:</b> At this setup, E-Ride can operate with an expected monthly profit of <b>{peso(scenario["profit"])}</b>. The model requires {peso(scenario["cost_user"], 2)} revenue per active user and currently earns {peso(scenario["revenue_user"], 2)}.</div>',
            unsafe_allow_html=True,
        )
    else:
        gap_user = scenario["cost_user"] - scenario["revenue_user"]
        st.markdown(
            f'<div class="callout danger"><b>Client message:</b> At this setup, E-Ride loses <b>{peso(abs(scenario["profit"]))}</b> per month. It is short by <b>{peso(gap_user, 2)} per active user</b>, so the owner must increase utilization, increase fleet productivity, add revenue, lower costs, or delay fleet expansion.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Operating capacity")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active users", f"{scenario['users']:,.0f}", f"{rides_user.mean:.0f} rides/user/month")
    s2.metric("Riders / vehicles", f"{scenario['riders']:,.0f}", f"{rides_rider.mean:.1f} requested rides/rider/day")
    s3.metric("Completed rides/month", f"{scenario['trips']:,.0f}", f"{scenario['completed_per_day']:,.0f} / day")
    s4.metric("Fleet utilization", pct(scenario["utilization"]), f"{scenario['completed_per_rider_day']:.1f} completed/rider/day")

    required_per_rider_day = break_even_completed_rides / max(riders.mean * days.mean, 1)
    feasible_note = ""
    if np.isfinite(break_even_completed_rides):
        if required_per_rider_day > rides_rider.mean * completion.mean / 100:
            feasible_note = " This is above current fleet capacity, so the model also needs more vehicles, more rides/rider/day, or more operating days."
        else:
            feasible_note = " This is within the current modeled fleet capacity."
        st.info(
            f"To break even at the current fare of {peso(scenario['fare'], 2)}, E-Ride needs about "
            f"{break_even_completed_rides:,.0f} completed rides/month, or "
            f"{break_even_completed_rides / max(days.mean, 1):,.0f} completed rides/day. "
            f"That equals {required_per_rider_day:.1f} completed rides per rider per day.{feasible_note}"
        )

    if scenario["demand"] >= scenario["capacity"]:
        unserved = scenario["demand"] - scenario["capacity"]
        st.warning(
            f"Fleet-constrained: requested demand exceeds capacity by {unserved:,.0f} rides/month. "
            "Profit will not improve from more user rides unless the business adds riders/vehicles, increases rider productivity, or expands operating days."
        )
    else:
        spare = scenario["capacity"] - scenario["demand"]
        st.success(
            f"Demand-constrained: the fleet can still absorb about {spare:,.0f} requested rides/month before vehicle capacity becomes the bottleneck."
        )

    st.markdown("### Scenario that covers rider wages")
    wage_per_rider_day = wage.mean * (1 + burden)
    scenario_rows = pd.DataFrame(
        [
            {
                "Scenario": "Current setup",
                "Active users": scenario["users"],
                "Riders": scenario["riders"],
                "Rides / user / month": rides_user.mean,
                "Requested rides / rider / day": rides_rider.mean,
                "Completed rides / rider / day": scenario["completed_per_rider_day"],
                "Fare revenue / rider / day": scenario["gross_per_rider_day"],
                "Wage cost / rider / day": wage_per_rider_day,
                "Wage coverage": scenario["gross_per_rider_day"] / max(wage_per_rider_day, 1),
                "Monthly profit / loss": scenario["profit"],
            },
            {
                "Scenario": "Profitable utilization case",
                "Active users": profitable_case["users"],
                "Riders": profitable_case["riders"],
                "Rides / user / month": 8.0,
                "Requested rides / rider / day": 20.0,
                "Completed rides / rider / day": profitable_case["completed_per_rider_day"],
                "Fare revenue / rider / day": profitable_case["gross_per_rider_day"],
                "Wage cost / rider / day": wage_per_rider_day,
                "Wage coverage": profitable_case["gross_per_rider_day"] / max(wage_per_rider_day, 1),
                "Monthly profit / loss": profitable_case["profit"],
            },
        ]
    )
    st.dataframe(
        scenario_rows.style.format(
            {
                "Active users": "{:,.0f}",
                "Riders": "{:,.0f}",
                "Rides / user / month": "{:,.1f}",
                "Requested rides / rider / day": "{:,.1f}",
                "Completed rides / rider / day": "{:,.1f}",
                "Fare revenue / rider / day": "PHP {:,.2f}",
                "Wage cost / rider / day": "PHP {:,.2f}",
                "Wage coverage": "{:.1f}x",
                "Monthly profit / loss": "PHP {:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "Interpretation: the current setup barely covers wage per rider-day before other costs. "
        "The profitable utilization case targets about 18 completed rides per rider per day, so fare revenue per rider-day is more than 2x the wage cost and the model turns profitable even with fleet, development, and cloud recovery switched on."
    )

    st.markdown("### Fare design")
    st.caption(
        f"Current fare: {peso(base_fare.mean)} base including {included_km:g} km, "
        f"{peso(per_km.mean, 2)}/km until {long_km_threshold:g} km, "
        f"{peso(long_per_km.mean, 2)}/km after that, "
        f"{peso(per_minute.mean, 2)}/minute after {included_minutes:g} minutes, "
        f"less {pct(eride_discount, 0)} electric discount."
    )
    short_distance_km = min(max(avg_km.mean - included_km, 0), max(long_km_threshold - included_km, 0))
    long_distance_km = max(avg_km.mean - long_km_threshold, 0)
    charged_minutes = max(avg_minutes.mean - included_minutes, 0)
    distance_charge = short_distance_km * per_km.mean + long_distance_km * long_per_km.mean
    time_charge = charged_minutes * per_minute.mean
    before_discount = base_fare.mean + distance_charge + time_charge
    fare_rows = pd.DataFrame(
        {
            "Fare component": ["Base fare", "Distance charge", "Time charge", "Electric discount", "Final passenger fare"],
            "Calculation": [
                f"Includes first {included_km:g} km",
                f"{short_distance_km:g} km x {peso(per_km.mean, 2)} + {long_distance_km:g} km x {peso(long_per_km.mean, 2)}",
                f"{charged_minutes:g} minutes x {peso(per_minute.mean, 2)}",
                f"{pct(eride_discount, 0)} x {peso(before_discount, 2)}",
                "Base + distance + time - discount",
            ],
            "PHP": [base_fare.mean, distance_charge, time_charge, -before_discount * eride_discount, scenario["fare"]],
        }
    )
    st.dataframe(fare_rows.style.format({"PHP": "PHP {:,.2f}"}), width="stretch", hide_index=True)

with tabs[1]:
    st.markdown("### Full monthly cost breakdown")
    build_recovery_cost = build_cost.mean / build_recovery if include_dev_recovery else 0.0
    dev_support_cost = dev_retainer.mean if include_dev_recovery else 0.0
    cost_rows = pd.DataFrame(
        {
            "Cost category": [
                "Rider base payroll",
                "Rider incentives",
                "Electricity",
                "Fleet acquisition / amortization",
                "Fleet maintenance, insurance and registration",
                "Development build recovery",
                "Ongoing dev support retainer",
                "Cloud and API usage",
                "Operations, support and admin",
                "Payment processing",
                "Tax / regulatory reserve",
            ],
            "Monthly PHP": [
                scenario["rider_salary"],
                scenario["incentive"],
                scenario["energy"],
                scenario["fleet_acquisition"],
                scenario["fleet_ops"],
                build_recovery_cost,
                dev_support_cost,
                scenario["technology"],
                scenario["operations"],
                scenario["payment"],
                scenario["reserve"],
            ],
        }
    )
    cost_rows["Share of cost"] = cost_rows["Monthly PHP"] / max(scenario["total_cost"], 1)
    cost_rows["PHP / completed ride"] = cost_rows["Monthly PHP"] / max(scenario["trips"], 1)
    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(
            cost_rows.style.format(
                {
                    "Monthly PHP": "PHP {:,.0f}",
                    "Share of cost": "{:.1%}",
                    "PHP / completed ride": "PHP {:,.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        cost_chart = (
            alt.Chart(cost_rows)
            .mark_bar(cornerRadiusEnd=5, color=theme["accent"])
            .encode(
                x=alt.X("Monthly PHP:Q", title="PHP per month"),
                y=alt.Y("Cost category:N", sort="-x", title=None),
                tooltip=["Cost category", alt.Tooltip("Monthly PHP:Q", format=",.0f")],
            )
            .properties(height=385, background=theme["surface"])
            .configure_axis(labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"])
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(cost_chart, width="stretch")

    st.markdown("### Development budget: PHP 1,000,000 spread across 6 months")
    dev_roles = pd.DataFrame(
        {
            "Role": [
                "Lead full-stack developer / PM",
                "Backend developer",
                "Frontend/mobile developer",
                "Cloud / DevOps engineer",
                "QA and release support",
                "Product / client coordination",
                "UI/UX and presentation support",
            ],
            "6-month allocation": [330_000, 180_000, 140_000, 120_000, 90_000, 80_000, 60_000],
        }
    )
    scale = build_cost.mean / 1_000_000 if build_cost.mean else 0
    dev_roles["Adjusted 6-month allocation"] = dev_roles["6-month allocation"] * scale
    dev_roles["Monthly allocation"] = dev_roles["Adjusted 6-month allocation"] / 6
    dev_roles["Share"] = dev_roles["Adjusted 6-month allocation"] / max(dev_roles["Adjusted 6-month allocation"].sum(), 1)
    st.dataframe(
        dev_roles.style.format(
            {
                "6-month allocation": "PHP {:,.0f}",
                "Adjusted 6-month allocation": "PHP {:,.0f}",
                "Monthly allocation": "PHP {:,.0f}",
                "Share": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The role split is a client-facing planning allocation. The total follows the editable Development build budget in the sidebar; default is PHP 1,000,000 over six months."
    )

    st.markdown("### Profit bridge")
    bridge_df = pd.DataFrame(
        {
            "Line": ["Fare and membership revenue", "Total monthly costs", "Net monthly profit / loss"],
            "PHP": [scenario["revenue"], -scenario["total_cost"], scenario["profit"]],
        }
    )
    bridge = (
        alt.Chart(bridge_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("PHP:Q", title="PHP per month"),
            y=alt.Y("Line:N", sort=None, title=None),
            color=alt.condition(alt.datum.PHP >= 0, alt.value(theme["positive"]), alt.value(theme["negative"])),
            tooltip=["Line", alt.Tooltip("PHP:Q", format=",.0f")],
        )
        .properties(height=175, background=theme["surface"])
        .configure_axis(labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"])
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(bridge, width="stretch")


with tabs[2]:
    st.markdown("### AWS / cloud cost tiers")
    st.write(
        "The current POC is expected to cost around PHP 1,500 because implemented features sit mostly inside free tiers. "
        "As users grow, map/search/route API calls are the main cost driver in the workbook-based model."
    )
    tier_users = np.array([20, 100, 500, 1_000, 3_750, 5_000, 10_000, 25_000])
    tier_names = [
        "POC pilot",
        "Small internal launch",
        "Campus/barangay launch",
        "Early city launch",
        "Current scenario",
        "Dense launch",
        "City scale",
        "Metro scale stress test",
    ]
    cloud_tiers = pd.DataFrame(
        {
            "Tier": tier_names,
            "Active users": tier_users,
            "Estimated monthly cloud/API cost": cloud_cost_from_workbook(tier_users),
        }
    )
    cloud_tiers["Cost / active user"] = cloud_tiers["Estimated monthly cloud/API cost"] / cloud_tiers["Active users"]
    st.dataframe(
        cloud_tiers.style.format(
            {
                "Active users": "{:,.0f}",
                "Estimated monthly cloud/API cost": "PHP {:,.0f}",
                "Cost / active user": "PHP {:,.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    max_scale_users = max(25_000, int(scenario["users"] * 2))
    scale_users = np.unique(
        np.concatenate(([0, 20, 100, 500, 1000, int(scenario["users"])], np.linspace(0, max_scale_users, 180).astype(int)))
    )
    cloud_scale = pd.DataFrame({"Active users": scale_users, "Cloud cost": cloud_cost_from_workbook(scale_users)})
    cloud_line = (
        alt.Chart(cloud_scale)
        .mark_line(color=theme["accent"], strokeWidth=3)
        .encode(
            x=alt.X("Active users:Q", title="Active users"),
            y=alt.Y("Cloud cost:Q", title="PHP per month"),
            tooltip=[alt.Tooltip("Active users:Q", format=",.0f"), alt.Tooltip("Cloud cost:Q", format=",.0f")],
        )
    )
    current_cloud_estimate = float(cloud_cost_from_workbook(scenario["users"]))
    current_cloud = (
        alt.Chart(pd.DataFrame({"Active users": [scenario["users"]], "Cloud cost": [current_cloud_estimate]}))
        .mark_point(color=theme["negative"], filled=True, size=120)
        .encode(x="Active users:Q", y="Cloud cost:Q", tooltip=["Active users", "Cloud cost"])
    )
    cloud_chart = (
        (cloud_line + current_cloud)
        .properties(height=320, background=theme["surface"])
        .configure_axis(labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"])
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(cloud_chart, width="stretch")
    st.caption(
        "Workbook basis: PHP 58/USD, 21 days, 10,000 free monthly events per priced API, and pilot usage ratios of 5 autocomplete, 2 place-detail, 1 geocoding and 2 route events per user/day. AWS account/budget/S3/SES/CloudWatch items remain low or optional at POC scale; map/search/route API usage is the growth driver."
    )


with tabs[3]:
    st.markdown("### Editable planning assumptions")
    assumptions = pd.DataFrame(
        [
            ["Active users", users.mean, "Internal scale case", "Monthly transacting users"],
            ["Rides / active user / month", rides_user.mean, "Planning assumption", "Not a verified public PH market average"],
            ["Rides / rider / day", rides_rider.mean, "Internal roadmap prior", "Capacity and utilization target"],
            ["Calculated passenger fare", expected_fare, "Editable fare engine", "Base + distance + time - electric discount"],
            ["Rider daily wage", wage.mean, "NWPC NCR-26", "PHP 695/day NCR non-agriculture effective July 18, 2025"],
            ["E-motorcycle", motorcycle_cost.mean, "User estimate", "Primary battery included; backup battery separate"],
            ["Development build", build_cost.mean, "Client budget", "Default PHP 1M over six months"],
            ["Cloud", current_cloud_estimate, "Dev workbook + POC guidance", "POC PHP 1,500; API usage grows with users"],
            ["Fixed operations", ops_fixed.mean, "Planning assumption", "Support/admin/ops excluding rider payroll"],
        ],
        columns=["Variable", "Value", "Basis", "Interpretation"],
    )
    st.dataframe(
        assumptions.style.format({"Value": "{:,.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Evidence and source quality")
    st.markdown(
        """
<div class="source-card"><b>Statutory wage - high confidence</b><br>
NWPC lists NCR Wage Order NCR-26, effective July 18, 2025: PHP 695/day for non-agriculture and PHP 658/day for agriculture and covered small establishments.<br>
<a href="https://nwpc.dole.gov.ph/ncr/" target="_blank">National Wages and Productivity Commission - NCR</a></div>

<div class="source-card"><b>Motorcycle-taxi fare benchmark - medium confidence / dated regulation</b><br>
The Philippine motorcycle-taxi pilot benchmark used PHP 50 for the first 2 km, then PHP 10/km through 7 km and PHP 15/km thereafter. Actual app prices and rules can differ today, so the dashboard keeps fare inputs editable.<br>
<a href="https://visor.ph/traffic/fare-for-pilot-run-of-motorcycle-taxis-is-p50-through-first-2km/" target="_blank">VISOR - pilot fare schedule</a></div>

<div class="source-card"><b>Supply / market scale - medium confidence</b><br>
The 2020 pilot registered 46,713 riders across participating operators. This supports supply scale only; it does not prove demand or user ride frequency for E-Ride.<br>
<a href="https://www.pna.gov.ph/articles/1093673" target="_blank">Philippine News Agency - registered pilot riders</a></div>

<div class="source-card"><b>Cloud - workbook and developer guidance</b><br>
The Dev Costing Foundation workbook supplies the cloud/API growth inputs. The developer guidance says the current POC should be around PHP 1,500 because implemented features are mostly covered by free tiers.</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Important: there is no credible public source for a stable Philippine motorcycle-taxi average rides per active user per month. Treat the rides/user/month input as a planning assumption until E-Ride pilot telemetry is available."
    )

    st.markdown("### Core equation")
    st.code(
        "completed rides = min(users x rides/user/month, riders x rides/rider/day x operating days) x completion rate\n"
        "profit = retained fares + user fees - payroll - incentives - energy - fleet - development - cloud - operations - fees/reserves\n"
        "required revenue/user = total monthly cost / active users\n"
        "needed rides/day = break-even completed monthly rides / operating days"
    )


st.markdown(
    '<p class="small-note">E-Ride client operating model - Philippine pesos - editable deterministic financial model.</p>',
    unsafe_allow_html=True,
)
