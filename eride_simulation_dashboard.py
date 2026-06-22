"""E-Ride Philippine e-motorcycle taxi unit economics simulator.

Run with: streamlit run eride_simulation_dashboard.py

This file is intentionally self-contained: assumptions, research notes, styling,
simulation, charts, and CSV export all live here. No workbook upload is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="E-Ride | Operating Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False, key="theme_dark")
theme = {
    "bg": "#08111f" if dark_mode else "#f5f7fb",
    "surface": "#101b2d" if dark_mode else "#ffffff",
    "surface2": "#17243a" if dark_mode else "#eef3f8",
    "text": "#f4f8fc" if dark_mode else "#142334",
    "muted": "#9cafc1" if dark_mode else "#607386",
    "border": "#263851" if dark_mode else "#dce5ed",
    "accent": "#62c9ee" if dark_mode else "#159dcc",
    "accent_soft": "#12344a" if dark_mode else "#e3f6fc",
    "positive": "#42d3a2" if dark_mode else "#0b9f72",
    "positive_soft": "#102f2d" if dark_mode else "#e4f7f0",
    "warning": "#ffcb70" if dark_mode else "#a86200",
    "warning_soft": "#352b1d" if dark_mode else "#fff6e5",
    "negative": "#ff8392" if dark_mode else "#d8495a",
    "accent_text": "#071722",
}

st.markdown(
    f"""
<style>
    :root {{
      --bg:{theme['bg']}; --surface:{theme['surface']}; --surface2:{theme['surface2']};
      --text:{theme['text']}; --muted:{theme['muted']}; --border:{theme['border']};
      --accent:{theme['accent']}; --accent-soft:{theme['accent_soft']};
      --positive:{theme['positive']}; --positive-soft:{theme['positive_soft']};
      --warning:{theme['warning']}; --warning-soft:{theme['warning_soft']}; --negative:{theme['negative']};
    }}
    html, body, [class*="css"] {{ font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
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
    header[data-testid="stHeader"] svg, [data-testid="stToolbar"] svg {{ color:var(--text) !important; fill:currentColor !important; }}
    [data-baseweb="input"] > div, [data-baseweb="select"] > div {{ background:var(--surface2); border-color:var(--border); }}
    [data-baseweb="input"] input, [data-baseweb="select"] span {{ color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; }}
    [data-testid="stNumberInput"] input {{
        pointer-events:auto !important; user-select:text !important; cursor:text !important;
        caret-color:var(--accent) !important;
    }}
    [data-baseweb="popover"], [role="listbox"] {{ background:var(--surface) !important; color:var(--text) !important; }}
    [role="option"] {{ color:var(--text) !important; background:var(--surface) !important; }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{ color:var(--text) !important; background:var(--accent-soft) !important; }}
    /* Streamlit controls need explicit colors; otherwise their native theme can
       remain dark after this app switches back to light mode. */
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
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    button[data-testid="baseButton-secondary"]:focus-visible {{
        color:var(--text) !important; background:var(--accent-soft) !important;
        border-color:var(--accent) !important; outline:2px solid var(--accent) !important;
    }}
    button[data-testid="baseButton-primary"], button[kind="primary"] {{
        color:{theme['accent_text']} !important; background:var(--accent) !important;
        border:1px solid var(--accent) !important;
    }}
    button[data-testid="baseButton-primary"] *, button[kind="primary"] * {{
        color:{theme['accent_text']} !important; -webkit-text-fill-color:{theme['accent_text']} !important;
    }}
    button[data-testid="baseButton-secondary"] *, button[kind="secondary"] *,
    .stDownloadButton > button * {{
        color:var(--text) !important; -webkit-text-fill-color:var(--text) !important;
    }}
    button[data-testid="baseButton-primary"]:hover, button[kind="primary"]:hover {{
        color:{theme['accent_text']} !important; filter:brightness(.94);
    }}
    button:disabled, .stButton > button:disabled, .stDownloadButton > button:disabled {{
        color:var(--muted) !important; background:var(--surface2) !important;
        border-color:var(--border) !important; opacity:.72 !important;
    }}
    /* Number-input steppers and icon-only buttons. */
    [data-testid="stNumberInput"] button, [data-baseweb="input"] button,
    button[aria-label="Open menu"], button[aria-label="Close"] {{
        color:var(--text) !important; background:var(--surface2) !important;
        border-color:var(--border) !important;
    }}
    [data-testid="stNumberInput"] button:hover, [data-baseweb="input"] button:hover {{
        color:var(--text) !important; background:var(--accent-soft) !important;
    }}
    button svg {{ color:currentColor !important; }}
    /* Radio/segmented controls and toggles. */
    [data-baseweb="radio"] label, [data-baseweb="checkbox"] label {{ color:var(--text) !important; }}
    button[aria-pressed="false"] {{ color:var(--text) !important; background:var(--surface) !important; border-color:var(--border) !important; }}
    button[aria-pressed="true"] {{ color:{theme['accent_text']} !important; background:var(--accent) !important; border-color:var(--accent) !important; }}
    /* Sliders and toggles must follow the app palette instead of Streamlit's
       native red/dark browser theme. */
    [data-testid="stSlider"] [role="slider"] {{
        background:var(--accent) !important; border-color:var(--accent) !important;
        box-shadow:0 0 0 2px var(--surface) !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background:var(--border); }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{ background:var(--accent) !important; }}
    [data-testid="stCheckbox"] [role="checkbox"] {{ border-color:var(--border) !important; }}
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {{ background:var(--accent) !important; }}
    button[role="switch"] {{ background:var(--surface2) !important; border:1px solid var(--border) !important; }}
    button[role="switch"][aria-checked="true"] {{ background:var(--accent) !important; border-color:var(--accent) !important; }}
    button[aria-label^="Help for"] {{ color:var(--text) !important; background:transparent !important; border:0 !important; }}
    button[aria-label^="Help for"] svg {{ color:var(--text) !important; }}
    [data-testid="stCheckbox"] label:has(input[aria-label*="Dark mode"]) > div:first-child {{
        background:var(--surface2) !important; border:1px solid var(--border) !important;
    }}
    [data-testid="stCheckbox"] label:has(input[aria-label*="Dark mode"]:checked) > div:first-child {{
        background:var(--accent) !important; border-color:var(--accent) !important;
    }}
    .hero {{ padding:1.45rem 1.6rem; border:1px solid var(--border); border-radius:18px;
        background:linear-gradient(135deg,var(--surface) 0%,var(--accent-soft) 145%);
        box-shadow:0 10px 28px #0000000c; margin-bottom:1rem; position:relative; overflow:hidden; }}
    .hero:after {{ content:'❄'; position:absolute; right:1.4rem; top:.15rem; font-size:6.8rem;
        color:var(--accent); opacity:.10; transform:rotate(12deg); }}
    .eyebrow {{ color:var(--accent); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }}
    .hero h1 {{ color:var(--text); margin:.28rem 0 .25rem; font-size:clamp(1.9rem,4vw,2.75rem); letter-spacing:-.045em; }}
    .hero p {{ color:var(--muted); max-width:790px; margin:0; font-size:.96rem; }}
    .client-badge {{ display:inline-block; margin-left:.55rem; padding:.2rem .55rem; border-radius:999px;
        background:var(--accent-soft); color:var(--accent); font-size:.68rem; letter-spacing:.08em; vertical-align:middle; }}
    [data-testid="stMetric"] {{ background:var(--surface); padding:1rem 1.05rem; border:1px solid var(--border);
        border-radius:14px; box-shadow:0 6px 18px #00000008; }}
    [data-testid="stMetricLabel"] p {{ color:var(--muted) !important; }}
    [data-testid="stMetricValue"] {{ color:var(--text); letter-spacing:-.035em; }}
    [data-testid="stMetricDelta"] {{ color:var(--muted); }}
    .callout {{ color:var(--text); background:var(--positive-soft); border:1px solid var(--positive);
        border-radius:14px; padding:1rem 1.15rem; }}
    .warning {{ background:var(--warning-soft); border-color:var(--warning); }}
    .source-card {{ color:var(--text); background:var(--surface); border:1px solid var(--border); border-radius:13px; padding:.9rem 1rem; margin:.55rem 0; }}
    .source-card a {{ color:var(--accent); }}
    div[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:13px; overflow:hidden; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid var(--border); }}
    .stTabs [data-baseweb="tab"] {{ color:var(--muted); background:transparent; border-radius:9px 9px 0 0; padding:.4rem .8rem; }}
    .stTabs [aria-selected="true"] {{ color:var(--accent) !important; background:var(--accent-soft); }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ background-color:var(--accent) !important; }}
    [data-testid="stExpander"] {{ background:var(--surface) !important; border-color:var(--border) !important; border-radius:12px; }}
    [data-testid="stExpander"] details, [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] summary > div {{ color:var(--text) !important; background:var(--surface2) !important; }}
    [data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover > div {{ background:var(--accent-soft) !important; }}
    [data-testid="stExpander"] summary *, [data-testid="stExpander"] summary svg {{ color:var(--text) !important; fill:currentColor !important; }}
    .stAlert {{ background:var(--surface); color:var(--text); border-color:var(--border); }}
    code {{ color:var(--text) !important; background:var(--surface2) !important; }}
    footer {{ visibility:hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class Dist:
    mean: float
    sd: float
    low: float = 0.0
    high: float | None = None


def peso(x: float, decimals: int = 0) -> str:
    if not np.isfinite(x):
        return "—"
    return f"₱{x:,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{100*x:.{decimals}f}%"


def draw(rng: np.random.Generator, d: Dist, n: int) -> np.ndarray:
    if d.sd <= 0:
        out = np.full(n, d.mean, dtype=float)
    else:
        out = rng.normal(d.mean, d.sd, n)
    return np.clip(out, d.low, d.high if d.high is not None else np.inf)


def ninput(label, value, step=1.0, min_value=0.0, help=None, key=None):
    return st.number_input(
        label, min_value=float(min_value), value=float(value), step=float(step),
        help=help, key=key, format="%.2f" if abs(step) < 1 else "%.0f"
    )


SIMULATION_SD_CONTROLS: list[dict] = []


def mean_sd(label, mean, sd, step=1.0, min_value=0.0, help=None, key=""):
    """Render the business value here; keep uncertainty controls in Simulation."""
    m = ninput(label, mean, step, min_value, help, f"{key}_m")
    sd_key = f"simulation_sd_{key}"
    s = float(st.session_state.get(sd_key, sd))
    SIMULATION_SD_CONTROLS.append({"label": label, "default": sd, "step": step, "key": sd_key})
    return Dist(m, s, min_value)


with st.sidebar:
    st.markdown("## ⚡ E-Ride model")
    st.caption("Type any operating scale. All amounts are monthly unless marked daily or per ride.")
    st.caption("💡 Click any number field to type a value directly; the −/+ buttons are optional.")
    seed = int(st.session_state.get("simulation_seed", 42))
    simulations = int(st.session_state.get("simulation_trials", 5000))

    with st.expander("Demand & fleet", expanded=True):
        users = mean_sd("Active users", 3750, 600, 100, 1, "Type any number of monthly transacting users.", "users")
        riders = mean_sd("Number of riders / vehicles", 50, 3, 1, 1, "Type any fleet size; one salaried rider per active vehicle.", "riders")
        rides_user = mean_sd("Rides per user / month", 4, 1, 1, 1, "Enter the expected number of complete rides taken by one active user in a month.", "rpu")
        rides_rider = mean_sd("Rides per rider / day", 10.0, 2.0, .5, .1, "Internal planning prior and capacity constraint.", "rpd")
        days = mean_sd("Operating days / month", 26.0, 1.0, 1, 1, None, "days")
        completion = mean_sd("Completion rate (%)", 90.0, 4.0, 1, 1, "Applied after matching demand to capacity.", "completion")
        avg_km = mean_sd("Average paid trip distance (km)", 6.0, 1.5, .5, .5, None, "km")

    with st.expander("Fare engine & rider pay", expanded=True):
        st.caption("Editable Angkas-style structure: base + distance + time, then an E-Ride discount.")
        base_fare = mean_sd("Base fare", 50, 5, 1, 1, "Planning benchmark for the first 2 km.", "basefare")
        included_km = ninput("Distance included in base fare (km)", 2, .5, 0, key="included_km")
        per_km = mean_sd("Charge / km after base", 10, 1.5, 1, 0, "Pilot benchmark through 7 km; actual app fares can vary.", "perkm")
        long_km_threshold = ninput("Long-distance threshold (km)", 7, .5, 0, key="long_km")
        long_per_km = mean_sd("Charge / km after threshold", 15, 2, 1, 0, "Documented pilot benchmark beyond 7 km.", "longperkm")
        avg_minutes = mean_sd("Average trip duration (minutes)", 25, 6, 1, 1, "Paid trip time from pickup to drop-off.", "minutes")
        included_minutes = ninput("Minutes included in base fare", 10, 1, 0, key="included_minutes")
        per_minute = mean_sd("Charge / additional minute", 1, .25, .25, 0, "Editable planning allowance, not a verified current Angkas tariff.", "perminute")
        eride_discount = ninput("E-Ride electric discount (%)", 10, 1, 0, "Discount applied to the calculated benchmark fare.", "fare_discount") / 100
        company_share = ninput("Company share of fare (%)", 100, 1, 1, "100% for owned fleet + salaried riders; 12–20% for a marketplace.", "share") / 100
        user_fee = ninput("Monthly membership / user fee", 0, 5, 0, "Optional non-fare revenue.", "ufee")
        wage = mean_sd("Rider daily base wage", 695, 25, 5, 1, "NCR non-agriculture statutory minimum is ₱695/day effective 18 July 2025.", "wage")
        burden = ninput("Employer burden / benefits (%)", 15, 1, 0, "Planning allowance for statutory and welfare costs.", "burden") / 100
        incentive_threshold = ninput("Daily gross incentive threshold", 3000, 100, 0, "Commission applies only to gross fares above this per-rider daily threshold.", "threshold")
        incentive_rate = ninput("Commission above threshold (%)", 20, 1, 0, key="inc_rate") / 100

    with st.expander("Vehicle & energy"):
        motorcycle_cost = mean_sd("E-motorcycle cost", 80000, 8000, 5000, 1, "User planning figure; primary battery assumed included.", "bike")
        backup_battery = mean_sd("Backup battery cost / bike", 20000, 3000, 1000, 0, "One backup battery per motorcycle.", "battery")
        asset_life = ninput("Asset recovery period (months)", 48, 1, 1, key="life")
        maintenance = mean_sd("Maintenance / bike", 1500, 350, 100, 0, "Internal range was ₱1k–₱2k per month.", "maint")
        insurance = mean_sd("Insurance, registration / bike", 500, 100, 50, 0, None, "ins")
        kwh_km = mean_sd("Energy use (kWh / km)", .045, .008, .005, .005, "Planning prior for small electric motorcycles; replace with fleet telemetry.", "kwh")
        power_rate = mean_sd("Electricity rate / kWh", 13.0, 1.5, .5, 1, "Blended charging assumption; varies by tariff and charging losses.", "power")

    with st.expander("Technology & operations"):
        build_cost = mean_sd("Initial development cost", 1652000, 165000, 50000, 0, "Audited roadmap recommended MVP budget.", "build")
        build_recovery = ninput("Development recovery period (months)", 36, 1, 1, key="build_life")
        dev_retainer = mean_sd("Ongoing development retainer", 55000, 10000, 5000, 0, "Working tier in Dev Costing Foundation.", "dev")
        poc_cloud_base = ninput("POC cloud baseline", 1500, 100, 0, "Developer-confirmed POC budget; workbook hosted-fallback estimate is approximately ₱1,450.", "poc_cloud")
        ops_fixed = mean_sd("Operations, support & admin", 120000, 25000, 5000, 0, "Excludes rider wages, dev, cloud, and vehicle costs.", "ops")
        support_user = mean_sd("Support cost / active user", 2.0, .5, .1, 0, None, "support")
        marketing = mean_sd("Marketing / active user", 5.0, 2.0, .5, 0, None, "mkt")

    with st.expander("Transaction & reserves"):
        digital_share = ninput("Digital payment share (%)", 35, 1, 0, key="digital") / 100
        payment_fee = ninput("Payment fee on digital fares (%)", 2.5, .1, 0, key="payfee") / 100
        regulatory_reserve = ninput("Tax / regulatory reserve on revenue (%)", 3.0, .5, 0, key="reserve") / 100


def excel_cloud_cost(active_users):
    """Pilot-cloud growth model using only Pilot Cloud Costs workbook inputs."""
    users_ = np.asarray(active_users, dtype=float)
    pilot_days = 21.0
    php_per_usd = 58.0
    free_events = 10_000.0
    # Workbook quantities for 20 users / 21 days imply 5, 2, 1 and 2
    # daily events per user respectively.
    usage_per_user_day = np.array([5.0, 2.0, 1.0, 2.0])
    usd_per_1000 = np.array([2.83, 5.0, 5.0, 5.0])
    events = np.expand_dims(users_, axis=-1) * pilot_days * usage_per_user_day
    overage = np.maximum(events - free_events, 0.0)
    variable_php = np.sum(overage / 1000.0 * usd_per_1000, axis=-1) * php_per_usd
    return poc_cloud_base + variable_php


rng = np.random.default_rng(seed)
n = simulations
samples = {
    "Active users": draw(rng, users, n),
    "Riders": draw(rng, riders, n),
    "Rides/user": draw(rng, rides_user, n),
    "Rides/rider/day": draw(rng, rides_rider, n),
    "Operating days": draw(rng, days, n),
    "Completion rate": np.clip(draw(rng, completion, n) / 100, .01, 1),
    "Trip distance": draw(rng, avg_km, n),
    "Trip minutes": draw(rng, avg_minutes, n),
    "Base fare": draw(rng, base_fare, n),
    "Per km": draw(rng, per_km, n),
    "Long-distance / km": draw(rng, long_per_km, n),
    "Per minute": draw(rng, per_minute, n),
    "Daily wage": draw(rng, wage, n),
    "E-bike cost": draw(rng, motorcycle_cost, n),
    "Backup battery": draw(rng, backup_battery, n),
    "Maintenance": draw(rng, maintenance, n),
    "Insurance": draw(rng, insurance, n),
    "Energy intensity": draw(rng, kwh_km, n),
    "Electricity rate": draw(rng, power_rate, n),
    "Initial development": draw(rng, build_cost, n),
    "Dev retainer": draw(rng, dev_retainer, n),
    "Fixed operations": draw(rng, ops_fixed, n),
    "Support / user": draw(rng, support_user, n),
    "Marketing / user": draw(rng, marketing, n),
}

samples["Average fare"] = np.maximum(
    samples["Base fare"],
    samples["Base fare"]
    + np.minimum(
        np.maximum(samples["Trip distance"] - included_km, 0),
        max(long_km_threshold - included_km, 0),
    ) * samples["Per km"]
    + np.maximum(samples["Trip distance"] - long_km_threshold, 0) * samples["Long-distance / km"]
    + np.maximum(samples["Trip minutes"] - included_minutes, 0) * samples["Per minute"],
) * (1 - eride_discount)

u = samples["Active users"]
r = samples["Riders"]
demand = u * samples["Rides/user"]
capacity = r * samples["Rides/rider/day"] * samples["Operating days"]
matched = np.minimum(demand, capacity)
trips = matched * samples["Completion rate"]
gross_fares = trips * samples["Average fare"]
revenue = gross_fares * company_share + u * user_fee

rider_salary = r * samples["Daily wage"] * samples["Operating days"] * (1 + burden)
gross_rider_day = gross_fares / np.maximum(r * samples["Operating days"], 1)
incentive = np.maximum(gross_rider_day - incentive_threshold, 0) * incentive_rate * r * samples["Operating days"]
energy = trips * samples["Trip distance"] * samples["Energy intensity"] * samples["Electricity rate"]
fleet = r * (samples["E-bike cost"] + samples["Backup battery"]) / asset_life
fleet += r * (samples["Maintenance"] + samples["Insurance"])
development = samples["Initial development"] / build_recovery + samples["Dev retainer"]
technology = excel_cloud_cost(u)
operations = samples["Fixed operations"] + u * (samples["Support / user"] + samples["Marketing / user"])
payment = gross_fares * company_share * digital_share * payment_fee
reserve = revenue * regulatory_reserve
total_cost = rider_salary + incentive + energy + fleet + development + technology + operations + payment + reserve
profit = revenue - total_cost

results = pd.DataFrame({
    "Active users": u, "Riders": r, "Demand rides": demand, "Capacity rides": capacity,
    "Completed rides": trips, "Average fare": samples["Average fare"], "Revenue": revenue,
    "Rider salary": rider_salary, "Rider incentive": incentive, "Energy": energy,
    "Fleet": fleet, "Development": development, "Technology": technology,
    "Operations": operations, "Payments & reserve": payment + reserve,
    "Total cost": total_cost, "Profit": profit,
    "Revenue / user": revenue / np.maximum(u, 1),
    "Required revenue / user": total_cost / np.maximum(u, 1),
    "Profit / user": profit / np.maximum(u, 1),
    "Fleet utilization": trips / np.maximum(capacity, 1),
})


def deterministic_model(test_fare: float) -> dict[str, float]:
    users_ = users.mean
    riders_ = riders.mean
    days_ = days.mean
    demand_ = users_ * rides_user.mean
    capacity_ = riders_ * rides_rider.mean * days_
    trips_ = min(demand_, capacity_) * completion.mean / 100
    gross_ = trips_ * test_fare
    rev_ = gross_ * company_share + users_ * user_fee
    salaries_ = riders_ * wage.mean * days_ * (1 + burden)
    gpd_ = gross_ / max(riders_ * days_, 1)
    inc_ = max(gpd_ - incentive_threshold, 0) * incentive_rate * riders_ * days_
    energy_ = trips_ * avg_km.mean * kwh_km.mean * power_rate.mean
    fleet_ = riders_ * (motorcycle_cost.mean + backup_battery.mean) / asset_life
    fleet_ += riders_ * (maintenance.mean + insurance.mean)
    dev_ = build_cost.mean / build_recovery + dev_retainer.mean
    tech_ = float(excel_cloud_cost(users_))
    ops_ = ops_fixed.mean + users_ * (support_user.mean + marketing.mean)
    pay_ = gross_ * company_share * digital_share * payment_fee
    reserve_ = rev_ * regulatory_reserve
    cost_ = salaries_ + inc_ + energy_ + fleet_ + dev_ + tech_ + ops_ + pay_ + reserve_
    return {
        "users": users_, "riders": riders_, "days": days_, "demand": demand_, "capacity": capacity_,
        "trips": trips_, "fare": test_fare, "gross_fares": gross_, "revenue": rev_,
        "rider_salary": salaries_, "incentive": inc_, "energy": energy_, "fleet": fleet_,
        "development": dev_, "technology": tech_, "operations": ops_, "payment": pay_,
        "reserve": reserve_, "total_cost": cost_, "profit": rev_ - cost_,
        "utilization": trips_ / max(capacity_, 1),
        "revenue_user": rev_ / max(users_, 1), "cost_user": cost_ / max(users_, 1),
        "profit_ride": (rev_ - cost_) / max(trips_, 1),
        "gross_rider_day": gross_ / max(riders_ * days_, 1),
    }


def deterministic_profit(test_fare: float) -> float:
    return deterministic_model(test_fare)["profit"]


lo, hi = 0.0, 2000.0
for _ in range(60):
    mid = (lo + hi) / 2
    if deterministic_profit(mid) >= 0:
        hi = mid
    else:
        lo = mid
break_even_fare = hi if deterministic_profit(hi) >= 0 else math.nan

deterministic_fare = (
    base_fare.mean
    + min(max(avg_km.mean - included_km, 0), max(long_km_threshold - included_km, 0)) * per_km.mean
    + max(avg_km.mean - long_km_threshold, 0) * long_per_km.mean
    + max(avg_minutes.mean - included_minutes, 0) * per_minute.mean
) * (1 - eride_discount)
scenario = deterministic_model(deterministic_fare)

mean_profit = float(results["Profit"].mean())
risk_loss = float((results["Profit"] < 0).mean())
required_user = float(results["Required revenue / user"].mean())
earned_user = float(results["Revenue / user"].mean())
monthly_gap_user = max(required_user - earned_user, 0)
expected_fare = float(results["Average fare"].mean())

st.markdown(
    """
<section class="hero">
  <div class="eyebrow">E-Ride decision dashboard <span class="client-badge">CLIENT VIEW</span></div>
  <h1>What must E-Ride earn per user?</h1>
  <p>A clear view of the revenue, fare and operating scale needed to keep the electric motorcycle service sustainable.</p>
</section>
""",
    unsafe_allow_html=True,
)

if company_share < .5:
    st.markdown('<div class="callout warning"><b>Marketplace warning:</b> At this revenue share, E-Ride still carries salaries and motorcycles but keeps only a small share of fares. That hybrid is usually structurally unprofitable.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Profit calculator", "Simulation", "Assumptions & evidence", "Model notes"])

with tabs[0]:
    st.caption("Deterministic base case · calculated directly from the editable mean settings in the sidebar")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly revenue", peso(scenario["revenue"]), f'{scenario["trips"]:,.0f} completed rides')
    c2.metric("Monthly total cost", peso(scenario["total_cost"]), f'{peso(scenario["cost_user"], 2)} / user')
    c3.metric("Monthly profit / loss", peso(scenario["profit"]), f'{peso(scenario["profit_ride"], 2)} / ride')
    c4.metric("Break-even fare", peso(break_even_fare, 2), f'current calculated fare {peso(scenario["fare"], 2)}')

    st.markdown("### Scenario settings")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active users", f'{scenario["users"]:,.0f}', f'{rides_user.mean:.0f} rides/user/month')
    s2.metric("Vehicles and riders", f'{scenario["riders"]:,.0f}', f'{rides_rider.mean:.1f} rides/rider/day')
    s3.metric("Passenger demand", f'{scenario["demand"]:,.0f} rides', "before completion")
    s4.metric("Fleet capacity", f'{scenario["capacity"]:,.0f} rides', f'{pct(scenario["utilization"])} utilized')

    capacity_rides_per_user = scenario["capacity"] / max(scenario["users"], 1)
    if scenario["demand"] >= scenario["capacity"]:
        unserved = scenario["demand"] - scenario["capacity"]
        st.warning(
            f"Fleet-constrained: demand exceeds capacity by {unserved:,.0f} rides/month. "
            f"At the current user count, profit stops increasing above approximately "
            f"{capacity_rides_per_user:.2f} requested rides per user/month unless riders, "
            f"rides per rider/day, or operating days also increase."
        )
    else:
        spare = scenario["capacity"] - scenario["demand"]
        st.info(
            f"Demand-constrained: the fleet has room for approximately {spare:,.0f} more requested rides/month. "
            "Increasing users or monthly rides can still increase completed rides and revenue."
        )

    st.caption(
        f"Current fare design: {peso(base_fare.mean)} base including {included_km:g} km, "
        f"then {peso(per_km.mean, 2)}/km through {long_km_threshold:g} km, {peso(long_per_km.mean, 2)}/km thereafter, "
        f"and {peso(per_minute.mean, 2)}/minute after "
        f"{included_minutes:g} minutes, less a {pct(eride_discount, 0)} electric-service discount. "
        f"Calculated fare for a {avg_km.mean:g} km, {avg_minutes.mean:g}-minute trip: {peso(scenario['fare'], 2)}."
    )
    with st.expander("Show fare calculation"):
        short_distance_km = min(max(avg_km.mean - included_km, 0), max(long_km_threshold - included_km, 0))
        long_distance_km = max(avg_km.mean - long_km_threshold, 0)
        charged_minutes = max(avg_minutes.mean - included_minutes, 0)
        distance_charge = short_distance_km * per_km.mean + long_distance_km * long_per_km.mean
        time_charge = charged_minutes * per_minute.mean
        fare_before_discount = base_fare.mean + distance_charge + time_charge
        discount_value = fare_before_discount * eride_discount
        fare_rows = pd.DataFrame({
            "Fare component": ["Base fare", "Distance charge", "Time charge", "E-Ride electric discount", "Final passenger fare"],
            "Calculation": [
                f"Includes first {included_km:g} km",
                f"{short_distance_km:g} km × {peso(per_km.mean,2)} + {long_distance_km:g} km × {peso(long_per_km.mean,2)}",
                f"{charged_minutes:g} min × {peso(per_minute.mean,2)}",
                f"{pct(eride_discount,0)} × {peso(fare_before_discount,2)}",
                "Base + distance + time − discount",
            ],
            "PHP": [base_fare.mean, distance_charge, time_charge, -discount_value, scenario["fare"]],
        })
        st.dataframe(fare_rows.style.format({"PHP": "₱{:,.2f}"}), width="stretch", hide_index=True)

    st.markdown("### The decision")
    if scenario["profit"] >= 0:
        st.markdown(
            f'<div class="callout"><b>Profitable base case:</b> This setting produces <b>{peso(scenario["profit"])}</b> monthly profit. '
            f'E-Ride earns {peso(scenario["revenue_user"],2)} per user against {peso(scenario["cost_user"],2)} required.</div>',
            unsafe_allow_html=True,
        )
    else:
        gap_user = max(scenario["cost_user"] - scenario["revenue_user"], 0)
        st.markdown(
            f'<div class="callout warning"><b>Unprofitable base case:</b> This setting loses <b>{peso(abs(scenario["profit"]))}</b> per month '
            f'and is short by {peso(gap_user,2)} per active user. The calculated fare is {peso(scenario["fare"],2)} versus a '
            f'{peso(break_even_fare,2)} break-even fare.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Monthly cost breakdown")
    cost_rows = pd.DataFrame({
        "Cost category": ["Rider base payroll", "Rider incentives", "Electricity", "Vehicles, batteries & maintenance",
                          "Development", "POC cloud & Google APIs", "Operations, support & marketing",
                          "Payment processing", "Tax / regulatory reserve"],
        "Monthly PHP": [scenario["rider_salary"], scenario["incentive"], scenario["energy"], scenario["fleet"],
                        scenario["development"], scenario["technology"], scenario["operations"],
                        scenario["payment"], scenario["reserve"]],
    })
    cost_rows["Share of cost"] = cost_rows["Monthly PHP"] / max(scenario["total_cost"], 1)
    cost_rows["PHP / completed ride"] = cost_rows["Monthly PHP"] / max(scenario["trips"], 1)
    left, right = st.columns([1.05, 1])
    with left:
        st.dataframe(
            cost_rows.style.format({"Monthly PHP": "₱{:,.0f}", "Share of cost": "{:.1%}", "PHP / completed ride": "₱{:,.2f}"}),
            width="stretch", hide_index=True,
        )
    with right:
        cost_chart = alt.Chart(cost_rows).mark_bar(cornerRadiusEnd=5, color=theme["accent"]).encode(
            x=alt.X("Monthly PHP:Q", title="PHP per month"),
            y=alt.Y("Cost category:N", sort="-x", title=None),
            tooltip=["Cost category", alt.Tooltip("Monthly PHP:Q", format=",.0f")],
        ).properties(height=330, background=theme["surface"]).configure_axis(
            labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"]
        ).configure_view(strokeOpacity=0)
        st.altair_chart(cost_chart, width="stretch")

    st.markdown("### Cloud-cost growth by active users")
    max_scale_users = max(10_000, int(scenario["users"] * 2))
    scale_users = np.unique(np.concatenate(([0, 20, 100, 250, 500, 1000, int(scenario["users"])], np.linspace(0, max_scale_users, 140).astype(int))))
    cloud_scale = pd.DataFrame({"Active users": scale_users, "Cloud cost": excel_cloud_cost(scale_users)})
    cloud_line = alt.Chart(cloud_scale).mark_line(color=theme["accent"], strokeWidth=3).encode(
        x=alt.X("Active users:Q", title="Active users"),
        y=alt.Y("Cloud cost:Q", title="PHP per 21-day pilot window"),
        tooltip=[alt.Tooltip("Active users:Q", format=",.0f"), alt.Tooltip("Cloud cost:Q", format=",.0f")],
    )
    current_cloud = alt.Chart(pd.DataFrame({"Active users": [scenario["users"]], "Cloud cost": [scenario["technology"]]})).mark_point(
        color=theme["negative"], filled=True, size=110
    ).encode(x="Active users:Q", y="Cloud cost:Q", tooltip=["Active users", "Cloud cost"])
    cloud_chart = (cloud_line + current_cloud).properties(height=280, background=theme["surface"]).configure_axis(
        labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"]
    ).configure_view(strokeOpacity=0)
    st.altair_chart(cloud_chart, width="stretch")
    checkpoints = sorted(set([20, 100, 500, 1000, int(scenario["users"]), 5000, 10000]))
    cloud_table = pd.DataFrame({"Active users": checkpoints, "Estimated cloud cost": excel_cloud_cost(np.array(checkpoints))})
    st.dataframe(cloud_table.style.format({"Active users": "{:,.0f}", "Estimated cloud cost": "₱{:,.0f}"}), width="stretch", hide_index=True)
    st.caption(
        "Workbook-only growth basis: 21 days, ₱58/USD, 10,000 free monthly events per API; "
        "5 autocomplete, 2 place-detail, 1 geocoding and 2 route events per user/day. "
        "The workbook's listed post-free-tier rates are extended linearly for planning beyond its stated 100k-event tier."
    )

    st.markdown("### Profit calculation")
    profit_bridge = pd.DataFrame({
        "Line": ["Fare and membership revenue", "Total monthly costs", "Net monthly profit / loss"],
        "PHP": [scenario["revenue"], -scenario["total_cost"], scenario["profit"]],
    })
    bridge = alt.Chart(profit_bridge).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("PHP:Q", title="PHP per month"), y=alt.Y("Line:N", sort=None, title=None),
        color=alt.condition(alt.datum.PHP >= 0, alt.value(theme["positive"]), alt.value(theme["negative"])),
        tooltip=["Line", alt.Tooltip("PHP:Q", format=",.0f")],
    ).properties(height=160, background=theme["surface"]).configure_axis(
        labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"]
    ).configure_view(strokeOpacity=0)
    st.altair_chart(bridge, width="stretch")

with tabs[1]:
    st.markdown("### Simulation settings")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.number_input("Random seed", min_value=0, value=42, step=1, key="simulation_seed")
    with sc2:
        st.select_slider("Simulation trials", [1000, 2500, 5000, 10000, 20000], value=5000, key="simulation_trials")
    with st.expander("Uncertainty assumptions (standard deviations)"):
        st.caption("These affect only the simulation. The Profit calculator uses the exact business values from the sidebar.")
        sd_columns = st.columns(3)
        for index, control in enumerate(SIMULATION_SD_CONTROLS):
            with sd_columns[index % 3]:
                step = float(control["step"])
                st.number_input(
                    f'{control["label"]} · stdev', min_value=0.0,
                    value=float(control["default"]), step=step, key=control["key"],
                    format="%.3f" if step < .01 else ("%.2f" if step < 1 else "%.0f"),
                )

    st.markdown("### Simulation results")
    st.caption(f"{simulations:,} uncertain operating months using the settings above")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Probability of loss", pct(risk_loss), "simulated months below ₱0")
    mc2.metric("Mean simulated profit", peso(mean_profit), f"median {peso(results['Profit'].median())}")
    mc3.metric("P10 downside", peso(results["Profit"].quantile(.10)), "90% of trials are higher")
    mc4.metric("P90 upside", peso(results["Profit"].quantile(.90)), "10% of trials are higher")

    st.markdown("### Profit distribution")
    hist = alt.Chart(results).mark_bar(color=theme["accent"], opacity=.88).encode(
        x=alt.X("Profit:Q", bin=alt.Bin(maxbins=55), title="Monthly profit / loss (PHP)"),
        y=alt.Y("count():Q", title="Trials"),
        tooltip=[alt.Tooltip("count():Q", title="Trials")],
    ).properties(height=350)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=theme["negative"], strokeWidth=2).encode(x="x:Q")
    profit_chart = (hist + zero).properties(background=theme["surface"]).configure_axis(
        labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"]
    ).configure_view(strokeOpacity=0)
    st.altair_chart(profit_chart, width="stretch")

    q = results["Profit"].quantile([.05, .10, .50, .90, .95])
    cols = st.columns(5)
    for col, label, val in zip(cols, ["P05 stress", "P10 downside", "Median", "P90 upside", "P95 upside"], q.values):
        col.metric(label, peso(float(val)))

    st.markdown("### What moves profit most?")
    sensitivities = []
    for name, arr in samples.items():
        if np.std(arr) > 0:
            corr = np.corrcoef(arr, profit)[0, 1]
            if np.isfinite(corr):
                sensitivities.append((name, corr))
    sens = pd.DataFrame(sensitivities, columns=["Variable", "Correlation"])
    sens["Absolute impact"] = sens["Correlation"].abs()
    sens = sens.nlargest(12, "Absolute impact").sort_values("Correlation")
    sens_chart = alt.Chart(sens).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("Correlation:Q", scale=alt.Scale(domain=[-1, 1]), title="Correlation with monthly profit"),
        y=alt.Y("Variable:N", sort=None, title=None),
        color=alt.condition(alt.datum.Correlation >= 0, alt.value(theme["positive"]), alt.value(theme["negative"])),
        tooltip=["Variable", alt.Tooltip("Correlation:Q", format=".2f")],
    ).properties(height=350, background=theme["surface"]).configure_axis(
        labelColor=theme["text"], titleColor=theme["muted"], gridColor=theme["border"]
    ).configure_view(strokeOpacity=0)
    st.altair_chart(sens_chart, width="stretch")
    st.caption("Correlation is a screening measure, not proof of causality. Large absolute values identify assumptions worth validating first.")

    export = results.round(4).to_csv(index=False).encode("utf-8")
    st.download_button("Download simulation trials (CSV)", export, "eride_simulation_trials.csv", "text/csv")

with tabs[2]:
    st.markdown("### Editable priors used in the dashboard")
    assumptions = pd.DataFrame([
        ["Active users", users.mean, users.sd, "Internal scale case", "50 riders × 300 rides/month ÷ 4 rides/user"],
        ["Rides / active user / month", rides_user.mean, rides_user.sd, "Internal audited roadmap", "Direct monthly ride-frequency input; needs pilot telemetry"],
        ["Rides / rider / day", rides_rider.mean, rides_rider.sd, "Internal audited roadmap", "Planning prior; capacity, not guaranteed demand"],
        ["Calculated passenger fare", expected_fare, float(results["Average fare"].std()), "Editable fare engine", "₱50 base + distance + time, less E-Ride electric discount"],
        ["Rider daily wage", wage.mean, wage.sd, "NWPC NCR-26", "₱695 non-agriculture; ₱658 agriculture/small retail"],
        ["E-motorcycle", motorcycle_cost.mean, motorcycle_cost.sd, "User estimate", "Primary battery included; backup separate"],
        ["Development build", build_cost.mean, build_cost.sd, "Audited roadmap", "Recommended MVP: ₱1.652m"],
        ["Ongoing dev retainer", dev_retainer.mean, dev_retainer.sd, "Dev Costing Foundation", "Working tier: ₱55k/month"],
        ["Fixed operations", ops_fixed.mean, ops_fixed.sd, "Audited roadmap", "50-driver app-ops range was ₱120k–₱250k"],
    ], columns=["Variable", "Mean", "Stdev", "Basis", "Interpretation"])
    st.dataframe(assumptions, width="stretch", hide_index=True)

    st.markdown("### Evidence and source quality")
    st.markdown(
        """
<div class="source-card"><b>Statutory wage · high confidence</b><br>
NWPC lists NCR Wage Order NCR-26, effective 18 July 2025: ₱695/day for non-agriculture and ₱658/day for agriculture and covered small establishments.<br>
<a href="https://nwpc.dole.gov.ph/ncr/" target="_blank">National Wages and Productivity Commission — NCR</a></div>

<div class="source-card"><b>Motorcycle-taxi fare benchmark · medium confidence / dated regulation</b><br>
The Philippine pilot benchmark started at ₱50 for the first 2 km, then ₱10/km through 7 km and ₱15/km thereafter. Actual app prices and rules can differ today. The dashboard therefore makes base fare, distance, duration, time charge, and E-Ride discount editable; its per-minute default is a planning assumption rather than a claimed Angkas tariff.<br>
<a href="https://visor.ph/traffic/fare-for-pilot-run-of-motorcycle-taxis-is-p50-through-first-2km/" target="_blank">VISOR — pilot fare schedule</a></div>

<div class="source-card"><b>Supply / market scale · medium confidence</b><br>
The 2020 pilot registered 46,713 riders across participating operators. This is evidence of supply scale, not daily demand or E-Ride's obtainable market.<br>
<a href="https://www.pna.gov.ph/articles/1093673" target="_blank">Philippine News Agency — registered pilot riders</a></div>

<div class="source-card"><b>Internal operating priors · planning confidence only</b><br>
The audited E-Ride roadmap assumes 10 trips/rider-day, 30 days/month, ₱125 fare, 4 trips/passenger-month, and staged operating-cost bands. The dashboard starts from those figures but exposes the mean and uncertainty because no public source reliably reports comparable operator-level daily demand.</div>

<div class="source-card"><b>Cloud · architecture estimate</b><br>
The Dev Costing Foundation workbook's Pilot Cloud Costs sheet supplies the dashboard's cloud-growth inputs: ₱58/USD, a 21-day pilot, 10,000 free monthly events per priced Google API, five searches per user/day, and its listed post-free-tier API rates. The ₱1,500 POC baseline reflects the developer-confirmed rounded hosted-fallback budget.</div>
""",
        unsafe_allow_html=True,
    )
    st.warning("Important research finding: credible public Philippine sources do not publish a stable mean and standard deviation for rides per active user or rides per rider-day. Those values are modeled priors, not measured market facts. Replace them with E-Ride pilot telemetry as soon as possible.")

with tabs[3]:
    st.markdown("### What this answers")
    st.write(
        "The main answer is **required monthly company revenue per active user**: all modeled monthly costs divided by monthly transacting users. "
        "It is compared with expected fare and membership revenue per user. The dashboard also solves for the average fare that makes expected profit zero."
    )
    st.markdown("### Core equation")
    st.code(
        "completed rides = min(users × rides/user/month, riders × rides/rider/day × operating days) × completion rate\n"
        "cloud cost = ₱1,500 POC baseline + workbook-priced API usage above each 10,000-event free cap\n"
        "profit = retained fares + user fees − payroll − incentives − energy − fleet − development − technology − operations − fees/reserves\n"
        "required revenue/user = total monthly cost ÷ active users"
    )
    st.markdown("### Modeling choices")
    st.write(
        "Independent truncated-normal draws mirror the supplied simulation workbook while preventing impossible negative counts and costs. "
        "Demand and supply are reconciled in every trial. Development and vehicles are amortized as economic costs, so the result tests whether the business can replace its assets—not merely survive this month's cash bill."
    )
    st.markdown("### Deliberate cautions")
    st.write(
        "This is a planning model, not a valuation or a regulatory opinion. Standard deviations express uncertainty, not historical volatility. "
        "Income tax, VAT treatment, financing interest, depot rent, battery degradation, downtime, safety incidents, and city-specific permits may need dedicated lines once commercial and legal structures are fixed."
    )
    st.markdown("### Recommended pilot measurement")
    st.write(
        "Track requests, accepts, completions, cancellations, occupied km, deadhead km, online hours, energy per km, revenue per rider-hour, support contacts, and 7/30-day passenger repeat rates. "
        "After 4–8 weeks, replace the dashboard priors with observed distributions and model weekday, weekend, peak, rain, and location effects separately."
    )

st.markdown('<p class="small-note">E-Ride decision model · Philippine pesos · Simulation results change with the settings in the Simulation tab.</p>', unsafe_allow_html=True)
