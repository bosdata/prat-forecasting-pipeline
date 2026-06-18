from __future__ import annotations
import streamlit as st

import io
import os
import re
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# SIMPLE LOGIN SECTION
# Created for PRAT Forecasting Pipeline
# ============================================================

USERS = {
    "admin": {
        "password": "Admin@2026",
        "role": "Admin"
    },
    "meal": {
        "password": "Meal@2026",
        "role": "MEAL Manager"
    },
    "viewer": {
        "password": "Viewer@2026",
        "role": "Viewer"
    }
}


def show_login_page():
    st.markdown(
        """
        <style>
        .login-title {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            margin-top: 40px;
            color: #ffffff;
        }
        .login-subtitle {
            text-align: center;
            font-size: 16px;
            color: #cfcfcf;
            margin-bottom: 25px;
        }
        .login-footer {
            position: fixed;
            bottom: 12px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 13px;
            color: #bdbdbd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-title">PRAT Forecasting Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-subtitle">Vibrant Village Foundation Registration Forecasting System</div>',
        unsafe_allow_html=True
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_clicked = st.form_submit_button("Login")

        if login_clicked:
            if username in USERS and password == USERS[username]["password"]:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = USERS[username]["role"]
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    if st.button("Forgot password?"):
        st.warning("Please contact the system administrator to reset your password.")

    st.markdown(
        """
        <div class="login-footer">
            PowerdBy: Bosdata Tech Team | Created by Brian Sifuna Obware
        </div>
        """,
        unsafe_allow_html=True
    )


def require_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        show_login_page()
        st.stop()


def show_logout_button():
    with st.sidebar:
        st.markdown("### User Access")
        st.write(f"**User:** {st.session_state.get('username', '')}")
        st.write(f"**Role:** {st.session_state.get('role', '')}")

        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["role"] = ""
            st.rerun()

    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 10px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 13px;
            color: #999999;
            z-index: 9999;
        ">
            PowerdBy: Bosdata Tech Team | Created by Brian Sifuna Obware
        </div>
        """,
        unsafe_allow_html=True
    )

APP_FOOTER = "Created by Bosdata Tech Team, Brian Sifuna Obware"
VIEW_ORDER = ["Daily", "Weekly", "Monthly", "Quarterly", "Semi-annual", "Annual"]

st.set_page_config(
    page_title="PRAT Forecasting Pipeline",
    page_icon="📈",
    layout="wide",
)

require_login()
show_logout_button()

st.markdown(
    """
    <style>
        .main-title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0.2rem;}
        .small-note {color: #777; font-size: 0.93rem;}
        .metric-card {border: 1px solid #30363d; border-radius: 12px; padding: 14px;}
        footer {visibility: hidden;}
        .bosdata-footer {text-align:center; color:#888; padding: 2rem 0 1rem 0; font-size:0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text_value(value):
    """Clean extra spaces while keeping missing values as missing."""
    if pd.isna(value):
        return value
    return " ".join(str(value).strip().split())


def latex_escape(text):
    text = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_") or "output"


def read_csv_uploaded(uploaded_file):
    return pd.read_csv(uploaded_file)


def detect_default_date_column(columns):
    priority = ["inserted_at", "created_at", "date", "registration_date", "registered_at", "timestamp"]
    lower_map = {c.lower(): c for c in columns}
    for p in priority:
        if p in lower_map:
            return lower_map[p]
    for c in columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "insert" in cl or "created" in cl:
            return c
    return columns[0] if columns else None


def detect_default_grouping_columns(columns):
    defaults = []
    for c in columns:
        if c.lower() in ["county", "sub_county", "ward"]:
            defaults.append(c)
    return defaults[:3]


def detect_default_id_columns(columns):
    defaults = []
    for c in columns:
        cl = c.lower()
        if cl in ["vvid", "id", "farmer_id", "national_id", "id_number", "phone_number", "phone"]:
            defaults.append(c)
    return defaults[:2]


def prepare_data(df_raw: pd.DataFrame, date_col: str, grouping_cols: list[str], id_cols: list[str]):
    df = df_raw.copy()
    original_rows = len(df)

    # Clean column names lightly, but do not rename user data columns silently.
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    changed_cells = 0
    for col in text_cols:
        before = df[col].copy()
        df[col] = df[col].apply(clean_text_value)
        changed_cells += int((before.fillna("__NA__") != df[col].fillna("__NA__")).sum())

    df["_registration_datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    try:
        df["_registration_datetime"] = df["_registration_datetime"].dt.tz_localize(None)
    except Exception:
        pass

    valid = df[df["_registration_datetime"].notna()].copy()
    invalid = df[df["_registration_datetime"].isna()].copy()
    valid = valid.sort_values("_registration_datetime")

    full_duplicate_count = int(df.duplicated().sum())
    duplicate_id_rows = pd.DataFrame()
    duplicate_id_count = 0

    selected_id_cols = [c for c in id_cols if c in df.columns]
    if selected_id_cols:
        key_df = df[selected_id_cols].copy()
        for col in selected_id_cols:
            key_df[col] = key_df[col].astype(str).str.strip()
        non_empty_key = key_df.replace({"": np.nan, "nan": np.nan, "None": np.nan}).notna().all(axis=1)
        dup_mask = key_df[non_empty_key].duplicated(keep=False)
        dup_index = key_df[non_empty_key][dup_mask].index
        duplicate_id_rows = df.loc[dup_index].copy()
        duplicate_id_count = int(len(duplicate_id_rows))

    missing_summary = (
        df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_values"})
    )
    missing_summary["missing_percent"] = np.where(
        len(df) > 0,
        (missing_summary["missing_values"] / len(df) * 100).round(2),
        0,
    )

    group_summaries = {}
    for col in grouping_cols:
        if col in valid.columns:
            # Build grouping summaries in a pandas-version-safe way.
            # Some pandas versions return columns as [col, "count"], while older ones
            # may return ["index", col]. Setting names directly avoids category errors.
            gdf = (
                valid[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .reset_index()
            )
            gdf.columns = [col, "registrations"]
            gdf["registrations"] = pd.to_numeric(gdf["registrations"], errors="coerce").fillna(0).astype(int)
            group_summaries[col] = gdf

    quality = {
        "original_rows": original_rows,
        "valid_date_rows": int(len(valid)),
        "invalid_date_rows": int(len(invalid)),
        "text_cells_cleaned": int(changed_cells),
        "full_duplicate_rows": full_duplicate_count,
        "duplicate_id_rows": duplicate_id_count,
        "date_min": valid["_registration_datetime"].min() if len(valid) else None,
        "date_max": valid["_registration_datetime"].max() if len(valid) else None,
        "selected_grouping_columns": grouping_cols,
        "selected_id_columns": selected_id_cols,
    }
    return df, valid, invalid, missing_summary, duplicate_id_rows, group_summaries, quality


def actual_series(valid: pd.DataFrame, view: str) -> pd.Series:
    if valid.empty:
        return pd.Series(dtype=float, name="registrations")

    dt = valid["_registration_datetime"]
    temp = valid.copy()

    if view == "Daily":
        s = temp.set_index("_registration_datetime").resample("D").size()
        idx = pd.date_range(s.index.min().floor("D"), s.index.max().floor("D"), freq="D")
        return s.reindex(idx, fill_value=0).rename("registrations")

    if view == "Weekly":
        return temp.set_index("_registration_datetime").resample("W-SUN").size().rename("registrations")

    if view == "Monthly":
        return temp.set_index("_registration_datetime").resample("MS").size().rename("registrations")

    if view == "Quarterly":
        return temp.set_index("_registration_datetime").resample("QS-JAN").size().rename("registrations")

    if view == "Semi-annual":
        years = dt.dt.year
        half_start_month = np.where(dt.dt.month <= 6, 1, 7)
        temp["_period_start"] = pd.to_datetime(
            years.astype(str) + "-" + pd.Series(half_start_month, index=temp.index).astype(str).str.zfill(2) + "-01"
        )
        s = temp.groupby("_period_start").size().sort_index()
        return s.rename("registrations")

    if view == "Annual":
        return temp.set_index("_registration_datetime").resample("YS").size().rename("registrations")

    raise ValueError(f"Unknown view: {view}")


def view_label(date_value, view: str) -> str:
    d = pd.Timestamp(date_value)
    if view == "Daily":
        return d.strftime("%d %b %Y")
    if view == "Weekly":
        return d.strftime("Week ending %d %b %Y")
    if view == "Monthly":
        return d.strftime("%b %Y")
    if view == "Quarterly":
        q = ((d.month - 1) // 3) + 1
        return f"Q{q} {d.year}"
    if view == "Semi-annual":
        return f"H1 {d.year}" if d.month <= 6 else f"H2 {d.year}"
    if view == "Annual":
        return str(d.year)
    return str(d.date())


def series_to_table(s: pd.Series, view: str) -> pd.DataFrame:
    if s.empty:
        return pd.DataFrame(columns=["period_start", "period", "registrations"])
    out = s.reset_index()
    out.columns = ["period_start", "registrations"]
    out["period"] = out["period_start"].apply(lambda x: view_label(x, view))
    return out[["period_start", "period", "registrations"]]


def current_pace_ratio(valid: pd.DataFrame) -> tuple[float, str]:
    """Estimate current-year pace using completed months only."""
    if valid.empty:
        return 1.0, "No valid dates were available; a neutral pace ratio of 1.0 was used."

    dt = valid["_registration_datetime"]
    last_date = dt.max()
    current_year = int(last_date.year)
    last_month = int(last_date.month)
    last_day = int(last_date.day)
    days_in_last_month = int(pd.Period(last_date, freq="M").days_in_month)

    complete_months = list(range(1, last_month + 1)) if last_day == days_in_last_month else list(range(1, last_month))
    if not complete_months:
        return 1.0, "There was no completed month in the current year; a neutral pace ratio of 1.0 was used."

    temp = valid.copy()
    temp["year"] = temp["_registration_datetime"].dt.year
    temp["month"] = temp["_registration_datetime"].dt.month

    current_total = int(temp[(temp["year"] == current_year) & (temp["month"].isin(complete_months))].shape[0])
    hist_years = sorted([y for y in temp["year"].unique() if y < current_year])
    hist_totals = []
    for y in hist_years:
        hist_totals.append(int(temp[(temp["year"] == y) & (temp["month"].isin(complete_months))].shape[0]))

    hist_nonzero = [v for v in hist_totals if v > 0]
    if not hist_nonzero:
        return 1.0, "There was no usable historical comparison for the completed months; a neutral pace ratio of 1.0 was used."

    hist_avg = float(np.mean(hist_nonzero))
    ratio = current_total / hist_avg if hist_avg else 1.0
    # Keep the ratio practical; extreme values are usually caused by short history.
    ratio = float(np.clip(ratio, 0.25, 3.0))
    months_text = ", ".join([pd.Timestamp(2000, m, 1).strftime("%b") for m in complete_months])
    note = (
        f"The pace ratio compared {months_text} {current_year} registrations "
        f"with the average of the same completed months in previous years. "
        f"The ratio used was {ratio:.3f}."
    )
    return ratio, note


def historical_baseline(valid: pd.DataFrame, view: str):
    temp = valid.copy()
    temp["year"] = temp["_registration_datetime"].dt.year
    temp["month"] = temp["_registration_datetime"].dt.month
    current_year = int(temp["year"].max()) if len(temp) else datetime.now().year
    hist = temp[temp["year"] < current_year].copy()
    if hist.empty:
        hist = temp.copy()

    if view == "Daily":
        hist["key"] = hist["_registration_datetime"].dt.month
        # Average daily count by calendar month. This avoids pretending every exact date has enough history.
        monthly_days = []
        for (year, month), g in hist.groupby([hist["_registration_datetime"].dt.year, hist["_registration_datetime"].dt.month]):
            days = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
            monthly_days.append({"key": month, "daily_rate": len(g) / days})
        if monthly_days:
            b = pd.DataFrame(monthly_days).groupby("key")["daily_rate"].mean().to_dict()
        else:
            b = {m: 0 for m in range(1, 13)}
        global_mean = np.mean(list(b.values())) if b else 0
        return b, global_mean

    if view == "Weekly":
        hist["key"] = hist["_registration_datetime"].dt.isocalendar().week.astype(int)
        b = hist.groupby("key").size().to_dict()
        # Convert total by week across years to average per historical year.
        n_years = max(1, hist["year"].nunique())
        b = {k: v / n_years for k, v in b.items()}
        global_mean = np.mean(list(b.values())) if b else 0
        return b, global_mean

    if view == "Monthly":
        hist["key"] = hist["month"]
        b = hist.groupby("key").size().to_dict()
        n_years = max(1, hist["year"].nunique())
        b = {k: v / n_years for k, v in b.items()}
        global_mean = np.mean(list(b.values())) if b else 0
        return b, global_mean

    if view == "Quarterly":
        hist["key"] = hist["_registration_datetime"].dt.quarter.astype(int)
        b = hist.groupby("key").size().to_dict()
        n_years = max(1, hist["year"].nunique())
        b = {k: v / n_years for k, v in b.items()}
        global_mean = np.mean(list(b.values())) if b else 0
        return b, global_mean

    if view == "Semi-annual":
        hist["key"] = np.where(hist["month"] <= 6, 1, 2)
        b = hist.groupby("key").size().to_dict()
        n_years = max(1, hist["year"].nunique())
        b = {k: v / n_years for k, v in b.items()}
        global_mean = np.mean(list(b.values())) if b else 0
        return b, global_mean

    if view == "Annual":
        annual = hist.groupby("year").size()
        global_mean = float(annual.mean()) if len(annual) else float(len(hist))
        return {}, global_mean

    return {}, 0


def next_periods(last_date: pd.Timestamp, view: str, years: int) -> pd.DatetimeIndex:
    years = int(max(1, min(5, years)))
    last_date = pd.Timestamp(last_date)

    if view == "Daily":
        return pd.date_range(last_date.floor("D") + pd.Timedelta(days=1), periods=365 * years, freq="D")
    if view == "Weekly":
        return pd.date_range(last_date + pd.Timedelta(days=1), periods=52 * years, freq="W-SUN")
    if view == "Monthly":
        start = (last_date + pd.offsets.MonthBegin(1)).normalize()
        return pd.date_range(start, periods=12 * years, freq="MS")
    if view == "Quarterly":
        start = (last_date + pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        return pd.date_range(start, periods=4 * years, freq="QS-JAN")
    if view == "Semi-annual":
        # Build H1/H2 period starts after the current date.
        if last_date.month <= 6:
            start = pd.Timestamp(last_date.year, 7, 1)
        else:
            start = pd.Timestamp(last_date.year + 1, 1, 1)
        periods = []
        d = start
        for _ in range(2 * years):
            periods.append(d)
            d = pd.Timestamp(d.year, 7, 1) if d.month == 1 else pd.Timestamp(d.year + 1, 1, 1)
        return pd.DatetimeIndex(periods)
    if view == "Annual":
        start = pd.Timestamp(last_date.year + 1, 1, 1)
        return pd.date_range(start, periods=years, freq="YS")
    return pd.DatetimeIndex([])


def seasonal_key(d: pd.Timestamp, view: str):
    d = pd.Timestamp(d)
    if view == "Daily":
        return d.month
    if view == "Weekly":
        return int(d.isocalendar().week)
    if view == "Monthly":
        return d.month
    if view == "Quarterly":
        return ((d.month - 1) // 3) + 1
    if view == "Semi-annual":
        return 1 if d.month <= 6 else 2
    return None


def forecast_for_view(valid: pd.DataFrame, view: str, years: int) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(columns=["period_start", "period", "low_scenario", "expected_forecast", "high_scenario"])

    last_date = valid["_registration_datetime"].max()
    future = next_periods(last_date, view, years)
    pace, _ = current_pace_ratio(valid)
    baseline, global_mean = historical_baseline(valid, view)

    rows = []
    for d in future:
        if view == "Annual":
            expected = global_mean * pace
        else:
            key = seasonal_key(d, view)
            expected = baseline.get(key, global_mean) * pace
            if view == "Daily":
                # Daily baseline is a daily rate by month.
                expected = expected
        expected = max(0, expected)
        expected_int = int(round(expected))
        rows.append(
            {
                "period_start": d,
                "period": view_label(d, view),
                "low_scenario": int(round(expected_int * 0.70)),
                "expected_forecast": expected_int,
                "high_scenario": int(round(expected_int * 1.35)),
            }
        )
    return pd.DataFrame(rows)


def plotly_actual_forecast(actual: pd.Series, forecast: pd.DataFrame, view: str):
    fig = go.Figure()
    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode="lines+markers",
                name="Actual registrations",
            )
        )
    if forecast is not None and not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["period_start"],
                y=forecast["expected_forecast"],
                mode="lines+markers",
                name="Expected forecast",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["period_start"],
                y=forecast["high_scenario"],
                mode="lines",
                name="High scenario",
                line=dict(width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["period_start"],
                y=forecast["low_scenario"],
                mode="lines",
                name="Low scenario",
                line=dict(width=1, dash="dot"),
            )
        )
    fig.update_layout(
        title=f"{view} actual registrations and future forecast",
        xaxis_title="Period",
        yaxis_title="Number of registrations",
        hovermode="x unified",
        height=470,
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


def save_matplotlib_plot(actual: pd.Series, forecast: pd.DataFrame, view: str, out_path: Path):
    plt.figure(figsize=(10, 4.8))
    if not actual.empty:
        plt.plot(actual.index, actual.values, marker="o", linewidth=1.4, label="Actual")
    if forecast is not None and not forecast.empty:
        plt.plot(forecast["period_start"], forecast["expected_forecast"], marker="o", linewidth=1.4, label="Expected forecast")
        plt.fill_between(
            forecast["period_start"],
            forecast["low_scenario"],
            forecast["high_scenario"],
            alpha=0.2,
            label="Planning range",
        )
    plt.title(f"{view} PRAT Farmer Registrations and Forecast")
    plt.xlabel("Period")
    plt.ylabel("Registrations")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_latex_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    use_df = df[columns].head(max_rows).copy()
    aligns = "l" + "r" * (len(columns) - 1)
    out = [f"\\begin{{tabular}}{{{aligns}}}", "\\toprule"]
    out.append(" & ".join(latex_escape(c.replace("_", " ").title()) for c in columns) + r" \\")
    out.append("\\midrule")
    for _, row in use_df.iterrows():
        vals = []
        for c in columns:
            val = row[c]
            if isinstance(val, (int, np.integer)):
                vals.append(f"{int(val):,}")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):,.2f}")
            else:
                vals.append(latex_escape(val))
        out.append(" & ".join(vals) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def generate_latex_report(project_name: str, date_col: str, quality: dict, all_actual_tables: dict, all_forecast_tables: dict, group_summaries: dict, missing_summary: pd.DataFrame, figure_files: dict, out_dir: Path) -> Path:
    tex_path = out_dir / "Forecasting_Report.tex"
    pace, pace_note = quality.get("pace_ratio", 1.0), quality.get("pace_note", "")
    date_min = quality.get("date_min")
    date_max = quality.get("date_max")
    date_range = "Not available"
    if pd.notna(date_min) and pd.notna(date_max):
        date_range = f"{pd.Timestamp(date_min).strftime('%d %B %Y')} to {pd.Timestamp(date_max).strftime('%d %B %Y')}"

    tex = []
    tex.append(r"\documentclass[12pt]{article}")
    tex.append(r"\usepackage[a4paper,margin=1in]{geometry}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{float}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{caption}")
    tex.append(r"\usepackage{setspace}")
    tex.append(r"\usepackage{titlesec}")
    tex.append(r"\usepackage{fancyhdr}")
    tex.append(r"\setstretch{1.2}")
    tex.append(r"\setlength{\headheight}{15pt}")
    tex.append(r"\pagestyle{fancy}")
    tex.append(r"\fancyhf{}")
    tex.append(r"\lhead{Time Series Forecasting Report}")
    tex.append(r"\rhead{Vibrant Village Foundation}")
    tex.append(r"\cfoot{\thepage}")
    tex.append(r"\begin{document}")
    tex.append(r"\begin{titlepage}")
    tex.append(r"\centering")
    tex.append(r"\vspace*{1.2cm}")
    tex.append(r"{\Large \textbf{Vibrant Village Foundation}\par}")
    tex.append(r"\vspace{1.2cm}")
    tex.append(r"{\LARGE \textbf{Forecasting PRAT Farmer Registrations for Planning and Decision-Making}\par}")
    tex.append(r"\vspace{0.8cm}")
    tex.append(r"{\large A Data Analysis Report Based on Registration Dates\par}")
    tex.append(r"\vspace{1.5cm}")
    tex.append(r"\begin{tabular}{rl}")
    tex.append(r"\textbf{Prepared by:} & Brian Sifuna Obware \\")
    tex.append(r"\textbf{Supervisor:} & Mr. Dickson Ochichi \\")
    tex.append(r"\textbf{Organization:} & Vibrant Village Foundation \\")
    tex.append(r"\textbf{Analysis tool:} & Python and Streamlit \\")
    tex.append(r"\textbf{Reporting format:} & Editable LaTeX, PDF, and interactive dashboard \\")
    tex.append(r"\textbf{Date:} & " + datetime.now().strftime("%B %Y") + r" \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\vfill")
    tex.append(r"{\small " + latex_escape(APP_FOOTER) + r"\par}")
    tex.append(r"\end{titlepage}")
    tex.append(r"\tableofcontents\newpage")
    tex.append(r"\listoffigures\newpage")
    tex.append(r"\listoftables\newpage")

    tex.append(r"\section{Executive Summary}")
    tex.append(
        "This report presents a time series analysis of PRAT farmer registration records. "
        f"The dataset contained \textbf{{{quality['original_rows']:,}}} records. "
        f"After checking the selected registration date field, \textbf{{{quality['valid_date_rows']:,}}} records had valid dates and "
        f"\textbf{{{quality['invalid_date_rows']:,}}} records had missing or invalid dates. "
        f"The valid registration period covered \textbf{{{latex_escape(date_range)}}}. "
        "The analysis summarised registrations at daily, weekly, monthly, quarterly, semi-annual, and annual levels, then prepared future planning forecasts for the same levels."
    )
    tex.append(
        "The results should be interpreted as programme planning estimates rather than exact predictions. "
        "Registration activity is influenced by field mobilisation, group meetings, training days, staff availability, reporting schedules, and programme priorities."
    )

    tex.append(r"\section{Objectives}")
    tex.append(r"\begin{enumerate}")
    tex.append(r"\item To clean and check the quality of the selected registration date field.")
    tex.append(r"\item To check whether there were missing dates, invalid dates, and duplicate records.")
    tex.append(r"\item To summarise registrations by daily, weekly, monthly, quarterly, semi-annual, and annual periods.")
    tex.append(r"\item To generate future forecasts for the same time levels for a selected duration of up to five years.")
    tex.append(r"\item To prepare outputs that can support decision-making, communication, and programme monitoring.")
    tex.append(r"\end{enumerate}")

    tex.append(r"\section{Data Preparation and Cleaning}")
    tex.append(
        f"The selected registration date column was \texttt{{{latex_escape(date_col)}}}. "
        "Text fields were cleaned by removing leading spaces, trailing spaces, and repeated internal spaces. "
        f"During this cleaning step, \textbf{{{quality['text_cells_cleaned']:,}}} text cells were adjusted. "
        "The date field was then converted into a proper date-time format. Records that could not be converted were separated from the time series analysis but documented in the data quality outputs."
    )

    tex.append(r"\begin{table}[H]\centering\caption{Data quality summary}")
    qdf = pd.DataFrame([
        ["Records reviewed", quality['original_rows']],
        ["Valid registration dates", quality['valid_date_rows']],
        ["Invalid or missing registration dates", quality['invalid_date_rows']],
        ["Text cells cleaned", quality['text_cells_cleaned']],
        ["Full-row duplicate records", quality['full_duplicate_rows']],
        ["Duplicate rows based on selected ID columns", quality['duplicate_id_rows']],
    ], columns=["Check", "Result"])
    tex.append(make_latex_table(qdf, ["Check", "Result"], max_rows=20))
    tex.append(r"\end{table}")

    tex.append(r"\subsection{Duplicate Checks}")
    selected_ids = quality.get("selected_id_columns", [])
    if selected_ids:
        tex.append(
            "Duplicate checking was done using the selected ID column(s): "
            + ", ".join([r"\texttt{" + latex_escape(c) + "}" for c in selected_ids])
            + ". "
            f"The check found \textbf{{{quality['duplicate_id_rows']:,}}} rows that shared the same selected ID combination. "
            "These rows were exported separately so that they could be reviewed before any correction or deletion was made."
        )
    else:
        tex.append("No ID column was selected for ID-based duplicate checking. Full-row duplicate checking was still completed.")

    tex.append(r"\subsection{Missing Values}")
    tex.append("The pipeline also counted missing values in every column. The table below shows the columns with the highest number of missing values.")
    top_missing = missing_summary.sort_values("missing_values", ascending=False).head(10).copy()
    tex.append(r"\begin{table}[H]\centering\caption{Top columns by missing values}")
    tex.append(make_latex_table(top_missing, ["column", "missing_values", "missing_percent"], max_rows=10))
    tex.append(r"\end{table}")

    tex.append(r"\section{Forecasting Method}")
    tex.append(
        "The forecast used a simple baseline seasonal planning method. It was not presented as an advanced machine learning model. "
        "The method first established the historical registration pattern, then adjusted the future values using the current registration pace. "
        f"{latex_escape(pace_note)} "
        "This approach was selected because the registration data was field-activity-driven and needed a method that programme staff could understand and discuss."
    )
    tex.append(r"\[")
    tex.append(r"\text{Expected forecast} = \text{Historical seasonal baseline} \times \text{Current pace ratio}")
    tex.append(r"\]")
    tex.append(
        "Low and high planning scenarios were created around the expected forecast. The low scenario used 70\% of the expected value, while the high scenario used 135\% of the expected value."
    )

    tex.append(r"\section{Actual Registration Time Plots}")
    for view in VIEW_ORDER:
        tex.append(r"\subsection{" + latex_escape(view) + " Registration Pattern}")
        tex.append(
            f"The {view.lower()} view was used to understand registration activity at the {view.lower()} planning level. "
            "This helped to show whether registrations were steady, irregular, concentrated, or slowing over time."
        )
        fig_name = figure_files.get((view, "actual"))
        if fig_name:
            tex.append(r"\begin{figure}[H]\centering")
            tex.append(r"\includegraphics[width=0.95\textwidth]{figures/" + latex_escape(Path(fig_name).name) + "}")
            tex.append(r"\caption[" + latex_escape(view) + " registration time plot]{" + latex_escape(view) + " registration time plot}")
            tex.append(r"\end{figure}")
        tbl = all_actual_tables[view].tail(12).copy()
        tex.append(r"\begin{table}[H]\centering\caption{" + latex_escape(view) + " registration summary, latest periods}")
        tex.append(make_latex_table(tbl, ["period", "registrations"], max_rows=12))
        tex.append(r"\end{table}")

    tex.append(r"\section{Future Forecast Results}")
    for view in VIEW_ORDER:
        tex.append(r"\subsection{" + latex_escape(view) + " Forecast}")
        tex.append(
            f"The {view.lower()} forecast was generated from the same cleaned registration data and the same baseline planning method. "
            "It gives low, expected, and high scenarios for planning."
        )
        fig_name = figure_files.get((view, "forecast"))
        if fig_name:
            tex.append(r"\begin{figure}[H]\centering")
            tex.append(r"\includegraphics[width=0.95\textwidth]{figures/" + latex_escape(Path(fig_name).name) + "}")
            tex.append(r"\caption[" + latex_escape(view) + " forecast time plot]{" + latex_escape(view) + " forecast time plot}")
            tex.append(r"\end{figure}")
        ftbl = all_forecast_tables[view].head(12).copy()
        tex.append(r"\begin{table}[H]\centering\caption{" + latex_escape(view) + " forecast table, first forecast periods}")
        tex.append(make_latex_table(ftbl, ["period", "low_scenario", "expected_forecast", "high_scenario"], max_rows=12))
        tex.append(r"\end{table}")

    if group_summaries:
        tex.append(r"\section{Selected Grouping Summaries}")
        tex.append("The selected grouping columns were used to show how registrations were distributed across programme categories or locations.")
        for col, gdf in group_summaries.items():
            tex.append(r"\subsection{Summary by " + latex_escape(col.replace('_', ' ').title()) + "}")
            tex.append(r"\begin{table}[H]\centering\caption{Registration summary by " + latex_escape(col.replace('_', ' ')) + "}")
            tex.append(make_latex_table(gdf.rename(columns={col: "category"}), ["category", "registrations"], max_rows=15))
            tex.append(r"\end{table}")

    tex.append(r"\section{Key Findings}")
    tex.append(r"\begin{enumerate}")
    tex.append(r"\item The registration data was suitable for time series analysis after the date field was cleaned and converted.")
    tex.append(r"\item The daily pattern was useful for supervision, but it was more irregular than the monthly, quarterly, semi-annual, and annual views.")
    tex.append(r"\item Monthly and quarterly outputs gave stronger planning signals because they reduced the noise found in daily registrations.")
    tex.append(r"\item The forecast should be used as a planning guide and should be updated when new registration data becomes available.")
    tex.append(r"\item Duplicate and missing value checks should be reviewed before official reporting, especially where selected ID columns show repeated records.")
    tex.append(r"\end{enumerate}")

    tex.append(r"\section{Recommendations}")
    tex.append(r"\begin{enumerate}")
    tex.append(r"\item The dashboard should be updated whenever new registration data is added.")
    tex.append(r"\item Daily and weekly views should be used mainly for operational supervision.")
    tex.append(r"\item Monthly, quarterly, semi-annual, and annual forecasts should be used for planning, budgeting, and communication.")
    tex.append(r"\item Duplicate records should be reviewed before any final reporting or decision-making.")
    tex.append(r"\item Future versions should include field activity dates, training dates, staff activity, and mobilisation plans to strengthen the forecast.")
    tex.append(r"\end{enumerate}")

    tex.append(r"\section{Limitations}")
    tex.append(
        "The forecast was based mainly on registration dates. It did not include field activity schedules, registration targets, staff availability, training days, or mobilisation plans. "
        "These factors can influence registration numbers. For this reason, the forecast should be treated as a decision-support estimate, not as a fixed prediction."
    )

    tex.append(r"\section{Conclusion}")
    tex.append(
        "The pipeline cleaned the uploaded time series data, checked date validity, checked duplicates, summarised registration activity at six time levels, and generated forecasts for the same levels. "
        "The editable LaTeX report, PDF report, CSV outputs, figures, and interactive dashboard provide a repeatable way of updating the analysis whenever new data is available."
    )

    tex.append(r"\appendix")
    tex.append(r"\section{Python Workflow Used}")
    tex.append(r"\begin{enumerate}")
    tex.append(r"\item Upload or load the CSV file.")
    tex.append(r"\item Select the registration date column.")
    tex.append(r"\item Select optional grouping columns and optional ID columns.")
    tex.append(r"\item Clean text fields by removing extra spaces.")
    tex.append(r"\item Convert the selected date column into a date-time field.")
    tex.append(r"\item Separate valid and invalid dates.")
    tex.append(r"\item Check full-row duplicates and duplicates based on selected ID columns.")
    tex.append(r"\item Aggregate valid registrations into daily, weekly, monthly, quarterly, semi-annual, and annual time series.")
    tex.append(r"\item Generate actual time plots for all six levels.")
    tex.append(r"\item Estimate future values using a baseline seasonal planning forecast.")
    tex.append(r"\item Generate forecast time plots and scenario tables for all six levels.")
    tex.append(r"\item Export CSV outputs, figures, editable LaTeX report, PDF report, and a zipped package.")
    tex.append(r"\end{enumerate}")
    tex.append(r"\vfill")
    tex.append(r"\begin{center}\small " + latex_escape(APP_FOOTER) + r"\end{center}")
    tex.append(r"\end{document}")

    tex_path.write_text("\n".join(tex), encoding="utf-8")
    return tex_path


def generate_pdf_report(pdf_path: Path, project_name: str, quality: dict, figure_files: dict, all_forecast_tables: dict):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    except Exception as e:
        pdf_path.write_text(f"PDF generation failed because reportlab is not installed: {e}", encoding="utf-8")
        return pdf_path

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallText", parent=styles["Normal"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    story.append(Paragraph("Vibrant Village Foundation", styles["Title"]))
    story.append(Paragraph("Forecasting PRAT Farmer Registrations for Planning and Decision-Making", styles["Heading1"]))
    story.append(Paragraph("Prepared by: Brian Sifuna Obware", styles["Normal"]))
    story.append(Paragraph("Supervisor: Mr. Dickson Ochichi", styles["Normal"]))
    story.append(Paragraph(APP_FOOTER, styles["SmallText"]))
    story.append(Spacer(1, 0.2 * inch))

    qdata = [
        ["Check", "Result"],
        ["Records reviewed", f"{quality['original_rows']:,}"],
        ["Valid registration dates", f"{quality['valid_date_rows']:,}"],
        ["Invalid/missing dates", f"{quality['invalid_date_rows']:,}"],
        ["Text cells cleaned", f"{quality['text_cells_cleaned']:,}"],
        ["Full-row duplicates", f"{quality['full_duplicate_rows']:,}"],
        ["Duplicate ID rows", f"{quality['duplicate_id_rows']:,}"],
    ]
    table = Table(qdata, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("Data Quality Summary", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Forecasting Method", styles["Heading2"]))
    story.append(Paragraph(
        "The forecast used a simple baseline seasonal planning method. Historical patterns were adjusted using the current registration pace. Low and high planning scenarios were created around the expected forecast.",
        styles["Normal"],
    ))

    for view in VIEW_ORDER:
        story.append(PageBreak())
        story.append(Paragraph(f"{view} Actual and Forecast View", styles["Heading2"]))
        actual_fig = figure_files.get((view, "actual"))
        forecast_fig = figure_files.get((view, "forecast"))
        for fig in [actual_fig, forecast_fig]:
            if fig and Path(fig).exists():
                story.append(Image(str(fig), width=6.8 * inch, height=3.2 * inch))
                story.append(Spacer(1, 0.1 * inch))
        ft = all_forecast_tables.get(view)
        if ft is not None and not ft.empty:
            tdf = ft[["period", "low_scenario", "expected_forecast", "high_scenario"]].head(8)
            data = [["Period", "Low", "Expected", "High"]] + tdf.astype(str).values.tolist()
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl)

    story.append(PageBreak())
    story.append(Paragraph("Conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "The pipeline generated actual time plots, forecast plots, data quality checks, CSV outputs, an editable LaTeX report, and this PDF report. The outputs should be updated whenever new registration data is available.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(APP_FOOTER, styles["SmallText"]))
    doc.build(story)
    return pdf_path


def build_outputs(df_raw, date_col, grouping_cols, id_cols, horizon_years, project_name="PRAT Farmer Registration Forecasting"):
    temp_dir = Path(tempfile.mkdtemp(prefix="bosdata_forecast_"))
    out_dir = temp_dir / "generated_outputs"
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "tables"
    report_dir = out_dir / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_clean, valid, invalid, missing_summary, duplicate_id_rows, group_summaries, quality = prepare_data(df_raw, date_col, grouping_cols, id_cols)
    pace, pace_note = current_pace_ratio(valid)
    quality["pace_ratio"] = pace
    quality["pace_note"] = pace_note

    all_actual_series = {}
    all_actual_tables = {}
    all_forecast_tables = {}
    figure_files = {}

    for view in VIEW_ORDER:
        s = actual_series(valid, view)
        f = forecast_for_view(valid, view, horizon_years)
        all_actual_series[view] = s
        all_actual_tables[view] = series_to_table(s, view)
        all_forecast_tables[view] = f

        all_actual_tables[view].to_csv(data_dir / f"actual_{safe_filename(view).lower()}.csv", index=False)
        f.to_csv(data_dir / f"forecast_{safe_filename(view).lower()}.csv", index=False)

        actual_fig = fig_dir / f"actual_{safe_filename(view).lower()}.png"
        forecast_fig = fig_dir / f"forecast_{safe_filename(view).lower()}.png"
        save_matplotlib_plot(s, pd.DataFrame(), view, actual_fig)
        save_matplotlib_plot(pd.Series(dtype=float), f, f"{view} Forecast", forecast_fig)
        figure_files[(view, "actual")] = actual_fig
        figure_files[(view, "forecast")] = forecast_fig

    df_clean.to_csv(data_dir / "cleaned_data.csv", index=False)
    invalid.to_csv(data_dir / "invalid_or_missing_dates.csv", index=False)
    duplicate_id_rows.to_csv(data_dir / "duplicate_id_records.csv", index=False)
    missing_summary.to_csv(data_dir / "missing_values_summary.csv", index=False)
    pd.DataFrame([quality]).to_csv(data_dir / "data_quality_summary.csv", index=False)
    for col, gdf in group_summaries.items():
        gdf.to_csv(data_dir / f"group_summary_by_{safe_filename(col).lower()}.csv", index=False)

    tex_path = generate_latex_report(project_name, date_col, quality, all_actual_tables, all_forecast_tables, group_summaries, missing_summary, figure_files, report_dir)
    pdf_path = generate_pdf_report(report_dir / "Forecasting_Report.pdf", project_name, quality, figure_files, all_forecast_tables)

    # Make a zip with everything generated.
    zip_path = temp_dir / "generated_forecasting_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in out_dir.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(out_dir))

    return {
        "temp_dir": temp_dir,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "data_dir": data_dir,
        "report_dir": report_dir,
        "zip_path": zip_path,
        "tex_path": tex_path,
        "pdf_path": pdf_path,
        "quality": quality,
        "valid": valid,
        "invalid": invalid,
        "missing_summary": missing_summary,
        "duplicate_id_rows": duplicate_id_rows,
        "group_summaries": group_summaries,
        "all_actual_series": all_actual_series,
        "all_actual_tables": all_actual_tables,
        "all_forecast_tables": all_forecast_tables,
        "figure_files": figure_files,
    }


# -------------------------- UI --------------------------
st.markdown('<div class="main-title">PRAT Farmer Registration Forecasting Pipeline</div>', unsafe_allow_html=True)
st.success('FIXED VERSION: multi-select optional columns, stable forecast view, PDF report, detailed data cleaning, duplicate checks, and Bosdata footer are active.')
st.markdown(
    '<div class="small-note">Upload time series data, clean it, check duplicates, generate future forecasts, plots, an editable LaTeX report, and a PDF report.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

sample_path = Path("sample_data") / "Farmer Data_For Forecasting - Sheet1.csv"
if uploaded is None and sample_path.exists():
    st.info("No file uploaded yet. The included PRAT sample data is loaded for demonstration.")
    df_raw = pd.read_csv(sample_path)
elif uploaded is not None:
    df_raw = read_csv_uploaded(uploaded)
else:
    st.warning("Upload a CSV file to begin.")
    st.stop()

if df_raw.empty:
    st.error("The uploaded file has no records.")
    st.stop()

columns = df_raw.columns.tolist()
def_date = detect_default_date_column(columns)

col1, col2, col3 = st.columns(3)
with col1:
    date_col = st.selectbox("Select registration/date column", columns, index=columns.index(def_date) if def_date in columns else 0)
with col2:
    grouping_cols = st.multiselect(
        "Optional grouping columns",
        options=[c for c in columns if c != date_col],
        default=[c for c in detect_default_grouping_columns(columns) if c != date_col],
        help="You can select more than one column, for example county, sub_county and ward.",
    )
with col3:
    id_cols = st.multiselect(
        "Optional ID columns for duplicate checking",
        options=[c for c in columns if c != date_col],
        default=[c for c in detect_default_id_columns(columns) if c != date_col],
        help="You can select one or more ID columns. The app checks repeated combinations.",
    )

col4, col5 = st.columns([1, 2])
with col4:
    horizon_years = st.slider("Forecast duration in years", min_value=1, max_value=5, value=1, step=1)
with col5:
    forecast_view = st.selectbox("Choose forecast view", VIEW_ORDER, index=0)

# Build results on every rerun so changing the view never clears the dashboard.
with st.spinner("Cleaning data, checking duplicates, building plots and generating report..."):
    results = build_outputs(df_raw, date_col, grouping_cols, id_cols, horizon_years)

quality = results["quality"]

st.subheader("Data quality summary")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Records", f"{quality['original_rows']:,}")
m2.metric("Valid dates", f"{quality['valid_date_rows']:,}")
m3.metric("Invalid dates", f"{quality['invalid_date_rows']:,}")
m4.metric("Text cells cleaned", f"{quality['text_cells_cleaned']:,}")
m5.metric("Full duplicates", f"{quality['full_duplicate_rows']:,}")
m6.metric("ID duplicate rows", f"{quality['duplicate_id_rows']:,}")

with st.expander("View cleaning, missing values and duplicate details", expanded=False):
    st.write("Selected grouping columns:", grouping_cols if grouping_cols else "None selected")
    st.write("Selected ID columns:", id_cols if id_cols else "None selected")
    st.dataframe(results["missing_summary"], use_container_width=True)
    if len(results["duplicate_id_rows"]):
        st.warning("Duplicate rows based on the selected ID column(s) were found. Review them before final reporting.")
        st.dataframe(results["duplicate_id_rows"].head(100), use_container_width=True)
    else:
        st.success("No duplicate rows were found based on the selected ID column(s).")

st.subheader(f"{forecast_view} actual time plot and future forecast")
actual = results["all_actual_series"][forecast_view]
forecast = results["all_forecast_tables"][forecast_view]
st.plotly_chart(plotly_actual_forecast(actual, forecast, forecast_view), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Latest {forecast_view.lower()} actual periods**")
    st.dataframe(results["all_actual_tables"][forecast_view].tail(20), use_container_width=True)
with c2:
    st.markdown(f"**Future {forecast_view.lower()} forecast**")
    st.dataframe(forecast.head(30), use_container_width=True)

st.subheader("All time plots and forecast views")
tabs = st.tabs(VIEW_ORDER)
for tab, view in zip(tabs, VIEW_ORDER):
    with tab:
        st.plotly_chart(
            plotly_actual_forecast(results["all_actual_series"][view], results["all_forecast_tables"][view], view),
            use_container_width=True,
            key=f"plot_{view}",
        )
        st.dataframe(results["all_forecast_tables"][view].head(30), use_container_width=True)

if results["group_summaries"]:
    st.subheader("Selected grouping summaries")
    g_tabs = st.tabs(list(results["group_summaries"].keys()))
    for tab, (col, gdf) in zip(g_tabs, results["group_summaries"].items()):
        with tab:
            st.dataframe(gdf, use_container_width=True)
            fig = go.Figure(go.Bar(x=gdf.iloc[:, 0].astype(str).head(20), y=gdf["registrations"].head(20)))
            fig.update_layout(title=f"Registrations by {col}", xaxis_title=col, yaxis_title="Registrations", height=420)
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    with open(results["zip_path"], "rb") as f:
        st.download_button(
            "Download full generated package",
            data=f.read(),
            file_name="forecasting_generated_package.zip",
            mime="application/zip",
        )
with d2:
    with open(results["tex_path"], "rb") as f:
        st.download_button(
            "Download editable LaTeX report",
            data=f.read(),
            file_name="Forecasting_Report.tex",
            mime="text/x-tex",
        )
with d3:
    with open(results["pdf_path"], "rb") as f:
        st.download_button(
            "Download PDF report",
            data=f.read(),
            file_name="Forecasting_Report.pdf",
            mime="application/pdf",
        )

st.markdown(f'<div class="bosdata-footer">{APP_FOOTER}</div>', unsafe_allow_html=True)
