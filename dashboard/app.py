"""LendScope | Historical LendingClub portfolio intelligence dashboard.

Run from the repository root with:
    python -m streamlit run dashboard/app.py

Reads the existing dbt-generated PostgreSQL marts. Does not mutate source data.
"""

from __future__ import annotations

import html
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

# Streamlit launches this file from dashboard/, so make project imports robust.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.database import get_engine  # noqa: E402


st.set_page_config(
    page_title="LendScope | Portfolio Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).with_name("styles.css")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

INK = "#eef0ff"
MUTED = "#929bb7"
BLUE = "#9ca9ff"
PURPLE = "#bc9cff"
CYAN = "#77d6e5"
GRID = "rgba(156,170,215,0.10)"
PLOT_BG = "rgba(0,0,0,0)"
GLOW = ["#8798ff", "#a5aaff", "#b7a0ef", "#83c5db", "#7d92c8", "#ab8fe3", "#597ca9"]


@st.cache_resource(show_spinner=False)
def database_engine():
    """Share the SQLAlchemy connection pool across Streamlit reruns."""
    return get_engine()


@st.cache_data(ttl=600, show_spinner=False)
def load_marts():
    """Load only small dbt summary tables, not the 2.26M-record raw dataset."""
    queries = {
        "vintage": """
            SELECT issue_year, grade, loan_count, total_loan_amount,
                   avg_interest_rate_pct, fully_paid_loans, charged_off_loans,
                   current_loans, resolved_loans
            FROM staging_analytics.mart_loan_vintage_grade
        """,
        "purpose": """
            SELECT issue_year, loan_purpose, loan_count, total_loan_amount,
                   avg_loan_amount, avg_interest_rate_pct, fully_paid_loans,
                   charged_off_loans, current_loans, resolved_loans
            FROM staging_analytics.mart_loan_purpose_year
        """,
        "states": """
            SELECT borrower_state, loan_count, total_loan_amount,
                   avg_loan_amount, origination_amount_share_pct,
                   charged_off_share_resolved_pct, fully_paid_loans,
                   charged_off_loans, current_loans, resolved_loans
            FROM staging_analytics.mart_loan_by_state
        """,
    }
    output = {}
    with database_engine().connect() as connection:
        for key, statement in queries.items():
            frame = pd.read_sql_query(text(statement), connection)
            number_columns = frame.select_dtypes(include="number").columns.tolist()
            # Some PostgreSQL NUMERIC columns arrive as Decimal objects.
            numeric_names = {
                "issue_year", "loan_count", "total_loan_amount", "avg_loan_amount",
                "avg_interest_rate_pct", "fully_paid_loans", "charged_off_loans",
                "current_loans", "resolved_loans", "origination_amount_share_pct",
                "charged_off_share_resolved_pct",
            }
            for column in set(number_columns) | (set(frame.columns) & numeric_names):
                if column in numeric_names:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            output[key] = frame
    return output


@st.cache_data(ttl=600, show_spinner=False)
def load_latest_ingestion():
    with database_engine().connect() as connection:
        record = connection.execute(text("""
            SELECT run_id, source_file, status, rows_read, rows_loaded,
                   started_at, completed_at
            FROM meta.ingestion_runs
            ORDER BY run_id DESC LIMIT 1
        """)).mappings().first()
    return dict(record) if record else None



@st.cache_data(ttl=600, show_spinner=False)
def load_raw_quality_snapshot():
    """Check the source at query time; do not silently treat stored claims as tests."""
    with database_engine().connect() as connection:
        totals = connection.execute(text("""
            SELECT
                COUNT(*) AS source_rows,
                COUNT(*) FILTER (WHERE id ~ '^[0-9]+$') AS loan_rows,
                COUNT(*) FILTER (WHERE id !~ '^[0-9]+$' OR id IS NULL)
                    AS non_loan_rows
            FROM raw.loans
        """)).mappings().one()
        columns = connection.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = 'loans'
        """)).scalar_one()
        history = pd.read_sql_query(text("""
            SELECT run_id, source_file, status, rows_read, rows_loaded,
                   started_at, completed_at
            FROM meta.ingestion_runs
            ORDER BY run_id DESC LIMIT 10
        """), connection)
    return {**dict(totals), "column_count": int(columns), "history": history}


def read_last_dbt_test_artifact():
    """Read an optional *local* result, not live test execution."""
    path = PROJECT_ROOT / "dbt" / "target" / "run_results.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tests = [r for r in data.get("results", [])
                 if r.get("unique_id", "").startswith("test.")]
        status_counts = Counter(str(r.get("status", "unknown")).lower()
                                for r in tests)
        return {
            "test_count": len(tests),
            "status_counts": status_counts,
            "last_updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M"),
        }
    except (OSError, ValueError, TypeError):
        return None


def display_loan_purpose(raw):
    return str(raw).replace("_", " ").title()


def grade_table(grades):
    """Seven grades in a compact HTML table, with no hidden final grade."""
    body_rows = []
    for row in grades.itertuples(index=False):
        chargeoff = (f"{row.resolved_chargeoff_pct:.2f}%"
                     if pd.notna(row.resolved_chargeoff_pct) else "N/A")
        body_rows.append(
            "<tr>"
            f"<td><span class='grade-pill'>{html.escape(str(row.grade))}</span></td>"
            f"<td>{int(row.loan_count):,}</td>"
            f"<td>{int(row.charged_off_loans):,}</td>"
            f"<td>{int(row.resolved_loans):,}</td>"
            f"<td>{html.escape(money(row.total_loan_amount))}</td>"
            f"<td>{chargeoff}</td>"
            "</tr>"
        )
    st.markdown(
        '<div class="compact-table-scroll"><table class="compact-data-table">'
        '<thead><tr><th>Grade</th><th>Loans</th><th>Charged off</th>'
        '<th>Resolved</th><th>Original lending</th><th>Charge-off share*</th>'
        '</tr></thead><tbody>' + "".join(body_rows) + '</tbody></table></div>',
        unsafe_allow_html=True,
    )

def money(value, compact=True):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if compact and abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if compact and abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.0f}"


def count(value):
    return f"{int(value):,}"


def rate(numerator, denominator):
    return (100.0 * numerator / denominator) if denominator else float("nan")


def configure_chart(fig, height=205, left=8, right=8, bottom=6, top=8):
    fig.update_layout(
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, Segoe UI, sans-serif", color=MUTED, size=11),
        showlegend=False,
        hoverlabel=dict(bgcolor="#252b48", bordercolor="#505a8a", font_color=INK),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def plot(fig, key):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def panel_header(title, subtitle="", tag=None):
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    tag_html = f'<span class="panel-tag">{html.escape(tag)}</span>' if tag else ""
    st.markdown(
        f'<div class="panel-heading"><div><div class="panel-title">{safe_title}</div>'
        f'<div class="panel-subtitle">{safe_subtitle}</div></div>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def metric_html(label, value, footnote, symbol="◈", highlight=False):
    """HTML for a KPI card, shared by the header grid and integrity checks."""
    special = " metric-special" if highlight else ""
    return (
        f'<div class="metric-card{special}">'
        f'<div class="metric-top"><span>{html.escape(label)}</span>'
        f'<span class="metric-symbol">{html.escape(symbol)}</span></div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-foot">{html.escape(footnote)}</div>'
        '</div>'
    )


def metric(label, value, footnote, symbol="◈", highlight=False):
    st.markdown(
        metric_html(label, value, footnote, symbol, highlight),
        unsafe_allow_html=True,
    )


def title_block(title, description, chip="HISTORICAL DATA • 2007–2018"):
    chip_html = f'<span class="top-chip">{html.escape(chip)}</span>' if chip else ""
    st.markdown(
        f'<div class="top-breadcrumb">LENDSCOPE <span> / </span> {html.escape(title.upper())}</div>'
        f'<div class="main-heading-row"><div class="page-header-copy"><h1 class="page-title">{html.escape(title)}</h1>'
        f'<div class="page-description">{html.escape(description)}</div></div>'
        f'{chip_html}</div>',
        unsafe_allow_html=True,
    )


def section_label(label, description=""):
    st.markdown(
        f'<div class="section-label"><span>{html.escape(label)}</span>'
        f'<span class="section-description">{html.escape(description)}</span></div>',
        unsafe_allow_html=True,
    )


def chart_trend(vintage, selected):
    annual = (vintage.groupby("issue_year", as_index=False)
              .agg(loan_count=("loan_count", "sum"), total_loan_amount=("total_loan_amount", "sum"))
              .sort_values("issue_year"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["issue_year"], y=annual["total_loan_amount"] / 1e9,
        mode="lines+markers", fill="tozeroy", fillcolor="rgba(140,154,255,.12)",
        line=dict(color=BLUE, width=3.5, shape="spline"),
        marker=dict(color="#c4caff", size=7, line=dict(width=2, color="#202640")),
        customdata=annual["loan_count"],
        hovertemplate="<b>%{x}</b><br>Originations: $%{y:.2f}B<br>Loans: %{customdata:,}<extra></extra>",
    ))
    if selected != "All years":
        row = annual[annual["issue_year"] == int(selected)]
        if not row.empty:
            fig.add_trace(go.Scatter(
                x=row["issue_year"], y=row["total_loan_amount"] / 1e9,
                mode="markers", marker=dict(color=CYAN, size=14, line=dict(color="#f1f2ff", width=2)),
                hovertemplate="Selected year: %{x}<extra></extra>",
            ))
    fig = configure_chart(fig, height=182, left=2, bottom=4)
    fig.update_yaxes(ticksuffix="B", tickprefix="$", rangemode="tozero")
    fig.update_xaxes(dtick=2, tickformat="d")
    return fig


def grade_summary(vintage):
    result = (vintage.groupby("grade", as_index=False)
              .agg(loan_count=("loan_count", "sum"), charged_off_loans=("charged_off_loans", "sum"),
                   resolved_loans=("resolved_loans", "sum"),
                   total_loan_amount=("total_loan_amount", "sum")))
    result["resolved_chargeoff_pct"] = (
        100 * result["charged_off_loans"] / result["resolved_loans"].replace(0, float("nan"))
    )
    return result.sort_values("grade")


def chart_grades(grades):
    fig = go.Figure(go.Bar(
        x=grades["grade"], y=grades["resolved_chargeoff_pct"],
        marker=dict(color=GLOW[:len(grades)], line=dict(width=0), cornerradius=6),
        customdata=grades["resolved_loans"],
        hovertemplate="Grade %{x}<br>Resolved charge-off share: %{y:.2f}%"
                      "<br>Resolved loans: %{customdata:,}<extra></extra>",
    ))
    fig = configure_chart(fig, height=182, left=4, right=5)
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(type="category")
    return fig


def purpose_summary(purpose, n=8):
    result = (purpose.groupby("loan_purpose", as_index=False)
              .agg(loan_count=("loan_count", "sum"), total_loan_amount=("total_loan_amount", "sum")))
    return result.nlargest(n, "total_loan_amount").sort_values("total_loan_amount")


def chart_purposes(purposes):
    fig = go.Figure(go.Bar(
        y=[name.replace("_", " ").title() for name in purposes["loan_purpose"]],
        x=purposes["total_loan_amount"] / 1e9,
        orientation="h",
        marker=dict(color="#98a7ff", opacity=.92, line=dict(width=0)),
        customdata=purposes["loan_count"],
        hovertemplate="%{y}<br>Original lending: $%{x:.2f}B"
                      "<br>Loans: %{customdata:,}<extra></extra>",
    ))
    fig = configure_chart(fig, height=186, left=8, bottom=4)
    fig.update_xaxes(tickprefix="$", ticksuffix="B", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10), automargin=True)
    return fig


def chart_states(states, limit=10):
    df = states.nlargest(limit, "total_loan_amount").sort_values("total_loan_amount")
    fig = go.Figure(go.Bar(
        x=df["origination_amount_share_pct"], y=df["borrower_state"], orientation="h",
        marker=dict(color="#9aa7ff"),
        customdata=df["total_loan_amount"],
        hovertemplate="%{y}<br>Portfolio share: %{x:.2f}%<br>Original lending: $%{customdata:,.0f}<extra></extra>",
    ))
    fig = configure_chart(fig, height=186, left=7)
    fig.update_xaxes(ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return fig


def region_table(states, n=12):
    df = states.nlargest(n, "total_loan_amount")[[
        "borrower_state", "loan_count", "total_loan_amount",
        "origination_amount_share_pct", "charged_off_share_resolved_pct"
    ]].rename(columns={
        "borrower_state": "State", "loan_count": "Loans",
        "total_loan_amount": "Original amount ($)",
        "origination_amount_share_pct": "Share (%)",
        "charged_off_share_resolved_pct": "Resolved charge-offs (%)",
    })
    st.dataframe(
        df, hide_index=True, use_container_width=True, height=min(240, 40 + len(df) * 26),
        column_config={
            "Loans": st.column_config.NumberColumn(format="%d"),
            "Original amount ($)": st.column_config.NumberColumn(format="$%,.0f"),
            "Share (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Resolved charge-offs (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )


def kpis(vintage, separate_from_charts=False):
    """Render the KPI row as one grid so Streamlit reserves its full height.

    Four separate st.columns with embedded HTML cards allowed the next chart
    row to start against the cards at some desktop viewport sizes.
    """
    total_loans = int(vintage["loan_count"].sum())
    total_amount = float(vintage["total_loan_amount"].sum())
    resolved = int(vintage["resolved_loans"].sum())
    charged = int(vintage["charged_off_loans"].sum())
    pct = rate(charged, resolved)
    spacing = " kpi-grid-spaced" if separate_from_charts else ""
    cards = [
        metric_html("Loan originations", count(total_loans), "Historical loan count", "▤"),
        metric_html("Original lending", money(total_amount), "Total issued loan value", "◇", highlight=True),
        metric_html("Average loan size", money(total_amount / total_loans if total_loans else None, False),
                    "Original principal per loan", "◎"),
        metric_html("Resolved charge-off share", f"{pct:.2f}%" if pd.notna(pct) else "N/A",
                    "Not a default probability", "◫"),
    ]
    st.markdown(
        f'<div class="kpi-grid{spacing}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def quality_screen(data):
    title_block("Data integrity", "Source checks, mart reconciliation, and ingestion history.")
    vintage, purpose, states = data["vintage"], data["purpose"], data["states"]
    mart_totals = {
        "Year & grade": int(vintage["loan_count"].sum()),
        "Geography": int(states["loan_count"].sum()),
        "Year & purpose": int(purpose["loan_count"].sum()),
    }
    in_sync = len(set(mart_totals.values())) == 1
    st.markdown(
        '<div class="kpi-grid integrity-grid">' + "".join(
            metric_html(name, count(value), "Records in analytics mart", "✓" if in_sync else "!")
            for name, value in mart_totals.items()
        ) + '</div>',
        unsafe_allow_html=True,
    )
    if in_sync:
        st.markdown(
            '<div class="status-banner status-success">All three analytics marts contain the same number of loans.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-banner status-error">Mart loan counts disagree. Rebuild and validate the dbt models.</div>',
            unsafe_allow_html=True,
        )

    try:
        raw = load_raw_quality_snapshot()
    except Exception as exc:
        raw = None
        st.warning(f"Raw source checks are unavailable: {exc}")

    if raw:
        with st.container(border=True, key="panel-raw-integrity"):
            panel_header("Raw source validation", "Computed from raw.loans when this page is opened", "LIVE QUERY")
            st.markdown(
                '<div class="quality-stat-grid">'
                f'<div><span>Source records</span><strong>{int(raw["source_rows"]):,}</strong></div>'
                f'<div><span>Identified loans</span><strong>{int(raw["loan_rows"]):,}</strong></div>'
                f'<div><span>Non-loan summary rows</span><strong>{int(raw["non_loan_rows"]):,}</strong></div>'
                f'<div><span>Raw columns</span><strong>{int(raw["column_count"]):,}</strong></div>'
                '</div>', unsafe_allow_html=True,
            )
            if (int(raw["loan_rows"]) == next(iter(mart_totals.values()))
                    and int(raw["source_rows"]) == int(raw["loan_rows"]) + int(raw["non_loan_rows"])):
                st.caption("Reconciled: raw loan records match every mart; metadata rows are retained in raw only.")
            else:
                st.error("The raw loan count or metadata classification does not reconcile with the marts.")
            if int(raw["column_count"]) != 151:
                st.warning("The raw schema differs from the 151-column source contract.")

        with st.expander("Ingestion history (latest 10 runs)", expanded=False):
            history = raw["history"].copy()
            if history.empty:
                st.info("No recorded ingestion runs.")
            else:
                history.columns = ["Run", "Source file", "Status", "Rows read",
                                   "Rows loaded", "Started", "Completed"]
                st.dataframe(history, hide_index=True, use_container_width=True, height=240)

    with st.container(border=True, key="panel-dbt-tests"):
        panel_header("dbt validation evidence", "Last available local dbt result file", "LOCAL ARTIFACT")
        artifact = read_last_dbt_test_artifact()
        if artifact is None:
            st.info("No local dbt result file found. Run the tests in PowerShell to generate one.")
        elif artifact["test_count"] == 0:
            st.info("The latest dbt command was not a test run. Run the full test suite to refresh this evidence.")
        else:
            statuses = artifact["status_counts"]
            passed = statuses.get("pass", 0)
            unsuccessful = artifact["test_count"] - passed
            summary = f"{passed}/{artifact['test_count']} tests passed in the latest recorded run"
            if unsuccessful:
                st.markdown(
                    f'<div class="status-banner status-warning">{html.escape(summary)}. Review the dbt test output.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="status-banner status-success">{html.escape(summary)}</div>',
                    unsafe_allow_html=True,
                )
        if artifact:
            st.caption(f"Local artifact last modified: {artifact['last_updated']}. This is not a live test rerun.")
        st.code(r"dbt test --project-dir .\dbt", language="powershell")


MODEL_NAMES = ("Original 6-feature", "Reduced 12-feature", "Reduced + Platt")
MODELING_REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def load_risk_reports():
    """Read small reports on each rerun; reject inconsistent evidence, never invent it."""
    report = json.loads((MODELING_REPORT_DIR / "final_test_evaluation.json").read_text(encoding="utf-8"))
    deciles = pd.read_csv(MODELING_REPORT_DIR / "final_test_deciles.csv")
    if not isinstance(report, dict) or not isinstance(report.get("models"), dict):
        raise ValueError("Evaluation must contain a model-results object.")
    if not isinstance(report.get("evaluated_at_utc"), str):
        raise ValueError("Evaluation timestamp is missing or invalid.")
    if (report["test_year"] != 2015 or report["test_loans"] != 283026
            or report["charged_off_loans"] != 42132
            or set(report["models"]) != set(MODEL_NAMES)):
        raise ValueError("The evaluation does not match the verified 2015 cohort.")
    expected = ((0.6419, 0.2230, 0.12302), (0.6565, 0.2364, 0.12238),
                (0.6565, 0.2364, 0.12222))
    if set(deciles["model"]) != set(MODEL_NAMES):
        raise ValueError("Decile model labels do not match the evaluation.")
    for name, values in zip(MODEL_NAMES, expected):
        scores = report["models"][name]
        if not isinstance(scores, dict):
            raise ValueError(f"{name}: expected numeric model results.")
        for field in ("roc_auc", "average_precision", "brier_score", "mean_predicted_pct",
                      "observed_pct", "prediction_bias_pp"):
            value = scores[field]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}: {field} must be a finite JSON number.")
        for field, expected_value, precision in zip(
                ("roc_auc", "average_precision", "brier_score"), values, (4, 4, 5)):
            if round(float(scores[field]), precision) != expected_value:
                raise ValueError(f"{name}: {field} differs from the verified result.")
        rows = deciles[deciles["model"] == name].sort_values("decile")
        if rows["decile"].tolist() != list(range(1, 11)):
            raise ValueError(f"{name}: expected exactly ten risk deciles.")
        numeric = rows[["loans", "charged_off", "predicted_pct", "observed_pct"]]
        if not all(math.isfinite(float(v)) for v in numeric.to_numpy().ravel()):
            raise ValueError("Deciles contain missing or non-finite values.")
        if (not rows["loans"].gt(0).all()
                or not rows["charged_off"].between(0, rows["loans"]).all()
                or not rows["predicted_pct"].between(0, 100).all()
                or not rows["observed_pct"].between(0, 100).all()
                or not (rows[["loans", "charged_off"]] % 1 == 0).all().all()
                or rows["loans"].sum() != report["test_loans"]
                or rows["charged_off"].sum() != report["charged_off_loans"]):
            raise ValueError("Decile counts or percentages are invalid.")
        if not ((100 * rows["charged_off"] / rows["loans"] - rows["observed_pct"]).abs() < 1e-8).all():
            raise ValueError("Observed decile rates do not reconcile with counts.")
        predicted = float((rows["predicted_pct"] * rows["loans"]).sum() / report["test_loans"])
        observed = 100 * report["charged_off_loans"] / report["test_loans"]
        for field, value in (("mean_predicted_pct", predicted), ("observed_pct", observed),
                             ("prediction_bias_pp", predicted - observed)):
            if not math.isclose(float(scores[field]), value, abs_tol=1e-8):
                raise ValueError(f"{name}: {field} does not reconcile with deciles.")
    return report, deciles


def load_reduced_importance():
    importance_path = MODELING_REPORT_DIR / "reduced_feature_importance.csv"
    frame = pd.read_csv(importance_path)
    meta = json.loads((MODELING_REPORT_DIR / "reduced_feature_importance_metadata.json").read_text(encoding="utf-8"))
    selection = json.loads((MODELING_REPORT_DIR / "feature_ablation_validation.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not isinstance(selection, dict):
        raise ValueError("Importance metadata and feature selection must be JSON objects.")
    if hashlib.sha256(importance_path.read_bytes()).hexdigest() != meta.get("report_sha256"):
        raise ValueError("Importance CSV does not match the SHA-256 recorded in its metadata.")
    reference_auc = meta.get("reference_auc")
    if type(reference_auc) not in (int, float) or not math.isfinite(reference_auc) or not 0 <= reference_auc <= 1:
        raise ValueError("Importance reference ROC-AUC must be a finite number between 0 and 1.")
    if (meta["model"] != "Reduced 12-feature" or meta["evaluation_year"] != 2014
            or meta["scoring"] != "roc_auc" or meta["sample_size"] != 20000
            or meta["n_repeats"] != 3 or len(frame) != 12
            or not frame["feature"].is_unique
            or set(frame["feature"]) != set(selection["reduced_features"])
            or set(frame["feature"]) != set(meta["features"])):
        raise ValueError("Importance report is not the reduced 12-feature validation evaluation.")
    values = frame[["mean_auc_drop", "std_auc_drop"]]
    if (not all(math.isfinite(float(v)) for v in values.to_numpy().ravel())
            or not frame["std_auc_drop"].ge(0).all()):
        raise ValueError("Invalid permutation importance values.")
    return frame.sort_values("mean_auc_drop"), meta


def risk_modelling_screen():
    title_block("Risk Modelling", "Frozen models evaluated on retrospective 2015 loan outcomes.",
                chip="2015 HOLDOUT · RESOLVED LOANS")
    st.markdown('<div class="status-banner status-warning">Research evaluation only. '
                'Not a fully point-in-time backtest or a production underwriting model.</div>',
                unsafe_allow_html=True)
    try:
        report, deciles = load_risk_reports()
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        st.error(f"2015 model reports are unavailable or inconsistent: {exc}")
        st.info("Restore final_test_evaluation.json and final_test_deciles.csv in reports/modeling, then refresh.")
        return

    calibrated = report["models"]["Reduced + Platt"]
    cards = [
        metric_html("2015 holdout loans", count(report["test_loans"]), "Resolved 36-month loans", "▤"),
        metric_html("Observed charge-offs", f"{calibrated['observed_pct']:.2f}%", "42,132 charged-off loans", "◫"),
        metric_html("Calibrated prediction", f"{calibrated['mean_predicted_pct']:.2f}%", "Mean · Reduced + Platt", "◇", True),
        metric_html("Prediction minus observed", f"{calibrated['prediction_bias_pp']:+.2f} pp", "Underprediction persists", "◎"),
    ]
    st.markdown('<div class="kpi-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
    with st.container(border=True, key="panel-risk-comparison"):
        panel_header("2015 holdout comparison", "Identical loans · models and calibrator frozen before test evaluation", "283,026 LOANS")
        body = []
        for name in MODEL_NAMES:
            m = report["models"][name]
            body.append(f"<tr><td>{html.escape(name)}</td><td>{m['roc_auc']:.4f}</td>"
                        f"<td>{m['average_precision']:.4f}</td><td>{m['brier_score']:.5f}</td>"
                        f"<td>{m['mean_predicted_pct']:.2f}%</td><td>{m['prediction_bias_pp']:+.2f} pp</td></tr>")
        st.markdown('<div class="compact-table-scroll"><table class="compact-data-table">'
                    '<thead><tr><th>Model</th><th>ROC-AUC ↑</th><th>AP ↑</th><th>Brier ↓</th>'
                    '<th>Mean predicted</th><th>Prediction bias</th></tr></thead><tbody>'
                    + ''.join(body) + '</tbody></table></div>', unsafe_allow_html=True)
        st.caption("ROC-AUC measures ranking; average precision (AP) summarises precision–recall. "
                   "Higher is better for both; lower Brier score is better. "
                   "Platt calibration preserves ranking here and slightly lowers Brier score, but does not remove underprediction.")

    left, right = st.columns([1.15, 1], gap="small")
    with left:
        with st.container(border=True, key="panel-risk-deciles"):
            panel_header("Predicted vs observed", "2015 holdout · decile 1 lowest risk, 10 highest", "HOLDOUT")
            selected = st.selectbox("Decile model", MODEL_NAMES, index=2, key="risk-decile-model")
            rows = deciles[deciles["model"] == selected].sort_values("decile")
            fig = go.Figure()
            for column, label, color in (("predicted_pct", "Predicted", BLUE),
                                         ("observed_pct", "Observed", CYAN)):
                fig.add_trace(go.Scatter(x=rows["decile"], y=rows[column], name=label,
                    mode="lines+markers", line=dict(color=color, width=2.5),
                    customdata=rows[["loans", "charged_off"]],
                    hovertemplate="Decile %{x}<br>" + label + ": %{y:.2f}%"
                                  "<br>Loans: %{customdata[0]:,}<br>Charged off: %{customdata[1]:,}<extra></extra>"))
            configure_chart(fig, height=300, top=32)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
            fig.update_xaxes(dtick=1, title_text="Predicted-risk decile")
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
            plot(fig, "risk-deciles")
            st.caption("Approximately equal-sized groups, ranked separately for each model; ties split by row order. "
                       "Observed means charged off among resolved loans, not all originated loans.")
    with right:
        with st.container(border=True, key="panel-risk-importance"):
            panel_header("12-feature permutation importance", "Frozen reduced model · 2014 validation sample", "VALIDATION")
            try:
                importance, meta = load_reduced_importance()
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                st.warning(f"Reduced-model importance unavailable: {exc}")
                st.caption("Restore the reduced importance CSV, metadata and feature-ablation report in reports/modeling.")
            else:
                fig = go.Figure(go.Bar(
                    x=importance["mean_auc_drop"],
                    y=importance["feature"].str.replace("_", " "), orientation="h",
                    marker_color=PURPLE,
                    error_x=dict(type="data", array=importance["std_auc_drop"], color=MUTED),
                    hovertemplate="%{y}<br>Mean ROC-AUC drop: %{x:.5f}<extra></extra>"))
                configure_chart(fig, height=340)
                fig.update_xaxes(title_text="ROC-AUC drop after shuffling", zeroline=True, zerolinecolor=GRID)
                fig.update_yaxes(showgrid=False, automargin=True)
                plot(fig, "risk-importance")
                st.caption(f"{meta['sample_size']:,} stratified 2014 loans · {meta['n_repeats']} shuffles · "
                           f"sample ROC-AUC {meta['reference_auc']:.4f}. Bars show mean drop ± shuffle standard deviation, "
                           "not confidence intervals. Correlated features can share importance; this is not causal evidence.")

    with st.expander("Selected model decile data", expanded=False):
        st.dataframe(rows[["decile", "loans", "charged_off", "predicted_pct", "observed_pct"]],
                     hide_index=True, use_container_width=True,
                     column_config={"predicted_pct": st.column_config.NumberColumn("Predicted (%)", format="%.2f"),
                                    "observed_pct": st.column_config.NumberColumn("Observed (%)", format="%.2f")})
    with st.container(border=True, key="panel-risk-methodology"):
        panel_header("Methodology & limitations", "Interpretation boundaries for this historical evaluation")
        st.markdown(
            "- **Time split:** trained on 2011–2013; model selection used 2014. Platt calibration used "
            "January–June 2014 and was checked on July–December 2014. Final evaluation used 2015 loans "
            "without refitting models or calibration on the test set.\n"
            "- **Resolved-only selection:** only Fully Paid and Charged Off 36-month loans are included. "
            "Unresolved loans are excluded, so these conditional outcomes may not represent all originations or applicants.\n"
            "- **Imperfect point-in-time design:** a retrospective extract does not establish that every feature "
            "and outcome would have been available at each historical decision date. The calendar split alone "
            "does not eliminate leakage or outcome-availability concerns.\n"
            "- **Underprediction:** the calibrated model averages 13.04% versus 14.89% observed "
            "(−1.84 percentage points). Calibration on 2014 does not guarantee calibration on 2015 or future loans.\n"
            "- **Scope:** descriptive research, not validated for lending decisions. Do not use the holdout "
            "to retune or select models. No confidence intervals or statistical significance claims are supplied."
        )
    with st.expander("Underlying reports & provenance", expanded=False):
        st.caption(f"Evaluation recorded: {report['evaluated_at_utc']}. Reports are read from the project's reports/modeling folder.")
        for name in ("final_test_evaluation.json", "final_test_deciles.csv", "feature_ablation_validation.json",
                     "reduced_feature_importance.csv", "reduced_feature_importance_metadata.json"):
            path = MODELING_REPORT_DIR / name
            try:
                contents = path.read_bytes()
            except OSError:
                st.caption(f"Unavailable: {name}")
            else:
                st.download_button(name, contents, file_name=name,
                                   mime="text/csv" if name.endswith(".csv") else "application/json",
                                   key="risk-download-" + name)


def main():
    with st.sidebar:
        st.markdown(
            '<div class="side-brand"><div class="brand-symbol">◈</div>'
            '<div><div class="brand-title">LendScope</div>'
            '<div class="brand-caption">CREDIT INTELLIGENCE</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["Overview", "Credit grades", "Geography", "Loan purposes", "Data quality", "Risk Modelling"],
                        label_visibility="collapsed")
        st.markdown('<div class="sidebar-label sidebar-label-lower">SYSTEM</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-status"><span class="status-dot"></span> PostgreSQL + dbt <span>LOCAL</span></div>', unsafe_allow_html=True)
        if st.button("↻  Refresh analytics", use_container_width=True):
            load_marts.clear()
            load_latest_ingestion.clear()
            load_raw_quality_snapshot.clear()
            st.rerun()
        st.markdown(
            '<div class="sidebar-foot"><strong>Historical dataset</strong><br>'
            'LendingClub accepted loans<br>June 2007 to December 2018<br>'
            '<span>All monetary values are USD.</span></div>',
            unsafe_allow_html=True,
        )

    if page == "Risk Modelling":
        with st.container(key="risk-layout"):
            risk_modelling_screen()
        return

    try:
        data = load_marts()
    except Exception as exc:
        title_block("Database unavailable", "Could not load the dbt reporting tables.")
        st.error(f"Database connection or model query failed: {exc}")
        st.code("docker compose up -d\ndbt run --project-dir .\\dbt", language="powershell")
        st.info("Start PostgreSQL, verify your .env, and ensure the three marts have been built.")
        st.stop()

    vintage, purpose, states = data["vintage"], data["purpose"], data["states"]
    available_years = sorted(int(y) for y in vintage["issue_year"].dropna().unique())
    if page in ("Overview", "Credit grades", "Loan purposes"):
        heading_col, controls_col = st.columns([3.6, 2.7], vertical_alignment="top", gap="small")
        with heading_col:
            descriptions = {
                "Overview": "Historical originations and observed credit outcomes, in one view.",
                "Credit grades": "Explore risk segmentation and observed repayment outcomes.",
                "Loan purposes": "Understand how and why borrowers used LendingClub financing.",
            }
            title_block("Portfolio overview" if page == "Overview" else page,
                        descriptions[page], chip=None)
        with controls_col:
            label_spacer, label_col = st.columns([0.92, 1.33], gap="small")
            with label_col:
                st.markdown('<div class="control-label-row">Reporting year <span class="control-help">⊝</span></div>',
                            unsafe_allow_html=True)
            with st.container(key="reporting-controls"):
                badge_col, year_col = st.columns([0.92, 1.33], vertical_alignment="top", gap="small")
                with badge_col:
                    st.markdown('<div class="aligned-date-badge">HISTORICAL DATA · 2007–2018</div>',
                                unsafe_allow_html=True)
                with year_col:
                    selected = st.selectbox(
                        "Reporting year",
                        ["All years"] + available_years,
                        index=0,
                        label_visibility="collapsed",
                        help="Filters loan totals, grades and purposes. The geography panel is all years.",
                    )
        st.markdown('<div class="page-header-spacer"></div>', unsafe_allow_html=True)
        v_filtered = vintage if selected == "All years" else vintage[vintage["issue_year"] == selected]
        p_filtered = purpose if selected == "All years" else purpose[purpose["issue_year"] == selected]
    else:
        selected = "All years"
        v_filtered, p_filtered = vintage, purpose

    if page == "Overview":
        kpis(v_filtered)
        section_label("PERFORMANCE OVERVIEW", "Data prepared with PostgreSQL and dbt")
        left, right = st.columns([1.4, 1], gap="small")
        with left:
            with st.container(border=True, key="panel-trend"):
                panel_header("Original lending trend", "Annual originated loan amounts, full history", "2007–2018")
                plot(chart_trend(vintage, selected), "chart_trend")
        with right:
            with st.container(border=True, key="panel-grades"):
                panel_header("Credit grade outcomes", "Charged-off share among resolved loans", str(selected).upper())
                plot(chart_grades(grade_summary(v_filtered)), "chart_grades")
        left2, right2 = st.columns([1.4, 1], gap="small")
        with left2:
            with st.container(border=True, key="panel-purposes"):
                panel_header("Borrowing purposes", "Largest categories by original lending", str(selected).upper())
                plot(chart_purposes(purpose_summary(p_filtered, 7)), "chart_purposes")
        with right2:
            with st.container(border=True, key="panel-states"):
                panel_header("Geographic concentration", "Top states, full historical period", "ALL YEARS")
                plot(chart_states(states, 7), "chart_states")
        st.caption("Full state-level table available under Geography. Geographic figures cover all historical years.")

    elif page == "Credit grades":
        kpis(v_filtered, separate_from_charts=True)
        left, right = st.columns([1.25, 1], gap="small")
        grade_df = grade_summary(v_filtered)
        with left:
            with st.container(border=True, key="panel-grades"):
                panel_header("Observed charge-offs", "Among resolved loans only", str(selected).upper())
                detail_chart = chart_grades(grade_df)
                detail_chart.update_layout(height=165)
                plot(detail_chart, "grade_detail")
        with right:
            with st.container(border=True, key="panel-grade-counts"):
                panel_header("Loan volumes by grade", "Number of loans issued", str(selected).upper())
                fig = go.Figure(go.Bar(x=grade_df["grade"], y=grade_df["loan_count"],
                                       marker_color=GLOW[:len(grade_df)],
                                       hovertemplate="Grade %{x}: %{y:,} loans<extra></extra>"))
                fig = configure_chart(fig, height=165)
                fig.update_yaxes(tickformat="~s", rangemode="tozero")
                plot(fig, "grade_count")
        with st.container(border=True, key="panel-grade-table"):
            panel_header("Grade summary", "Resolved-loan share is not a calibrated default probability")
            grade_table(grade_df)

    elif page == "Geography":
        title_block("Geographic exposure", "Historical origination amounts and observed outcomes by US state.")
        st.caption("All historical years combined. Year filtering is unavailable for the current geography mart.")
        with st.container(border=True, key="panel-geography"):
            panel_header("State concentrations", "Percentage of total original lending", "ALL YEARS")
            plot(chart_states(states, 10), "geography_detail")
        with st.container(border=True, key="panel-geo-table"):
            panel_header("State portfolio table", "Historical original loan amounts, not live balances")
            region_table(states, len(states))

    elif page == "Loan purposes":
        kpis(v_filtered, separate_from_charts=True)
        with st.container(border=True, key="panel-purpose-detail"):
            panel_header("Lending by purpose", "Top loan purposes by original lending amount", str(selected).upper())
            detail_chart = chart_purposes(purpose_summary(p_filtered, 8))
            detail_chart.update_layout(height=186)
            plot(detail_chart, "purpose_detail")
        with st.container(border=True, key="panel-purpose-table"):
            panel_header("Borrowing purpose table", "Totals aggregated from the year-and-purpose dbt mart")
            df = (p_filtered.groupby("loan_purpose", as_index=False)
                  .agg(loans=("loan_count", "sum"), original_amount=("total_loan_amount", "sum"))
                  .sort_values("original_amount", ascending=False))
            df = df.rename(columns={"loan_purpose": "Loan purpose", "loans": "Loans",
                                    "original_amount": "Original lending"})
            df["Loan purpose"] = df["Loan purpose"].map(display_loan_purpose)
            df["Loans"] = df["Loans"].map(lambda n: f"{int(n):,}")
            df["Original lending"] = df["Original lending"].map(money)
            st.dataframe(df, hide_index=True, use_container_width=True, height=248)

    else:
        # A real Streamlit container owns the spacing between all quality sections.
        with st.container(key="quality-layout"):
            quality_screen(data)

    st.markdown(
        '<div class="page-footer">LENDSCOPE <span>•</span> BUILT WITH PYTHON, POSTGRESQL, DBT & STREAMLIT'
        '<span class="footer-right">Historic data • Not investment advice</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
