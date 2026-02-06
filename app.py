# app.py — FINAL STREAMLIT DASHBOARD (Notebook-parity + 2026 predictions + Tableau link)
# Fixes:
# - Option 1 caching for sklearn models (underscore args in forecast_year)
# - Unique widget keys to prevent StreamlitDuplicateElementId
# - Tableau link compatible with older Streamlit (NO st.link_button, uses st.markdown)

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Sudan Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(".")

# ---------------------------
# Core data files
# ---------------------------
FOOD_MASTER = ROOT / "regional_food_security_master.csv"
DISP_FACT = ROOT / "geo_admin1_snapshots_2023_2024_INTENSITY.csv"
HOTSPOTS = ROOT / "report_hotspots_top5_by_date.csv"  # optional

ADMIN1_GEOJSON = ROOT / "sdn_admin_boundaries_unzipped" / "sdn_admin1.geojson"

# Tableau (LINK ONLY)
TABLEAU_LINK = "https://public.tableau.com/app/profile/clariss.wangari/viz/tableaudispacementintensity/SudanIDPIntensityHotspots"


# ---------------------------
# Helpers
# ---------------------------
def safe_read_csv(p: Path, **kwargs):
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as e:
        st.error(f"Could not read {p}: {e}")
        return None


def df_search_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df is None or df.empty or not q:
        return df
    q = q.lower().strip()
    try:
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(q, na=False)).any(axis=1)
        return df[mask]
    except Exception:
        return df


# ---------------------------
# Food panel (IPC)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_food_master(p: Path):
    df = safe_read_csv(p, low_memory=False)
    if df is None:
        return None

    if "From" in df.columns:
        df = df[~df["From"].astype(str).str.startswith("#")].copy()
        df["From"] = pd.to_datetime(df["From"], errors="coerce")
        df = df.dropna(subset=["From"]).copy()
        df["month"] = df["From"].dt.to_period("M").dt.to_timestamp()

    for c in ["Country", "Phase"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "Area" in df.columns or "state" in df.columns:
        area = df["Area"] if "Area" in df.columns else pd.Series([None] * len(df))
        state = df["state"] if "state" in df.columns else pd.Series([None] * len(df))
        df["admin"] = area.where(area.notna(), state).astype(str).str.strip()

    if "Percentage" in df.columns:
        df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def build_food_panel(df: pd.DataFrame):
    req = {"Country", "admin", "month", "Phase", "Percentage"}
    if df is None or not req.issubset(df.columns):
        return None

    clean = df.dropna(subset=["Country", "admin", "month", "Phase", "Percentage"]).copy()

    panel = (
        clean.pivot_table(
            index=["Country", "admin", "month"],
            columns="Phase",
            values="Percentage",
            aggfunc="first",
        )
        .reset_index()
    )

    panel.columns = [
        ("phase_" + str(c).replace("+", "plus").replace(" ", "").lower())
        if c not in ["Country", "admin", "month"]
        else c
        for c in panel.columns
    ]

    panel = panel.sort_values(["Country", "admin", "month"]).copy()

    panel["month_num"] = panel["month"].dt.month
    panel["sin_month"] = np.sin(2 * np.pi * panel["month_num"] / 12)
    panel["cos_month"] = np.cos(2 * np.pi * panel["month_num"] / 12)

    if "phase_3plus" in panel.columns:
        panel["ipc3plus_lag1"] = panel.groupby(["Country", "admin"])["phase_3plus"].shift(1)
        panel["ipc3plus_lag2"] = panel.groupby(["Country", "admin"])["phase_3plus"].shift(2)
        panel["ipc3plus_lag3"] = panel.groupby(["Country", "admin"])["phase_3plus"].shift(3)

        panel["ipc3plus_roll3"] = (
            panel.groupby(["Country", "admin"])["phase_3plus"]
            .apply(lambda s: s.shift(1).rolling(3).mean())
            .reset_index(level=[0, 1], drop=True)
        )

        panel["ipc3plus_next"] = panel.groupby(["Country", "admin"])["phase_3plus"].shift(-1)
        panel["ipc3plus_delta_next"] = panel["ipc3plus_next"] - panel["phase_3plus"]
        panel["worsen_next_2pp"] = (panel["ipc3plus_delta_next"] >= 0.02).astype(int)

    return panel


# ---------------------------
# Displacement data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_displacement_fact(p: Path):
    df = safe_read_csv(p, low_memory=False)
    if df is None:
        return None

    for dcol in ["report_date", "date", "Date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            if dcol != "report_date":
                df["report_date"] = df[dcol]
            break

    num_cols = [
        "idps",
        "households",
        "rank",
        "hotspot_intensity_rank",
        "intensity_idps_minmax",
        "intensity_idps_share",
        "intensity_idps_global",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


# ---------------------------
# Admin boundaries + labeled maps
# ---------------------------
@st.cache_data(show_spinner=False)
def load_admin1(geojson_path: Path):
    if not geojson_path.exists():
        return None, None
    gdf = gpd.read_file(str(geojson_path))
    gdf["adm1_pcode"] = gdf["adm1_pcode"].astype(str).str.strip()

    keep_codes = [f"SD{str(i).zfill(2)}" for i in range(1, 19)]
    gdf = gdf[gdf["adm1_pcode"].isin(keep_codes)].copy()

    name_candidates = ["adm1_en", "adm1_name", "adm1_name_en", "name", "state_name", "STATE_EN"]
    name_col = next((c for c in name_candidates if c in gdf.columns), None)
    if name_col is None:
        raise ValueError(f"No state name column found in admin file. Columns: {list(gdf.columns)}")

    return gdf, name_col


def plot_admin1_map(gdf, name_col, snap_df, value_col, title):
    m = gdf.merge(snap_df, left_on="adm1_pcode", right_on="state_code", how="left")
    m[value_col] = pd.to_numeric(m[value_col], errors="coerce").fillna(0)

    label_points = m.geometry.representative_point()

    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    m.plot(
        column=value_col,
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"},
        edgecolor="black",
        linewidth=0.5,
    )

    for (x, y), label in zip(zip(label_points.x, label_points.y), m[name_col].astype(str)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    return fig, m


# ---------------------------
# Modeling: train + forecast 2026
# ---------------------------
@st.cache_resource(show_spinner=False)
def train_models(panel: pd.DataFrame):
    model_frame = panel.dropna(subset=["phase_3plus"]).copy()
    model_frame["ipc3plus_next"] = model_frame.groupby(["Country", "admin"])["phase_3plus"].shift(-1)
    model_frame["ipc3plus_delta_next"] = model_frame["ipc3plus_next"] - model_frame["phase_3plus"]
    model_frame["worsen_next_2pp"] = (model_frame["ipc3plus_delta_next"] >= 0.02).astype(int)

    # Critical: drop rows where y is NaN
    model_df = model_frame.dropna(subset=["ipc3plus_next"]).copy()

    feat_cols = [
        "phase_3plus",
        "ipc3plus_lag1",
        "ipc3plus_lag2",
        "ipc3plus_lag3",
        "ipc3plus_roll3",
        "sin_month",
        "cos_month",
    ]

    missing = [c for c in feat_cols if c not in model_df.columns]
    if missing:
        raise ValueError(f"Missing required features for modeling: {missing}")

    X = model_df[feat_cols]
    y_reg = model_df["ipc3plus_next"]
    y_clf = model_df["worsen_next_2pp"]

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feat_cols,
            )
        ]
    )

    reg_model = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
    clf_model = Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=300, random_state=42))])

    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)

    return reg_model, clf_model, feat_cols


def month_features(dt: pd.Timestamp):
    m = int(dt.month)
    return np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)


@st.cache_data(show_spinner=False)
def forecast_year(panel_df: pd.DataFrame, _reg_model, _clf_model, year: int = 2026):
    hist = panel_df.sort_values(["Country", "admin", "month"]).copy()

    series_map = {}
    for (c, a), g in hist.groupby(["Country", "admin"]):
        g2 = g.dropna(subset=["phase_3plus"]).sort_values("month")
        series_map[(c, a)] = list(zip(pd.to_datetime(g2["month"]).tolist(), g2["phase_3plus"].tolist()))

    preds = []

    for mth in range(1, 13):
        target_month = pd.Timestamp(year=year, month=mth, day=1)

        for (c, a), series in series_map.items():
            series = [(d, v) for d, v in series if d < target_month]
            if len(series) == 0:
                continue

            current_val = series[-1][1]
            lag1 = series[-1][1] if len(series) >= 1 else np.nan
            lag2 = series[-2][1] if len(series) >= 2 else np.nan
            lag3 = series[-3][1] if len(series) >= 3 else np.nan
            roll3 = np.mean([series[-1][1], series[-2][1], series[-3][1]]) if len(series) >= 3 else np.nan

            sin_m, cos_m = month_features(target_month)

            X_one = pd.DataFrame(
                [
                    {
                        "phase_3plus": float(current_val),
                        "ipc3plus_lag1": float(lag1) if pd.notna(lag1) else np.nan,
                        "ipc3plus_lag2": float(lag2) if pd.notna(lag2) else np.nan,
                        "ipc3plus_lag3": float(lag3) if pd.notna(lag3) else np.nan,
                        "ipc3plus_roll3": float(roll3) if pd.notna(roll3) else np.nan,
                        "sin_month": sin_m,
                        "cos_month": cos_m,
                    }
                ]
            )

            pred_next = float(_reg_model.predict(X_one)[0])
            pred_next = max(0.0, min(1.0, pred_next))  # adjust to 100 if needed

            prob_worsen = float(_clf_model.predict_proba(X_one)[:, 1][0])

            preds.append(
                {
                    "Country": c,
                    "admin": a,
                    "month": target_month,
                    "ipc3plus_pred": pred_next,
                    "prob_worsen_2pp": prob_worsen,
                }
            )

            series_map[(c, a)] = series + [(target_month, pred_next)]

    return pd.DataFrame(preds).sort_values(["Country", "admin", "month"]).reset_index(drop=True)


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Display controls")
    show_tables = st.checkbox("Show tables", True, key="sidebar_show_tables")

# ---------------------------
# Header + search
# ---------------------------
st.title("Sudan Early Warning & Humanitarian Risk Dashboard")
query = st.text_input("Search dashboard (state, admin, phase, idps, keyword)", "", key="global_search").strip()

tabs = st.tabs(
    [
        "Overview",
        "Food Report",
        "Displacement (Tables)",
        "Maps (LIVE + State Names)",
        "Predictions (2026)",
        "Tableau (Link only)",
    ]
)

# ---------------------------
# Overview
# ---------------------------
with tabs[0]:
    st.markdown(
        """
### What’s inside
- Food Report (IPC Phase 3+)
- Displacement (tables) + year filter
- LIVE geospatial maps (with state names)
- 2026 predictions (IPC3+ forecast + probability of worsening ≥2pp)
- Tableau link (opens in a new tab)
"""
    )

# ---------------------------
# Food
# ---------------------------
with tabs[1]:
    st.subheader("Food Security Report (IPC Phase 3+)")

    df_food = load_food_master(FOOD_MASTER)
    if df_food is None:
        st.error("Missing `regional_food_security_master.csv` in the project folder.")
    else:
        panel = build_food_panel(df_food)
        if panel is None:
            st.error("Food CSV exists but required columns are missing: Country, Area/state, From, Phase, Percentage.")
        else:
            st.success(f"Food panel ready: {panel.shape[0]:,} rows × {panel.shape[1]} columns")

            if "phase_3plus" in panel.columns:
                c1, c2, c3 = st.columns(3)

                country = c1.selectbox(
                    "Country",
                    sorted(panel["Country"].dropna().unique()),
                    key="food_country",
                )

                admins = sorted(panel.loc[panel["Country"] == country, "admin"].dropna().unique())
                admin = c2.selectbox(
                    "Admin",
                    admins,
                    key="food_admin",
                )

                months = sorted(
                    panel.loc[(panel["Country"] == country) & (panel["admin"] == admin), "month"].dropna().unique()
                )
                month = c3.selectbox(
                    "Month",
                    months,
                    key="food_month",
                )

                snap_raw = panel[
                    (panel["Country"] == country) & (panel["admin"] == admin) & (panel["month"] == month)
                ].copy()

                st.markdown("### Next-month outcome (observed from data)")
                if snap_raw.empty:
                    st.error("No data for that Country/Admin/Month selection.")
                else:
                    row = snap_raw.iloc[0]
                    if ("ipc3plus_next" in snap_raw.columns) and pd.notna(row.get("ipc3plus_next", np.nan)):
                        current = float(row["phase_3plus"])
                        nextm = float(row["ipc3plus_next"])
                        delta = nextm - current
                        worsened = int(delta >= 0.02)

                        st.write(f"Current IPC3+ share: {current:.3f}")
                        st.write(f"Next-month IPC3+ (observed): {nextm:.3f}")
                        st.write(f"Change (next - current): {delta:.3f}")
                        st.write(f"Outcome: {'WORSENED (≥ +2pp)' if worsened == 1 else 'DID NOT WORSEN'}")
                    else:
                        st.info(
                            "Cannot compute observed next-month outcome for this selection. "
                            "Either this is the latest month available for that admin, or the next month is missing."
                        )

                snap = df_search_filter(snap_raw, query)
                if show_tables:
                    st.dataframe(snap, use_container_width=True, hide_index=True)

# ---------------------------
# Displacement tables
# ---------------------------
with tabs[2]:
    st.subheader("Displacement Report (Admin-1 Intensity)")

    df_disp = load_displacement_fact(DISP_FACT)
    if df_disp is None:
        st.error("Missing `geo_admin1_snapshots_2023_2024_INTENSITY.csv` in the project folder.")
    else:
        if "report_date" not in df_disp.columns or df_disp["report_date"].isna().all():
            st.error("No valid report_date values found in displacement data.")
        else:
            df_disp = df_disp.copy()
            df_disp["year"] = df_disp["report_date"].dt.year

            years = sorted(df_disp["year"].dropna().unique())
            sel_year = st.selectbox("Select year", years, index=len(years) - 1, key="disp_year")

            df_year = df_disp[df_disp["year"] == sel_year].copy()
            view = df_search_filter(df_year, query)

            if show_tables:
                st.dataframe(view.head(500), use_container_width=True)

        if HOTSPOTS.exists():
            df_hot = safe_read_csv(HOTSPOTS, low_memory=False)
            if df_hot is not None and show_tables:
                for dcol in ["report_date", "date", "Date"]:
                    if dcol in df_hot.columns:
                        df_hot[dcol] = pd.to_datetime(df_hot[dcol], errors="coerce")
                        if dcol != "report_date":
                            df_hot["report_date"] = df_hot[dcol]
                        break

                if "report_date" in df_hot.columns:
                    df_hot["year"] = df_hot["report_date"].dt.year
                    df_hot_view = df_hot[df_hot["year"] == sel_year].copy()
                else:
                    df_hot_view = df_hot.copy()

                st.markdown("#### Top-5 hotspots by date")
                st.dataframe(df_search_filter(df_hot_view, query).head(500), use_container_width=True)

# ---------------------------
# Maps
# ---------------------------
with tabs[3]:
    st.subheader("Maps (LIVE + State Names)")

    gdf_admin, name_col = load_admin1(ADMIN1_GEOJSON)
    if gdf_admin is None:
        st.error(f"Missing admin boundaries file: {ADMIN1_GEOJSON}")
        st.stop()

    df_disp = load_displacement_fact(DISP_FACT)
    if df_disp is None:
        st.error("Missing displacement fact file.")
        st.stop()

    dates = sorted(df_disp["report_date"].dropna().unique())
    if not dates:
        st.error("No valid report_date values found in displacement data.")
        st.stop()

    sel_date = st.selectbox("Select report date", dates, key="maps_date")
    snap = df_disp[df_disp["report_date"] == sel_date].copy()

    st.markdown("### Displacement intensity (global)")
    fig1, m1 = plot_admin1_map(
        gdf_admin,
        name_col,
        snap,
        value_col="intensity_idps_global",
        title=f"Sudan displacement intensity (global) — {pd.to_datetime(sel_date).date()}",
    )
    st.pyplot(fig1, use_container_width=True)

    if "idps" in m1.columns and m1["idps"].notna().any():
        idx = m1["idps"].idxmax()
        st.caption(f"Highest IDPs: {m1.loc[idx, name_col]} ({m1.loc[idx,'adm1_pcode']}) = {m1.loc[idx,'idps']:,.0f}")

    st.divider()

    st.markdown("### Hotspots (rank 1–5)")
    fig2, m2 = plot_admin1_map(
        gdf_admin,
        name_col,
        snap,
        value_col="hotspot_intensity_rank",
        title=f"Sudan hotspots (rank 1–5) — {pd.to_datetime(sel_date).date()}",
    )
    st.pyplot(fig2, use_container_width=True)

    top5 = m2[m2["hotspot_intensity_rank"].between(1, 5)].copy()
    if len(top5) > 0:
        top5 = top5.sort_values("hotspot_intensity_rank", ascending=True)[
            [name_col, "adm1_pcode", "hotspot_intensity_rank"]
        ]
        st.write("Top 5 hotspots (rank 1 = highest):")
        st.dataframe(top5, use_container_width=True, hide_index=True)
    else:
        st.info("No hotspot ranks 1..5 found for this date.")

# ---------------------------
# Predictions (2026)
# ---------------------------
with tabs[4]:
    st.subheader("Predictions for 2026 (IPC Phase 3+)")

    df_food = load_food_master(FOOD_MASTER)
    if df_food is None:
        st.error("Missing `regional_food_security_master.csv` in the project folder.")
        st.stop()

    panel = build_food_panel(df_food)
    if panel is None or "phase_3plus" not in panel.columns:
        st.error("Food panel not ready or missing `phase_3plus` column.")
        st.stop()

    with st.spinner("Training models (cached) and generating 2026 forecast..."):
        reg_model, clf_model, _ = train_models(panel)
        pred_2026 = forecast_year(panel, reg_model, clf_model, year=2026)

    st.success(f"2026 predictions ready: {pred_2026.shape[0]:,} rows")

    c1, c2, c3 = st.columns([2, 2, 2])

    country = c1.selectbox(
        "Country",
        sorted(pred_2026["Country"].dropna().unique()),
        key="pred_country",
    )

    admins = sorted(pred_2026.loc[pred_2026["Country"] == country, "admin"].dropna().unique())
    admin = c2.selectbox(
        "Admin",
        admins,
        key="pred_admin",
    )

    metric = c3.selectbox(
        "Metric",
        ["ipc3plus_pred", "prob_worsen_2pp"],
        key="pred_metric",
    )

    view = pred_2026[(pred_2026["Country"] == country) & (pred_2026["admin"] == admin)].copy()
    view = df_search_filter(view, query)

    st.markdown("### 2026 forecast trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(view["month"], view[metric], marker="o")
    ax.set_title(f"{metric} — {country} / {admin}")
    ax.set_xlabel("Month")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    if show_tables:
        st.markdown("### Forecast table")
        st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download all 2026 predictions (CSV)",
        data=pred_2026.to_csv(index=False).encode("utf-8"),
        file_name="ipc3plus_predictions_2026.csv",
        mime="text/csv",
        key="pred_download",
    )

# ---------------------------
# Tableau (Link only) — FIXED (no link_button)
# ---------------------------
with tabs[5]:
    st.subheader("Tableau (Link only)")

    st.markdown(
        """
Tableau Public embedding is frequently blocked by browser/network policy.
This dashboard is provided as a direct link instead.
        """
    )

    # Works on ALL Streamlit versions
    st.markdown(f"[Open Tableau dashboard (new tab)]({TABLEAU_LINK})")
    st.write(TABLEAU_LINK)
