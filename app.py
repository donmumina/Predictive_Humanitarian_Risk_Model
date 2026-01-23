from pathlib import Path
import re
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Sudan Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(".")

# Core outputs from your notebook
FOOD_MASTER = ROOT / "regional_food_security_master.csv"
DISP_FACT = ROOT / "geo_admin1_snapshots_2023_2024_INTENSITY.csv"
HOTSPOTS = ROOT / "report_hotspots_top5_by_date.csv"  # optional

# PNG outputs from notebook
ARTIFACTS = ROOT / "artifacts"
EDA_FOOD = ARTIFACTS / "eda_food"
EDA_GENERIC = ARTIFACTS / "eda"

MAPS_RAW = ROOT / "maps_out_idps_raw"
MAPS_INTENSITY = ROOT / "maps_out_intensity_global"
MAPS_HOTSPOTS = ROOT / "maps_out_hotspots"

# Optional health files
HEALTH_FILES = [
    ROOT / "cleaned_ipc_status.csv",
    ROOT / "health_status.csv",
    ROOT / "data" / "cleaned_ipc_status.csv",
    ROOT / "data" / "health_status.csv",
]

# Tableau: offline-safe (no iframe)
TABLEAU_LINK = "https://public.tableau.com/app/profile/clariss.wangari/viz/tableaudispacementintensity/SudanIDPIntensityHotspots?publish=yes"
TABLEAU_PREVIEW = ROOT / "tableau_preview.png"  # optional screenshot file


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

def list_pngs(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.png") if p.is_file()])

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
def build_food_panel(df):
    req = {"Country", "admin", "month", "Phase", "Percentage"}
    if not req.issubset(df.columns):
        return None

    clean = df.dropna(subset=["Country", "admin", "month", "Phase", "Percentage"]).copy()

    panel = (
        clean.pivot_table(
            index=["Country", "admin", "month"],
            columns="Phase",
            values="Percentage",
            aggfunc="first"
        )
        .reset_index()
    )

    panel.columns = [
        ("phase_" + str(c).replace("+", "plus").replace(" ", "").lower())
        if c not in ["Country", "admin", "month"] else c
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
        "idps", "households", "rank",
        "hotspot_intensity_rank",
        "intensity_idps_minmax",
        "intensity_idps_share",
        "intensity_idps_global",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

@st.cache_data(show_spinner=False)
def load_health():
    for p in HEALTH_FILES:
        if p.exists():
            df = safe_read_csv(p, low_memory=False)
            if df is not None:
                return df, p
    return None, None


def df_search_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """Row-wise substring search across all columns (safe, simple, good enough)."""
    if df is None or df.empty or not q:
        return df
    q = q.lower().strip()
    try:
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(q, na=False)).any(axis=1)
        return df[mask]
    except Exception:
        return df


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Display controls")
    img_width = st.slider("PNG width (px)", 400, 1600, 1000, 50)
    show_png = st.checkbox("Show PNGs", True)
    show_tables = st.checkbox("Show tables", True)

# ---------------------------
# Header + search
# ---------------------------
st.title("Sudan Early Warning & Humanitarian Risk Dashboard")
query = st.text_input("Search dashboard (state, admin, phase, idps, keyword)", "").strip()


tabs = st.tabs([
    "Overview",
    "Food Report",
    "Displacement Report",
    "Geospatial Maps (PNG)",
    "Health Status",
    "Tableau (Clickable Link)",
])

# ---- Overview
with tabs[0]:
    st.markdown("""
### What’s inside
- **Food Report:** IPC Phase 3+ panel + early warning fields (lags, rolling, next month, worsening flag)
- **Displacement Report:** admin-1 intensity panel + optional hotspot report
- **Geospatial Maps:** PNG map gallery
- **Health:** optional
- **Tableau:** clickable link + optional local preview (no iframe)
""")

    st.code("""
Expected files/folders:
- regional_food_security_master.csv
- geo_admin1_snapshots_2023_2024_INTENSITY.csv
- report_hotspots_top5_by_date.csv (optional)
- artifacts/ (feature_importance.png, shap_summary.png, eda_food/, eda/)
- maps_out_idps_raw/
- maps_out_intensity_global/
- maps_out_hotspots/
- tableau_preview.png (optional)
""".strip())

# ---- Food
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
                country = c1.selectbox("Country", sorted(panel["Country"].dropna().unique()))
                admins = sorted(panel.loc[panel["Country"] == country, "admin"].dropna().unique())
                admin = c2.selectbox("Admin", admins)
                months = sorted(panel.loc[(panel["Country"] == country) & (panel["admin"] == admin), "month"].dropna().unique())
                month = c3.selectbox("Month", months)

                snap = panel[(panel["Country"] == country) & (panel["admin"] == admin) & (panel["month"] == month)].copy()
                snap = df_search_filter(snap, query)

                if show_tables:
                    st.dataframe(snap, use_container_width=True, hide_index=True)

            if show_png:
                st.markdown("#### Food artifacts (PNG)")
                for p in [ARTIFACTS / "feature_importance.png", ARTIFACTS / "shap_summary.png"]:
                    if p.exists():
                        st.image(str(p), caption=p.name, width=img_width)

                for folder, label in [(EDA_FOOD, "EDA Food"), (EDA_GENERIC, "EDA Generic")]:
                    pngs = list_pngs(folder)
                    if pngs:
                        with st.expander(f"{label} ({len(pngs)} images)"):
                            for p in pngs:
                                st.image(str(p), caption=str(p), width=img_width)

# ---- Displacement
with tabs[2]:
    st.subheader("Displacement Report (Admin-1 Intensity)")

    df_disp = load_displacement_fact(DISP_FACT)
    if df_disp is None:
        st.error("Missing `geo_admin1_snapshots_2023_2024_INTENSITY.csv` in the project folder.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("States", int(df_disp["state_code"].nunique()) if "state_code" in df_disp.columns else 0)
        c2.metric("Dates", int(df_disp["report_date"].nunique()) if "report_date" in df_disp.columns else 0)
        c3.metric("Rows", len(df_disp))

        view = df_disp.copy()
        view = df_search_filter(view, query)

        if show_tables:
            st.dataframe(view.head(500), use_container_width=True)

        if HOTSPOTS.exists():
            df_hot = safe_read_csv(HOTSPOTS, low_memory=False)
            if df_hot is not None:
                df_hot = df_search_filter(df_hot, query)
                if show_tables:
                    st.markdown("#### Top-5 hotspots by date")
                    st.dataframe(df_hot.head(500), use_container_width=True)

# ---- Maps
with tabs[3]:
    st.subheader("Geospatial Maps (PNG)")

    maps = {
        "Raw IDPs": list_pngs(MAPS_RAW),
        "Global intensity": list_pngs(MAPS_INTENSITY),
        "Top-5 hotspots": list_pngs(MAPS_HOTSPOTS),
    }

    if not any(len(v) for v in maps.values()):
        st.warning("No map PNGs found. Run the notebook map export to populate `maps_out_*` folders.")
    else:
        kind = st.selectbox("Map type", list(maps.keys()))
        files = maps[kind]

        if files:
            def label(p):
                m = re.search(r"(\\d{4}-\\d{2}-\\d{2})", p.name)
                return m.group(1) if m else p.name

            options = {label(p): p for p in files}
            choice = st.selectbox("Date / file", list(options.keys()))
            st.image(str(options[choice]), caption=str(options[choice]), use_container_width=True)

# ---- Health
with tabs[4]:
    st.subheader("Health Status (Optional)")
    hdf, hpath = load_health()
    if hdf is None:
        st.warning("No health file detected. Add `cleaned_ipc_status.csv` or `health_status.csv` to show this panel.")
    else:
        st.success(f"Loaded: {hpath}")
        view = df_search_filter(hdf.copy(), query)
        if show_tables:
            st.dataframe(view.head(500), use_container_width=True)

# ---- Tableau (clickable + searchable)
with tabs[5]:
    st.subheader("Tableau (Clickable Link)")
    st.caption("Tableau embedding is blocked on many networks. This section is link-only, so it works reliably.")

    # Clickable button
    st.link_button("Open Tableau dashboard", TABLEAU_LINK)

    # Clickable text link (also copyable)
    st.markdown(f"[Open Tableau dashboard in browser]({TABLEAU_LINK})")
    st.write(TABLEAU_LINK)

    st.divider()

    # Offline preview for environments where Tableau is blocked
    if TABLEAU_PREVIEW.exists():
        st.markdown("#### Local preview (recommended for offline / blocked networks)")
        st.image(str(TABLEAU_PREVIEW), caption="tableau_preview.png", use_container_width=True)
        with open(TABLEAU_PREVIEW, "rb") as f:
            st.download_button("Download preview PNG", f, file_name="tableau_preview.png")
    else:
        st.info("Optional: save a screenshot/export as `tableau_preview.png` in this folder to show a preview here.")
