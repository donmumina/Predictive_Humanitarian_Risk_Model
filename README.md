# Sudan Food Insecurity & Displacement Early Warning Prototype

## Project Overview

This project presents an end-to-end analytical prototype for **anticipatory humanitarian decision-making** in Sudan. It integrates food insecurity indicators (IPC phases) and internal displacement data to generate **early warning signals, short-term forecasts, displacement pressure rankings, and spatial hotspot intelligence** at the Admin-1 (state) level.

The analysis is designed to support **forward-looking prioritization** by identifying areas at risk of deterioration before conditions escalate into humanitarian emergencies.

---

## Problem Context

Sudan's humanitarian context is highly volatile, with rapid and interacting changes in:

* Food insecurity severity (IPC phases),
* Internal displacement patterns,
* Conflict dynamics and seasonal stressors.

Most existing monitoring systems rely on **reactive snapshots** of conditions. This project shifts toward a **predictive and operational early-warning approach** that enables:

* Proactive planning,
* Better resource prioritization,
* Timely and targeted humanitarian response.

---

## Objectives

The project is structured around two core analytical objectives:

### 1. Food Insecurity Forecasting (Admin-1 Level)

* Forecast next-month IPC Phase 3+ (IPC3+) prevalence.
* Identify states at risk of **meaningful short-term deterioration**.
* Provide interpretable, uncertainty-aware early warning signals.

### 2. Displacement Pressure & Hotspot Intelligence

* Rank states by total and relative IDP burden.
* Identify spatial concentration hotspots.
* Produce time-comparable outputs suitable for mapping and dashboards.

---

## Data Sources

The analysis integrates multiple datasets, including:

* IPC food insecurity phase distributions,
* Internal displacement (DTM) datasets,
* Population and household data,
* Administrative boundary (GeoJSON) data for Sudan.

All datasets are cleaned, standardized, and aligned to **Admin-1 geography and monthly time steps**.

---

## Methodology Overview

This project follows a structured **CRISP-DM-aligned analytical pipeline**:

### 1. Data Ingestion & Cleaning

* Standardization of geographic identifiers (country, state, codes).
* Parsing and validation of reporting dates.
* Removal of metadata rows and inconsistent records.
* Numeric coercion and missing-value handling.

### 2. Feature Engineering

* Construction of a Country–Admin–Month panel.
* Generation of IPC phase share features.
* Temporal features (lags, rolling averages).
* Seasonality encoding using cyclical month features.

### 3. Modeling

Two predictive tasks are addressed:

**Regression**

* Forecast next-month IPC Phase 3+ prevalence.
* Baseline models: Ridge Regression, Random Forest, Gradient Boosting.
* Best-performing tuned model: Random Forest (R² ≈ 0.43).

**Classification**

* Predict whether IPC3+ will worsen by ≥ 2 percentage points.
* Baseline models: Logistic Regression, Random Forest, Gradient Boosting.
* Best-performing model: Random Forest Classifier (AUC ≈ 0.68).

### 4. Evaluation & Explainability

* Time-based train/test split to avoid leakage.
* Task-appropriate metrics (RMSE, R², AUC, F1).
* SHAP analysis for global and local model interpretability.

### 5. Geospatial & Operational Outputs

* Admin-1 displacement intensity maps.
* Hotspot (Top-5) ranking maps.
* GIS-ready CSVs and Tableau-ready tables.

---

## Key Outputs

This project produces:

* Cleaned and standardized analytical datasets,
* Short-term food insecurity forecasts,
* Displacement pressure and hotspot rankings,
* Time-series and distribution visualizations,
* GIS- and dashboard-ready CSV outputs.

All outputs are designed to be **operationally usable** by humanitarian analysts and decision-makers.

---

## Key Conclusions

* Food insecurity exhibits **strong temporal persistence**, with current IPC3+ levels and recent trends dominating forecasts.
* Tree-based models outperform linear baselines, indicating **nonlinear dynamics** in humanitarian risk.
* Short-term forecasting is feasible but inherently uncertain in conflict settings.
* SHAP explanations confirm domain intuition and improve model transparency.
* Predictive outputs are best suited for **early warning and prioritization**, not deterministic decision-making.

---

## Recommendations

### Analytical

* Integrate exogenous drivers (prices, rainfall, conflict events).
* Extend forecasts to multi-month horizons.
* Explore probabilistic and scenario-based forecasting.

### Operational

* Use classification outputs as **early warning flags**.
* Pair predictions with SHAP explanations for analyst review.
* Embed outputs into dashboards for routine monitoring.

### Policy & Humanitarian Use

* Treat outputs as **decision-support tools**.
* Prioritize areas showing both high IPC3+ levels and upward trends.
* Update models regularly as new IPC rounds are released.

---

## Limitations

* Results depend on data quality and reporting frequency.
* Predictive components do not capture all political or conflict dynamics.
* Outputs should be interpreted alongside qualitative assessments.

---

## How to Use This Project

1. Run the notebook **top to bottom** to reproduce results.
2. Review intermediate outputs for validation.
3. Use exported CSVs and maps for dashboards or GIS tools.
4. Adapt thresholds and assumptions to operational needs.

---
Group_3_Project/
├── Group_3_Project.ipynb          # Main analysis notebook
├── README.md                      # This file
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   └── processed/                 # Cleaned and processed data
├── outputs/                       # Analysis outputs
│   ├── forecasts/                 # Prediction results
│   ├── visualizations/            # Charts and maps
│   └── gis_ready/                 # GIS-ready files
└── utils/                         # Utility functions

---

## Authors
**Group 3**
- Michael Mumina (Scrum Master)
- Sharon Nyakeya
- Bryan Njogu
- Ashley Kibwogo
- Tanveer Chege
- Claris Wangari
- Priscillah Giriama
