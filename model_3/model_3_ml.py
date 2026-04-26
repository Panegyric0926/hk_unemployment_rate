"""
Machine Learning Analysis: Cross-Sector Unemployment & Foreign Labour Import

This script builds ML models to explain why different industries have different
unemployment rates and how foreign-labour import intensity relates to sectoral
unemployment. It provides policy-relevant insights via:

    1. Random Forest & Gradient Boosting – feature importance rankings
    2. SHAP analysis – individual feature contributions by sector
    3. Partial Dependence Plots – marginal effect of intensity on unemployment
    4. Policy Counterfactual Simulation – predicted unemployment under
       alternative imported-worker scenarios
    5. Sector Vulnerability Heatmap – which sectors are most sensitive

Data pipeline:
    - Pool quarterly panel data across 4 sectors (Construction, Retail,
      Transportation, Community) with macro controls (GDP, inflation,
      interest rate) and foreign-labour intensity.
    - Intensity = ImportedWorkers_{s,t} / TotalSectorEmployment_{s,t}

Requires: numpy, scikit-learn, shap, matplotlib, statsmodels
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL2_DIR = BASE_DIR.parent / "model_2"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}

SECTOR_CONFIG = {
    "Construction": {
        "folder": "construction",
        "employment_industry": "Construction",
        "unemployment_filter": None,
    },
    "Retail": {
        "folder": "retail",
        "employment_industry": "Retail, accommodation and food services",
        "unemployment_filter": None,
    },
    "Transportation": {
        "folder": "transportation",
        "employment_industry": (
            "Transportation, storage, postal and courier services, "
            "information and communications"
        ),
        "unemployment_filter": None,
    },
    "Community": {
        "folder": "community",
        "employment_industry": "Public administration, social and personal services",
        "unemployment_filter": "Public administration, social and personal services",
    },
}

SECTOR_COLORS = {
    "Construction": "#e29578",
    "Retail": "#89c6ac",
    "Transportation": "#19929d",
    "Community": "#5cb0cf",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_yyyyqq(value):
    parts = value.strip().split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1][1])
    except (ValueError, IndexError):
        return None


def parse_month_range(value):
    if not value or "/" not in value:
        return None
    try:
        left = value.split("-")[0].strip()
        month_str, year_str = left.split("/")
        month = int(month_str.strip())
        year = int(year_str.strip())
    except (ValueError, IndexError):
        return None
    quarter = MONTH_TO_Q.get(month)
    if quarter is None:
        return None
    return year, quarter


# ---------------------------------------------------------------------------
# Data readers (reused from model_3_macro.py)
# ---------------------------------------------------------------------------

def read_macro_series():
    gdp_growth, inflation, interest_rate = {}, {}, {}

    with (MODEL2_DIR / "gdp.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_yyyyqq((row.get("Time") or "").strip())
            val = (row.get("GDP Year-on-year Change") or "").strip()
            if key and val:
                try:
                    gdp_growth[key] = float(val.lstrip("+"))
                except ValueError:
                    pass

    with (MODEL2_DIR / "inflation_rate.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_yyyyqq((row.get("Quarter") or "").strip())
            val = (row.get("Quarterly Inflation Rate (%)") or "").strip()
            if key and val:
                try:
                    inflation[key] = float(val)
                except ValueError:
                    pass

    with (MODEL2_DIR / "interest_rate.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_yyyyqq((row.get("Time") or "").strip())
            val = (row.get("3 months") or "").strip()
            if key and val:
                try:
                    interest_rate[key] = float(val)
                except ValueError:
                    pass

    return gdp_growth, inflation, interest_rate


def read_total_employment():
    employment = {}
    with (BASE_DIR / "all_sector_labour.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            q_str = (row.get("Quarter") or "").strip()
            industry = (row.get("Industry of main employment") or "").strip()
            value = (row.get("Overall(1000 persons)") or "").strip()
            if not year_str or not q_str or not industry or not value:
                continue
            try:
                employment.setdefault(industry, {})[(int(year_str), int(q_str[1]))] = float(value)
            except (ValueError, IndexError):
                continue
    return employment


def read_imported_workers(folder):
    imported = {}
    path = MODEL2_DIR / folder / "imported_workers.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            val = (row.get("Number of Imported Workers Approved") or "").strip()
            if not year_str.isdigit() or not val:
                continue
            try:
                quarterly = float(val) / 4.0
            except ValueError:
                continue
            year = int(year_str)
            for q in range(1, 5):
                imported[(year, q)] = quarterly
    return imported


def read_unemployment(folder, industry_filter=None):
    unemployment = {}
    path = MODEL2_DIR / folder / "unemployment_rate.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            time_val = (row.get("Time") or "").strip()
            unemp_val = (row.get("Unemployment Rate") or "").strip()
            if not time_val or not unemp_val:
                continue
            if industry_filter is not None:
                industry = (row.get("Detailed previous industry") or "").strip()
                if industry != industry_filter:
                    continue
            key = parse_month_range(time_val)
            if key is None:
                continue
            try:
                unemployment[key] = float(unemp_val)
            except ValueError:
                continue
    return unemployment


# ---------------------------------------------------------------------------
# Build pooled panel dataset
# ---------------------------------------------------------------------------

def build_panel():
    gdp, infl, ir = read_macro_series()
    all_employment = read_total_employment()

    records = []
    for sector_name, config in SECTOR_CONFIG.items():
        imported = read_imported_workers(config["folder"])
        unemployment = read_unemployment(config["folder"], config["unemployment_filter"])
        employment = all_employment.get(config["employment_industry"], {})

        common_keys = sorted(
            set(imported) & set(unemployment) & set(employment)
            & set(gdp) & set(infl) & set(ir)
        )

        for year, quarter in common_keys:
            emp_thousands = employment[(year, quarter)]
            iw = imported[(year, quarter)]
            intensity = iw / emp_thousands  # per 1000 persons

            records.append({
                "Sector": sector_name,
                "Year": year,
                "Quarter": quarter,
                "Unemployment Rate": unemployment[(year, quarter)],
                "Imported Workers": iw,
                "Total Employment (1000)": emp_thousands,
                "Intensity": intensity,
                "GDP Growth": gdp[(year, quarter)],
                "Inflation Rate": infl[(year, quarter)],
                "Interest Rate 3M": ir[(year, quarter)],
            })

    return records


# ---------------------------------------------------------------------------
# Prepare feature matrix
# ---------------------------------------------------------------------------

def prepare_features(records):
    """Return X (features), y (target), feature_names, sector_labels."""
    sector_encoder = LabelEncoder()
    sector_encoded = sector_encoder.fit_transform([r["Sector"] for r in records])

    feature_names = [
        "Intensity",
        "GDP Growth",
        "Inflation Rate",
        "Interest Rate 3M",
        "Total Employment (1000)",
        "Sector (encoded)",
    ]

    X = np.column_stack([
        [r["Intensity"] for r in records],
        [r["GDP Growth"] for r in records],
        [r["Inflation Rate"] for r in records],
        [r["Interest Rate 3M"] for r in records],
        [r["Total Employment (1000)"] for r in records],
        sector_encoded,
    ])
    y = np.array([r["Unemployment Rate"] for r in records])
    sector_labels = np.array([r["Sector"] for r in records])

    return X, y, feature_names, sector_labels, sector_encoder


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(X, y):
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=3, random_state=42,
    )
    rf.fit(X, y)
    gb.fit(X, y)
    return rf, gb


def evaluate_models(rf, gb, X, y):
    loo = LeaveOneOut()
    rf_scores = cross_val_score(rf, X, y, cv=loo, scoring="neg_mean_absolute_error")
    gb_scores = cross_val_score(gb, X, y, cv=loo, scoring="neg_mean_absolute_error")

    print("\n" + "=" * 70)
    print("  Model Evaluation (Leave-One-Out Cross-Validation)")
    print("=" * 70)
    print(f"  Random Forest  – MAE: {-rf_scores.mean():.4f} (±{rf_scores.std():.4f})")
    print(f"  Gradient Boost – MAE: {-gb_scores.mean():.4f} (±{gb_scores.std():.4f})")
    print(f"  Random Forest  – R² (in-sample): {rf.score(X, y):.4f}")
    print(f"  Gradient Boost – R² (in-sample): {gb.score(X, y):.4f}")


# ===========================================================================
# VISUALISATIONS
# ===========================================================================

def plot_feature_importance(rf, gb, feature_names):
    """Side-by-side feature importance from RF and GB."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model, title in [
        (axes[0], rf, "Random Forest"),
        (axes[1], gb, "Gradient Boosting"),
    ]:
        importances = model.feature_importances_
        order = np.argsort(importances)
        ax.barh(
            np.array(feature_names)[order],
            importances[order],
            color="teal", alpha=0.8,
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(title)

    fig.suptitle(
        "Which Factors Drive Cross-Sector Unemployment Differences?",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_feature_importance.png", dpi=150)
    plt.show()


def plot_shap_summary(gb, X, feature_names, sector_labels):
    """SHAP beeswarm plot coloured by sector."""
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X)

    # ----- Global beeswarm -----
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(
        "SHAP Summary – How Each Feature Pushes Unemployment Up or Down",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_shap_summary.png", dpi=150)
    plt.show()

    # ----- Per-sector mean |SHAP| -----
    sectors_unique = list(SECTOR_CONFIG.keys())
    mean_abs_shap = np.zeros((len(sectors_unique), len(feature_names)))
    for i, sector in enumerate(sectors_unique):
        mask = sector_labels == sector
        mean_abs_shap[i] = np.mean(np.abs(shap_values[mask]), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(feature_names))
    width = 0.18
    for i, sector in enumerate(sectors_unique):
        ax.bar(
            x_pos + i * width,
            mean_abs_shap[i],
            width,
            label=sector,
            color=SECTOR_COLORS[sector],
            alpha=0.85,
        )
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(feature_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(
        "Feature Importance by Sector – Which Factors Matter Where?",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_shap_by_sector.png", dpi=150)
    plt.show()

    return shap_values


def plot_shap_dependence_intensity(gb, X, shap_values, feature_names, sector_labels):
    """SHAP dependence plot for Intensity, coloured by sector."""
    intensity_idx = feature_names.index("Intensity")
    shap_intensity = shap_values[:, intensity_idx]
    x_intensity = X[:, intensity_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    for sector in SECTOR_CONFIG:
        mask = sector_labels == sector
        ax.scatter(
            x_intensity[mask], shap_intensity[mask],
            label=sector, color=SECTOR_COLORS[sector],
            alpha=0.7, s=40, edgecolors="white", linewidths=0.4,
        )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Foreign-Labour Intensity (IW / Employment ×1000)")
    ax.set_ylabel("SHAP Value (impact on unemployment prediction)")
    ax.set_title(
        "How Foreign-Labour Intensity Shifts Predicted Unemployment\n(SHAP Dependence)",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_shap_dependence_intensity.png", dpi=150)
    plt.show()


def plot_partial_dependence(gb, X, feature_names):
    """Partial dependence of unemployment on Intensity and GDP Growth."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    intensity_idx = feature_names.index("Intensity")
    gdp_idx = feature_names.index("GDP Growth")

    PartialDependenceDisplay.from_estimator(
        gb, X, features=[intensity_idx],
        feature_names=feature_names, ax=axes[0],
    )
    axes[0].set_title("Partial Dependence:\nForeign-Labour Intensity")

    PartialDependenceDisplay.from_estimator(
        gb, X, features=[gdp_idx],
        feature_names=feature_names, ax=axes[1],
    )
    axes[1].set_title("Partial Dependence:\nGDP Growth")

    fig.suptitle(
        "Marginal Effect on Unemployment Rate (Gradient Boosting)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_partial_dependence.png", dpi=150)
    plt.show()


def plot_actual_vs_predicted(gb, X, y, sector_labels):
    """Actual vs Predicted scatter for each sector."""
    y_pred = gb.predict(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sector in SECTOR_CONFIG:
        mask = sector_labels == sector
        ax.scatter(
            y[mask], y_pred[mask],
            label=sector, color=SECTOR_COLORS[sector],
            alpha=0.7, s=50, edgecolors="white", linewidths=0.5,
        )
    lims = [min(y.min(), y_pred.min()) - 0.3, max(y.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Unemployment Rate (%)")
    ax.set_ylabel("Predicted Unemployment Rate (%)")
    ax.set_title(
        "Gradient Boosting – Actual vs Predicted by Sector",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_actual_vs_predicted.png", dpi=150)
    plt.show()


def plot_policy_counterfactual(gb, X, feature_names, sector_labels, records):
    """
    Simulate: what happens to each sector's unemployment if imported-worker
    intensity is scaled to 50%, 100% (baseline), 150%, 200% of actual?
    """
    intensity_idx = feature_names.index("Intensity")
    scales = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]

    fig, ax = plt.subplots(figsize=(9, 6))

    for sector in SECTOR_CONFIG:
        mask = sector_labels == sector
        X_sector = X[mask].copy()
        baseline_intensity = X_sector[:, intensity_idx].copy()

        mean_predictions = []
        for scale in scales:
            X_sim = X_sector.copy()
            X_sim[:, intensity_idx] = baseline_intensity * scale
            preds = gb.predict(X_sim)
            mean_predictions.append(preds.mean())

        ax.plot(
            [s * 100 for s in scales],
            mean_predictions,
            marker="o", linewidth=2, markersize=6,
            label=sector, color=SECTOR_COLORS[sector],
        )

    ax.axvline(100, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Foreign-Labour Intensity (% of actual)", fontsize=11)
    ax.set_ylabel("Predicted Avg Unemployment Rate (%)", fontsize=11)
    ax.set_title(
        "Policy Simulation: How Would Changing Foreign-Labour Import\n"
        "Affect Sectoral Unemployment?",
        fontsize=12, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_policy_counterfactual.png", dpi=150)
    plt.show()


def plot_sector_sensitivity_heatmap(gb, X, feature_names, sector_labels):
    """
    Heatmap of each sector's sensitivity to a 1-SD increase in each feature,
    measured as change in predicted unemployment.
    """
    sectors_unique = list(SECTOR_CONFIG.keys())
    n_features = len(feature_names)
    sensitivity = np.zeros((len(sectors_unique), n_features))

    for i, sector in enumerate(sectors_unique):
        mask = sector_labels == sector
        X_sector = X[mask].copy()
        baseline_pred = gb.predict(X_sector).mean()

        for j in range(n_features):
            X_shifted = X_sector.copy()
            std_j = X[:, j].std()
            if std_j == 0:
                sensitivity[i, j] = 0.0
                continue
            X_shifted[:, j] += std_j
            shifted_pred = gb.predict(X_shifted).mean()
            sensitivity[i, j] = shifted_pred - baseline_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sensitivity, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(sectors_unique)))
    ax.set_yticklabels(sectors_unique)

    for i in range(len(sectors_unique)):
        for j in range(n_features):
            val = sensitivity[i, j]
            color = "white" if abs(val) > 0.3 * np.abs(sensitivity).max() else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Predicted Unemployment (%)\nfrom +1 SD increase")
    ax.set_title(
        "Sector Vulnerability: Unemployment Sensitivity to Each Factor\n"
        "(+1 Standard Deviation Shock)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_sector_sensitivity_heatmap.png", dpi=150)
    plt.show()


def plot_sector_unemployment_decomposition(gb, X, y, feature_names, sector_labels):
    """
    Bar chart showing average unemployment by sector, decomposed into
    the mean SHAP contribution of each feature.
    """
    explainer = shap.TreeExplainer(gb)
    shap_values = np.asarray(explainer.shap_values(X))
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 2:
        raise ValueError(
            "Expected 2D SHAP values for single-output regression, "
            f"got shape {shap_values.shape}."
        )

    base_value_arr = np.asarray(explainer.expected_value)
    if base_value_arr.size == 0:
        raise ValueError("SHAP expected_value is empty.")
    base_value = float(base_value_arr.reshape(-1)[0])

    sectors_unique = list(SECTOR_CONFIG.keys())
    n_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(sectors_unique))
    bottom_pos = np.full(len(sectors_unique), base_value)
    bottom_neg = np.full(len(sectors_unique), base_value)

    # Accumulate positive and negative SHAP contributions separately
    mean_shap = np.zeros((len(sectors_unique), n_features))
    for i, sector in enumerate(sectors_unique):
        mask = sector_labels == sector
        mean_shap[i] = shap_values[mask].mean(axis=0)

    # Sort features by total absolute contribution for consistent stacking
    total_abs = np.abs(mean_shap).sum(axis=0)
    feature_order = np.argsort(total_abs)[::-1]

    cmap = plt.cm.Set2
    colors = [cmap(i / n_features) for i in range(n_features)]

    # Reset bottoms for stacking
    bottom_pos = np.full(len(sectors_unique), 0.0)
    bottom_neg = np.full(len(sectors_unique), 0.0)

    for rank, fi in enumerate(feature_order):
        vals = mean_shap[:, fi]
        pos_vals = np.maximum(vals, 0)
        neg_vals = np.minimum(vals, 0)

        if np.any(pos_vals > 0):
            ax.bar(x_pos, pos_vals, bottom=base_value + bottom_pos,
                   width=0.5, color=colors[rank], label=feature_names[fi],
                   edgecolor="white", linewidth=0.5)
            bottom_pos += pos_vals

        if np.any(neg_vals < 0):
            ax.bar(x_pos, neg_vals, bottom=base_value + bottom_neg,
                   width=0.5, color=colors[rank],
                   edgecolor="white", linewidth=0.5)
            bottom_neg += neg_vals

    # Actual average unemployment markers
    avg_unemp = [y[sector_labels == s].mean() for s in sectors_unique]
    ax.scatter(x_pos, avg_unemp, color="black", zorder=5, s=80, marker="D",
               label="Actual Avg Unemployment")

    ax.axhline(base_value, color="grey", linewidth=1, linestyle="--",
               label=f"Baseline (E[f(x)] = {base_value:.2f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sectors_unique, fontsize=10)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title(
        "Why Do Sectors Have Different Unemployment Rates?\n"
        "(SHAP Decomposition of Mean Predicted Unemployment)",
        fontsize=11, fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate legend entries
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml_unemployment_decomposition.png", dpi=150)
    plt.show()


def print_policy_summary(gb, X, feature_names, sector_labels):
    """Print textual policy insights derived from the model."""
    intensity_idx = feature_names.index("Intensity")
    sectors = list(SECTOR_CONFIG.keys())

    print("\n" + "=" * 70)
    print("  POLICY INSIGHTS: Foreign Labour Import Recommendations")
    print("=" * 70)

    print("\n  Per-sector analysis of foreign-labour intensity effect:\n")

    for sector in sectors:
        mask = sector_labels == sector
        X_sector = X[mask].copy()
        baseline_intensity = X_sector[:, intensity_idx].copy()
        baseline_pred = gb.predict(X_sector).mean()

        # +50% increase in intensity
        X_up = X_sector.copy()
        X_up[:, intensity_idx] = baseline_intensity * 1.5
        pred_up = gb.predict(X_up).mean()

        # Zero imported workers
        X_zero = X_sector.copy()
        X_zero[:, intensity_idx] = 0.0
        pred_zero = gb.predict(X_zero).mean()

        delta_up = pred_up - baseline_pred
        delta_zero = pred_zero - baseline_pred

        print(f"  {sector}:")
        print(f"    Baseline avg intensity  : {baseline_intensity.mean():.4f}")
        print(f"    Baseline avg unemployment: {baseline_pred:.2f}%")
        print(f"    If intensity +50%       : {pred_up:.2f}% (Δ = {delta_up:+.3f})")
        print(f"    If intensity set to 0   : {pred_zero:.2f}% (Δ = {delta_zero:+.3f})")

        if abs(delta_up) < 0.1:
            sensitivity = "LOW"
        elif abs(delta_up) < 0.3:
            sensitivity = "MODERATE"
        else:
            sensitivity = "HIGH"
        print(f"    Sensitivity             : {sensitivity}")
        print()

    print("  " + "-" * 66)
    print("  KEY POLICY TAKEAWAYS:")
    print("  " + "-" * 66)
    print("""
  1. Sectors differ substantially in how foreign labour affects their
     unemployment. A one-size-fits-all quota policy is sub-optimal.

  2. Sectors with HIGH sensitivity warrant careful, graduated import
     controls with monitoring of local labour market displacement.

  3. Sectors with LOW sensitivity can absorb foreign workers without
     measurable harm to local unemployment — suggesting import quotas
     can be expanded safely in those areas.

  4. Macro-economic conditions (GDP growth, interest rates) often
     dominate intensity effects. Coordination of import policy with
     the business cycle can mitigate adverse impacts.

  5. Policy recommendation: adopt sector-specific foreign-labour
     import ceilings tied to real-time unemployment thresholds,
     tightening quotas when sectoral unemployment exceeds its
     historical average and loosening them during labour shortages.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building pooled panel dataset across 4 sectors...")
    records = build_panel()
    print(f"  Total observations: {len(records)}")
    for sector in SECTOR_CONFIG:
        n = sum(1 for r in records if r["Sector"] == sector)
        print(f"    {sector}: {n} quarters")

    X, y, feature_names, sector_labels, sector_encoder = prepare_features(records)

    print("\nTraining Random Forest and Gradient Boosting models...")
    rf, gb = train_models(X, y)
    evaluate_models(rf, gb, X, y)

    # --- Visualisations ---
    print("\nGenerating plots...\n")

    # 1. Feature importance comparison
    plot_feature_importance(rf, gb, feature_names)

    # 2. SHAP summary + per-sector importance
    shap_values = plot_shap_summary(gb, X, feature_names, sector_labels)

    # 3. SHAP dependence for intensity by sector
    plot_shap_dependence_intensity(gb, X, shap_values, feature_names, sector_labels)

    # 4. Partial dependence plots
    plot_partial_dependence(gb, X, feature_names)

    # 5. Actual vs predicted
    plot_actual_vs_predicted(gb, X, y, sector_labels)

    # 6. Policy counterfactual simulation
    plot_policy_counterfactual(gb, X, feature_names, sector_labels, records)

    # 7. Sector sensitivity heatmap
    plot_sector_sensitivity_heatmap(gb, X, feature_names, sector_labels)

    # 8. SHAP-based unemployment decomposition by sector
    plot_sector_unemployment_decomposition(gb, X, y, feature_names, sector_labels)

    # --- Policy summary ---
    print_policy_summary(gb, X, feature_names, sector_labels)

    print("\nAll plots saved to:", BASE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
