"""
Cross-Sector Heterogeneity Comparison Model (Model 3)

Step 1 – Compute foreign-labour intensity for each sector:

    Intensity_{s,t} = ImportedWorkers_{s,t} / TotalSectorEmployment_{s,t}

Step 2 – Simple regression of impact on intensity:

    ImpactStrength_{s,t} = gamma0 + gamma1 * Intensity_{s,t} + epsilon_{s,t}

    where ImpactStrength is measured by the sector unemployment rate.

Step 3 – Compare gamma1 across sectors to reveal cross-sector heterogeneity.

Notes:
    - Imported workers are annual approvals divided evenly across four quarters.
    - Total sector employment comes from C&SD (all_sector_labour.csv, unit: 1 000 persons).
    - Intensity is expressed as imported workers per 1 000 employed persons.
    - Community unemployment uses "Public administration, social and personal services".
    - Retail unemployment uses "Retail, accommodation and food services".
"""

from pathlib import Path
import csv

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

BASE_DIR = Path(__file__).resolve().parent
MODEL2_DIR = BASE_DIR.parent / "model_2"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}

# Mapping from sector label to the industry name in all_sector_labour.csv
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


# ---------------------------------------------------------------------------
# Parsing helpers (same convention as model_2 scripts)
# ---------------------------------------------------------------------------

def parse_month_range(value):
    """Parse '1/2017 - 3/2017' into (year, quarter)."""
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
# Data readers
# ---------------------------------------------------------------------------

def read_total_employment():
    """Read total sector employment from all_sector_labour.csv.

    Returns dict: {industry_name: {(year, quarter): employment_in_thousands}}
    """
    employment = {}
    with (BASE_DIR / "all_sector_labour.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            q_str = (row.get("Quarter") or "").strip()
            industry = (row.get("Industry of main employment") or "").strip()
            value = (row.get("Overall(1000 persons)") or "").strip()
            if not year_str or not q_str or not industry or not value:
                continue
            try:
                year = int(year_str)
                quarter = int(q_str[1])  # "Q1" -> 1
                emp = float(value)
            except (ValueError, IndexError):
                continue
            employment.setdefault(industry, {})[(year, quarter)] = emp
    return employment


def read_imported_workers(sector_folder):
    """Read annual imported workers and spread evenly across quarters.

    Returns dict: {(year, quarter): quarterly_workers}
    """
    imported_workers = {}
    path = MODEL2_DIR / sector_folder / "imported_workers.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            value = (row.get("Number of Imported Workers Approved") or "").strip()
            if not year_str.isdigit() or not value:
                continue
            try:
                quarterly = float(value) / 4.0
            except ValueError:
                continue
            year = int(year_str)
            for q in range(1, 5):
                imported_workers[(year, q)] = quarterly
    return imported_workers


def read_unemployment(sector_folder, industry_filter=None):
    """Read quarterly unemployment rate for a sector.

    If *industry_filter* is given (community sector), only rows whose
    'Detailed previous industry' matches are kept.

    Returns dict: {(year, quarter): unemployment_rate}
    """
    unemployment = {}
    path = MODEL2_DIR / sector_folder / "unemployment_rate.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_value = (row.get("Time") or "").strip()
            unemp_value = (row.get("Unemployment Rate") or "").strip()
            if not time_value or not unemp_value:
                continue
            if industry_filter is not None:
                industry = (row.get("Detailed previous industry") or "").strip()
                if industry != industry_filter:
                    continue
            key = parse_month_range(time_value)
            if key is None:
                continue
            try:
                unemployment[key] = float(unemp_value)
            except ValueError:
                continue
    return unemployment


# ---------------------------------------------------------------------------
# Per-sector analysis
# ---------------------------------------------------------------------------

def analyse_sector(sector_name, config, all_employment):
    """Compute intensity and run simple OLS for one sector.

    Returns (dataset, ols_results) or (None, None) if data is insufficient.
    """
    folder = config["folder"]
    emp_industry = config["employment_industry"]
    unemp_filter = config["unemployment_filter"]

    imported = read_imported_workers(folder)
    unemployment = read_unemployment(folder, unemp_filter)
    employment = all_employment.get(emp_industry, {})

    common_keys = sorted(
        set(imported) & set(unemployment) & set(employment)
    )

    if len(common_keys) < 3:
        print(f"\n[{sector_name}] Insufficient overlapping observations "
              f"({len(common_keys)}). Skipping.")
        return None, None

    dataset = []
    for year, quarter in common_keys:
        intensity = imported[(year, quarter)] / employment[(year, quarter)]
        dataset.append({
            "Quarter": f"{year} Q{quarter}",
            "Unemployment Rate": unemployment[(year, quarter)],
            "Imported Workers": imported[(year, quarter)],
            "Total Employment (1000)": employment[(year, quarter)],
            "Intensity": intensity,
        })

    y = np.array([r["Unemployment Rate"] for r in dataset])
    x = sm.add_constant(np.array([r["Intensity"] for r in dataset]))
    results = sm.OLS(y, x).fit()

    return dataset, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sector_report(sector_name, dataset, results):
    print("\n" + "=" * 80)
    print(f"  {sector_name}  –  Intensity → Unemployment Simple Regression")
    print("=" * 80)

    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'ImpWork':>10} "
        f"{'Employ(k)':>10} {'Intensity':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in dataset:
        print(
            f"{r['Quarter']:<12} "
            f"{r['Unemployment Rate']:>10.2f} "
            f"{r['Imported Workers']:>10.2f} "
            f"{r['Total Employment (1000)']:>10.1f} "
            f"{r['Intensity']:>12.6f}"
        )

    print(f"\nObservations: {len(dataset)}  "
          f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})")

    print(
        results.summary(
            yname="Unemployment Rate",
            xname=["gamma0 (Intercept)", "gamma1 (Intensity)"],
        )
    )

    gamma0, gamma1 = results.params
    p0, p1 = results.pvalues
    sig = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else ""))
    print(f"\n  gamma0 (Intercept)   coef = {gamma0:>12.6f}   p = {p0:.4f}")
    print(f"  gamma1 (Intensity)   coef = {gamma1:>12.6f}   p = {p1:.4f}  {sig}")
    print(f"  R^2       = {results.rsquared:.4f}")
    print(f"  Adj. R^2  = {results.rsquared_adj:.4f}")


def print_comparison(sector_results):
    print("\n" + "=" * 80)
    print("  Cross-Sector Comparison Summary")
    print("=" * 80)

    header = (
        f"{'Sector':<18} {'gamma1':>12} {'p-value':>10} {'Sig':>5} "
        f"{'R^2':>8} {'Adj.R^2':>8} {'N':>5}"
    )
    print(header)
    print("-" * len(header))

    ranked = sorted(
        sector_results,
        key=lambda s: abs(s["gamma1"]),
        reverse=True,
    )

    for s in ranked:
        sig = (
            "***" if s["p"] < 0.01
            else ("**" if s["p"] < 0.05
                  else ("*" if s["p"] < 0.10 else ""))
        )
        print(
            f"{s['sector']:<18} {s['gamma1']:>12.6f} {s['p']:>10.4f} {sig:>5} "
            f"{s['r2']:>8.4f} {s['adj_r2']:>8.4f} {s['n']:>5}"
        )

    print("\nInterpretation:")
    print("  Sectors are ranked by |gamma1| (absolute coefficient magnitude).")
    print("  A larger positive gamma1 implies that higher foreign-labour")
    print("  intensity is associated with a stronger rise in unemployment,")
    print("  consistent with increased labour-supply competition.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_employment = read_total_employment()

    sector_results = []
    plot_data = []   # for QQ and scatter plots
    for sector_name, config in SECTOR_CONFIG.items():
        dataset, results = analyse_sector(sector_name, config, all_employment)
        if dataset is None:
            continue
        print_sector_report(sector_name, dataset, results)
        sector_results.append({
            "sector": sector_name,
            "gamma1": results.params[1],
            "p": results.pvalues[1],
            "r2": results.rsquared,
            "adj_r2": results.rsquared_adj,
            "n": int(results.nobs),
        })
        plot_data.append({
            "sector": sector_name,
            "dataset": dataset,
            "results": results,
        })

    if sector_results:
        print_comparison(sector_results)

    if not plot_data:
        return

    sectors = [d["sector"] for d in plot_data]
    n = len(sectors)
    ncols = 2
    nrows = (n + 1) // ncols

    # ── QQ-plots ──────────────────────────────────────────────────────────────
    fig_qq, axes_qq = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_qq = axes_qq.flatten()
    for i, d in enumerate(plot_data):
        qqplot(d["results"].resid, line="s", alpha=0.6, ax=axes_qq[i])
        axes_qq[i].set_title(d["sector"])
    for j in range(i + 1, len(axes_qq)):
        axes_qq[j].set_visible(False)
    fig_qq.suptitle("Model 3 (Baseline) – Normal QQ-Plots of Residuals", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── Scatter plots: Intensity vs Unemployment Rate with regression line ────
    fig_sc, axes_sc = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_sc = axes_sc.flatten()
    for i, d in enumerate(plot_data):
        x = np.array([r["Intensity"] for r in d["dataset"]])
        y = np.array([r["Unemployment Rate"] for r in d["dataset"]])
        gamma0, gamma1 = d["results"].params
        p1 = d["results"].pvalues[1]
        sig = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else ""))
        ax = axes_sc[i]
        ax.scatter(x, y, alpha=0.7, s=30)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, gamma0 + gamma1 * x_line, color="red", linewidth=1.5)
        ax.set_xlabel("Intensity (IW / Employment)")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.set_title(d["sector"])
        ax.annotate(
            f"γ₁ = {gamma1:.4f}{sig}\np = {p1:.3f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
    for j in range(i + 1, len(axes_sc)):
        axes_sc[j].set_visible(False)
    fig_sc.suptitle(
        "Model 3 (Baseline) – Intensity vs Unemployment Rate by Sector",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    # ── Combined: all 4 sectors on one plot ───────────────────────────────────
    COMBINED_COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]
    fig_comb, ax_comb = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(plot_data):
        x = np.array([r["Intensity"] for r in d["dataset"]])
        y = np.array([r["Unemployment Rate"] for r in d["dataset"]])
        gamma0, gamma1 = d["results"].params[0], d["results"].params[1]
        p1 = d["results"].pvalues[1]
        sig = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else ""))
        color = COMBINED_COLORS[i % len(COMBINED_COLORS)]
        ax_comb.scatter(x, y, alpha=0.45, s=25, color=color)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax_comb.plot(
            x_line, gamma0 + gamma1 * x_line, color=color, linewidth=2,
            label=f"{d['sector']}  γ₁ = {gamma1:.4f}{sig}",
        )
    ax_comb.set_xlabel("Intensity (IW / Employment)")
    ax_comb.set_ylabel("Unemployment Rate (%)")
    ax_comb.set_title("Model 3 (Baseline) – All Sectors: Intensity vs Unemployment Rate")
    ax_comb.legend(fontsize=9)
    ax_comb.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
