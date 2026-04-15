"""
Cross-Sector Heterogeneity Model with Macro Controls (Model 3 Macro)

Instead of using Model 2 residuals, this model directly includes macro
variables as controls alongside foreign-labour intensity:

    UnemploymentRate_{s,t} = gamma0
        + gamma1 * Intensity_{s,t}
        + gamma2 * GDP Growth_t
        + gamma3 * Inflation Rate_t
        + gamma4 * Interest Rate (3M)_t
        + epsilon_{s,t}

gamma1 now captures the intensity effect after controlling for the macro
cycle, without needing sector-specific controls from Model 2.

Notes:
    - Imported workers are annual approvals divided evenly across four quarters.
    - Total sector employment comes from C&SD (all_sector_labour.csv, unit: 1 000 persons).
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
# Data readers
# ---------------------------------------------------------------------------

def read_macro_series():
    gdp_growth = {}
    inflation = {}
    interest_rate = {}

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
# Per-sector analysis
# ---------------------------------------------------------------------------

def analyse_sector(sector_name, config, all_employment, gdp, infl, ir):
    folder = config["folder"]
    emp_industry = config["employment_industry"]
    unemp_filter = config["unemployment_filter"]

    imported = read_imported_workers(folder)
    unemployment = read_unemployment(folder, unemp_filter)
    employment = all_employment.get(emp_industry, {})

    common_keys = sorted(
        set(imported) & set(unemployment) & set(employment)
        & set(gdp) & set(infl) & set(ir)
    )

    if len(common_keys) < 6:
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
            "GDP Growth": gdp[(year, quarter)],
            "Inflation Rate": infl[(year, quarter)],
            "Interest Rate 3M": ir[(year, quarter)],
        })

    y = np.array([r["Unemployment Rate"] for r in dataset])
    X = np.column_stack([
        [r["Intensity"] for r in dataset],
        [r["GDP Growth"] for r in dataset],
        [r["Inflation Rate"] for r in dataset],
        [r["Interest Rate 3M"] for r in dataset],
    ])
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()

    return dataset, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sector_report(sector_name, dataset, results):
    print("\n" + "=" * 80)
    print(f"  {sector_name}  –  Intensity + Macro Controls → Unemployment")
    print("=" * 80)

    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'Intensity':>12} "
        f"{'GDPGrowth':>10} {'Inflation':>10} {'IntRate3M':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in dataset:
        print(
            f"{r['Quarter']:<12} "
            f"{r['Unemployment Rate']:>10.2f} "
            f"{r['Intensity']:>12.6f} "
            f"{r['GDP Growth']:>10.1f} "
            f"{r['Inflation Rate']:>10.2f} "
            f"{r['Interest Rate 3M']:>10.2f}"
        )

    print(f"\nObservations: {len(dataset)}  "
          f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})")

    print(
        results.summary(
            yname="Unemployment Rate",
            xname=[
                "gamma0 (Intercept)",
                "gamma1 (Intensity)",
                "gamma2 (GDP Growth)",
                "gamma3 (Inflation Rate)",
                "gamma4 (Interest Rate 3M)",
            ],
        )
    )

    labels = [
        "gamma0 (Intercept)",
        "gamma1 (Intensity)",
        "gamma2 (GDP Growth)",
        "gamma3 (Inflation Rate)",
        "gamma4 (Interest Rate 3M)",
    ]
    print("\nCoefficient Estimates:")
    for label, coef, pvalue in zip(labels, results.params, results.pvalues):
        sig = "***" if pvalue < 0.01 else ("**" if pvalue < 0.05 else ("*" if pvalue < 0.10 else ""))
        print(f"  {label:<30}  coef = {coef:>12.6f}   p = {pvalue:.4f}  {sig}")

    print(f"\n  R^2         = {results.rsquared:.4f}")
    print(f"  Adj. R^2    = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.4f}  (p = {results.f_pvalue:.4f})")


def print_comparison(sector_results):
    print("\n" + "=" * 80)
    print("  Cross-Sector Comparison Summary (with Macro Controls)")
    print("=" * 80)

    header = (
        f"{'Sector':<18} {'gamma1':>12} {'p-value':>10} {'Sig':>5} "
        f"{'R^2':>8} {'Adj.R^2':>8} {'N':>5}"
    )
    print(header)
    print("-" * len(header))

    ranked = sorted(sector_results, key=lambda s: abs(s["gamma1"]), reverse=True)

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
    print("  gamma1 is the intensity effect after controlling for GDP growth,")
    print("  inflation, and interest rate. A positive gamma1 means higher")
    print("  foreign-labour intensity is associated with higher unemployment")
    print("  once the macro cycle is accounted for.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gdp, infl, ir = read_macro_series()
    all_employment = read_total_employment()

    sector_results = []
    plot_data = []   # for QQ and scatter plots
    for sector_name, config in SECTOR_CONFIG.items():
        dataset, results = analyse_sector(
            sector_name, config, all_employment, gdp, infl, ir
        )
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
    fig_qq.suptitle(
        "Model 3 (Macro-Controlled) – Normal QQ-Plots of Residuals", fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # ── Scatter plots: Intensity vs Unemployment Rate with regression line ────
    fig_sc, axes_sc = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_sc = axes_sc.flatten()
    for i, d in enumerate(plot_data):
        x = np.array([r["Intensity"] for r in d["dataset"]])
        y = np.array([r["Unemployment Rate"] for r in d["dataset"]])
        gamma0, gamma1 = d["results"].params[0], d["results"].params[1]
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
        "Model 3 (Macro-Controlled) – Intensity vs Unemployment Rate by Sector",
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
    ax_comb.set_title("Model 3 (Macro-Controlled) – All Sectors: Intensity vs Unemployment Rate")
    ax_comb.legend(fontsize=9)
    ax_comb.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()