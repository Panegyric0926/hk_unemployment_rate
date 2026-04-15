"""
OLS Regression Model 2: Retail sector

    Retail Unemployment Rate_{s,t} = beta0
        + beta1 * ImportedWorkers_{s,t}
        + beta2 * GDP Growth_t
        + beta3 * Inflation Rate_t
        + beta4 * Interest Rate (3M)_t
        + beta5 * Retail Index_{s,t}
        + beta6 * Average Reception Index_{s,t}
        + epsilon_{s,t}

Notes:
    - Common macro shocks come from the model_2 folder.
    - Imported workers are annual approvals divided evenly across quarters.
    - Retail index uses the quarterly average of the monthly Value index of
      retail sales.
    - Reception index uses the average of Retail, Accommodation services,
      and Food services for each quarter.
    - No imputation is needed because all required years are available.
"""

from pathlib import Path
import csv

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent

MONTH_TO_Q = {
    "Jan": 1,
    "Feb": 1,
    "Mar": 1,
    "Apr": 2,
    "May": 2,
    "Jun": 2,
    "Jul": 3,
    "Aug": 3,
    "Sep": 3,
    "Oct": 4,
    "Nov": 4,
    "Dec": 4,
}


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

    quarter = {1: 1, 4: 2, 7: 3, 10: 4}.get(month)
    if quarter is None:
        return None
    return year, quarter


def parse_year_quarter_pair(year_value, quarter_value):
    year_text = (year_value or "").strip()
    if not year_text.isdigit():
        return None
    quarter_tokens = (quarter_value or "").strip().split()
    if not quarter_tokens:
        return None
    quarter_text = quarter_tokens[0]
    if not quarter_text.startswith("Q"):
        return None
    try:
        return int(year_text), int(quarter_text[1:])
    except ValueError:
        return None


def read_macro_series():
    gdp_growth = {}
    inflation = {}
    interest_rate = {}

    with (MODEL_DIR / "gdp.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_yyyyqq((row.get("Time") or "").strip())
            value = (row.get("GDP Year-on-year Change") or "").strip()
            if key is None or not value:
                continue
            try:
                gdp_growth[key] = float(value.lstrip("+"))
            except ValueError:
                continue

    with (MODEL_DIR / "inflation_rate.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_yyyyqq((row.get("Quarter") or "").strip())
            value = (row.get("Quarterly Inflation Rate (%)") or "").strip()
            if key is None or not value:
                continue
            try:
                inflation[key] = float(value)
            except ValueError:
                continue

    with (MODEL_DIR / "interest_rate.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_yyyyqq((row.get("Time") or "").strip())
            value = (row.get("3 months") or "").strip()
            if key is None or not value:
                continue
            try:
                interest_rate[key] = float(value)
            except ValueError:
                continue

    return gdp_growth, inflation, interest_rate


def read_retail_series():
    unemployment = {}
    imported_workers = {}
    retail_index = {}
    reception_index = {}

    with (BASE_DIR / "unemployment_rate.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_value = (row.get("Time") or "").strip()
            unemployment_value = (row.get("Unemployment Rate") or "").strip()
            if not time_value or not unemployment_value:
                continue
            key = parse_month_range(time_value)
            if key is None:
                continue
            try:
                unemployment[key] = float(unemployment_value)
            except ValueError:
                continue

    with (BASE_DIR / "imported_workers.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            value = (row.get("Number of Imported Workers Approved") or "").strip()
            if not year_str.isdigit() or not value:
                continue
            try:
                quarterly_workers = float(value) / 4.0
            except ValueError:
                continue
            year = int(year_str)
            for quarter in range(1, 5):
                imported_workers[(year, quarter)] = quarterly_workers

    monthly_retail_index = {}
    with (BASE_DIR / "retail_index.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            month_str = (row.get("Month") or "").strip()
            value = (row.get("Value index of retail sales") or "").strip()
            if not year_str.isdigit() or month_str not in MONTH_TO_Q or not value:
                continue
            key = (int(year_str), MONTH_TO_Q[month_str])
            monthly_retail_index.setdefault(key, []).append(float(value))

    for key, values in monthly_retail_index.items():
        retail_index[key] = sum(values) / len(values)

    with (BASE_DIR / "reception_index.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_year_quarter_pair(row.get("Year"), row.get("Quarter"))
            if key is None:
                continue
            retail_value = (row.get("Retail") or "").strip()
            accommodation_value = (row.get("Accommodation services") or "").strip()
            food_value = (row.get("Food services") or "").strip()
            if not retail_value or not accommodation_value or not food_value:
                continue
            try:
                reception_index[key] = (
                    float(retail_value)
                    + float(accommodation_value)
                    + float(food_value)
                ) / 3.0
            except ValueError:
                continue

    return unemployment, imported_workers, retail_index, reception_index


def build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    retail_index,
    reception_index,
):
    all_keys = sorted(
        set(unemployment)
        & set(imported_workers)
        & set(gdp_growth)
        & set(inflation)
        & set(interest_rate)
        & set(retail_index)
        & set(reception_index)
    )

    dataset = []
    for year, quarter in all_keys:
        dataset.append({
            "Quarter": f"{year} Q{quarter}",
            "Unemployment Rate": unemployment[(year, quarter)],
            "Imported Workers": imported_workers[(year, quarter)],
            "GDP Growth": gdp_growth[(year, quarter)],
            "Inflation Rate": inflation[(year, quarter)],
            "Interest Rate 3M": interest_rate[(year, quarter)],
            "Retail Index": retail_index[(year, quarter)],
            "Reception Index Avg": reception_index[(year, quarter)],
        })

    return dataset


def fit_ols(dataset):
    y_values = np.array([row["Unemployment Rate"] for row in dataset])
    x_values = np.column_stack([
        [row["Imported Workers"] for row in dataset],
        [row["GDP Growth"] for row in dataset],
        [row["Inflation Rate"] for row in dataset],
        [row["Interest Rate 3M"] for row in dataset],
        [row["Retail Index"] for row in dataset],
        [row["Reception Index Avg"] for row in dataset],
    ])
    x_values = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_values)
    return model.fit()


def print_regression_report(dataset, results):
    print("Dataset used in retail regression:")
    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'ImpWork':>10} {'GDPGrowth':>10} "
        f"{'Inflation':>10} {'IntRate3M':>10} {'RetailIdx':>10} {'ReceptionAvg':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in dataset:
        print(
            f"{row['Quarter']:<12} "
            f"{row['Unemployment Rate']:>10.2f} "
            f"{row['Imported Workers']:>10.2f} "
            f"{row['GDP Growth']:>10.1f} "
            f"{row['Inflation Rate']:>10.2f} "
            f"{row['Interest Rate 3M']:>10.2f} "
            f"{row['Retail Index']:>10.2f} "
            f"{row['Reception Index Avg']:>13.2f}"
        )

    print(
        f"\nObservations: {len(dataset)}  "
        f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})"
    )

    print("\n" + "=" * 80)
    print("Retail OLS Regression Results")
    print("=" * 80)
    print(
        results.summary(
            yname="Retail Unemployment Rate",
            xname=[
                "const (beta0)",
                "Imported Workers (beta1)",
                "GDP Growth (beta2)",
                "Inflation Rate (beta3)",
                "Interest Rate 3M (beta4)",
                "Retail Index (beta5)",
                "Reception Index Avg (beta6)",
            ],
        )
    )

    print("\nCoefficient Estimates:")
    labels = [
        "beta0 (Intercept)",
        "beta1 (Imported Workers)",
        "beta2 (GDP Growth)",
        "beta3 (Inflation Rate)",
        "beta4 (Interest Rate 3M)",
        "beta5 (Retail Index)",
        "beta6 (Reception Index Avg)",
    ]
    for label, coef, pvalue in zip(labels, results.params, results.pvalues):
        significance = "***" if pvalue < 0.01 else ("**" if pvalue < 0.05 else ("*" if pvalue < 0.10 else ""))
        print(f"  {label:<33}  coef = {coef:>12.6f}   p = {pvalue:.4f}  {significance}")

    print(f"\n  R^2         = {results.rsquared:.4f}")
    print(f"  Adj. R^2    = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.4f}  (p = {results.f_pvalue:.4f})")


(
    gdp_growth,
    inflation,
    interest_rate,
) = read_macro_series()

(
    unemployment,
    imported_workers,
    retail_index,
    reception_index,
) = read_retail_series()

dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    retail_index,
    reception_index,
)
results = fit_ols(dataset)
print_regression_report(dataset, results)

# QQ-plot of residuals
fig = qqplot(results.resid, line="s", alpha=0.6)
fig.suptitle("Retail (Model 2) – Normal QQ-Plot of Residuals", fontsize=11)
plt.tight_layout()
plt.show()