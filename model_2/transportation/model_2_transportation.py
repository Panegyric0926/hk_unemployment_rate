"""
OLS Regression Model 2: Transportation sector

    Transportation Unemployment Rate_{s,t} = beta0
        + beta1 * ImportedWorkers_{s,t}
        + beta2 * GDP Growth_t
        + beta3 * Inflation Rate_t
        + beta4 * Interest Rate (3M)_t
        + beta5 * Air Cargo Output_{s,t}
        + beta6 * Total Tourists_{s,t}
        + epsilon_{s,t}

Notes:
    - Common macro shocks come from the model_2 folder.
    - Transportation controls come from the transportation folder.
    - Imported workers are annual approvals divided evenly across quarters.
    - Tourist control is Overnight visitors + Same-day visitors.
    - Annual transportation controls are divided evenly across quarters to align
      with the quarterly unemployment series.
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

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}


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


def read_transportation_series():
    unemployment = {}
    imported_workers = {}
    air_cargo_output = {}
    tourist_total = {}

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

    with (BASE_DIR / "air_cargo_output.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            value = (row.get("Air Cargo Throughput") or "").strip()
            if not year_str.isdigit() or not value:
                continue
            try:
                quarterly_output = float(value) / 4.0
            except ValueError:
                continue
            year = int(year_str)
            for quarter in range(1, 5):
                air_cargo_output[(year, quarter)] = quarterly_output

    with (BASE_DIR / "tourist.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            overnight_str = (row.get("Overnight visitors") or "").strip()
            same_day_str = (row.get("Same-day visitors") or "").strip()
            if not year_str.isdigit() or not overnight_str or not same_day_str:
                continue
            try:
                yearly_total = float(overnight_str) + float(same_day_str)
                quarterly_total = yearly_total / 4.0
            except ValueError:
                continue
            year = int(year_str)
            for quarter in range(1, 5):
                tourist_total[(year, quarter)] = quarterly_total

    return unemployment, imported_workers, air_cargo_output, tourist_total


def build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    air_cargo_output,
    tourist_total,
):
    all_keys = sorted(
        set(unemployment)
        & set(imported_workers)
        & set(gdp_growth)
        & set(inflation)
        & set(interest_rate)
        & set(air_cargo_output)
        & set(tourist_total)
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
            "Air Cargo Output": air_cargo_output[(year, quarter)],
            "Tourist Total": tourist_total[(year, quarter)],
        })

    return dataset


def fit_ols(dataset):
    y_values = np.array([row["Unemployment Rate"] for row in dataset])
    x_values = np.column_stack([
        [row["Imported Workers"] for row in dataset],
        [row["GDP Growth"] for row in dataset],
        [row["Inflation Rate"] for row in dataset],
        [row["Interest Rate 3M"] for row in dataset],
        [row["Air Cargo Output"] for row in dataset],
        [row["Tourist Total"] for row in dataset],
    ])
    x_values = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_values)
    return model.fit()


def print_regression_report(dataset, results):
    print("Dataset used in transportation regression:")
    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'ImpWork':>10} {'GDPGrowth':>10} "
        f"{'Inflation':>10} {'IntRate3M':>10} {'AirCargo':>10} {'Tourists':>13}"
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
            f"{row['Air Cargo Output']:>10.2f} "
            f"{row['Tourist Total']:>13.2f}"
        )

    print(
        f"\nObservations: {len(dataset)}  "
        f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})"
    )

    print("\n" + "=" * 80)
    print("Transportation OLS Regression Results")
    print("=" * 80)
    print(
        results.summary(
            yname="Transportation Unemployment Rate",
            xname=[
                "const (beta0)",
                "Imported Workers (beta1)",
                "GDP Growth (beta2)",
                "Inflation Rate (beta3)",
                "Interest Rate 3M (beta4)",
                "Air Cargo Output (beta5)",
                "Total Tourists (beta6)",
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
        "beta5 (Air Cargo Output)",
        "beta6 (Total Tourists)",
    ]
    for label, coef, pvalue in zip(labels, results.params, results.pvalues):
        significance = "***" if pvalue < 0.01 else ("**" if pvalue < 0.05 else ("*" if pvalue < 0.10 else ""))
        print(f"  {label:<31}  coef = {coef:>12.6f}   p = {pvalue:.4f}  {significance}")

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
    air_cargo_output,
    tourist_total,
) = read_transportation_series()

dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    air_cargo_output,
    tourist_total,
)
results = fit_ols(dataset)
print_regression_report(dataset, results)

# QQ-plot of residuals
fig = qqplot(results.resid, line="s", alpha=0.6)
fig.suptitle("Transportation (Model 2) – Normal QQ-Plot of Residuals", fontsize=11)
plt.tight_layout()
plt.show()