"""
OLS Regression Model 2: Construction sector

    Construction Unemployment Rate_{s,t} = beta0
        + beta1 * ImportedWorkersPer1KWorkers_{s,t}
        + beta2 * GDP Growth_t
        + beta3 * Inflation Rate_t
        + beta4 * Interest Rate (3M)_t
        + beta5 * Gross Value of Construction Works_{s,t}
        + beta6 * Compensation_{s,t}
        + beta7 * Actual Expenditure_{s,t}
        + epsilon_{s,t}

Notes:
    - Common macro shocks come from the model_2 folder.
    - Construction controls come from the construction folder.
    - Imported workers are annual approvals divided equally across quarters.
    - Imported workers are scaled per 1,000 construction workers using the
      yearly number of persons directly engaged from the compensation file.
        - Compensation is annual, so each year's total is divided evenly across
            four quarters.
        - The script prints two model runs:
            1. Historical sample only, excluding 2025 entirely.
            2. Sample including 2025 after forecasting missing annual compensation
                 and construction workers.
"""

from pathlib import Path
import csv

import numpy as np
import statsmodels.api as sm


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


def parse_year_quarter_number(value):
    parts = value.strip().split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def aggregate_to_annual(series):
    annual_values = {}
    for (year, _quarter), value in series.items():
        annual_values[year] = annual_values.get(year, 0.0) + value
    return annual_values


def forecast_annual_value(target_by_year, predictor_by_year, forecast_year):
    common_years = sorted(set(target_by_year) & set(predictor_by_year))
    if len(common_years) < 2:
        raise ValueError("Need at least two annual observations to forecast missing values.")

    x_values = np.array([predictor_by_year[year] for year in common_years], dtype=float)
    y_values = np.array([target_by_year[year] for year in common_years], dtype=float)
    design = sm.add_constant(x_values)
    fitted = sm.OLS(y_values, design).fit()

    forecast_input = sm.add_constant(np.array([predictor_by_year[forecast_year]], dtype=float), has_constant="add")
    forecast_value = float(fitted.predict(forecast_input)[0])
    return forecast_value, fitted


def build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    gross_value,
    compensation,
    actual_expenditure,
    sector_workers,
):
    all_keys = sorted(
        set(unemployment)
        & set(imported_workers)
        & set(gdp_growth)
        & set(inflation)
        & set(interest_rate)
        & set(gross_value)
        & set(compensation)
        & set(actual_expenditure)
        & set(sector_workers)
    )

    dataset = []
    for key in all_keys:
        year, quarter = key
        imported_per_1k = (imported_workers[key] / sector_workers[key]) * 1000.0
        dataset.append({
            "Quarter": f"{year} Q{quarter}",
            "Unemployment Rate": unemployment[key],
            "ImportedWorkers per 1K": imported_per_1k,
            "GDP Growth": gdp_growth[key],
            "Inflation Rate": inflation[key],
            "Interest Rate 3M": interest_rate[key],
            "Gross Value": gross_value[key],
            "Compensation": compensation[key],
            "Actual Expenditure": actual_expenditure[key],
        })

    return dataset


def fit_ols(dataset):
    y_values = np.array([row["Unemployment Rate"] for row in dataset])
    x_values = np.column_stack([
        [row["ImportedWorkers per 1K"] for row in dataset],
        [row["GDP Growth"] for row in dataset],
        [row["Inflation Rate"] for row in dataset],
        [row["Interest Rate 3M"] for row in dataset],
        [row["Gross Value"] for row in dataset],
        [row["Compensation"] for row in dataset],
        [row["Actual Expenditure"] for row in dataset],
    ])
    x_values = sm.add_constant(x_values)

    model = sm.OLS(y_values, x_values)
    return model.fit()


def print_regression_report(title, dataset, results, forecast_notes=None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Dataset used in construction regression:")
    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'IW/1K':>10} {'GDPGrowth':>10} "
        f"{'Inflation':>10} {'IntRate3M':>10} {'GrossValue':>12} {'Comp':>10} {'ActExp':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in dataset:
        print(
            f"{row['Quarter']:<12} "
            f"{row['Unemployment Rate']:>10.2f} "
            f"{row['ImportedWorkers per 1K']:>10.4f} "
            f"{row['GDP Growth']:>10.1f} "
            f"{row['Inflation Rate']:>10.2f} "
            f"{row['Interest Rate 3M']:>10.2f} "
            f"{row['Gross Value']:>12.1f} "
            f"{row['Compensation']:>10.1f} "
            f"{row['Actual Expenditure']:>10.1f}"
        )

    print(
        f"\nObservations: {len(dataset)}  "
        f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})"
    )
    if forecast_notes:
        for note in forecast_notes:
            print(note)

    print("\nOLS Regression Results")
    print(results.summary(
        yname="Construction Unemployment Rate",
        xname=[
            "const (beta0)",
            "ImportedWorkers per 1K (beta1)",
            "GDP Growth (beta2)",
            "Inflation Rate (beta3)",
            "Interest Rate 3M (beta4)",
            "Gross Value (beta5)",
            "Compensation (beta6)",
            "Actual Expenditure (beta7)",
        ],
    ))

    print("\nCoefficient Estimates:")
    labels = [
        "beta0 (Intercept)",
        "beta1 (ImportedWorkers per 1K)",
        "beta2 (GDP Growth)",
        "beta3 (Inflation Rate)",
        "beta4 (Interest Rate 3M)",
        "beta5 (Gross Value)",
        "beta6 (Compensation)",
        "beta7 (Actual Expenditure)",
    ]
    for label, coef, pvalue in zip(labels, results.params, results.pvalues):
        significance = "***" if pvalue < 0.01 else ("**" if pvalue < 0.05 else ("*" if pvalue < 0.10 else ""))
        print(f"  {label:<34}  coef = {coef:>12.6f}   p = {pvalue:.4f}  {significance}")

    print(f"\n  R^2         = {results.rsquared:.4f}")
    print(f"  Adj. R^2    = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.4f}  (p = {results.f_pvalue:.4f})")


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


def read_construction_series():
    unemployment = {}
    imported_workers = {}
    gross_value = {}
    actual_expenditure = {}
    annual_compensation = {}
    annual_workers = {}

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

    with (BASE_DIR / "gross_value_of_construction_works.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_yyyyqq((row.get("Quarter") or "").strip())
            value = (row.get("Main Contractor") or "").strip()
            if key is None or not value:
                continue
            try:
                gross_value[key] = float(value)
            except ValueError:
                continue

    with (BASE_DIR / "actual_expenditure.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = parse_year_quarter_number((row.get("YEAR") or "").strip())
            value = (row.get("TOTAL") or "").strip()
            if key is None or not value:
                continue
            try:
                actual_expenditure[key] = float(value)
            except ValueError:
                continue

    with (BASE_DIR / "compensation.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            compensation_value = (
                row.get("Compensation of employees and payments to labour-only sub-contractors (HK$ million)")
                or ""
            ).strip()
            workers_value = (row.get("Number of persons directly engaged") or "").strip()
            if not year_str.isdigit() or not compensation_value or not workers_value:
                continue
            year = int(year_str)
            try:
                annual_compensation[year] = float(compensation_value)
                annual_workers[year] = float(workers_value)
            except ValueError:
                continue

    compensation = {}
    workers = {}
    for year, yearly_compensation in annual_compensation.items():
        yearly_workers = annual_workers.get(year)
        if yearly_workers is None:
            continue
        quarterly_compensation = yearly_compensation / 4.0
        for quarter in range(1, 5):
            compensation[(year, quarter)] = quarterly_compensation
            workers[(year, quarter)] = yearly_workers

    return (
        unemployment,
        imported_workers,
        gross_value,
        actual_expenditure,
        compensation,
        workers,
        annual_compensation,
        annual_workers,
    )


(
    gdp_growth,
    inflation,
    interest_rate,
) = read_macro_series()

(
    unemployment,
    imported_workers,
    gross_value,
    actual_expenditure,
    compensation,
    sector_workers,
    annual_compensation,
    annual_workers,
) = read_construction_series()


annual_gross_value = aggregate_to_annual(gross_value)
annual_actual_expenditure = aggregate_to_annual(actual_expenditure)

historical_dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    gross_value,
    compensation,
    actual_expenditure,
    sector_workers,
)
historical_results = fit_ols(historical_dataset)

compensation_forecast_2025 = None
workers_forecast_2025 = None

if 2025 not in annual_compensation and 2025 in annual_gross_value:
    compensation_forecast_2025, compensation_forecast_model = forecast_annual_value(
        annual_compensation,
        annual_gross_value,
        2025,
    )
    annual_compensation[2025] = compensation_forecast_2025
    quarterly_compensation = compensation_forecast_2025 / 4.0
    for quarter in range(1, 5):
        compensation[(2025, quarter)] = quarterly_compensation

if 2025 not in annual_workers and 2025 in annual_actual_expenditure:
    workers_forecast_2025, workers_forecast_model = forecast_annual_value(
        annual_workers,
        annual_actual_expenditure,
        2025,
    )
    annual_workers[2025] = workers_forecast_2025
    for quarter in range(1, 5):
        sector_workers[(2025, quarter)] = workers_forecast_2025


forecast_dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    gross_value,
    compensation,
    actual_expenditure,
    sector_workers,
)
forecast_results = fit_ols(forecast_dataset)

print_regression_report(
    "Construction OLS Regression Results: Historical Sample Through 2024",
    historical_dataset,
    historical_results,
)

forecast_notes = []
if compensation_forecast_2025 is not None:
    forecast_notes.append(
        "Forecasted 2025 annual compensation from annual gross value "
        f"= {compensation_forecast_2025:.1f} HK$ million"
    )
if workers_forecast_2025 is not None:
    forecast_notes.append(
        "Forecasted 2025 annual construction workers from annual actual expenditure "
        f"= {workers_forecast_2025:.1f} persons"
    )

print_regression_report(
    "Construction OLS Regression Results: Including Forecasted 2025",
    forecast_dataset,
    forecast_results,
    forecast_notes=forecast_notes,
)