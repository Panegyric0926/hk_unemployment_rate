"""
OLS Regression Model 2: Community sector

    Community Unemployment Rate_{s,t} = beta0
        + beta1 * ImportedWorkers_{s,t}
        + beta2 * GDP Growth_t
        + beta3 * Inflation Rate_t
        + beta4 * Interest Rate (3M)_t
        + beta5 * Government Expenditure_{s,t}
        + beta6 * Above65 Population_{s,t}
        + epsilon_{s,t}

Notes:
    - Common macro shocks come from the model_2 folder.
    - Imported workers are annual approvals divided evenly across quarters.
    - Government expenditure is the annual sum of Services for the Elderly and
      Rehabilitation & Medical Social Services, divided evenly across quarters.
    - Above65 Population is the annual sum across all 65+ age groups and is
      repeated across quarters as a stock variable.
        - The script prints two model runs:
            1. Real-data-only sample starting in 2018.
            2. Sample including 2017 after forecasting the missing 2017 annual
                 government expenditure from annual 65+ population.
"""

from pathlib import Path
import csv
import sys

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent

VALID_INDUSTRIES = [
    "Public administration, social and personal services",
    "Human health and social work activities",
    "Social work activities",
]

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


def forecast_annual_value(target_by_year, predictor_by_year, forecast_year):
    common_years = sorted(set(target_by_year) & set(predictor_by_year))
    if len(common_years) < 2:
        raise ValueError("Need at least two annual observations to forecast missing values.")

    x_values = np.array([predictor_by_year[year] for year in common_years], dtype=float)
    y_values = np.array([target_by_year[year] for year in common_years], dtype=float)
    design = sm.add_constant(x_values)
    fitted = sm.OLS(y_values, design).fit()

    forecast_input = sm.add_constant(
        np.array([predictor_by_year[forecast_year]], dtype=float),
        has_constant="add",
    )
    forecast_value = float(fitted.predict(forecast_input)[0])
    return forecast_value, fitted


def resolve_selected_industry():
    if len(sys.argv) > 1:
        selected_industry = " ".join(sys.argv[1:]).strip()
    else:
        print("Select the industry for the community unemployment series:")
        for index, industry in enumerate(VALID_INDUSTRIES, start=1):
            print(f"  {index}. {industry}")
        user_input = input("Enter the number or exact industry name: ").strip()
        if user_input.isdigit():
            option = int(user_input)
            if 1 <= option <= len(VALID_INDUSTRIES):
                selected_industry = VALID_INDUSTRIES[option - 1]
            else:
                raise ValueError("Selected industry number is out of range.")
        else:
            selected_industry = user_input

    if selected_industry not in VALID_INDUSTRIES:
        valid_values = "; ".join(VALID_INDUSTRIES)
        raise ValueError(f"Invalid industry selection: {selected_industry}. Choose one of: {valid_values}")

    return selected_industry


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


def read_community_series(selected_industry):
    unemployment = {}
    imported_workers = {}
    annual_above_65 = {}
    annual_government_expenditure = {}

    with (BASE_DIR / "unemployment_rate.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_value = (row.get("Time") or "").strip()
            industry_value = (row.get("Detailed previous industry") or "").strip()
            unemployment_value = (row.get("Unemployment Rate") or "").strip()
            if not time_value or not unemployment_value or industry_value != selected_industry:
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

    with (BASE_DIR / "above_65.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("Year") or "").strip()
            population_value = (row.get("Population") or "").strip()
            if not year_str.isdigit() or not population_value:
                continue
            try:
                annual_above_65[int(year_str)] = annual_above_65.get(int(year_str), 0.0) + float(population_value)
            except ValueError:
                continue

    with (BASE_DIR / "government_expenditure.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_str = (row.get("YEAR") or "").strip()
            elderly_value = (row.get("Services for the Elderly") or "").strip()
            rehab_value = (row.get("Rehabilitation & Medical Social Services") or "").strip()
            if not year_str.isdigit() or not elderly_value or not rehab_value:
                continue
            try:
                annual_government_expenditure[int(year_str)] = float(elderly_value) + float(rehab_value)
            except ValueError:
                continue

    above_65_population = {}
    government_expenditure = {}

    for year, total_population in annual_above_65.items():
        for quarter in range(1, 5):
            above_65_population[(year, quarter)] = total_population

    for year, total_expenditure in annual_government_expenditure.items():
        quarterly_expenditure = total_expenditure / 4.0
        for quarter in range(1, 5):
            government_expenditure[(year, quarter)] = quarterly_expenditure

    return (
        unemployment,
        imported_workers,
        above_65_population,
        government_expenditure,
        annual_above_65,
        annual_government_expenditure,
    )


def build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    above_65_population,
    government_expenditure,
):
    all_keys = sorted(
        set(unemployment)
        & set(imported_workers)
        & set(gdp_growth)
        & set(inflation)
        & set(interest_rate)
        & set(above_65_population)
        & set(government_expenditure)
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
            "Government Expenditure": government_expenditure[(year, quarter)],
            "Above65 Population": above_65_population[(year, quarter)],
        })

    return dataset


def fit_ols(dataset):
    y_values = np.array([row["Unemployment Rate"] for row in dataset])
    x_values = np.column_stack([
        [row["Imported Workers"] for row in dataset],
        [row["GDP Growth"] for row in dataset],
        [row["Inflation Rate"] for row in dataset],
        [row["Interest Rate 3M"] for row in dataset],
        [row["Government Expenditure"] for row in dataset],
        [row["Above65 Population"] for row in dataset],
    ])
    x_values = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_values)
    return model.fit()


def print_regression_report(title, dataset, results, forecast_note=None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Dataset used in community regression:")
    header = (
        f"{'Quarter':<12} {'UnempRate':>10} {'ImpWork':>10} {'GDPGrowth':>10} "
        f"{'Inflation':>10} {'IntRate3M':>10} {'GovExp':>10} {'Above65':>10}"
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
            f"{row['Government Expenditure']:>10.2f} "
            f"{row['Above65 Population']:>10.1f}"
        )

    print(
        f"\nObservations: {len(dataset)}  "
        f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})"
    )
    if forecast_note:
        print(forecast_note)

    print("\nOLS Regression Results")
    print(
        results.summary(
            yname="Community Unemployment Rate",
            xname=[
                "const (beta0)",
                "Imported Workers (beta1)",
                "GDP Growth (beta2)",
                "Inflation Rate (beta3)",
                "Interest Rate 3M (beta4)",
                "Government Expenditure (beta5)",
                "Above65 Population (beta6)",
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
        "beta5 (Government Expenditure)",
        "beta6 (Above65 Population)",
    ]
    for label, coef, pvalue in zip(labels, results.params, results.pvalues):
        significance = "***" if pvalue < 0.01 else ("**" if pvalue < 0.05 else ("*" if pvalue < 0.10 else ""))
        print(f"  {label:<37}  coef = {coef:>12.6f}   p = {pvalue:.4f}  {significance}")

    print(f"\n  R^2         = {results.rsquared:.4f}")
    print(f"  Adj. R^2    = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.4f}  (p = {results.f_pvalue:.4f})")


selected_industry = resolve_selected_industry()

(
    gdp_growth,
    inflation,
    interest_rate,
) = read_macro_series()

(
    unemployment,
    imported_workers,
    above_65_population,
    government_expenditure,
    annual_above_65,
    annual_government_expenditure,
) = read_community_series(selected_industry)

historical_dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    above_65_population,
    government_expenditure,
)
historical_results = fit_ols(historical_dataset)

government_expenditure_forecast_2017 = None
if 2017 not in annual_government_expenditure and 2017 in annual_above_65:
    government_expenditure_forecast_2017, _forecast_model = forecast_annual_value(
        annual_government_expenditure,
        annual_above_65,
        2017,
    )
    annual_government_expenditure[2017] = government_expenditure_forecast_2017
    quarterly_expenditure = government_expenditure_forecast_2017 / 4.0
    for quarter in range(1, 5):
        government_expenditure[(2017, quarter)] = quarterly_expenditure

dataset = build_dataset(
    unemployment,
    imported_workers,
    gdp_growth,
    inflation,
    interest_rate,
    above_65_population,
    government_expenditure,
)
results = fit_ols(dataset)

forecast_note = None
if government_expenditure_forecast_2017 is not None:
    forecast_note = (
        "Forecasted 2017 annual government expenditure from annual 65+ population "
        f"= {government_expenditure_forecast_2017:.1f}"
    )

print_regression_report(
    f"Community OLS Regression Results: Real Data Only From 2018 ({selected_industry})",
    historical_dataset,
    historical_results,
)

print_regression_report(
    f"Community OLS Regression Results: Including Predicted 2017 ({selected_industry})",
    dataset,
    results,
    forecast_note=forecast_note,
)

# QQ-plots of residuals (real-data-only and imputed-2017 samples side by side)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
qqplot(historical_results.resid, line="s", alpha=0.6, ax=axes[0])
axes[0].set_title("Real Data Only (from 2018)")
qqplot(results.resid, line="s", alpha=0.6, ax=axes[1])
axes[1].set_title("Including Predicted 2017")
fig.suptitle(
    f"Community (Model 2) – Normal QQ-Plots of Residuals\n({selected_industry})",
    fontsize=10,
)
plt.tight_layout()
plt.show()