"""
OLS Regression Model:

    Unemployment Rate_t = β0
        + β1 * (ImportedWorkers_t / TotalEmployment_t × 100%)
        + β2 * GDP Growth_t
        + β3 * Inflation Rate_t
        + β4 * Interest Rate (3M)_t
        + β5 * LFPR_t
        + ε

Data sources:
    unemployment_rate.csv  — Unemployment Rate, Employed Persons, LFPR
    imported_workers.csv   — Yearly imported workers (divided by 4 → quarterly)
    gdp.csv                — GDP year-on-year growth rate
    inflation_rate.csv     — Quarterly inflation rate
    interest_rate.csv      — HIBOR 3-month rate
"""

import csv
import numpy as np
import statsmodels.api as sm


# ── Helper: parse "2017 Q1" → (2017, 1) ─────────────────────────────────────
def parse_yyyyqq(s):
    parts = s.strip().split()
    if len(parts) != 2:
        return None
    try:
        year = int(parts[0])
        q = int(parts[1][1])
        return (year, q)
    except (ValueError, IndexError):
        return None


# ── 1. Unemployment rate, employed persons, LFPR ─────────────────────────────
#    Time format: "1/2017 - 3/2017"
unemployment = {}   # (year, q) → unemployment rate (both sexes)
employment   = {}   # (year, q) → employed persons (thousands, both sexes)
lfpr         = {}   # (year, q) → labour force participation rate (both sexes)

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}

with open("unemployment_rate.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time = (row.get("Time") or "").strip()
        if not time or "/" not in time:
            continue
        try:
            # e.g. "1/2017 - 3/2017"  →  start month = 1, year = 2017
            left = time.split("-")[0].strip()         # "1/2017"
            month_str, year_str = left.split("/")
            month = int(month_str.strip())
            year  = int(year_str.strip())
            q = MONTH_TO_Q.get(month)
            if q is None:
                continue
        except (ValueError, IndexError):
            continue

        try:
            u_rate   = float((row.get("Unemployment Rate Both") or "").strip())
            emp      = float((row.get("Employed Persons Both")  or "").strip())
            lfpr_val = float((row.get("Labour Force Participation Rate Both") or "").strip())
        except ValueError:
            continue

        key = (year, q)
        unemployment[key] = u_rate
        employment[key]   = emp
        lfpr[key]         = lfpr_val


# ── 2. Imported workers (yearly → quarterly, equal distribution) ──────────────
imported_workers = {}  # (year, q) → quarterly imported workers (actual count)

with open("imported_workers.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        year_str    = (row.get("Year") or "").strip()
        workers_str = (row.get("Number of Imported Workers Approved") or "").strip()
        if not year_str.isdigit() or not workers_str:
            continue
        year             = int(year_str)
        quarterly_count  = float(workers_str) / 4.0     # distribute evenly
        for q in range(1, 5):
            imported_workers[(year, q)] = quarterly_count


# ── 3. GDP year-on-year growth rate ──────────────────────────────────────────
gdp_growth = {}  # (year, q) → % change

with open("gdp.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time       = (row.get("Time") or "").strip()
        change_str = (row.get("GDP Year-on-year Change") or "").strip()
        key = parse_yyyyqq(time)
        if key is None or not change_str:
            continue
        try:
            gdp_growth[key] = float(change_str.lstrip("+"))
        except ValueError:
            continue


# ── 4. Quarterly inflation rate ───────────────────────────────────────────────
inflation = {}  # (year, q) → quarterly inflation rate (%)

with open("inflation_rate.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        quarter  = (row.get("Quarter") or "").strip()
        rate_str = (row.get("Quarterly Inflation Rate (%)") or "").strip()
        key = parse_yyyyqq(quarter)
        if key is None or not rate_str:
            continue
        try:
            inflation[key] = float(rate_str)
        except ValueError:
            continue


# ── 5. HIBOR 3-month interest rate ────────────────────────────────────────────
interest_rate = {}  # (year, q) → 3-month rate (%)

with open("interest_rate.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time     = (row.get("Time") or "").strip()
        rate_str = (row.get("3 months") or "").strip()
        key = parse_yyyyqq(time)
        if key is None or not rate_str:
            continue
        try:
            interest_rate[key] = float(rate_str)
        except ValueError:
            continue


# ── 6. Assemble aligned dataset ───────────────────────────────────────────────
all_keys = sorted(
    set(unemployment)
    & set(employment)
    & set(lfpr)
    & set(imported_workers)
    & set(gdp_growth)
    & set(inflation)
    & set(interest_rate)
)

dataset = []
for key in all_keys:
    year, q = key
    emp       = employment[key]          # thousands
    iw        = imported_workers[key]    # actual count (yearly / 4)

    # Convert employment to actual headcount for ratio calculation
    iw_ratio = (iw / (emp * 1_000)) * 100.0   # as percentage

    dataset.append({
        "Quarter":                f"{year} Q{q}",
        "Unemployment Rate":      unemployment[key],
        "ImportedWorkers Ratio":  iw_ratio,
        "GDP Growth":             gdp_growth[key],
        "Inflation Rate":         inflation[key],
        "Interest Rate 3M":       interest_rate[key],
        "LFPR":                   lfpr[key],
    })


# ── 7. OLS Regression ─────────────────────────────────────────────────────────
Y = np.array([r["Unemployment Rate"]     for r in dataset])
X = np.column_stack([
    [r["ImportedWorkers Ratio"]  for r in dataset],   # β1
    [r["GDP Growth"]             for r in dataset],   # β2
    [r["Inflation Rate"]         for r in dataset],   # β3
    [r["Interest Rate 3M"]       for r in dataset],   # β4
    [r["LFPR"]                   for r in dataset],   # β5
])
X = sm.add_constant(X)    # prepends the intercept column (β0)

model   = sm.OLS(Y, X)
results = model.fit()


# ── 8. Print dataset ──────────────────────────────────────────────────────────
print("Dataset used in regression:")
header = (f"{'Quarter':<12} {'UnempRate':>10} {'IW Ratio%':>10} "
          f"{'GDPGrowth':>10} {'Inflation':>10} {'IntRate3M':>10} {'LFPR':>8}")
print(header)
print("-" * len(header))
for r in dataset:
    print(
        f"{r['Quarter']:<12} "
        f"{r['Unemployment Rate']:>10.2f} "
        f"{r['ImportedWorkers Ratio']:>10.5f} "
        f"{r['GDP Growth']:>10.1f} "
        f"{r['Inflation Rate']:>10.2f} "
        f"{r['Interest Rate 3M']:>10.2f} "
        f"{r['LFPR']:>8.1f}"
    )

print(f"\nObservations: {len(dataset)}  "
      f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})")


# ── 9. Print regression results ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("OLS Regression Results")
print("=" * 65)
print(results.summary(
    yname="Unemployment Rate",
    xname=["const (β0)",
           "ImportedWorkers/TotalEmp×100% (β1)",
           "GDP Growth (β2)",
           "Inflation Rate (β3)",
           "Interest Rate 3M (β4)",
           "LFPR (β5)"]
))

# ── 10. Coefficient summary ───────────────────────────────────────────────────
print("\nCoefficient Estimates:")
labels = ["β0 (Intercept)",
          "β1 (IW Ratio)",
          "β2 (GDP Growth)",
          "β3 (Inflation Rate)",
          "β4 (Interest Rate 3M)",
          "β5 (LFPR)"]
for label, coef, pval in zip(labels, results.params, results.pvalues):
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
    print(f"  {label:<28}  coef = {coef:>10.6f}   p = {pval:.4f}  {sig}")

print(f"\n  R²          = {results.rsquared:.4f}")
print(f"  Adj. R²     = {results.rsquared_adj:.4f}")
print(f"  F-statistic = {results.fvalue:.4f}  (p = {results.f_pvalue:.4f})")
