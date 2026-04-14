"""
Enhanced Cross-Sector Heterogeneity Model (Model 3 Enhanced)

Uses residuals from each sector's Model 2 regression as the impact measure,
removing the confounding effect of the business cycle.

    Step 1 – Re-estimate each sector's Model 2 to obtain residuals:
             residual_{s,t} = actual unemployment - predicted unemployment

    Step 2 – Compute foreign-labour intensity:
             Intensity_{s,t} = ImportedWorkers_{s,t} / TotalSectorEmployment_{s,t}

    Step 3 – Regress residuals on intensity:
             residual_{s,t} = gamma0 + gamma1 * Intensity_{s,t} + epsilon_{s,t}

    Step 4 – Compare gamma1 across sectors.

The residuals strip out macro-cycle effects (GDP, inflation, interest rates)
and sector-specific activity controls, so gamma1 now captures the association
between foreign-labour intensity and *unexplained* unemployment variation.
"""

from pathlib import Path
import csv

import numpy as np
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent
MODEL2_DIR = BASE_DIR.parent / "model_2"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}
MONTH_STR_TO_Q = {
    "Jan": 1, "Feb": 1, "Mar": 1,
    "Apr": 2, "May": 2, "Jun": 2,
    "Jul": 3, "Aug": 3, "Sep": 3,
    "Oct": 4, "Nov": 4, "Dec": 4,
}


# ===================================================================
# Shared parsing helpers
# ===================================================================

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


def parse_year_quarter_pair(year_value, quarter_value):
    year_text = (year_value or "").strip()
    if not year_text.isdigit():
        return None
    quarter_tokens = (quarter_value or "").strip().split()
    if not quarter_tokens or not quarter_tokens[0].startswith("Q"):
        return None
    try:
        return int(year_text), int(quarter_tokens[0][1:])
    except ValueError:
        return None


# ===================================================================
# Shared macro data reader
# ===================================================================

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


# ===================================================================
# Total sector employment (from C&SD all_sector_labour.csv)
# ===================================================================

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


# ===================================================================
# Per-sector imported workers reader (shared)
# ===================================================================

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


# ===================================================================
# Per-sector unemployment reader (shared)
# ===================================================================

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


# ===================================================================
# Construction Model 2 replication
# ===================================================================

def run_construction_model2(gdp, infl, ir):
    folder = MODEL2_DIR / "construction"
    unemployment = read_unemployment("construction")
    imported = read_imported_workers("construction")

    gross_value = {}
    with (folder / "gross_value_of_construction_works.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_yyyyqq((row.get("Quarter") or "").strip())
            val = (row.get("Main Contractor") or "").strip()
            if key and val:
                try:
                    gross_value[key] = float(val)
                except ValueError:
                    pass

    actual_expenditure = {}
    with (folder / "actual_expenditure.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_year_quarter_number((row.get("YEAR") or "").strip())
            val = (row.get("TOTAL") or "").strip()
            if key and val:
                try:
                    actual_expenditure[key] = float(val)
                except ValueError:
                    pass

    annual_compensation = {}
    annual_workers = {}
    with (folder / "compensation.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            comp_val = (row.get(
                "Compensation of employees and payments to labour-only sub-contractors (HK$ million)"
            ) or "").strip()
            work_val = (row.get("Number of persons directly engaged") or "").strip()
            if not year_str.isdigit() or not comp_val or not work_val:
                continue
            year = int(year_str)
            try:
                annual_compensation[year] = float(comp_val)
                annual_workers[year] = float(work_val)
            except ValueError:
                continue

    # Forecast 2025 compensation & workers if missing
    annual_gv = {}
    for (y, _q), v in gross_value.items():
        annual_gv[y] = annual_gv.get(y, 0.0) + v
    annual_ae = {}
    for (y, _q), v in actual_expenditure.items():
        annual_ae[y] = annual_ae.get(y, 0.0) + v

    if 2025 not in annual_compensation and 2025 in annual_gv:
        forecast, _ = _forecast_annual(annual_compensation, annual_gv, 2025)
        annual_compensation[2025] = forecast
    if 2025 not in annual_workers and 2025 in annual_ae:
        forecast, _ = _forecast_annual(annual_workers, annual_ae, 2025)
        annual_workers[2025] = forecast

    compensation = {}
    sector_workers = {}
    for year, comp in annual_compensation.items():
        workers = annual_workers.get(year)
        if workers is None:
            continue
        for q in range(1, 5):
            compensation[(year, q)] = comp / 4.0
            sector_workers[(year, q)] = workers

    keys = sorted(
        set(unemployment) & set(imported) & set(gdp) & set(infl) & set(ir)
        & set(gross_value) & set(compensation) & set(actual_expenditure)
        & set(sector_workers)
    )

    rows = []
    for k in keys:
        iw_per1k = (imported[k] / sector_workers[k]) * 1000.0
        rows.append({
            "key": k,
            "y": unemployment[k],
            "x": [iw_per1k, gdp[k], infl[k], ir[k],
                  gross_value[k], compensation[k], actual_expenditure[k]],
            "imported": imported[k],
        })

    return _fit_and_extract(rows)


# ===================================================================
# Retail Model 2 replication
# ===================================================================

def run_retail_model2(gdp, infl, ir):
    folder = MODEL2_DIR / "retail"
    unemployment = read_unemployment("retail")
    imported = read_imported_workers("retail")

    monthly_ri = {}
    with (folder / "retail_index.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            month_str = (row.get("Month") or "").strip()
            val = (row.get("Value index of retail sales") or "").strip()
            if not year_str.isdigit() or month_str not in MONTH_STR_TO_Q or not val:
                continue
            key = (int(year_str), MONTH_STR_TO_Q[month_str])
            monthly_ri.setdefault(key, []).append(float(val))
    retail_index = {k: sum(v) / len(v) for k, v in monthly_ri.items()}

    reception_index = {}
    with (folder / "reception_index.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = parse_year_quarter_pair(row.get("Year"), row.get("Quarter"))
            if key is None:
                continue
            r = (row.get("Retail") or "").strip()
            a = (row.get("Accommodation services") or "").strip()
            fd = (row.get("Food services") or "").strip()
            if not r or not a or not fd:
                continue
            try:
                reception_index[key] = (float(r) + float(a) + float(fd)) / 3.0
            except ValueError:
                continue

    keys = sorted(
        set(unemployment) & set(imported) & set(gdp) & set(infl) & set(ir)
        & set(retail_index) & set(reception_index)
    )

    rows = []
    for k in keys:
        rows.append({
            "key": k,
            "y": unemployment[k],
            "x": [imported[k], gdp[k], infl[k], ir[k],
                  retail_index[k], reception_index[k]],
            "imported": imported[k],
        })

    return _fit_and_extract(rows)


# ===================================================================
# Transportation Model 2 replication
# ===================================================================

def run_transportation_model2(gdp, infl, ir):
    folder = MODEL2_DIR / "transportation"
    unemployment = read_unemployment("transportation")
    imported = read_imported_workers("transportation")

    air_cargo = {}
    with (folder / "air_cargo_output.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            val = (row.get("Air Cargo Throughput") or "").strip()
            if not year_str.isdigit() or not val:
                continue
            quarterly = float(val) / 4.0
            year = int(year_str)
            for q in range(1, 5):
                air_cargo[(year, q)] = quarterly

    tourist_total = {}
    with (folder / "tourist.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            overnight = (row.get("Overnight visitors") or "").strip()
            same_day = (row.get("Same-day visitors") or "").strip()
            if not year_str.isdigit() or not overnight or not same_day:
                continue
            quarterly = (float(overnight) + float(same_day)) / 4.0
            year = int(year_str)
            for q in range(1, 5):
                tourist_total[(year, q)] = quarterly

    keys = sorted(
        set(unemployment) & set(imported) & set(gdp) & set(infl) & set(ir)
        & set(air_cargo) & set(tourist_total)
    )

    rows = []
    for k in keys:
        rows.append({
            "key": k,
            "y": unemployment[k],
            "x": [imported[k], gdp[k], infl[k], ir[k],
                  air_cargo[k], tourist_total[k]],
            "imported": imported[k],
        })

    return _fit_and_extract(rows)


# ===================================================================
# Community Model 2 replication
# ===================================================================

def run_community_model2(gdp, infl, ir):
    folder = MODEL2_DIR / "community"
    selected_industry = "Public administration, social and personal services"
    unemployment = read_unemployment("community", selected_industry)
    imported = read_imported_workers("community")

    annual_above_65 = {}
    with (folder / "above_65.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("Year") or "").strip()
            val = (row.get("Population") or "").strip()
            if not year_str.isdigit() or not val:
                continue
            year = int(year_str)
            try:
                annual_above_65[year] = annual_above_65.get(year, 0.0) + float(val)
            except ValueError:
                continue

    annual_gov_exp = {}
    with (folder / "government_expenditure.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_str = (row.get("YEAR") or "").strip()
            elderly = (row.get("Services for the Elderly") or "").strip()
            rehab = (row.get("Rehabilitation & Medical Social Services") or "").strip()
            if not year_str.isdigit() or not elderly or not rehab:
                continue
            try:
                annual_gov_exp[int(year_str)] = float(elderly) + float(rehab)
            except ValueError:
                continue

    # Forecast 2017 government expenditure if missing
    if 2017 not in annual_gov_exp and 2017 in annual_above_65:
        forecast, _ = _forecast_annual(annual_gov_exp, annual_above_65, 2017)
        annual_gov_exp[2017] = forecast

    above_65 = {}
    for year, total in annual_above_65.items():
        for q in range(1, 5):
            above_65[(year, q)] = total

    gov_exp = {}
    for year, total in annual_gov_exp.items():
        quarterly = total / 4.0
        for q in range(1, 5):
            gov_exp[(year, q)] = quarterly

    keys = sorted(
        set(unemployment) & set(imported) & set(gdp) & set(infl) & set(ir)
        & set(above_65) & set(gov_exp)
    )

    rows = []
    for k in keys:
        rows.append({
            "key": k,
            "y": unemployment[k],
            "x": [imported[k], gdp[k], infl[k], ir[k],
                  gov_exp[k], above_65[k]],
            "imported": imported[k],
        })

    return _fit_and_extract(rows)


# ===================================================================
# Helpers
# ===================================================================

def _forecast_annual(target_by_year, predictor_by_year, forecast_year):
    common = sorted(set(target_by_year) & set(predictor_by_year))
    x = np.array([predictor_by_year[y] for y in common], dtype=float)
    y = np.array([target_by_year[y] for y in common], dtype=float)
    fitted = sm.OLS(y, sm.add_constant(x)).fit()
    xf = sm.add_constant(
        np.array([predictor_by_year[forecast_year]], dtype=float),
        has_constant="add",
    )
    return float(fitted.predict(xf)[0]), fitted


def _fit_and_extract(rows):
    """Fit Model 2 OLS and return (keys, residuals, imported_workers, model2_results)."""
    y = np.array([r["y"] for r in rows])
    X = sm.add_constant(np.array([r["x"] for r in rows]))
    results = sm.OLS(y, X).fit()
    residuals = results.resid
    keys = [r["key"] for r in rows]
    imported = [r["imported"] for r in rows]
    return keys, residuals, imported, results


# ===================================================================
# Sector configuration
# ===================================================================

SECTOR_CONFIG = {
    "Construction": {
        "runner": "run_construction_model2",
        "employment_industry": "Construction",
    },
    "Retail": {
        "runner": "run_retail_model2",
        "employment_industry": "Retail, accommodation and food services",
    },
    "Transportation": {
        "runner": "run_transportation_model2",
        "employment_industry": (
            "Transportation, storage, postal and courier services, "
            "information and communications"
        ),
    },
    "Community": {
        "runner": "run_community_model2",
        "employment_industry": "Public administration, social and personal services",
    },
}

RUNNERS = {
    "run_construction_model2": run_construction_model2,
    "run_retail_model2": run_retail_model2,
    "run_transportation_model2": run_transportation_model2,
    "run_community_model2": run_community_model2,
}


# ===================================================================
# Reporting
# ===================================================================

def print_sector_report(sector_name, dataset, model2_results, model3_results):
    print("\n" + "=" * 80)
    print(f"  {sector_name}  –  Intensity → Model 2 Residual Regression")
    print("=" * 80)

    print(f"\n  Model 2 summary: R^2 = {model2_results.rsquared:.4f}, "
          f"Adj. R^2 = {model2_results.rsquared_adj:.4f}, "
          f"N = {int(model2_results.nobs)}")

    header = (
        f"{'Quarter':<12} {'Residual':>10} {'ImpWork':>10} "
        f"{'Employ(k)':>10} {'Intensity':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in dataset:
        print(
            f"{r['Quarter']:<12} "
            f"{r['Residual']:>10.4f} "
            f"{r['Imported Workers']:>10.2f} "
            f"{r['Total Employment (1000)']:>10.1f} "
            f"{r['Intensity']:>12.6f}"
        )

    print(f"\nObservations: {len(dataset)}  "
          f"(from {dataset[0]['Quarter']} to {dataset[-1]['Quarter']})")

    print(
        model3_results.summary(
            yname="Model 2 Residual",
            xname=["gamma0 (Intercept)", "gamma1 (Intensity)"],
        )
    )

    gamma0, gamma1 = model3_results.params
    p0, p1 = model3_results.pvalues
    sig = "***" if p1 < 0.01 else ("**" if p1 < 0.05 else ("*" if p1 < 0.10 else ""))
    print(f"\n  gamma0 (Intercept)   coef = {gamma0:>12.6f}   p = {p0:.4f}")
    print(f"  gamma1 (Intensity)   coef = {gamma1:>12.6f}   p = {p1:.4f}  {sig}")
    print(f"  R^2       = {model3_results.rsquared:.4f}")
    print(f"  Adj. R^2  = {model3_results.rsquared_adj:.4f}")


def print_comparison(sector_results):
    print("\n" + "=" * 80)
    print("  Cross-Sector Comparison Summary (Enhanced: using Model 2 residuals)")
    print("=" * 80)

    header = (
        f"{'Sector':<18} {'gamma1':>12} {'p-value':>10} {'Sig':>5} "
        f"{'R^2':>8} {'Adj.R^2':>8} {'N':>5} {'M2 R^2':>8}"
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
            f"{s['r2']:>8.4f} {s['adj_r2']:>8.4f} {s['n']:>5} {s['m2_r2']:>8.4f}"
        )

    print("\nInterpretation:")
    print("  Impact strength is now measured by Model 2 residuals (cycle-adjusted).")
    print("  A positive gamma1 means higher foreign-labour intensity is associated")
    print("  with higher *unexplained* unemployment (after controlling for macro")
    print("  conditions and sector-specific activity), supporting the hypothesis")
    print("  of labour-supply competition from imported workers.")


# ===================================================================
# Main
# ===================================================================

def main():
    gdp, infl, ir = read_macro_series()
    all_employment = read_total_employment()

    sector_results = []

    for sector_name, config in SECTOR_CONFIG.items():
        runner = RUNNERS[config["runner"]]
        emp_industry = config["employment_industry"]
        employment = all_employment.get(emp_industry, {})

        keys, residuals, imported_arr, model2_results = runner(gdp, infl, ir)

        # Build dataset with intensity
        dataset = []
        valid_indices = []
        for i, k in enumerate(keys):
            if k not in employment:
                continue
            emp = employment[k]
            intensity = imported_arr[i] / emp
            dataset.append({
                "Quarter": f"{k[0]} Q{k[1]}",
                "Residual": residuals[i],
                "Imported Workers": imported_arr[i],
                "Total Employment (1000)": emp,
                "Intensity": intensity,
            })
            valid_indices.append(i)

        if len(dataset) < 3:
            print(f"\n[{sector_name}] Insufficient data. Skipping.")
            continue

        y = np.array([d["Residual"] for d in dataset])
        x = sm.add_constant(np.array([d["Intensity"] for d in dataset]))
        model3_results = sm.OLS(y, x).fit()

        print_sector_report(sector_name, dataset, model2_results, model3_results)
        sector_results.append({
            "sector": sector_name,
            "gamma1": model3_results.params[1],
            "p": model3_results.pvalues[1],
            "r2": model3_results.rsquared,
            "adj_r2": model3_results.rsquared_adj,
            "n": len(dataset),
            "m2_r2": model2_results.rsquared,
        })

    if sector_results:
        print_comparison(sector_results)


if __name__ == "__main__":
    main()
