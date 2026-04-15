# Labour Market Analysis: Impact of Imported Workers on Unemployment

This repository contains a series of OLS regression models examining how imported (foreign) labour affects unemployment rates in Hong Kong, both at the economy-wide level and across four specific industry sectors.

---

## Research Question

> Does the inflow of imported workers affect sectoral and aggregate unemployment rates, and does this effect differ across industries?

---

## Repository Structure

```
.
├── model_1.py                       # Economy-wide baseline model
├── quarterly_cpi.py                 # CPI quarterly aggregation helper
├── plot_data_overview.py            # Exploratory data visualisations (heat-map, scatter matrix, time series)
├── unemployment_rate.csv            # Economy-wide quarterly unemployment, employment, LFPR
├── imported_workers.csv             # Annual imported worker approvals (economy-wide)
├── gdp.csv                          # GDP year-on-year growth rate
├── inflation_rate.csv               # Quarterly inflation rate
├── interest_rate.csv                # HIBOR 3-month rate
├── cpi.csv                          # Consumer Price Index (monthly)
│
├── model_2/                         # Sector-specific models
│   ├── gdp.csv
│   ├── inflation_rate.csv
│   ├── interest_rate.csv
│   ├── plot_unemployment_comparison.py   # Chart comparing unemployment across sectors
│   ├── construction/
│   │   └── model_2_construction.py
│   ├── retail/
│   │   └── model_2_retail.py
│   ├── transportation/
│   │   └── model_2_transportation.py
│   └── community/
│       └── model_2_community.py
│
└── model_3/                         # Cross-sector heterogeneity models
    ├── model_3.py                   # Baseline (no controls)
    ├── model_3_macro.py             # Macro-controlled
    ├── model_3_enhance.py           # Two-step residual-based
    ├── plot_reverse_causality.py    # Reverse causality diagnostic
    ├── all_sector_labour.csv        # Sector-level employment (C&SD)
    └── MODEL_3_COMPARISON.md        # Detailed comparison of Model 3 variants
```

---

## Models

### Model 1 — Economy-Wide Baseline (`model_1.py`)

A single OLS regression using aggregate Hong Kong data:

$$
\text{UnemploymentRate}_t = \beta_0 + \beta_1 \cdot \frac{\text{ImportedWorkers}_t}{\text{TotalEmployment}_t} + \beta_2 \cdot \text{GDPGrowth}_t + \beta_3 \cdot \text{Inflation}_t + \beta_4 \cdot \text{InterestRate}_t + \beta_5 \cdot \text{LFPR}_t + \varepsilon_t
$$

**Data sources:** `unemployment_rate.csv`, `imported_workers.csv`, `gdp.csv`, `inflation_rate.csv`, `interest_rate.csv`

A **QQ-plot** of residuals is displayed after the regression summary.

---

### Model 2 — Sector-Specific OLS

Each sector model adds industry-specific activity controls on top of the common macro variables.

| Sector | Key Sector Controls | Script |
|---|---|---|
| **Construction** | Gross value of construction works, compensation, actual expenditure | `model_2/construction/model_2_construction.py` |
| **Retail** | Retail sales value index, reception index (retail/accommodation/food) | `model_2/retail/model_2_retail.py` |
| **Transportation** | Air cargo output, tourist arrivals (overnight + same-day) | `model_2/transportation/model_2_transportation.py` |
| **Community** | Government expenditure on elderly/social services, population aged 65+ | `model_2/community/model_2_community.py` |

All sector models share the same macro controls: **GDP growth**, **inflation rate**, **HIBOR 3-month interest rate**.

Imported workers are annual approval figures divided equally across the four quarters of each year. Each sector model prints two runs where applicable: one on the historical sample only, and one extended to include a forecast year.

Each Model 2 script also displays a **QQ-plot** of OLS residuals to assess normality. Scripts with two runs (Construction, Community) show a side-by-side panel.

---

### Model 3 — Cross-Sector Heterogeneity

Three variants compare the effect of **foreign-labour intensity** across sectors:

$$
\text{Intensity}_{s,t} = \frac{\text{ImportedWorkers}_{s,t}}{\text{TotalSectorEmployment}_{s,t}}
$$

| Variant | Dependent Variable | Controls | Script |
|---|---|---|---|
| **Baseline** | Sector unemployment rate | None | `model_3/model_3.py` |
| **Macro-controlled** | Sector unemployment rate | GDP growth, inflation, interest rate | `model_3/model_3_macro.py` |
| **Two-step residual** | Model 2 residuals (unexplained unemployment) | Macro + sector activity (via Model 2) | `model_3/model_3_enhance.py` |

The key coefficient `gamma1` measures how strongly foreign-labour intensity affects unemployment. Comparing `gamma1` across Construction, Retail, Transportation, and Community reveals cross-sector heterogeneity. See [model_3/MODEL_3_COMPARISON.md](model_3/MODEL_3_COMPARISON.md) for a detailed explanation of the differences between variants.

Each Model 3 script also displays:
- A **2×2 QQ-plot grid** – one panel per sector, checking normality of OLS residuals.
- A **2×2 Intensity vs Unemployment scatter grid** – showing observed data, an OLS trend line, and the γ₁ coefficient with its p-value. For `model_3_enhance.py` the y-axis is the Model 2 residual.

---

## Data Notes

- **Time frequency:** Quarterly (unemployment, macro variables); annual figures for imported workers and some sector controls are divided evenly across four quarters.
- **Quarter format in CSVs:** `"1/2017 - 3/2017"` (start month / year – end month / year).
- **CPI:** `quarterly_cpi.py` aggregates monthly CPI data into quarterly averages.
- **Construction (2025):** Annual compensation and construction workers are forecast from trend to keep 2025 quarters in-sample.
- **Community (2017):** Missing government expenditure is imputed from the annual 65+ population series.

---

## Dependencies

```
numpy
statsmodels
matplotlib
pandas  (optional – only needed if you extend plot_data_overview.py with pandas-based analysis)
```

Install with:

```bash
pip install numpy statsmodels matplotlib
```

---

## Running the Models

```bash
# Economy-wide model
python model_1.py

# Sector-specific models (run from project root)
python model_2/construction/model_2_construction.py
python model_2/retail/model_2_retail.py
python model_2/transportation/model_2_transportation.py
python model_2/community/model_2_community.py

# Cross-sector comparison plots
python model_2/plot_unemployment_comparison.py

# Model 3 variants
python model_3/model_3.py
python model_3/model_3_macro.py
python model_3/model_3_enhance.py
python model_3/plot_reverse_causality.py

# Data overview visualisations
python plot_data_overview.py
```

---

## Data Overview Visualisations (`plot_data_overview.py`)

A standalone script that produces six exploratory charts from all available data without running any regression:

| # | Chart | Description |
|---|---|---|
| 1 | **Economy-wide time series** | Unemployment rate, imported-worker ratio, GDP growth, inflation, and HIBOR 3M in a 5-panel stacked plot. |
| 2 | **Sector unemployment time series** | All four sector unemployment rates on one chart, with the economy-wide rate as a dashed reference. |
| 3 | **Annual imported-worker approvals by sector** | Grouped bar chart showing each sector's annual approvals side-by-side. |
| 4 | **Correlation heat-map** | Pearson correlation matrix for the six Model 1 variables (unemployment, IW ratio, GDP growth, inflation, interest rate, LFPR). |
| 5 | **Pairwise scatter matrix** | Every variable pair as a scatter plot; diagonal panels show histograms. |
| 6 | **IW ratio vs unemployment** | Scatter plot coloured by year with an OLS trend line, illustrating the raw association between foreign-labour intensity and unemployment. |

Run from the project root; no additional dependencies beyond `numpy` and `matplotlib` are required.
