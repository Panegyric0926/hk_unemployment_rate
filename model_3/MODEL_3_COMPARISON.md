# Model 3 Variants — Comparison

This document explains the three variants of the **Cross-Sector Heterogeneity Model** (Model 3) and the methodological differences between them.

---

## Overview

All three models share the same research question:

> Does foreign-labour intensity affect unemployment differently across sectors (Construction, Retail, Transportation, Community)?

They all compute **foreign-labour intensity** the same way:

$$
\text{Intensity}_{s,t} = \frac{\text{ImportedWorkers}_{s,t}}{\text{TotalSectorEmployment}_{s,t}}
$$

and all fit a regression with `gamma1` as the key coefficient measuring intensity's effect.

The critical difference is **what each model uses as the dependent variable** and **how it controls for macroeconomic conditions**.

---

## Side-by-Side Comparison

| Dimension | `model_3.py` | `model_3_macro.py` | `model_3_enhance.py` |
|---|---|---|---|
| **Dependent variable** | Sector unemployment rate | Sector unemployment rate | **Model 2 residuals** (unexplained unemployment) |
| **Macro controls** | None | GDP growth, Inflation, Interest rate | Embedded inside Model 2 step |
| **Sector-specific controls** | None | None | Embedded inside Model 2 step |
| **Estimation steps** | 1 (single OLS) | 1 (single OLS) | **2** (re-run Model 2, then OLS on residuals) |
| **Confounding addressed** | None | Business cycle only | Business cycle + sector activity |
| **Complexity** | Lowest | Medium | Highest |

---

## Model Details

### `model_3.py` — Simple Baseline

**Equation:**

$$
\text{UnemploymentRate}_{s,t} = \gamma_0 + \gamma_1 \cdot \text{Intensity}_{s,t} + \varepsilon_{s,t}
$$

**What it does:**  
Directly regresses the raw unemployment rate on foreign-labour intensity, with no controls at all.

**Strength:** Simple and transparent.  

---

### `model_3_macro.py` — Macro-Controlled

**Equation:**

$$
\text{UnemploymentRate}_{s,t} = \gamma_0 + \gamma_1 \cdot \text{Intensity}_{s,t} + \gamma_2 \cdot \text{GDPGrowth}_t + \gamma_3 \cdot \text{Inflation}_t + \gamma_4 \cdot \text{InterestRate}_t + \varepsilon_{s,t}
$$

**What it does:**  
Adds GDP growth, inflation, and the 3-month interest rate as covariates in a single OLS regression. This removes the common business cycle from `gamma1`.

**Strength:** Accounts for macro-cycle confounding without requiring a two-step procedure.  

---

### `model_3_enhance.py` — Residual-Based (Two-Step)

**Step 1 — Re-run each sector's Model 2:**

$$
\text{UnemploymentRate}_{s,t} = \beta_0 + \beta_1 \cdot \text{ImportedWorkers}_{s,t} + \text{MacroControls}_t + \text{SectorControls}_{s,t} + u_{s,t}
$$

Compute the residual:

$$
\hat{u}_{s,t} = \text{Actual}_{s,t} - \hat{\text{Predicted}}_{s,t}
$$

**Step 2 — Regress residuals on intensity:**

$$
\hat{u}_{s,t} = \gamma_0 + \gamma_1 \cdot \text{Intensity}_{s,t} + \varepsilon_{s,t}
$$

**What it does:**  
The Model 2 residual is unemployment that cannot be explained by macro conditions *and* by sector-specific activity variables (e.g. gross construction value, retail sales index, tourist arrivals, elderly population). The intensity effect `gamma1` therefore operates on **pure unexplained variation**.

**Strength:** Most rigorous control for confounding. `gamma1` isolates the labour-supply competition channel from both macro and sector noise.  