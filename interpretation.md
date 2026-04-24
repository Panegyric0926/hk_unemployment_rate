# Regression Results and Interpretation

This document summarizes and interprets the regression outputs in:

- [output/model1.txt](output/model1.txt)
- [output/model_2_community.txt](output/model_2_community.txt)
- [output/model_2_construction.txt](output/model_2_construction.txt)
- [output/model_2_retail.txt](output/model_2_retail.txt)
- [output/model_2_transportation.txt](output/model_2_transportation.txt)
- [output/model_3_macro.txt](output/model_3_macro.txt)

Unless stated otherwise, coefficients should be read as **conditional associations**, not as proof of causality. A positive coefficient means higher unemployment is associated with a higher value of that regressor after controlling for the other variables in the same model. A negative coefficient means the opposite.

Significance guide used below:

- `***` for $p < 0.01$
- `**` for $p < 0.05$
- `*` for $p < 0.10$

---

## Model 1: Economy-Wide Baseline

Source: [output/model1.txt](output/model1.txt)

### Documented result

| Item | Value |
|---|---|
| Sample | 36 quarters, 2017 Q1 to 2025 Q4 |
| Dependent variable | Economy-wide unemployment rate |
| $R^2$ | 0.6829 |
| Adjusted $R^2$ | 0.6300 |
| F-statistic | 12.9213, $p < 0.001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| ImportedWorkers/TotalEmp $\times 100\%$ | -2.2023 | 0.1076 | Higher imported-worker share is associated with lower unemployment, but the estimate is not statistically reliable at conventional levels. |
| GDP Growth | -0.0695 | 0.0194 | Stronger GDP growth is associated with lower unemployment. |
| Inflation Rate | -0.1120 | 0.5292 | Negative sign, but no reliable relationship in this sample. |
| Interest Rate 3M | -0.6133 | 0.0000 | Higher short-term interest rates are associated with lower unemployment in the sample period. |
| LFPR | -0.5463 | 0.0003 | Higher labour-force participation is associated with lower unemployment. |

### Interpretation

At the aggregate level, the model fits the data reasonably well: about 68% of the variation in unemployment is explained by the included regressors. The imported-worker ratio is **negative but not statistically significant**, so Model 1 does **not** provide strong evidence that a higher economy-wide imported-worker share raises unemployment.

The more robust signals come from the macro controls. GDP growth is significantly negative, which is consistent with unemployment falling when the economy expands. The interest-rate coefficient is also significantly negative in this sample, which is less standard from a macroeconomic perspective and likely reflects the particular timing of Hong Kong's post-pandemic recovery rather than a simple structural effect. LFPR is significantly negative as well, suggesting that quarters with stronger labour-market participation were also quarters with lower unemployment.

Substantively, Model 1 points to a broad conclusion: **macro conditions appear to explain economy-wide unemployment better than the imported-worker ratio does**.

### Diagnostic note

Durbin-Watson is 1.053, which suggests positive serial correlation in the residuals. The Jarque-Bera test is significant, so residual normality is imperfect. The condition number is also fairly high, so coefficient precision may be affected by scaling or collinearity. These diagnostics do not invalidate the model, but they do mean the coefficient estimates should be interpreted with caution.

---

## Model 2: Community Sector

Source: [output/model_2_community.txt](output/model_2_community.txt)

### Documented result

| Item | Value |
|---|---|
| Sample | 32 quarters, 2018 Q1 to 2025 Q4 |
| Dependent variable | Community unemployment rate |
| $R^2$ | 0.6697 |
| Adjusted $R^2$ | 0.5904 |
| F-statistic | 8.4476, $p < 0.001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| Imported Workers | 0.000826 | 0.0446 | More imported workers are associated with higher community-sector unemployment. |
| GDP Growth | -0.0454 | 0.0393 | Stronger aggregate growth is associated with lower community unemployment. |
| Inflation Rate | -0.0205 | 0.8667 | No meaningful relationship detected. |
| Interest Rate 3M | -0.3254 | 0.0013 | Higher rates are associated with lower community unemployment in the sample. |
| Government Expenditure | 0.000708 | 0.2026 | Positive sign, but not statistically reliable. |
| Above65 Population | -0.000987 | 0.6335 | No meaningful relationship detected. |

### Interpretation

This sector model suggests a **positive and statistically significant association** between imported workers and unemployment in community-related services. Given the variable scaling, the effect is economically small per worker but becomes more visible over larger changes in imported-worker approvals. This is evidence that, within this sector, imported labour moved with higher unemployment rather than lower unemployment.

However, the result should not be read mechanically as displacement. A positive coefficient can also arise if imported-worker approvals increase during periods when the sector is already under staffing or restructuring pressure. In other words, imported labour may be responding to sector stress rather than causing unemployment by itself.

GDP growth remains protective, and the interest-rate coefficient is again negative and significant. The two community-specific controls, government expenditure and population aged 65+, are not significant, which suggests that most explanatory power here comes from the imported-worker series plus macro conditions rather than from the added demographic and fiscal controls.

### Diagnostic note

Residual normality looks acceptable here, but Durbin-Watson is only 1.135, so some positive autocorrelation likely remains. The condition number is high, which means the scale and overlap of regressors may still reduce coefficient stability.

---

## Model 2: Construction Sector

Source: [output/model_2_construction.txt](output/model_2_construction.txt)

### Documented result

| Item | Value |
|---|---|
| Sample | 32 quarters, 2017 Q1 to 2024 Q4 |
| Dependent variable | Construction unemployment rate |
| $R^2$ | 0.6875 |
| Adjusted $R^2$ | 0.5964 |
| F-statistic | 7.5438, $p = 0.0001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| ImportedWorkers per 1K | 0.3706 | 0.0397 | Higher imported-worker intensity is associated with higher construction unemployment. |
| GDP Growth | -0.1829 | 0.0214 | Stronger GDP growth is associated with lower construction unemployment. |
| Inflation Rate | -0.3432 | 0.3733 | No meaningful relationship detected. |
| Interest Rate 3M | -0.8792 | 0.0114 | Higher rates are associated with lower construction unemployment in the sample. |
| Gross Value | 0.000004 | 0.9651 | No detectable relationship. |
| Compensation | -0.000612 | 0.0890 | Weak evidence that higher compensation is associated with lower unemployment. |
| Actual Expenditure | 0.000109 | 0.2720 | No reliable relationship detected. |

### Interpretation

Construction shows one of the clearest positive imported-worker coefficients among the sector models. The sign is positive and statistically significant, which means that quarters with a higher imported-worker count per 1,000 workers also tended to have higher construction unemployment after controlling for macro conditions and sector activity.

This is consistent with a labour-supply competition story, but the sector is also exposed to strong timing and policy effects. Imported workers may be brought in precisely when project pipelines change, skill bottlenecks widen, or firms are adjusting their labour mix. The coefficient therefore supports a positive association, but it still does not cleanly separate cause from policy response.

GDP growth again reduces unemployment, while the interest-rate coefficient is significantly negative. Among the construction-specific controls, only compensation is even weakly significant, with a negative sign that is intuitive: better-paid construction periods may coincide with tighter labour demand and lower unemployment.

### Diagnostic note

This model has the most serious stability warning among the Model 2 sector runs. Durbin-Watson is 0.896, residual normality is rejected, and the condition number is extremely high at $1.84 \times 10^6$. The signs are still informative, but the exact magnitudes should be treated cautiously because multicollinearity and scaling problems are likely present.

---

## Model 2: Retail Sector

Source: [output/model_2_retail.txt](output/model_2_retail.txt)

### Documented result

| Item | Value |
|---|---|
| Sample | 36 quarters, 2017 Q1 to 2025 Q4 |
| Dependent variable | Retail unemployment rate |
| $R^2$ | 0.8620 |
| Adjusted $R^2$ | 0.8334 |
| F-statistic | 30.1855, $p < 0.001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| Imported Workers | 0.000387 | 0.0245 | More imported workers are associated with higher retail unemployment. |
| GDP Growth | -0.0633 | 0.1866 | Negative sign, but not statistically reliable. |
| Inflation Rate | -0.0158 | 0.9467 | No meaningful relationship detected. |
| Interest Rate 3M | -0.6241 | 0.0006 | Higher rates are associated with lower retail unemployment in the sample. |
| Retail Index | 0.0321 | 0.2993 | No reliable relationship detected. |
| Reception Index Avg | -0.1181 | 0.0007 | Better reception activity is associated with lower retail unemployment. |

### Interpretation

Retail has the strongest overall model fit among the Model 2 sector regressions, with an $R^2$ above 0.86. The imported-worker coefficient is positive and statistically significant, indicating that higher imported-worker approvals are associated with higher retail unemployment after controlling for macro and sector demand variables.

The most important sector-specific control is the reception index average, which is strongly negative. That is an intuitive result: when retail, accommodation, and food-service activity improves, unemployment in the broader retail-related sector falls. By contrast, the retail sales index itself is not significant once the other controls are in the model.

This combination suggests that labour demand linked to consumer-facing service activity matters more than the goods-sales index alone, while imported workers still retain a positive partial association with unemployment. The retail result therefore supports the view that foreign labour and domestic unemployment can move together in this sector, especially when broader service conditions are weak.

### Diagnostic note

The residual distribution looks well behaved relative to the other models, although Durbin-Watson at 1.254 still suggests some positive autocorrelation. The condition number is elevated but not as extreme as in Construction or Transportation.

---

## Model 2: Transportation Sector

Source: [output/model_2_transportation.txt](output/model_2_transportation.txt)

### Documented result

| Item | Value |
|---|---|
| Sample | 36 quarters, 2017 Q1 to 2025 Q4 |
| Dependent variable | Transportation unemployment rate |
| $R^2$ | 0.8540 |
| Adjusted $R^2$ | 0.8237 |
| F-statistic | 28.2604, $p < 0.001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| Imported Workers | 0.000784 | 0.0789 | Positive association, but only marginally significant. |
| GDP Growth | -0.0266 | 0.3760 | No reliable relationship detected. |
| Inflation Rate | 0.0916 | 0.5041 | No meaningful relationship detected. |
| Interest Rate 3M | -0.3451 | 0.0037 | Higher rates are associated with lower transportation unemployment in the sample. |
| Air Cargo Output | 0.001909 | 0.3184 | No reliable relationship detected. |
| Total Tourists | -0.0000001724 | 0.0000 | Higher tourist arrivals are strongly associated with lower transportation unemployment. |

### Interpretation

Transportation tells a slightly different story from the other sectors. The imported-worker coefficient is still positive, but it is only significant at the 10% level. That makes the evidence suggestive rather than firm. On its own, this does not justify a strong claim that imported workers systematically raise transportation unemployment.

The dominant variable here is tourism. The coefficient on total tourists is strongly negative and highly significant, which is exactly what would be expected in a sector tied to travel, mobility, and visitor demand. Once tourist arrivals recover, transportation unemployment falls sharply.

This means the transportation labour market appears to be driven more by demand recovery than by imported labour. The imported-worker effect may exist, but it is secondary compared with the tourism channel in this specification.

### Diagnostic note

Durbin-Watson rises to 1.551, which is better than in the other sector models but still short of the ideal benchmark near 2. The condition number is extremely high at $2.54 \times 10^8$, so the exact coefficient magnitudes are again vulnerable to scaling and multicollinearity concerns.

---

## Model 3: Cross-Sector Heterogeneity with Macro Controls

Source: [output/model_3_macro.txt](output/model_3_macro.txt)

This output contains two pieces of information:

1. A full sector-level OLS report for **Community**.
2. A comparison table that reports the key `gamma1` intensity coefficient for all four sectors.

### Community detailed macro-controlled result

| Item | Value |
|---|---|
| Sample | 36 quarters, 2017 Q1 to 2025 Q4 |
| Dependent variable | Community unemployment rate |
| Key regressor | Foreign-labour intensity |
| $R^2$ | 0.5275 |
| Adjusted $R^2$ | 0.4665 |
| F-statistic | 8.6507, $p = 0.0001$ |

| Variable | Coefficient | p-value | Interpretation of sign |
|---|---:|---:|---|
| Intensity | 1.2294 | 0.0107 | Higher foreign-labour intensity is associated with higher unemployment in Community. |
| GDP Growth | -0.0575 | 0.0064 | Stronger GDP growth is associated with lower unemployment. |
| Inflation Rate | -0.0745 | 0.5689 | No reliable relationship detected. |
| Interest Rate 3M | -0.2699 | 0.0002 | Higher rates are associated with lower unemployment in the sample. |

### Interpretation

The Community sector remains positive and significant even when the imported-worker variable is normalized as **foreign-labour intensity** rather than entered as a raw level. This matters because it shows the positive association is not just a scale effect. Relative exposure to imported labour is itself associated with higher unemployment in the sector.

At the same time, the explanatory power is lower than in the richer Model 2 sector regressions, which is expected because Model 3 Macro controls only for economy-wide conditions and does not include sector-specific activity variables.

### Cross-sector comparison summary

| Sector | `gamma1` on intensity | p-value | Significance | $R^2$ | Adjusted $R^2$ | Interpretation |
|---|---:|---:|---|---:|---:|---|
| Community | 1.2294 | 0.0107 | `**` | 0.5275 | 0.4665 | Strong positive association between foreign-labour intensity and unemployment. |
| Transportation | 0.7327 | 0.0352 | `**` | 0.4335 | 0.3604 | Positive association exists, but the model fit is weaker than in other sectors. |
| Construction | 0.3668 | 0.1988 | none | 0.5745 | 0.5195 | Positive sign, but not statistically reliable once only macro controls are used. |
| Retail | 0.2492 | 0.0199 | `**` | 0.6290 | 0.5812 | Positive and statistically significant, though smaller than Community and Transportation. |

### Interpretation of the cross-sector summary

The comparison table is the clearest evidence of **sector heterogeneity**. The estimated unemployment effect of foreign-labour intensity is largest in Community, next in Transportation, then Construction, and smallest in Retail.

Three sectors, Community, Transportation, and Retail, show a statistically significant positive `gamma1`. Construction stays positive but loses significance, which implies that its imported-labour relationship is more sensitive to model specification. In practical terms, the unemployment association with foreign-labour intensity is not uniform across the economy. It appears strongest in service sectors where staffing composition and labour-market segmentation may be more important.

### Diagnostic note

For the detailed Community regression, the condition number is low and stable relative to the Model 2 sector equations, but Durbin-Watson is only 0.775, which suggests notable positive autocorrelation. The coefficient signs are still useful, but standard OLS inference should be read conservatively.

---

## Overall Interpretation Across All Attached Results

Three broad patterns emerge from the attached outputs.

### 1. Imported labour does not show a strong unemployment effect at the aggregate level

In Model 1, the economy-wide imported-worker ratio is not statistically significant. This means the data do not support a strong claim that imported labour raises unemployment across the whole labour market once GDP growth, inflation, interest rates, and LFPR are controlled for.

### 2. Sector-level results are more positive and more heterogeneous

In Model 2, the imported-worker coefficient is positive and significant in Community, Construction, and Retail, and positive but only marginal in Transportation. In Model 3 Macro, foreign-labour intensity is significantly positive in Community, Transportation, and Retail. This suggests that the labour-market relationship is **sector-specific rather than economy-wide**.

### 3. Demand conditions still matter a great deal

Sector demand indicators remain important. Reception activity matters in Retail, tourist arrivals matter strongly in Transportation, and GDP growth is often negative and significant across models. A sensible reading is that unemployment is shaped by both labour supply conditions and sector demand, with their relative importance differing by industry.

---

## Cautions and Limitations

- These are OLS associations, not causal estimates.
- Sample sizes are small, mostly 32 to 36 quarterly observations.
- Several models show positive serial correlation, which can understate standard errors.
- Some sector equations have very large condition numbers, especially Construction and Transportation, so coefficient magnitudes may be unstable.
- Imported-worker counts are based on annual approvals spread across quarters, which introduces measurement simplification.

Taken together, the attached results support a balanced conclusion: **imported labour is not a robust predictor of aggregate unemployment, but it is positively associated with unemployment in several individual sectors, especially Community and Retail, while Transportation is more strongly tied to tourism demand and Construction is sensitive to specification.**