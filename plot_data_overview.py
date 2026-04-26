"""
Data Overview Visualisations

Produces a suite of exploratory data science plots from all available
data in this repository:

  1. Time-series panel  – Aggregate unemployment, GDP growth, inflation,
                          interest rate, and imported workers (economy-wide).
  2. Sector unemployment time series  – All four sector unemployment rates
                                        on one chart.
  3. Sector imported-workers bar chart  – Annual approvals by sector.
  4. Correlation heat-map  – Pearson correlations for the Model 1 regressors.
  5. Pairwise scatter matrix  – Scatter plots for every pair of Model 1
                                 variables.
  6. Imported-workers vs unemployment scatter  – Economy-wide,
                                                  coloured by time period.

Run from the project root:
    python plot_data_overview.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR = Path(__file__).resolve().parent
MODEL2_DIR = BASE_DIR / "model_2"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}
MONTH_STR_TO_Q = {
    "Jan": 1, "Feb": 1, "Mar": 1,
    "Apr": 2, "May": 2, "Jun": 2,
    "Jul": 3, "Aug": 3, "Sep": 3,
    "Oct": 4, "Nov": 4, "Dec": 4,
}

SECTOR_COLORS = {
    "Construction": "#e29578",
    "Retail": "#89c6ac",
    "Transportation": "#19929d",
    "Community": "#5cb0cf",
}

MACRO_COLORS = {
    "Imported Worker Ratio": SECTOR_COLORS["Construction"],
    "GDP Year-on-Year Growth": SECTOR_COLORS["Retail"],
    "Quarterly Inflation Rate": SECTOR_COLORS["Community"],
    "HIBOR 3-Month Interest Rate": SECTOR_COLORS["Transportation"],
}

HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "sector_diverging",
    [SECTOR_COLORS["Construction"], "#f7f4ef", SECTOR_COLORS["Retail"]],
)


# ── helpers ───────────────────────────────────────────────────────────────────

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
    q = MONTH_TO_Q.get(month)
    return (year, q) if q else None


def parse_yyyyqq(value):
    parts = (value or "").strip().split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1][1])
    except (ValueError, IndexError):
        return None


def key_to_float(key):
    year, q = key
    return year + (q - 1) / 4.0


# ── 1. Economy-wide data ──────────────────────────────────────────────────────

unemp_agg = {}
emp_agg = {}
lfpr_agg = {}
with (BASE_DIR / "unemployment_rate.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        k = parse_month_range((row.get("Time") or "").strip())
        if k is None:
            continue
        try:
            unemp_agg[k] = float((row.get("Unemployment Rate Both") or "").strip())
            emp_agg[k] = float((row.get("Employed Persons Both") or "").strip())
            lfpr_agg[k] = float((row.get("Labour Force Participation Rate Both") or "").strip())
        except ValueError:
            pass

imported_agg = {}
with (BASE_DIR / "imported_workers.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        y_str = (row.get("Year") or "").strip()
        v = (row.get("Number of Imported Workers Approved") or "").strip()
        if y_str.isdigit() and v:
            year = int(y_str)
            for q in range(1, 5):
                imported_agg[(year, q)] = float(v) / 4.0

gdp_growth = {}
with (BASE_DIR / "gdp.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        k = parse_yyyyqq((row.get("Time") or "").strip())
        v = (row.get("GDP Year-on-year Change") or "").strip()
        if k and v:
            try:
                gdp_growth[k] = float(v.lstrip("+"))
            except ValueError:
                pass

inflation = {}
with (BASE_DIR / "inflation_rate.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        k = parse_yyyyqq((row.get("Quarter") or "").strip())
        v = (row.get("Quarterly Inflation Rate (%)") or "").strip()
        if k and v:
            try:
                inflation[k] = float(v)
            except ValueError:
                pass

interest_rate = {}
with (BASE_DIR / "interest_rate.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        k = parse_yyyyqq((row.get("Time") or "").strip())
        v = (row.get("3 months") or "").strip()
        if k and v:
            try:
                interest_rate[k] = float(v)
            except ValueError:
                pass

# Build aligned Model 1 dataset
m1_keys = sorted(
    set(unemp_agg) & set(emp_agg) & set(imported_agg)
    & set(gdp_growth) & set(inflation) & set(interest_rate)
)
m1_t = [key_to_float(k) for k in m1_keys]
m1_unemp = [unemp_agg[k] for k in m1_keys]
m1_iw = [(imported_agg[k] / (emp_agg[k] * 1_000)) * 100.0 for k in m1_keys]
m1_gdp = [gdp_growth[k] for k in m1_keys]
m1_infl = [inflation[k] for k in m1_keys]
m1_ir = [interest_rate[k] for k in m1_keys]
m1_lfpr = [lfpr_agg[k] for k in m1_keys]


# ── 2. Sector unemployment ────────────────────────────────────────────────────

SECTOR_UNEMPLOYMENT_FILTERS = {
    "Construction": (MODEL2_DIR / "construction" / "unemployment_rate.csv", None),
    "Retail": (MODEL2_DIR / "retail" / "unemployment_rate.csv", None),
    "Transportation": (MODEL2_DIR / "transportation" / "unemployment_rate.csv", None),
    "Community": (
        MODEL2_DIR / "community" / "unemployment_rate.csv",
        "Public administration, social and personal services",
    ),
}

sector_unemp = {}
for sector, (path, filter_val) in SECTOR_UNEMPLOYMENT_FILTERS.items():
    data = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if filter_val is not None:
                if (row.get("Detailed previous industry") or "").strip() != filter_val:
                    continue
            k = parse_month_range((row.get("Time") or "").strip())
            v = (row.get("Unemployment Rate") or "").strip()
            if k and v:
                try:
                    data[k] = float(v)
                except ValueError:
                    pass
    sector_unemp[sector] = data


# ── 3. Sector imported workers (annual totals) ────────────────────────────────

SECTOR_FOLDERS = ["construction", "retail", "transportation", "community"]
sector_iw_annual = {}
for folder in SECTOR_FOLDERS:
    annual = {}
    path = MODEL2_DIR / folder / "imported_workers.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_str = (row.get("Year") or "").strip()
            v = (row.get("Number of Imported Workers Approved") or "").strip()
            if y_str.isdigit() and v:
                try:
                    annual[int(y_str)] = float(v)
                except ValueError:
                    pass
    sector_iw_annual[folder.capitalize()] = annual


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Time-series panel (economy-wide macro + imported workers)
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(5, 1, figsize=(16, 30), sharex=True, constrained_layout=True)

# Helper function to apply styling to all plots to keep code clean
def style_ax(ax, title, ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    # This adds 10% padding to the top/bottom so lines aren't cut off
    ax.margins(y=0.1)

axes1[0].plot(m1_t, m1_unemp, color="steelblue", linewidth=1.5)
style_ax(axes1[0], "Economy-Wide Unemployment Rate", "Unemployment (%)")

axes1[1].bar(
    [k[0] + (k[1] - 1) / 4 for k in m1_keys],
    m1_iw,
    width=0.2,
    color=MACRO_COLORS["Imported Worker Ratio"],
    alpha=0.8,
)
style_ax(axes1[1], "Imported Worker Ratio (Economy-Wide)", "IW / Total (%)")
axes1[1].grid(True, alpha=0.3)

axes1[2].plot(
    m1_t,
    m1_gdp,
    color=MACRO_COLORS["GDP Year-on-Year Growth"],
    linewidth=1.5,
)
axes1[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
style_ax(axes1[2], "GDP Year-on-Year Growth", "GDP YoY Change (%)")
axes1[2].set_title("GDP Year-on-Year Growth")
axes1[2].grid(True, alpha=0.3)

axes1[3].plot(
    m1_t,
    m1_infl,
    color=MACRO_COLORS["Quarterly Inflation Rate"],
    linewidth=1.5,
)
axes1[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
style_ax(axes1[3], "Quarterly Inflation Rate", "Inflation (%)")

axes1[4].plot(
    m1_t,
    m1_ir,
    color=MACRO_COLORS["HIBOR 3-Month Interest Rate"],
    linewidth=1.5,
)
axes1[4].axhline(0, color="black", linewidth=0.8, linestyle="--")
style_ax(axes1[4], "HIBOR 3-Month Interest Rate", "HIBOR 3M (%)")
# axes1[4].set_xlabel("Year")
axes1[4].grid(True, alpha=0.3)

# fig1.suptitle("Economy-Wide Time Series Overview", fontsize=13)
#plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Sector unemployment time series (all four sectors + economy-wide)
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(12, 5))

# Economy-wide as reference
agg_t = [key_to_float(k) for k in sorted(unemp_agg)]
agg_u = [unemp_agg[k] for k in sorted(unemp_agg)]
ax2.plot(agg_t, agg_u, color="black", linewidth=1.2, linestyle="--",
         label="Economy-Wide", zorder=5)

for sector, data in sector_unemp.items():
    if not data:
        continue
    keys_s = sorted(data)
    t_s = [key_to_float(k) for k in keys_s]
    u_s = [data[k] for k in keys_s]
    ax2.plot(t_s, u_s, linewidth=1.5, color=SECTOR_COLORS[sector], label=sector)

ax2.set_xlabel("Year")
ax2.set_ylabel("Unemployment Rate (%)")
ax2.set_title("Unemployment Rate by Sector and Economy-Wide")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Annual imported-workers approvals by sector (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════════════

all_years_iw = sorted(
    set(y for v in sector_iw_annual.values() for y in v)
)
sector_names_iw = list(sector_iw_annual.keys())
n_sectors = len(sector_names_iw)
x_pos = np.arange(len(all_years_iw))
bar_width = 0.8 / n_sectors

fig3, ax3 = plt.subplots(figsize=(12, 5))
for j, (sector, annual) in enumerate(sector_iw_annual.items()):
    values = [annual.get(y, 0) for y in all_years_iw]
    ax3.bar(
        x_pos + j * bar_width - 0.4 + bar_width / 2,
        values,
        width=bar_width,
        label=sector,
        color=SECTOR_COLORS[sector],
        alpha=0.85,
    )

ax3.set_xticks(x_pos)
ax3.set_xticklabels([str(y) for y in all_years_iw], rotation=45, ha="right")
ax3.set_xlabel("Year")
ax3.set_ylabel("Imported Workers Approved")
ax3.set_title("Annual Imported Worker Approvals by Sector")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 – Correlation heat-map (Model 1 variables)
# ══════════════════════════════════════════════════════════════════════════════

var_names = [
    "Unemp\nRate",
    "IW\nRatio%",
    "GDP\nGrowth",
    "Inflation",
    "Interest\nRate",
    "LFPR",
]
data_matrix = np.array([m1_unemp, m1_iw, m1_gdp, m1_infl, m1_ir, m1_lfpr])
corr_matrix = np.corrcoef(data_matrix)

fig4, ax4 = plt.subplots(figsize=(7, 6))
cax = ax4.imshow(corr_matrix, vmin=-1, vmax=1, cmap=HEATMAP_CMAP, aspect="auto")
plt.colorbar(cax, ax=ax4, label="Pearson r")

ax4.set_xticks(range(len(var_names)))
ax4.set_yticks(range(len(var_names)))
ax4.set_xticklabels(var_names, fontsize=9)
ax4.set_yticklabels(var_names, fontsize=9)

for row in range(len(var_names)):
    for col in range(len(var_names)):
        val = corr_matrix[row, col]
        text_color = "white" if abs(val) > 0.6 else "black"
        ax4.text(col, row, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, color=text_color)

ax4.set_title("Correlation Heat-Map – Model 1 Variables")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Pairwise scatter matrix (Model 1 variables)
# ══════════════════════════════════════════════════════════════════════════════

var_labels = ["Unemp Rate", "IW Ratio%", "GDP Growth",
              "Inflation", "Interest Rate", "LFPR"]
data_cols = [np.array(m1_unemp), np.array(m1_iw), np.array(m1_gdp),
             np.array(m1_infl), np.array(m1_ir), np.array(m1_lfpr)]
nv = len(var_labels)

fig5, axes5 = plt.subplots(nv, nv, figsize=(12, 11))
for r in range(nv):
    for c in range(nv):
        ax = axes5[r][c]
        if r == c:
            ax.hist(data_cols[r], bins=15, color="steelblue", alpha=0.7)
        else:
            ax.scatter(data_cols[c], data_cols[r], s=8, alpha=0.5, color="steelblue")
        if r == nv - 1:
            ax.set_xlabel(var_labels[c], fontsize=7)
        if c == 0:
            ax.set_ylabel(var_labels[r], fontsize=7)
        ax.tick_params(labelsize=6)

fig5.suptitle("Pairwise Scatter Matrix – Model 1 Variables", fontsize=12)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 – Imported-worker ratio vs unemployment (coloured by time period)
# ══════════════════════════════════════════════════════════════════════════════

iw_arr = np.array(m1_iw)
un_arr = np.array(m1_unemp)
t_arr = np.array(m1_t)

# Fit a simple trend line
poly = np.polyfit(iw_arr, un_arr, 1)

fig6, ax6 = plt.subplots(figsize=(8, 5))
sc = ax6.scatter(iw_arr, un_arr, c=t_arr, cmap="viridis", s=40, alpha=0.8, zorder=3)
plt.colorbar(sc, ax=ax6, label="Year")
x_fit = np.linspace(iw_arr.min(), iw_arr.max(), 200)
ax6.plot(x_fit, np.polyval(poly, x_fit), color="red", linewidth=1.5, linestyle="--",
         label=f"OLS trend  slope={poly[0]:.2f}")
ax6.set_xlabel("Imported Worker Ratio (% of Employment)")
ax6.set_ylabel("Unemployment Rate (%)")
ax6.set_title("Imported Worker Ratio vs Economy-Wide Unemployment\n(colour = year)")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
