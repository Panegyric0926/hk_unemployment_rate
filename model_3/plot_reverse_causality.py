"""
Reverse-Causality Visualisation

Shows that imported-worker approvals are demand-driven: workers are imported
when unemployment is already low (boom), and approvals collapse when
unemployment is high (bust).

Produces a dual-axis time-series plot for each sector with:
    - Left axis:  Unemployment rate (%)
    - Right axis: Imported workers approved (quarterly)

The two series move in opposite directions, illustrating why the naïve
Model 3 gamma1 is negative.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL2_DIR = BASE_DIR.parent / "model_2"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}

SECTOR_CONFIG = {
    "Construction": {
        "folder": "construction",
        "unemployment_filter": None,
    },
    "Retail": {
        "folder": "retail",
        "unemployment_filter": None,
    },
    "Transportation": {
        "folder": "transportation",
        "unemployment_filter": None,
    },
    "Community": {
        "folder": "community",
        "unemployment_filter": "Public administration, social and personal services",
    },
}


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


def quarter_to_label(year, quarter):
    return f"{year}\nQ{quarter}" if quarter == 1 else f"Q{quarter}"


def quarter_to_float(year, quarter):
    return year + (quarter - 1) / 4.0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Reverse Causality: Imported Workers Are Approved When Unemployment Is Already Low",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    colors_unemp = "#d62728"   # red
    colors_import = "#1f77b4"  # blue

    for ax, (sector_name, config) in zip(axes.flat, SECTOR_CONFIG.items()):
        unemployment = read_unemployment(config["folder"], config["unemployment_filter"])
        imported = read_imported_workers(config["folder"])

        common_keys = sorted(set(unemployment) & set(imported))

        x_vals = [quarter_to_float(y, q) for y, q in common_keys]
        unemp_vals = [unemployment[k] for k in common_keys]
        import_vals = [imported[k] for k in common_keys]

        # Left axis: unemployment
        ax.set_title(sector_name, fontsize=12, fontweight="bold")
        ax.plot(x_vals, unemp_vals, color=colors_unemp, linewidth=1.8,
                label="Unemployment Rate (%)", zorder=3)
        ax.set_ylabel("Unemployment Rate (%)", color=colors_unemp, fontsize=9)
        ax.tick_params(axis="y", labelcolor=colors_unemp, labelsize=8)
        ax.set_xlabel("")

        # Shade COVID period
        ax.axvspan(2020.0, 2022.0, alpha=0.08, color="gray", label="COVID period")

        # Shade ESDS period (Enhanced Supplementary Labour Scheme expansion)
        ax.axvspan(2023.0, 2025.75, alpha=0.08, color="blue", label="Post-ESDS expansion")

        # Right axis: imported workers
        ax2 = ax.twinx()
        ax2.bar(x_vals, import_vals, width=0.2, alpha=0.5, color=colors_import,
                label="Imported Workers (quarterly)", zorder=2)
        ax2.set_ylabel("Imported Workers (quarterly approvals)", color=colors_import, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=colors_import, labelsize=8)

        # X ticks: show only Q1 of each year
        all_years = sorted(set(y for y, _ in common_keys))
        tick_positions = [quarter_to_float(y, 1) for y in all_years]
        tick_labels = [str(y) for y in all_years]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)

        # Add annotation arrows for key periods
        max_unemp_key = max(common_keys, key=lambda k: unemployment[k])
        min_unemp_key = None
        for k in common_keys:
            if k[0] >= 2023:
                if min_unemp_key is None or unemployment[k] < unemployment[min_unemp_key]:
                    min_unemp_key = k

        # Annotate peak unemployment
        peak_x = quarter_to_float(*max_unemp_key)
        peak_y = unemployment[max_unemp_key]
        peak_iw = imported[max_unemp_key]
        ax.annotate(
            f"High unemp ({peak_y}%)\nLow approvals ({peak_iw:.0f})",
            xy=(peak_x, peak_y),
            xytext=(peak_x + 0.8, peak_y + 0.5),
            fontsize=7,
            color=colors_unemp,
            arrowprops=dict(arrowstyle="->", color=colors_unemp, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors_unemp, alpha=0.8),
        )

        # Annotate boom period with high approvals
        if min_unemp_key is not None:
            boom_x = quarter_to_float(*min_unemp_key)
            boom_y = unemployment[min_unemp_key]
            boom_iw = imported[min_unemp_key]
            ax2.annotate(
                f"Low unemp ({boom_y}%)\nHigh approvals ({boom_iw:.0f})",
                xy=(boom_x, boom_iw),
                xytext=(boom_x - 1.5, boom_iw * 0.85),
                fontsize=7,
                color=colors_import,
                arrowprops=dict(arrowstyle="->", color=colors_import, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors_import, alpha=0.8),
            )

        ax.grid(axis="y", alpha=0.3)

    # Shared legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors_unemp, lw=1.8, label="Unemployment Rate (%)"),
        Patch(facecolor=colors_import, alpha=0.5, label="Imported Workers (quarterly)"),
        Patch(facecolor="gray", alpha=0.15, label="COVID period (2020–2022)"),
        Patch(facecolor="blue", alpha=0.15, label="Post-ESDS expansion (2023+)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_path = BASE_DIR / "plot_reverse_causality.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
