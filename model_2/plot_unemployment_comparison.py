from pathlib import Path
import csv

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "unemployment_rate_comparison.png"
COMMUNITY_INDUSTRY = "Human health and social work activities"

MONTH_TO_Q = {1: 1, 4: 2, 7: 3, 10: 4}


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


def quarter_sort_key(label):
    year_text, quarter_text = label.split()
    return int(year_text), int(quarter_text[1:])


def normalize_industry(value):
    return " ".join((value or "").split())


def read_simple_sector_series(file_path):
    series = {}
    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_value = (row.get("Time") or "").strip()
            rate_value = (row.get("Unemployment Rate") or "").strip()
            key = parse_month_range(time_value)
            if key is None or not rate_value:
                continue
            try:
                series[key] = float(rate_value)
            except ValueError:
                continue
    return series


def read_community_series(file_path, selected_industry):
    series = {}
    target = normalize_industry(selected_industry)
    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_value = (row.get("Time") or "").strip()
            industry_value = normalize_industry(row.get("Detailed previous industry"))
            rate_value = (row.get("Unemployment Rate") or "").strip()
            key = parse_month_range(time_value)
            if key is None or not rate_value or industry_value != target:
                continue
            try:
                series[key] = float(rate_value)
            except ValueError:
                continue
    return series


sector_series = {
    "Construction": read_simple_sector_series(BASE_DIR / "construction" / "unemployment_rate.csv"),
    "Transportation": read_simple_sector_series(BASE_DIR / "transportation" / "unemployment_rate.csv"),
    "Retail": read_simple_sector_series(BASE_DIR / "retail" / "unemployment_rate.csv"),
    "Community: Human health and social work activities": read_community_series(
        BASE_DIR / "community" / "unemployment_rate.csv",
        COMMUNITY_INDUSTRY,
    ),
}

all_keys = sorted({key for values in sector_series.values() for key in values})
x_labels = [f"{year} Q{quarter}" for year, quarter in all_keys]

plt.figure(figsize=(15, 7))
for label, series in sector_series.items():
    y_values = [series.get(key) for key in all_keys]
    plt.plot(x_labels, y_values, marker="o", linewidth=2, markersize=4, label=label)

plt.title("Unemployment Rate Comparison Across Sectors", fontsize=14)
plt.xlabel("Quarter")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(x_labels[::2], rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)

print(f"Saved plot to: {OUTPUT_PATH}")