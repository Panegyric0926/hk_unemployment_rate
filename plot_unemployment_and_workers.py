from pathlib import Path
import csv

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "unemployment_vs_approved_workers.png"

UNEMPLOYMENT_COLOR = "#e29578"
WORKERS_COLOR = "#e5f2e5"
BLACK = "#000000"


def parse_quarter_from_range(value):
    if not value or "/" not in value:
        return None
    try:
        left = value.split("-")[0].strip()
        month_str, year_str = left.split("/")
        month = int(month_str.strip())
        year = int(year_str.strip())
        quarter = ((month - 1) // 3) + 1
        return year, quarter
    except (ValueError, IndexError):
        return None


def read_quarterly_unemployment(file_path):
    quarterly_rates = {}
    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            quarter_key = parse_quarter_from_range((row.get("Time") or "").strip())
            rate_text = (row.get("Unemployment Rate Both") or "").strip()
            if quarter_key is None or not rate_text:
                continue
            try:
                quarterly_rates[quarter_key] = float(rate_text)
            except ValueError:
                continue

    return quarterly_rates


def read_quarterly_approved_workers(file_path):
    quarterly_workers = {}
    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year_text = (row.get("Year") or "").strip()
            workers_text = (row.get("Number of Imported Workers Approved") or "").strip()
            if not year_text.isdigit() or not workers_text:
                continue
            try:
                year = int(year_text)
                quarterly_workers.update(
                    {
                        (year, quarter): float(workers_text) / 4
                        for quarter in range(1, 5)
                    }
                )
            except ValueError:
                continue
    return quarterly_workers


def format_quarter_label(period):
    year, quarter = period
    return f"{year} Q{quarter}"


def style_axis(axis):
    axis.tick_params(axis="both", colors=BLACK, labelcolor=BLACK)
    axis.xaxis.label.set_color(BLACK)
    axis.yaxis.label.set_color(BLACK)
    for spine in axis.spines.values():
        spine.set_color(BLACK)


def main():
    unemployment_by_quarter = read_quarterly_unemployment(BASE_DIR / "unemployment_rate.csv")
    approved_workers_by_quarter = read_quarterly_approved_workers(BASE_DIR / "imported_workers.csv")

    periods = sorted(set(unemployment_by_quarter) & set(approved_workers_by_quarter))
    if not periods:
        raise ValueError("No overlapping quarters were found in the source files.")

    positions = list(range(len(periods)))
    labels = [format_quarter_label(period) for period in periods]
    unemployment_values = [unemployment_by_quarter[period] for period in periods]
    approved_worker_values = [approved_workers_by_quarter[period] for period in periods]

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, workers_axis = plt.subplots(figsize=(12, 7))
    unemployment_axis = workers_axis.twinx()

    bars = workers_axis.bar(
        positions,
        approved_worker_values,
        width=0.65,
        color=WORKERS_COLOR,
        edgecolor=WORKERS_COLOR,
        linewidth=1.0,
        label="Approved workers per quarter",
        zorder=1,
    )

    unemployment_axis.set_zorder(workers_axis.get_zorder() + 1)
    unemployment_axis.patch.set_visible(False)
    line, = unemployment_axis.plot(
        positions,
        unemployment_values,
        color=UNEMPLOYMENT_COLOR,
        linewidth=3,
        #marker="o",
        markersize=7,
        markerfacecolor=UNEMPLOYMENT_COLOR,
        markeredgecolor=BLACK,
        markeredgewidth=1.0,
        label="Unemployment rate",
        zorder=3,
    )

    fig.patch.set_facecolor("white")
    workers_axis.set_facecolor("white")
    workers_axis.set_title(
        "Hong Kong Quarterly Unemployment Rate vs Approved Imported Workers",
        color=BLACK,
        pad=12,
    )
    workers_axis.set_xlabel("Quarter", color=BLACK)
    workers_axis.set_ylabel("Approved workers per quarter", color=BLACK)
    unemployment_axis.set_ylabel("Unemployment rate (%)", color=BLACK)
    workers_axis.set_xticks(positions)
    workers_axis.set_xticklabels(labels, rotation=45, ha="right")
    workers_axis.set_xlim(min(positions) - 0.75, max(positions) + 0.75)
    workers_axis.grid(axis="y", color=BLACK, alpha=0.15, linewidth=0.8)
    workers_axis.set_axisbelow(True)

    style_axis(workers_axis)
    style_axis(unemployment_axis)

    legend = workers_axis.legend(
        handles=[line, bars],
        labels=["Unemployment rate", "Approved workers per quarter"],
        loc="upper left",
        frameon=True,
    )
    legend.get_frame().set_edgecolor(BLACK)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    for text in legend.get_texts():
        text.set_color(BLACK)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()