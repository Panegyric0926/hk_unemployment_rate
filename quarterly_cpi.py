import csv
from collections import defaultdict

MONTH_TO_QUARTER = {
    "Jan": 1, "Feb": 1, "Mar": 1,
    "Apr": 2, "May": 2, "Jun": 2,
    "Jul": 3, "Aug": 3, "Sep": 3,
    "Oct": 4, "Nov": 4, "Dec": 4,
}

# Read CPI data
quarterly_cpi = defaultdict(list)

with open("cpi.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        year = (row.get("Year") or "").strip()
        month = (row.get("Month") or "").strip()
        cpi_str = (row.get("Consumer Price Index") or "").strip()
        if not year.isdigit() or not month or not cpi_str:
            continue
        year = int(year)
        quarter = MONTH_TO_QUARTER.get(month)
        if quarter is None:
            continue
        quarterly_cpi[(year, quarter)].append(float(cpi_str))

# Calculate average CPI per quarter
quarterly_avg = {
    key: sum(values) / len(values)
    for key, values in quarterly_cpi.items()
}

# Sort quarters
sorted_quarters = sorted(quarterly_avg.keys())

# Calculate quarterly inflation rate starting from 2017 Q1
output_rows = []

for i, (year, q) in enumerate(sorted_quarters):
    if (year, q) < (2017, 1):
        continue
    current_cpi = quarterly_avg[(year, q)]
    prev_quarter = sorted_quarters[i - 1]
    prev_cpi = quarterly_avg[prev_quarter]
    inflation_rate = (current_cpi / prev_cpi - 1) * 100
    output_rows.append({
        "Quarter": f"{year} Q{q}",
        "Avg CPI": round(current_cpi, 2),
        "Quarterly Inflation Rate (%)": round(inflation_rate, 2),
    })

# Print to console
print(f"{'Quarter':<12} {'Avg CPI':>10} {'Quarterly Inflation Rate':>25}")
print("-" * 50)
for row in output_rows:
    print(f"{row['Quarter']:<12} {row['Avg CPI']:>10.2f} {row['Quarterly Inflation Rate (%)']:>24.2f}%")

# Save to CSV
output_file = "inflation_rate.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Quarter", "Avg CPI", "Quarterly Inflation Rate (%)"])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nResults saved to {output_file}")
