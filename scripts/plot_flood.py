from __future__ import annotations

import argparse
import calendar
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
TYPES = ("Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood")
COLORS = ("#2563a6", "#159b8b", "#e59832", "#8a64ae")
MONTHS = list(calendar.month_abbr)[1:]
MONEY_FIELDS = ("damage_property_usd", "damage_crops_usd")
CASUALTY_FIELDS = ("deaths_direct", "deaths_indirect", "injuries_direct", "injuries_indirect")
REQUIRED = {"year", "event_type", "begin_time_est", "duration_hours", "flood_cause",
            "cpi_base_month", *MONEY_FIELDS, *CASUALTY_FIELDS}
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")


@dataclass
class FloodData:
    records: int = 0
    base_month: str = ""
    annual_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    annual_damage: dict = field(default_factory=lambda: defaultdict(lambda: [Decimal(0), Decimal(0)]))
    annual_casualties: dict = field(default_factory=lambda: defaultdict(Counter))
    monthly_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    durations: dict = field(default_factory=lambda: defaultdict(list))
    causes: dict = field(default_factory=lambda: defaultdict(Counter))
    cause_years: set = field(default_factory=set)
    missing_causes: int = 0
    missing_casualties: Counter = field(default_factory=Counter)

    @property
    def years(self) -> list[int]:
        return list(range(min(self.annual_counts), max(self.annual_counts) + 1))

    @property
    def period(self) -> str:
        return f"{self.years[0]}-{self.years[-1]}"


def read_data(path: Path) -> FloodData:
    data = FloodData()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                year, kind = int(row["year"]), row["event_type"]
                if kind not in TYPES:
                    raise ValueError(f"Unknown event type: {kind}")
                begin = datetime.fromisoformat(row["begin_time_est"])
                if begin.utcoffset() != timedelta(hours=-5):
                    raise ValueError("begin_time_est must use fixed UTC-05:00")
                month = begin.month
                base_month = row["cpi_base_month"].strip()
                if not base_month or (data.base_month and base_month != data.base_month):
                    raise ValueError("CPI base month must be populated and consistent")
                data.base_month = base_month
                for index, name in enumerate(MONEY_FIELDS):
                    amount = Decimal(row[name])
                    if not amount.is_finite() or amount < 0:
                        raise ValueError(f"Invalid {name}")
                    data.annual_damage[year][index] += amount
                for name in CASUALTY_FIELDS:
                    value = row[name].strip()
                    if value:
                        count = int(value)
                        if count < 0:
                            raise ValueError(f"Negative {name}")
                        data.annual_casualties[year][name] += count
                    else:
                        data.missing_casualties[name] += 1
                duration = float(row["duration_hours"])
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("Invalid duration_hours")
                data.durations[kind].append(duration)
                data.annual_counts[year][kind] += 1
                data.monthly_counts[kind][month] += 1
                cause = row["flood_cause"].strip()
                if cause:
                    data.causes[cause][month] += 1
                    data.cause_years.add(year)
                else:
                    data.missing_causes += 1
                data.records += 1
            except (ValueError, ArithmeticError, AttributeError, TypeError) as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error
    if not data.records:
        raise ValueError("The input CSV has no event records")
    return data


def annual_axis(ax, data: FloodData) -> None:
    ax.set_xlim(data.years[0] - 0.7, data.years[-1] + 0.7)
    ax.set_xticks(data.years[::2])
    ax.set_xlabel("Source year")
    if data.years[0] < 2007:
        ax.axvspan(data.years[0] - 0.7, min(2006.5, data.years[-1] + 0.7),
                   color="#cbd5e1", alpha=0.3, zorder=0)


def finish(fig, output: Path, name: str, dpi: int, note: str) -> Path:
    fig.text(0.06, 0.02, note, ha="left", va="bottom", fontsize=9, color="#526173")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    target = output / name
    fig.savefig(target, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {target}")
    return target


def draw_charts(data: FloodData, output: Path, dpi: int) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 16,
                         "axes.axisbelow": True, "axes.grid": True,
                         "grid.alpha": 0.2, "grid.color": "#94a3b8"})
    output.mkdir(parents=True, exist_ok=True)
    files = []
    years = data.years
    source_note = f"{data.records:,} event records; {data.period}."
    early_note = (" Shading: pre-2007 records heavily filtered for missing damage."
                  if years[0] < 2007 else "")

    fig, ax = plt.subplots(figsize=(12, 6))
    property_damage = [float(data.annual_damage[y][0]) / 1e9 for y in years]
    crop_damage = [float(data.annual_damage[y][1]) / 1e9 for y in years]
    ax.bar(years, property_damage, color=COLORS[0], label="Property")
    ax.bar(years, crop_damage, bottom=property_damage, color=COLORS[1], label="Crops")
    ax.set_title("Annual flood losses")
    ax.set_ylabel(f"Billion USD, at {data.base_month} prices")
    ax.legend(frameon=False)
    annual_axis(ax, data)
    files.append(finish(fig, output, "01_annual_loss.png", dpi,
                        source_note + " CPI-adjusted property + crop losses." + "\n"
                        + "Zero values include single missing damage fields filled during cleaning." + early_note))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0] * len(years)
    for kind, color in zip(TYPES, COLORS):
        counts = [data.annual_counts[y][kind] for y in years]
        ax.bar(years, counts, bottom=bottom, color=color, label=kind)
        bottom = [a + b for a, b in zip(bottom, counts)]
    ax.set_title("Annual flood event records")
    ax.set_ylabel("Event records")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.legend(frameon=False, ncol=2)
    annual_axis(ax, data)
    files.append(finish(fig, output, "02_annual_event_count.png", dpi,
                        source_note + " Records are not distinct weather episodes.\n" + early_note.strip()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Annual recorded flood casualties", fontsize=17, fontweight="bold")
    for ax, prefix, label, color in zip(axes, ("deaths", "injuries"),
                                        ("Deaths", "Injuries"), COLORS):
        values = [sum(data.annual_casualties[y][f"{prefix}_{mode}"]
                      for mode in ("direct", "indirect")) for y in years]
        ax.plot(years, values, marker="o", markersize=4, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        annual_axis(ax, data)
    missing_note = "; ".join(f"{name}: {count:,}" for name, count in data.missing_casualties.items()) or "none"
    files.append(finish(fig, output, "03_annual_casualties.png", dpi,
                        "Direct + indirect counts; missing values excluded from sums, not filled with zero.\n"
                        + f"Missing fields ({missing_note})." + early_note))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Monthly flood event records by type", fontsize=17, fontweight="bold")
    for ax, kind, color in zip(axes.flat, TYPES, COLORS):
        counts = [data.monthly_counts[kind][m] for m in range(1, 13)]
        ax.bar(range(1, 13), counts, color=color)
        ax.set_title(f"{kind} (n={sum(counts):,})", fontsize=13)
        ax.set_xticks(range(1, 13), MONTHS, rotation=45)
        ax.tick_params(labelbottom=True)
        ax.set_ylabel("Event records")
        ax.yaxis.set_major_formatter(THOUSANDS)
    files.append(finish(fig, output, "04_monthly_event_types.png", dpi,
                        source_note + " Start month in fixed EST (UTC-05:00).\n"
                        + "Panels use separate count scales so small categories remain visible."))

    fig, ax = plt.subplots(figsize=(12, 6))
    active_types = [kind for kind in TYPES if data.durations[kind]]
    boxes = ax.boxplot([data.durations[kind] for kind in active_types],
                       tick_labels=active_types, patch_artist=True, showfliers=False,
                       medianprops={"color": "#172b4d", "linewidth": 2})
    for patch, kind in zip(boxes["boxes"], active_types):
        patch.set_facecolor(COLORS[TYPES.index(kind)])
        patch.set_alpha(0.7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 2, 6, 24, 72, 240, 744], ["0", "1", "2", "6", "24", "72", "240", "744"])
    ax.set_title("Recorded flood duration by type")
    ax.set_ylabel("Duration (hours); linear below 1 h, logarithmic above")
    ax.set_ylim(0, max(max(values) for values in data.durations.values()) * 1.1 or 1)
    zeros = sum(value == 0 for values in data.durations.values() for value in values)
    files.append(finish(fig, output, "05_duration_by_type.png", dpi,
                        f"Boxes: 25th-75th percentiles; center: median; whiskers: 1.5 x IQR; outliers hidden.\n"
                        + f"Zero durations retained ({zeros:,} records). Near-month-long records may reflect reporting boundaries."))

    fig, ax = plt.subplots(figsize=(14, 7))
    if data.causes:
        causes = sorted(data.causes, key=lambda c: sum(data.causes[c].values()), reverse=True)
        counts = [[data.causes[c][m] for m in range(1, 13)] for c in causes]
        shares = [[100 * value / sum(row) for value in row] for row in counts]
        peak = max(max(row) for row in shares)
        heat = ax.imshow(shares, cmap="Blues", aspect="auto", vmin=0, vmax=peak,
                         interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(12), MONTHS)
        ax.set_yticks(range(len(causes)), [f"{cause} (n={sum(row):,})"
                                          for cause, row in zip(causes, counts)])
        ax.set_xlabel("Start month (fixed EST)")
        for i, row in enumerate(counts):
            for j, count in enumerate(row):
                ax.text(j, i, f"{count:,}", ha="center", va="center", fontsize=9,
                        color="white" if shares[i][j] > peak * 0.55 else "#172b4d")
        fig.colorbar(heat, ax=ax, pad=0.02).set_label("Share of records within each cause (%)")
        coverage = f"Known-cause records cover source years {min(data.cause_years)}-{max(data.cause_years)}."
    else:
        ax.text(0.5, 0.5, "No records with a known flood cause", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        coverage = "No known-cause records."
    ax.set_title("Flood causes by month", pad=18)
    files.append(finish(fig, output, "06_flood_cause_by_month.png", dpi,
                        "Color: monthly share within each cause. Cell labels: event record counts. " + coverage + "\n"
                        + f"Missing causes excluded: {data.missing_causes:,} ({data.missing_causes / data.records:.1%}). "
                        + "Coastal/lakeshore causes are unavailable; early-year coverage is incomplete."))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_data/flood_final_CPI.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    data = read_data(args.input)
    files = draw_charts(data, args.output_dir, args.dpi)
    print(f"Done: {data.records:,} records, {data.period}, CPI base {data.base_month}, {len(files)} charts.")


if __name__ == "__main__":
    main()
