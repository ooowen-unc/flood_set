from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TYPES = ["Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood"]
COLORS = ["#2563a6", "#159b8b", "#e59832", "#8a64ae"]


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Pairwise ranks avoid a SciPy dependency and preserve missingness."""
    pairs = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return pairs.rank().corr().iloc[0, 1]


def prepare_counties(annual: pd.DataFrame, counties: pd.DataFrame,
                     start: int, end: int, minimum: int) -> pd.DataFrame:
    subset = annual[annual.flood_category.eq("inland")
                    & annual.analysis_year.between(start, end)]
    if subset.empty:
        raise ValueError("No inland county-year observations in the requested period")
    values = ["event_records", "property_loss_adjusted_usd", "material_loss_adjusted_usd"]
    result = subset.groupby("county_fips")[values].sum(min_count=1)
    result["observed_years"] = subset.groupby("county_fips").analysis_year.nunique()
    result["largest_year_loss"] = subset.groupby("county_fips").material_loss_adjusted_usd.max()
    context = counties[counties.nri_hazard_prefix.eq("IFLD")].set_index("county_fips")
    result = result.join(context, how="left", validate="one_to_one")
    result = result[result.observed_years.ge(minimum)].copy()
    if len(result) < 8:
        raise ValueError("Too few counties for four-group comparisons; lower --min-observed-years")
    result["mean_loss"] = result.material_loss_adjusted_usd / result.observed_years
    result["mean_property_loss"] = result.property_loss_adjusted_usd / result.observed_years
    assets = result.nri_buildvalue_adjusted_usd.where(result.nri_buildvalue_adjusted_usd.gt(0))
    result["asset_loss_pct"] = result.mean_property_loss / assets * 100
    result["largest_year_share_pct"] = (result.largest_year_loss
                                       / result.material_loss_adjusted_usd.where(
                                           result.material_loss_adjusted_usd.gt(0)) * 100)
    result["eal_group"] = pd.qcut(result.nri_ealb_adjusted_usd, 4,
                                  labels=["Q1", "Q2", "Q3", "Q4"])
    return result


def save(fig, output: Path, filename: str, dpi: int, title: str,
         subtitle: str, note: str) -> None:
    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10.5, color="#526173")
    lines = []
    for paragraph in note.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=145))
    fig.text(0.045, 0.02, "\n".join(lines), fontsize=8.5, color="#526173", va="bottom")
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.885))
    target = output / filename
    fig.savefig(target, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {target}")


def relationship_charts(data: pd.DataFrame, output: Path, dpi: int,
                        start: int, end: int, minimum: int, base: str) -> None:
    sample = f"{len(data):,} inland counties; {start}-{end}; at least {minimum} years with retained records"
    note = ("Averages use only years with retained records; absent county-years are not filled with zero. "
            f"USD at {base} prices. NRI assets/population are static snapshots.\n"
            "Historical periods overlap NRI inputs; associations are not independent prediction validation or causal effects.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, outcome, label in zip(axes, ["mean_loss", "asset_loss_pct"],
                                  ["Mean retained-year material loss (USD)",
                                   "Mean retained-year property loss / NRI building value (%)"]):
        pairs = data[["nri_risks", "nri_population", outcome]].dropna()
        points = ax.scatter(pairs.nri_risks, pairs[outcome], s=15, alpha=0.5,
                            c=np.log10(pairs.nri_population.clip(lower=1)), cmap="viridis")
        ax.set_xlabel("NRI inland flood risk score")
        ax.set_ylabel(label)
        ax.set_xlim(-2, 102)
        ax.set_yscale("symlog", linthresh=1000 if outcome == "mean_loss" else 0.0001)
        ax.set_ylim(bottom=0)
        bins = pd.cut(pairs.nri_risks, [0, 20, 40, 60, 80, 100], include_lowest=True)
        medians = pairs.groupby(bins, observed=True)[outcome].median()
        centers = [interval.mid for interval in medians.index]
        ax.plot(centers, medians, color="#e76f51", marker="o", linewidth=2,
                label="Median in 20-point risk bands")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"n={len(pairs):,}; Spearman rho={spearman(pairs.nri_risks, pairs[outcome]):.3f}")
        fig.colorbar(points, ax=ax, pad=0.02).set_label("log10(NRI population)")
    save(fig, output, "07_nri_risk_vs_loss.png", dpi,
         "Flood risk, loss scale, and relative asset burden", sample,
         note + "\nMaterial loss = property + crops. Y scales are linear near zero and logarithmic above; zero losses are retained.")

    fig, ax = plt.subplots(figsize=(13, 7))
    pairs = data[data.mean_property_loss.gt(0) & data.nri_ealb_adjusted_usd.gt(0)].copy()
    pairs = pairs.dropna(subset=["nri_ealb_adjusted_usd", "mean_property_loss", "largest_year_share_pct"])
    points = ax.scatter(pairs.nri_ealb_adjusted_usd, pairs.mean_property_loss,
                        c=pairs.largest_year_share_pct, cmap="YlOrRd", vmin=0, vmax=100,
                        s=22, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    low = min(pairs.nri_ealb_adjusted_usd.min(), pairs.mean_property_loss.min()) * 0.6
    high = max(pairs.nri_ealb_adjusted_usd.max(), pairs.mean_property_loss.max()) * 2
    ax.plot([low, high], [low, high], "--", color="#526173", linewidth=1,
            label="Equal dollar amount (reference only)")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("NRI expected annual building loss (USD)")
    ax.set_ylabel("Mean retained-year NOAA property loss (USD)")
    ax.legend(frameon=False, loc="upper left")
    # Label large losses with spaced offsets rather than all outliers.
    for (_, row), offset in zip(pairs.nlargest(5, "mean_property_loss").iterrows(),
                                 [(8, 18), (8, -22), (8, 5), (-100, 16), (-105, -20)]):
        ax.annotate(f"{row.nri_county}, {row.nri_stateabbrv}",
                    (row.nri_ealb_adjusted_usd, row.mean_property_loss),
                    xytext=offset, textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#64748b", "linewidth": 0.7})
    fig.colorbar(points, ax=ax).set_label("Largest loss year / period material loss (%)")
    rho = spearman(pairs.nri_ealb_adjusted_usd, pairs.mean_property_loss)
    save(fig, output, "08_nri_eal_vs_observed_loss.png", dpi,
         "Expected annual building loss and historical property loss",
         f"{sample}; {len(pairs):,} positive pairs; Spearman rho={rho:.3f}",
         note + f"\n{len(data) - len(pairs):,} zero/missing pairs omitted on log axes. NOAA property and NRI building losses differ in scope.")

    fig, ax = plt.subplots(figsize=(12, 7))
    pairs = data[["nri_afreq", "event_records", "observed_years"]].dropna()
    pairs = pairs[pairs.nri_afreq.gt(0) & pairs.event_records.gt(0)]
    points = ax.scatter(pairs.nri_afreq, pairs.event_records, c=pairs.observed_years,
                        cmap="viridis", s=18, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NRI annualized inland flood frequency (events/year)")
    ax.set_ylabel(f"Retained event records, {start}-{end} (total)")
    fig.colorbar(points, ax=ax).set_label("Years with retained records")
    rho = spearman(pairs.nri_afreq, pairs.event_records)
    save(fig, output, "09_nri_frequency_vs_records.png", dpi,
         "NRI flood frequency and retained historical records",
         f"{sample}; {len(pairs):,} positive pairs; Spearman rho={rho:.3f}",
         note + "\nDifferent periods and screening rules: retained counts are not a complete annual occurrence rate.")

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), sharey=True)
    for column, group in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        subset = data[data.eal_group.eq(group)]
        for row, predictor, label, color in [(0, "nri_sovi_score", "Social vulnerability score", COLORS[0]),
                                              (1, "nri_resl_score", "Community resilience score", COLORS[1])]:
            ax = axes[row, column]
            pairs = subset[[predictor, "asset_loss_pct"]].dropna()
            ax.scatter(pairs[predictor], pairs.asset_loss_pct, s=14, alpha=0.5, color=color)
            ax.set_yscale("symlog", linthresh=0.0001)
            ax.set_ylim(bottom=0)
            ax.set_xlim(-2, 102)
            ax.set_xlabel(label, fontsize=10)
            ax.set_title(f"Building EAL {group}\nn={len(pairs):,}; rho={spearman(pairs[predictor], pairs.asset_loss_pct):.3f}", fontsize=11)
            if column == 0:
                ax.set_ylabel("Mean retained-year property loss\n/ NRI building value (%)", fontsize=10)
    save(fig, output, "10_vulnerability_resilience_by_eal.png", dpi,
         "Social vulnerability and resilience within EAL groups", sample,
         note + "\nEAL Q1-Q4: county quartiles, lowest to highest. Shared y scale retains zeros; grouping is not full adjustment for confounding.")


def annotate_bars(ax, bars, values, fmt="{:.2f}%") -> None:
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + ax.get_xlim()[1] * 0.012,
                bar.get_y() + bar.get_height() / 2, fmt.format(value), va="center", fontsize=10)


def draw_nested_donut(ax, outer: pd.Series, inner: pd.Series) -> None:
    """Normalize each ring independently, retaining true slice sizes and colors."""
    for values, radius, width in [(outer, 1.0, 0.26), (inner, 0.69, 0.24)]:
        ax.pie(values, colors=COLORS, startangle=90, counterclock=False,
               radius=radius, wedgeprops={"width": width, "edgecolor": "white", "linewidth": 2})
    # Exact values live in the shared key, avoiding crowded labels on inner slices.
    ax.text(0, 0.09, "4", ha="center", va="center", fontsize=35,
            fontweight="bold", color="#172b4d")
    ax.text(0, -0.16, "FLOOD TYPES", ha="center", va="center", fontsize=10, color="#64748b")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def type_donut_charts(summary: pd.DataFrame, output: Path, dpi: int,
                      base: str, sample: str, note: str) -> None:
    """Figures 11/12 use concentric rings with a fixed palette and exact shares."""
    designs = [
        ("11_flood_type_record_vs_loss_share.png", "Four flood types: frequency and economic impact",
         ["records", "material"], ["EVENT RECORDS", "TOTAL MATERIAL LOSS"],
         [f"{summary.records.sum():,}", f"${summary.material.sum() / 1e9:.2f}B"],
         ["retained records", f"USD at {base} prices"], ["Outer: record share", "Inner: loss share"]),
        ("12_property_crop_losses_by_type.png", "Four flood types: property and crop losses",
         ["property", "crops"], ["PROPERTY LOSS", "CROP LOSS"],
         [f"${summary.property.sum() / 1e9:.2f}B", f"${summary.crops.sum() / 1e9:.2f}B"],
         [f"USD at {base} prices"] * 2, ["Outer: property share", "Inner: crop share"]),
    ]
    for filename, title, fields, headings, centers, units, headers in designs:
        fig = plt.figure(figsize=(14, 9))
        fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=20,
                 fontweight="bold", color="#172b4d")
        fig.text(0.5, 0.925, sample, ha="center", va="top", fontsize=10.5, color="#64748b")
        ax = fig.add_axes((0.09, 0.35, 0.52, 0.54))
        draw_nested_donut(ax, summary[fields[0]], summary[fields[1]])
        for y, ring, heading, total, unit in zip([0.77, 0.54], ["OUTER RING", "INNER RING"],
                                                headings, centers, units):
            fig.text(0.68, y, ring, fontsize=9, fontweight="bold", color="#64748b")
            fig.text(0.68, y - 0.045, heading, fontsize=12, fontweight="bold", color="#172b4d")
            fig.text(0.68, y - 0.105, total, fontsize=25, fontweight="bold", color="#172b4d")
            fig.text(0.68, y - 0.15, unit, fontsize=10, color="#64748b")

        key = fig.add_axes((0.12, 0.105, 0.76, 0.20))
        key.set(xlim=(0, 1), ylim=(0, 1))
        key.axis("off")
        shares = [summary[field] / summary[field].sum() * 100 for field in fields]
        if filename.startswith("12"):
            # Preserve the former within-type composition alongside the new pies.
            headers = headers + ["Crop share\nwithin each type"]
            shares.append(summary.crops / summary.material.where(summary.material.gt(0)) * 100)
            positions = [0.48, 0.72, 0.98]
        else:
            positions = [0.62, 0.95]
        key.text(0.025, 1.17, "FLOOD TYPE", color="#64748b", fontsize=9, fontweight="bold")
        for position, header in zip(positions, headers):
            key.text(position, 1.17, header, ha="right", color="#64748b", fontsize=9, fontweight="bold")
        for index, (kind, color) in enumerate(zip(TYPES, COLORS)):
            y = 0.83 - index * 0.23
            key.scatter(0.018, y, s=90, color=color, marker="o")
            key.text(0.055, y, kind, va="center", fontsize=11, color="#172b4d")
            for position, series in zip(positions, shares):
                value = series.loc[kind]
                label = f"{value:.2f}%" if pd.notna(value) else "n/a"
                key.text(position, y, label, ha="right", va="center", fontsize=11,
                         fontweight="bold", color="#172b4d")
            if index < 3:
                key.plot([0.015, 0.99], [y - 0.115] * 2, color="#e2e8f0", linewidth=0.7)
        denominator_note = ("Each ring independently totals 100%; outer = record share, inner = material loss share. Exact shares are listed below."
                            if filename.startswith("11") else
                            "Outer = property loss share; inner = crop loss share. Each ring totals 100% independently; the last column uses loss within each flood type.")
        footnotes = "\n".join(textwrap.fill(line, width=155)
                              for line in (denominator_note + "\n" + note).splitlines())
        fig.text(0.05, 0.016, footnotes, fontsize=8.3, color="#64748b", va="bottom")
        target = output / filename
        fig.savefig(target, dpi=dpi, facecolor="white")
        plt.close(fig)
        print(f"Saved {target}")


def basic_charts(events: pd.DataFrame, output: Path, dpi: int, base: str) -> None:
    summary = events.groupby("event_type").agg(
        records=("event_id", "size"), material=("damage_material_adjusted_usd", "sum"),
        property=("damage_property_adjusted_usd", "sum"), crops=("damage_crops_adjusted_usd", "sum"))
    summary = summary.reindex(TYPES, fill_value=0)
    summary["record_share"] = summary.records / summary.records.sum() * 100
    summary["loss_share"] = summary.material / summary.material.sum() * 100
    period = f"{events.analysis_year.min()}-{events.analysis_year.max()}"
    sample = f"All {len(events):,} event records, including unmatched events; {period}"
    note = (f"Property + crop losses, CPI-adjusted to {base} prices. Counts describe retained records, not distinct weather episodes.\n"
            "Source screening removed records with both loss fields missing and filled a single missing component with zero.")

    type_donut_charts(summary, output, dpi, base, sample, note)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    positive = events.damage_material_adjusted_usd.gt(0)
    shares = (positive.groupby(events.event_type).mean().reindex(TYPES).fillna(0) * 100)
    medians = (events[positive].groupby("event_type").damage_material_adjusted_usd.median()
               .reindex(TYPES))
    axes[0].set_xlim(0, 100)
    bars = axes[0].barh(TYPES, shares, color=COLORS)
    annotate_bars(axes[0], bars, shares, "{:.1f}%")
    axes[0].xaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_xlabel("Records with positive material loss (%)")
    axes[0].set_title("How often is a positive loss reported?", fontsize=13)
    axes[1].barh(TYPES, medians, color=COLORS)
    axes[1].set_xscale("log")
    axes[1].set_xlim(max(medians.min() / 3, 1), medians.max() * 5)
    for i, value in enumerate(medians):
        axes[1].text(value * 1.12, i, f"${value:,.0f}", va="center", fontsize=10)
    axes[1].set_xlabel(f"Median positive material loss (USD, {base} prices; log scale)")
    axes[1].set_title("Size of a typical positive-loss record", fontsize=13)
    for ax in axes:
        ax.invert_yaxis()
    save(fig, output, "13_positive_loss_frequency_and_size.png", dpi,
         "Reported loss frequency and typical positive loss", sample,
         note + "\nZero-loss records remain in the frequency denominator; conditional medians use positive-loss records only.")

    losses = events.damage_material_adjusted_usd.sort_values().to_numpy()
    n = len(losses)
    x = np.arange(n + 1) / n * 100
    cumulative = np.r_[0, np.cumsum(losses)] / losses.sum() * 100
    top_n = max(1, int(np.ceil(n * 0.01)))
    boundary = n - top_n
    top_share = 100 - cumulative[boundary]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, cumulative, color=COLORS[0], linewidth=2.5, label="Observed cumulative loss")
    ax.plot([0, 100], [0, 100], "--", color="#94a3b8", label="Equal losses across records")
    ax.axvline(x[boundary], color="#e59832", linestyle=":")
    ax.scatter(x[boundary], cumulative[boundary], color="#e59832", zorder=3)
    ax.annotate(f"Highest-loss 1% of records\ncontribute {top_share:.1f}% of total loss",
                (x[boundary], cumulative[boundary]), xytext=(42, 57), fontsize=12,
                arrowprops={"arrowstyle": "->", "color": "#e59832"})
    ax.set(xlim=(0, 101), ylim=(0, 101),
           xlabel="Share of records, ordered from lowest to highest loss (%)",
           ylabel="Cumulative share of material loss (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, output, "14_flood_loss_concentration.png", dpi,
         "How concentrated are flood losses?", sample, note)

    print("Flood type shares (%):")
    print(summary[["records", "record_share", "loss_share"]].round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/analyzed_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-observed-years", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.start_year > args.end_year or args.min_observed_years < 1 or args.dpi < 1:
        parser.error("Use an ordered year range, positive minimum observed years, and positive DPI")
    events = pd.read_csv(args.data_dir / "flood_nri_events.csv", dtype={"county_fips": "string"}, low_memory=False)
    annual = pd.read_csv(args.data_dir / "flood_nri_county_year.csv", dtype={"county_fips": "string"}, low_memory=False)
    counties = pd.read_csv(args.data_dir / "nri_flood_counties.csv", dtype={"county_fips": "string"}, low_memory=False)
    metadata = json.loads((args.data_dir / "analysis_metadata.json").read_text(encoding="utf-8-sig"))
    base = metadata["parameters"]["base_month"]
    for table in [events, annual]:
        if table.cpi_base_month.isna().any() or set(table.cpi_base_month) != {base}:
            raise ValueError("Mixed or missing CPI base months; compare only consistent price levels")
    if set(counties.nri_price_target_month.dropna()) != {base}:
        raise ValueError("NRI and event CPI targets differ")
    data = prepare_counties(annual, counties, args.start_year, args.end_year, args.min_observed_years)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.axisbelow": True,
                         "axes.grid": True, "grid.alpha": 0.2, "grid.color": "#94a3b8"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relationship_charts(data, args.output_dir, args.dpi, args.start_year, args.end_year,
                        args.min_observed_years, base)
    basic_charts(events, args.output_dir, args.dpi, base)
    print(f"Done: 8 charts; {len(events):,} events; {len(data):,} county comparison samples; CPI base {base}.")


if __name__ == "__main__":
    main()
