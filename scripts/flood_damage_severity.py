"""
Flood damage vs. severity, NOAA Storm Events (Flood + Flash Flood only). Default period 1999-2025.

Question: how does flood property damage scale with how severe a flood is?

Storm Events has no magnitude field for floods, so severity is measured with proxies:
    Footprint   number of counties hit by the same flood episode (EPISODE_ID)
    Extent      straight-line distance between the flood's begin and end points, miles
    Depth       water depth quoted in the NWS narrative ("3 feet of water"), feet
                -> only ~3% of events mention a depth; treat as a sub-sample
    Duration    END_DATE_TIME - BEGIN_DATE_TIME, hours
    Cause       FLOOD_CAUSE (heavy rain, tropical system, dam/levee break, ...)
    Type        Flash Flood vs Flood

Period: set with --start-year / --end-year (default 1999-2025, matching the team's data).
Caveat for years before 2007: NWS changed how Storm Data is entered in late 2006. Before
2007 about 60-70% of flood damage fields are blank (counted here as $0) and almost no
flood has coordinates, so damage in 1999-2006 is under-recorded and extent is missing.
Events recorded by NWS forecast zone (CZ_TYPE "Z", mostly before 2007) get no county FIPS.

Usage (from the repo root):
    python scripts/flood_damage_severity.py --raw-dir data/raw --cpi CPIAUCSL.csv

    --raw-dir   folder with StormEvents_details-*.csv or .csv.gz (searched recursively)
    --cpi       FRED CPIAUCSL csv (monthly). Damage is converted to the prices of
                --base-month (default 2026-08, the same base as the team's
                adjust_flood_cpi.py) using each event's begin month.
                If the file is missing, damage stays in nominal dollars.
    --out       output folder (default: outputs/flood_severity)

Outputs:
    flood_events.csv.gz             one row per flood event with every severity proxy
    tables/*.csv                    one summary table per proxy + stats.csv
    figures/*.png                   charts for slides
    FINDINGS.md                     the numbers written out, regenerated on every run
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

YEAR_MIN, YEAR_MAX = 1999, 2025   # overridden by --start-year / --end-year
BASE_MONTH = "2026-08"   # same price base as scripts/adjust_flood_cpi.py
FLOOD_TYPES = ["Flash Flood", "Flood"]

COLS = ["EVENT_ID", "EPISODE_ID", "YEAR", "STATE", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "CZ_NAME",
        "EVENT_TYPE", "BEGIN_DATE_TIME", "END_DATE_TIME",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS", "DEATHS_DIRECT", "INJURIES_DIRECT", "FLOOD_CAUSE",
        "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON", "EVENT_NARRATIVE"]

# 50 states + DC; drops marine zones and territories
STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "DISTRICT OF COLUMBIA", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS",
    "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND",
    "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA",
    "NEBRASKA", "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH",
    "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
}

# Chart styling: one sequential blue ramp (light = mild, dark = severe)
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- load
def find_detail_files(raw_dir):
    pats = ["**/StormEvents_details*.csv.gz", "**/StormEvents_details*.csv"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(raw_dir, p), recursive=True)})
    keep = []
    for f in files:
        m = re.search(r"_d(\d{4})_", os.path.basename(f))
        if m and YEAR_MIN <= int(m.group(1)) <= YEAR_MAX:
            keep.append(f)
    return keep


def parse_damage(series):
    """'2.50M' -> 2_500_000.0, '.1M' -> 100_000.0, '10.00K' -> 10_000.0. Blank stays NaN."""
    s = series.astype("string").str.strip().str.upper()
    parts = s.str.extract(r"^([0-9]*\.?[0-9]*)([HKMB]?)$")
    num = pd.to_numeric(parts[0].replace("", np.nan), errors="coerce")
    mult = parts[1].map({"": 1.0, "H": 1e2, "K": 1e3, "M": 1e6, "B": 1e9})
    out = num * mult
    out = out.where(~(num.isna() & parts[1].isin(["K", "M", "B", "H"])), 0.0)  # bare "K" = 0
    return out.astype(float)


def cpi_factors(cpi_path, base_month):
    """Return {(year, month): multiplier to base-month prices}, or None if there is no CPI file."""
    if not cpi_path or not os.path.exists(cpi_path):
        return None
    cpi = pd.read_csv(cpi_path)
    date_col, val_col = cpi.columns[0], cpi.columns[1]
    when = pd.to_datetime(cpi[date_col])
    vals = pd.to_numeric(cpi[val_col], errors="coerce")
    monthly = pd.Series(vals.values, index=list(zip(when.dt.year, when.dt.month)))
    # Oct 2025 CPI was never published (government shutdown). Like adjust_flood_cpi.py,
    # fill a single missing month with the mean of its two neighbours.
    monthly = monthly.fillna((monthly.shift(1) + monthly.shift(-1)) / 2).dropna()
    by, bm = (int(x) for x in base_month.split("-"))
    if (by, bm) not in monthly.index:
        raise ValueError(f"CPI file has no value for base month {base_month}")
    return (monthly[(by, bm)] / monthly).to_dict()


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


DEPTH_RE = (r"(\d+(?:\.\d+)?)\s*(?:to\s*(\d+(?:\.\d+)?)\s*)?"
            r"(feet|foot|ft|inches|inch)\s+(?:of\s+)?(?:water|deep)")


def depth_from_narrative(text):
    """'2 to 4 feet of water' -> 4.0 ft; '18 inches of water' -> 1.5 ft; no mention -> NaN."""
    m = text.fillna("").str.extract(DEPTH_RE, flags=re.I)
    val = pd.to_numeric(m[1].fillna(m[0]), errors="coerce")     # upper end of a range
    # .astype(bool) keeps this a real boolean on older pandas (2.x returns object dtype here)
    inches = m[2].fillna("").str.lower().str.startswith("inch").astype(bool)
    ft = val.where(~inches, val / 12)
    return ft.where(ft <= 40)                                   # drop implausible values


def load(raw_dir, cpi_path, base_month=BASE_MONTH):
    files = find_detail_files(raw_dir)
    if not files:
        raise SystemExit(f"No StormEvents_details files for {YEAR_MIN}-{YEAR_MAX} under {raw_dir}")
    parts = []
    for f in files:
        x = pd.read_csv(f, usecols=COLS, low_memory=False)
        parts.append(x[x["EVENT_TYPE"].isin(FLOOD_TYPES)])
    df = pd.concat(parts, ignore_index=True)
    df = df[df["YEAR"].between(YEAR_MIN, YEAR_MAX) & df["STATE"].isin(STATES)].copy()

    fmt = "%d-%b-%y %H:%M:%S"
    start = pd.to_datetime(df["BEGIN_DATE_TIME"], format=fmt, errors="coerce")
    factors = cpi_factors(cpi_path, base_month)
    if factors is None:
        df["cpi_factor"] = 1.0
        dollars = "nominal dollars (no CPI file found)"
    else:
        keys = list(zip(start.dt.year, start.dt.month))
        df["cpi_factor"] = [factors.get(k, np.nan) for k in keys]
        if df["cpi_factor"].isna().any():
            raise ValueError("Some event months have no CPI value")
        dollars = f"{pd.Timestamp(base_month + '-01'):%b %Y} dollars (CPI-U)"
    df["damage_property"] = parse_damage(df["DAMAGE_PROPERTY"]).fillna(0.0) * df["cpi_factor"]
    df["damage_crops"] = parse_damage(df["DAMAGE_CROPS"]).fillna(0.0) * df["cpi_factor"]
    df["damage"] = df["damage_property"]                       # main measure
    # blank damage = no estimate made (common before 2007). Keep the event (deaths, footprint,
    # totals need it) but leave it out of per-event damage averages and shares.
    df["damage_reported"] = df["DAMAGE_PROPERTY"].notna()

    end = pd.to_datetime(df["END_DATE_TIME"], format=fmt, errors="coerce")
    df["begin_time"] = start
    df["duration_h"] = (end - start).dt.total_seconds() / 3600
    df["extent_mi"] = haversine_miles(df["BEGIN_LAT"], df["BEGIN_LON"], df["END_LAT"], df["END_LON"])
    df["depth_ft"] = depth_from_narrative(df["EVENT_NARRATIVE"])
    df["episode_counties"] = df.groupby("EPISODE_ID")["EVENT_ID"].transform("count")
    # only county-type records (CZ_TYPE "C") carry a county FIPS; forecast zones ("Z") do not
    fips = (df["STATE_FIPS"].astype(int) * 1000 + df["CZ_FIPS"].astype(int)).astype(str).str.zfill(5)
    df["county_fips"] = fips.where(df["CZ_TYPE"] == "C")
    df["FLOOD_CAUSE"] = df["FLOOD_CAUSE"].fillna("Unknown")

    print(f"Loaded {len(df):,} flood events ({', '.join(FLOOD_TYPES)}) from {len(files)} files, "
          f"{YEAR_MIN}-{YEAR_MAX}; {dollars}")
    return df, dollars


# ----------------------------------------------------------------------------- stats
def summarize(x, bin_col="bin", count_name="events"):
    g = x.groupby(bin_col, observed=True)
    r = x[x["damage_reported"]].groupby(bin_col, observed=True)["damage"]
    out = pd.DataFrame({
        count_name: g.size(),
        "with_damage_estimate": g["damage_reported"].sum(),
        "pct_with_damage": r.apply(lambda s: (s > 0).mean() * 100),
        "mean_damage": r.mean(),
        "median_damage_if_damaged": g["damage"].apply(lambda s: s[s > 0].median()),
        "total_damage_M": g["damage"].sum() / 1e6,
        "deaths": g["DEATHS_DIRECT"].sum(),
        "injuries": g["INJURIES_DIRECT"].sum(),
    })
    out[f"deaths_per_1000_{count_name}"] = out["deaths"] / out[count_name] * 1000
    out[f"share_of_{count_name}_pct"] = out[count_name] / out[count_name].sum() * 100
    out["share_of_damage_pct"] = out["total_damage_M"] / out["total_damage_M"].sum() * 100
    out["share_of_deaths_pct"] = out["deaths"] / max(out["deaths"].sum(), 1) * 100
    return out.reset_index().rename(columns={bin_col: "bin"})


def fit_row(name, x, col, unit):
    """Spearman on all rows + log-log fit on rows with damage > 0 and severity > 0."""
    x = x[x[col].notna() & x["damage_reported"]]
    rho, p = stats.spearmanr(x[col], x["damage"])
    d = x[(x["damage"] > 0) & (x[col] > 0)]
    fit = stats.linregress(np.log(d[col]), np.log(d["damage"]))
    return {"proxy": name, "column": col, "unit": unit, "n": len(x), "n_damaged": len(d),
            "spearman_rho_all": rho, "spearman_p_all": p,
            "elasticity": fit.slope, "elasticity_ci95_low": fit.slope - 1.96 * fit.stderr,
            "elasticity_ci95_high": fit.slope + 1.96 * fit.stderr, "r2": fit.rvalue ** 2}


# ----------------------------------------------------------------------------- charts
def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def money(v):
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def approx(x):
    """Round a ratio to 2 significant figures for titles: 497 -> '500'."""
    return f"{float(f'{x:.2g}'):,.0f}"


def finish(fig, ax, title, subtitle, path):
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar_chart(tab, title, subtitle, ylabel, value_col, path, fmt=money, axis_fmt=None):
    n = len(tab)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    x = np.arange(n)
    vals = tab[value_col].values
    ax.bar(x, vals, color=RAMP[-n:], width=0.62, edgecolor="white", linewidth=1.5)
    ax.set_yscale("log")
    for xi, v in zip(x, vals):
        if v > 0:
            ax.annotate(fmt(v), (xi, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, tab["bin"].astype(str))
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    style(ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    tick = axis_fmt or fmt
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: tick(v)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    finish(fig, ax, title, subtitle, path)


def cause_chart(tab, title, subtitle, path):
    tab = tab.sort_values("mean_damage")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    y = np.arange(len(tab))
    ax.barh(y, tab["mean_damage"], color=RAMP[3], height=0.6, edgecolor="white", linewidth=1.5)
    ax.set_xscale("log")
    for yi, v, n in zip(y, tab["mean_damage"], tab["events"]):
        ax.annotate(f"{money(v)}  (n={n:,})", (v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, tab["bin"])
    ax.set_xlim(right=ax.get_xlim()[1] * 20)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Average property damage per event (log scale)", color=INK_2, fontsize=9)
    style(ax, grid_axis="x")
    finish(fig, ax, title, subtitle, path)


def concentration_chart(conc, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    y = np.arange(len(conc))
    h = 0.34
    for i, (col, name, color) in enumerate([("share_of_events_pct", "Share of flood events", RAMP[1]),
                                            ("share_of_damage_pct", "Share of flood damage", RAMP[5])]):
        vals = conc[col].values
        ax.barh(y + (i - 0.5) * h, vals, height=h - 0.04, color=color, label=name,
                edgecolor="white", linewidth=1)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.1f}%" if v < 10 else f"{v:.0f}%", (v, yi + (i - 0.5) * h), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, conc["group"])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of all flood events / damage", color=INK_2, fontsize=9)
    style(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    finish(fig, ax, title, subtitle, path)


# ----------------------------------------------------------------------------- main
def main():
    global YEAR_MIN, YEAR_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cpi", default="CPIAUCSL.csv")
    ap.add_argument("--out", default="outputs/flood_severity")
    ap.add_argument("--base-month", default=BASE_MONTH, help="CPI price base, YYYY-MM")
    ap.add_argument("--start-year", type=int, default=YEAR_MIN)
    ap.add_argument("--end-year", type=int, default=YEAR_MAX)
    args = ap.parse_args()
    YEAR_MIN, YEAR_MAX = args.start_year, args.end_year
    tab_dir, fig_dir = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df, dollars = load(args.raw_dir, args.cpi, args.base_month)
    span = f"{YEAR_MIN}-{YEAR_MAX}"

    # event-level file for teammates (map, AI summaries, ...)
    keep = ["EVENT_ID", "EPISODE_ID", "YEAR", "begin_time", "STATE", "county_fips", "CZ_NAME",
            "EVENT_TYPE", "FLOOD_CAUSE", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
            "damage_property", "damage_crops", "damage_reported", "DEATHS_DIRECT", "INJURIES_DIRECT",
            "duration_h", "extent_mi", "depth_ft", "episode_counties", "cpi_factor"]
    df[keep].to_csv(os.path.join(args.out, "flood_events.csv.gz"), index=False)

    # 1) footprint: one row per episode
    ep = df.groupby("EPISODE_ID").agg(counties=("EVENT_ID", "count"), damage=("damage", "sum"),
                                      damage_reported=("damage_reported", "max"),
                                      DEATHS_DIRECT=("DEATHS_DIRECT", "sum"),
                                      INJURIES_DIRECT=("INJURIES_DIRECT", "sum"))
    ep["bin"] = pd.cut(ep["counties"], [1, 2, 4, 8, 16, np.inf], right=False,
                       labels=["1 county", "2-3", "4-7", "8-15", "16+ counties"])
    t_foot = summarize(ep, count_name="episodes")

    # 2) extent, 3) depth, 4) duration: one row per event
    ext = df[df["extent_mi"].notna()].copy()
    ext["bin"] = pd.cut(ext["extent_mi"], [0, 0.5, 2, 5, 10, np.inf], right=False,
                        labels=["< 0.5 mi", "0.5-2 mi", "2-5 mi", "5-10 mi", "10+ mi"])
    t_ext = summarize(ext)

    dep = df[df["depth_ft"].notna()].copy()
    dep["bin"] = pd.cut(dep["depth_ft"], [0, 1, 2, 4, 8, np.inf], right=False,
                        labels=["< 1 ft", "1-2 ft", "2-4 ft", "4-8 ft", "8+ ft"])
    t_dep = summarize(dep)

    dur = df[df["duration_h"] >= 0].copy()
    dur["bin"] = pd.cut(dur["duration_h"], [0, 3, 12, 24, 72, np.inf], right=False,
                        labels=["< 3 h", "3-12 h", "12-24 h", "1-3 days", "> 3 days"])
    t_dur = summarize(dur)

    # 5) cause, 6) type
    t_cause = summarize(df.assign(bin=df["FLOOD_CAUSE"])).sort_values("mean_damage", ascending=False)
    t_type = summarize(df.assign(bin=df["EVENT_TYPE"]))

    tables = {"by_episode_footprint": t_foot, "by_extent": t_ext, "by_water_depth": t_dep,
              "by_duration": t_dur, "by_cause": t_cause, "by_type": t_type}
    for name, tab in tables.items():
        tab.round(3).to_csv(os.path.join(tab_dir, f"{name}.csv"), index=False)

    st = pd.DataFrame([
        fit_row("Episode footprint", ep.rename(columns={"counties": "episode_counties"}), "episode_counties", "counties"),
        fit_row("Extent", df, "extent_mi", "miles"),
        fit_row("Water depth (narrative)", df, "depth_ft", "feet"),
        fit_row("Duration", df, "duration_h", "hours"),
    ])
    st.to_csv(os.path.join(tab_dir, "stats.csv"), index=False)

    # concentration of damage in the largest events
    rep = df[df["damage_reported"]]
    dmg = rep["damage"].sort_values(ascending=False).values
    total = dmg.sum()
    conc = pd.DataFrame([{"group": lbl, "share_of_events_pct": q * 100,
                          "share_of_damage_pct": dmg[: max(int(len(dmg) * q), 1)].sum() / total * 100}
                         for lbl, q in [("Costliest 0.1% of events", 0.001),
                                        ("Costliest 1% of events", 0.01),
                                        ("Costliest 10% of events", 0.10)]])
    trop = rep["FLOOD_CAUSE"].eq("Heavy Rain / Tropical System")
    conc.loc[len(conc)] = {"group": "Floods from tropical systems",
                           "share_of_events_pct": trop.mean() * 100,
                           "share_of_damage_pct": rep.loc[trop, "damage"].sum() / total * 100}
    conc.round(2).to_csv(os.path.join(tab_dir, "concentration.csv"), index=False)

    # charts
    r_foot = t_foot["mean_damage"].iloc[-1] / t_foot["mean_damage"].iloc[0]
    bar_chart(t_foot, f"Floods that hit 16+ counties cost ~{approx(r_foot)}x more than one-county floods",
              f"Average property damage per flood episode by number of counties affected, {span}, {dollars}. Log scale.",
              "Average damage per episode", "mean_damage",
              os.path.join(fig_dir, "fig1_damage_by_footprint.png"))
    r_dth = t_foot["deaths_per_1000_episodes"].iloc[-1] / t_foot["deaths_per_1000_episodes"].iloc[0]
    bar_chart(t_foot, f"Wide floods are ~{approx(r_dth)}x as deadly per episode",
              f"Direct deaths per 1,000 flood episodes by number of counties affected, {span}. Log scale.",
              "Deaths per 1,000 episodes", "deaths_per_1000_episodes",
              os.path.join(fig_dir, "fig2_deaths_by_footprint.png"),
              fmt=lambda v: f"{v:,.1f}", axis_fmt=lambda v: f"{v:,g}")
    bar_chart(t_ext, f"Floods spanning 10+ miles cost ~{approx(t_ext['mean_damage'].iloc[-1] / t_ext['mean_damage'].iloc[0])}x more",
              f"Average property damage per event by distance between flood start and end points, {span}, {dollars}. Log scale.",
              "Average damage per event", "mean_damage",
              os.path.join(fig_dir, "fig3_damage_by_extent.png"))
    bar_chart(t_dep, "Deeper water, bigger losses",
              f"Average property damage per event by water depth quoted in the NWS narrative "
              f"(n={int(t_dep['events'].sum()):,}, {t_dep['events'].sum() / len(df) * 100:.0f}% of events), {span}. Log scale.",
              "Average damage per event", "mean_damage",
              os.path.join(fig_dir, "fig4_damage_by_depth.png"))
    bar_chart(t_dur, "Longer floods cost more",
              f"Average property damage per event by duration, {span}, {dollars}. Log scale.",
              "Average damage per event", "mean_damage",
              os.path.join(fig_dir, "fig5_damage_by_duration.png"))
    top = t_cause.iloc[0]
    rain = t_cause.set_index("bin").loc["Heavy Rain"]
    cause_chart(t_cause, f"Tropical-system floods cost ~{approx(top['mean_damage'] / rain['mean_damage'])}x a typical rain flood",
                f"Average property damage per event by flood cause, {span}, {dollars}.",
                os.path.join(fig_dir, "fig6_damage_by_cause.png"))
    concentration_chart(conc, f"The costliest 1% of floods account for {conc['share_of_damage_pct'].iloc[1]:.0f}% of damage",
                        f"Flood and flash flood events, {span}, {dollars}.",
                        os.path.join(fig_dir, "fig7_concentration.png"))

    write_findings(args.out, dollars, df, tables, st, conc)
    print(f"Done. Outputs in {args.out}/ (tables, figures, FINDINGS.md, event-level csv)")


def write_findings(out_dir, dollars, df, tables, st, conc):
    pre, post = df[df["YEAR"] < 2007], df[df["YEAR"] >= 2007]
    pre_blank = pre["DAMAGE_PROPERTY"].isna().mean() * 100 if len(pre) else 0
    post_blank = post["DAMAGE_PROPERTY"].isna().mean() * 100 if len(post) else 0
    pre_zone = (pre["CZ_TYPE"] != "C").mean() * 100 if len(pre) else 0
    t_foot, t_ext, t_dep, t_dur, t_cause, t_type = (tables[k] for k in
        ["by_episode_footprint", "by_extent", "by_water_depth", "by_duration", "by_cause", "by_type"])
    s = st.set_index("proxy")
    c = conc.set_index("group")
    tt = t_type.set_index("bin")

    def md(tab, label, count="events"):
        lines = [f"| {label} | {count.capitalize()} | % with damage | Avg damage | Median (if damaged) | Deaths | Deaths / 1k {count} |",
                 "|---|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            med = money(r["median_damage_if_damaged"]) if pd.notna(r["median_damage_if_damaged"]) else "-"
            lines.append(f"| {r['bin']} | {int(r[count]):,} | {r['pct_with_damage']:.0f}% | {money(r['mean_damage'])} | "
                         f"{med} | {int(r['deaths']):,} | {r[f'deaths_per_1000_{count}']:.1f} |")
        return "\n".join(lines)

    ff, fl = tt.loc["Flash Flood"], tt.loc["Flood"]
    text = f"""# Flood damage vs. severity: findings ({YEAR_MIN}-{YEAR_MAX})

Generated by `scripts/flood_damage_severity.py`. Event types: Flood + Flash Flood, 50 states + DC.
Money is property damage in {dollars}. Every number is recomputed on each run; do not edit by hand.

**{len(df):,} flood events** in **{df['EPISODE_ID'].nunique():,} episodes**, {money(df['damage'].sum())} property damage, {int(df['DEATHS_DIRECT'].sum()):,} direct deaths.

## Headline

- **Footprint is the strongest severity signal.** A flood episode that hits 16+ counties averages {money(t_foot['mean_damage'].iloc[-1])} in damage, vs {money(t_foot['mean_damage'].iloc[0])} for a one-county flood (~{approx(t_foot['mean_damage'].iloc[-1] / t_foot['mean_damage'].iloc[0])}x). Damage grows faster than footprint: elasticity {s.loc['Episode footprint', 'elasticity']:.2f} (95% CI {s.loc['Episode footprint', 'elasticity_ci95_low']:.2f}-{s.loc['Episode footprint', 'elasticity_ci95_high']:.2f}), so doubling the counties hit multiplies damage by about {2 ** s.loc['Episode footprint', 'elasticity']:.1f}x.
- **Wide floods are also far deadlier:** {t_foot['deaths_per_1000_episodes'].iloc[-1]:.0f} deaths per 1,000 episodes at 16+ counties vs {t_foot['deaths_per_1000_episodes'].iloc[0]:.0f} for one county. 16+ county episodes are {t_foot['share_of_episodes_pct'].iloc[-1]:.1f}% of episodes but {t_foot['share_of_damage_pct'].iloc[-1]:.0f}% of damage and {t_foot['share_of_deaths_pct'].iloc[-1]:.0f}% of deaths.
- **Extent and depth point the same way.** Floods spanning 10+ miles average {money(t_ext['mean_damage'].iloc[-1])} vs {money(t_ext['mean_damage'].iloc[0])} under half a mile. Where the narrative quotes a depth, 8+ ft floods average {money(t_dep['mean_damage'].iloc[-1])} vs {money(t_dep['mean_damage'].iloc[0])} under 1 ft.
- **Cause matters:** floods from tropical systems average {money(t_cause.iloc[0]['mean_damage'])} per event; they are {c.loc['Floods from tropical systems', 'share_of_events_pct']:.1f}% of events and {c.loc['Floods from tropical systems', 'share_of_damage_pct']:.0f}% of damage.
- **Extremely concentrated:** the costliest 1% of events (with a damage estimate) carry {c.loc['Costliest 1% of events', 'share_of_damage_pct']:.0f}% of all flood damage; the costliest 0.1% carry {c.loc['Costliest 0.1% of events', 'share_of_damage_pct']:.0f}%.
- **Flash floods kill more:** {int(ff['deaths']):,} deaths ({ff['deaths_per_1000_events']:.1f} per 1k events) vs {int(fl['deaths']):,} for river/areal floods ({fl['deaths_per_1000_events']:.1f} per 1k).

## By episode footprint (counties hit by one flood episode)

{md(t_foot, 'Counties', 'episodes')}

## By extent (distance between start and end points)

{md(t_ext, 'Extent')}

## By water depth quoted in the narrative ({int(t_dep['events'].sum()):,} events, {t_dep['events'].sum() / len(df) * 100:.1f}% of all)

{md(t_dep, 'Depth')}

## By duration

{md(t_dur, 'Duration')}

## By cause

{md(t_cause, 'Cause')}

## Flash Flood vs Flood

{md(t_type, 'Type')}

## Statistics (log-log fit on events with damage > 0)

| Proxy | Unit | Spearman ρ (all) | Elasticity (95% CI) | R² | n with damage |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {p} | {r['unit']} | {r['spearman_rho_all']:.2f} | {r['elasticity']:.2f} "
        f"({r['elasticity_ci95_low']:.2f} to {r['elasticity_ci95_high']:.2f}) | {r['r2']:.2f} | {int(r['n_damaged']):,} |"
        for p, r in s.iterrows()) + """

Elasticity = % change in damage for a 1% change in the proxy. R² values are modest: each proxy explains part of the variation, and damage also depends on what the water hits (homes, roads, farmland).

## Caveats
""" + (f"""
- **Years before 2007 are under-recorded.** {pre_blank:.0f}% of {YEAR_MIN}-2006 flood events have a blank damage field, vs {post_blank:.0f}% from 2007 on. Blank = no estimate: those events are kept for counts, deaths and flood size, add $0 to totals, and are left out of per-event damage averages and shares,
  and {pre_zone:.0f}% of {YEAR_MIN}-2006 events were recorded by forecast zone rather than county (no county FIPS). Treat {YEAR_MIN}-2006 damage totals as a lower bound.""" if YEAR_MIN < 2007 else "") + """
- **No direct flood magnitude.** Storm Events has no flood gauge height or rainfall field; all six measures are proxies.
- **Footprint and extent are partly scale, not intensity.** A wide flood can be shallow. They measure how much area is exposed.
- **Depth covers few events** and comes from free-text narratives parsed with a regex. An LLM could extract depth (and rainfall, rescues, homes flooded) from many more narratives; this is a natural AI extension.
- **Damage is a NWS estimate, not insured loss.** Many events carry $0 because no estimate was made; Helene (2024) is far below published totals.
- **Averages are driven by a few huge events** (e.g. Harvey 2017). Medians of damaged events are shown next to them.
- **Deaths use DEATHS_DIRECT only.**
"""
    with open(os.path.join(out_dir, "FINDINGS.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
