"""
Where did federal flood-mitigation money go? FEMA HMA projects x recorded flood damage x NRI vulnerability.

Inputs
    --hma     OpenFEMA HazardMitigationAssistanceProjects.csv (v4)
    --events  outputs/flood_severity/flood_events.csv.gz  (flood_damage_severity.py)
    --nri     NRI_Table_Counties.csv  (original table; the FINAL one drops SOVI)
    --cpi     CPIAUCSL.csv            (money converted to Aug 2026 dollars, like the rest of the repo)

Definitions
    Flood mitigation project = HMA project type 200-204, 207 (buyouts, relocation, elevation, floodproofing,
        mitigation reconstruction), 303 (floodplain/stream restoration), 403-405 (stormwater, local flood
        control), 500 (berms, levees), plus any FMA / SRL / RFC (flood-only programs) project that is not
        a management cost.
    Funded = status Closed, Approved, Obligated, Awarded or Completed; program fiscal years = the events period.
    Money  = federal share obligated; if missing or 0, project amount x cost-share percentage.
    Priority county = recorded flood damage in the top 25% of counties, NRI social vulnerability
        High or Very high (top 40%), and federal flood mitigation < 1 cent per $1 of recorded damage.

Usage (from the repo root):
    python3 scripts/flood_mitigation.py --hma ~/Downloads/HazardMitigationAssistanceProjects.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

FLOOD_TYPES = {"200", "201", "202", "203", "204", "207", "303", "403", "404", "405", "500"}
FLOOD_PROGRAMS = {"FMA", "SRL", "RFC"}
FUNDED = {"Closed", "Approved", "Obligated", "Awarded", "Completed"}
TYPE_GROUP = {"200": "Buyouts & relocation", "201": "Buyouts & relocation",
              "202": "Elevation & floodproofing", "203": "Elevation & floodproofing",
              "204": "Elevation & floodproofing", "207": "Elevation & floodproofing",
              "303": "Floodplain restoration", "403": "Drainage & flood control",
              "404": "Drainage & flood control", "405": "Drainage & flood control",
              "500": "Drainage & flood control"}
QUINTILES = ["Very low", "Low", "Moderate", "High", "Very high"]
START, END = 2007, 2025   # reset from the events file in main()
LINK = {}                 # linkage diagnostics, filled in load_projects()
BASE_MONTH = "2026-08"

SHADES = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c9c8c2"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"


def money(v):
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def title(fig, text, sub, note=None):
    fig.suptitle(text, x=0.02, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, sub, ha="left", fontsize=9.5, color=INK_2)
    if note:
        fig.text(0.02, 0.005, note, ha="left", fontsize=7.5, color=MUTED)


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path, top=0.86, bottom=0.08):
    fig.subplots_adjust(top=top, bottom=bottom)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- data
def cpi_factor_by_year(cpi_path):
    cpi = pd.read_csv(cpi_path)
    when = pd.to_datetime(cpi.iloc[:, 0])
    val = pd.to_numeric(cpi.iloc[:, 1], errors="coerce")
    base = val[when == pd.Timestamp(BASE_MONTH + "-01")].iloc[0]
    annual = val.groupby(when.dt.year).mean()   # annual mean skips the missing Oct 2025 value
    return (base / annual).to_dict()


def load_projects(hma_path, cpi_path):
    h = pd.read_csv(hma_path, low_memory=False)
    h["code"] = h["projectType"].str.extract(r"^(\d+)\.")[0]
    flood = h["code"].isin(FLOOD_TYPES) | (h["programArea"].isin(FLOOD_PROGRAMS) & ~h["code"].isin({"700", "701"}))
    f = h[flood & h["status"].isin(FUNDED) & h["programFy"].between(START, END)].copy()
    fed = f["federalShareObligated"].where(f["federalShareObligated"] > 0,
                                           f["projectAmount"] * f["costSharePercentage"].fillna(0.75))
    f["federal"] = fed * f["programFy"].map(cpi_factor_by_year(cpi_path))
    f["type_group"] = f["code"].map(TYPE_GROUP).fillna("Other flood-program work")
    n_all = len(f)
    f = f.dropna(subset=["countyCode", "stateNumberCode"])
    f["county_fips"] = (f["stateNumberCode"] * 1000 + f["countyCode"]).astype(int).astype(str).str.zfill(5)
    LINK["no_disaster_number"] = int(f["disasterNumber"].isna().sum())
    LINK["statewide"] = int((f["countyCode"] == 0).sum())
    LINK["connecticut"] = int((f["stateNumberCode"] == 9).sum())
    LINK["total"] = n_all
    print(f"{n_all:,} funded flood-mitigation projects FY{START}-{END}; {len(f):,} with a county "
          f"({money(f['federal'].sum())} federal, Aug 2026 dollars)")
    return f


def county_table(projects, events_path, nri_path):
    e = pd.read_csv(events_path, dtype={"county_fips": str})
    dmg = e.groupby("county_fips").agg(damage=("damage_property", "sum"), deaths=("DEATHS_DIRECT", "sum"),
                                       flood_events=("EVENT_ID", "size"))
    mit = projects.groupby("county_fips").agg(mitigation=("federal", "sum"), projects=("federal", "size"))
    nri = pd.read_csv(nri_path, usecols=["STCOFIPS", "COUNTY", "STATEABBRV", "POPULATION", "SOVI_SCORE",
                                         "IFLD_EALB", "CFLD_EALB"])
    nri["county_fips"] = nri["STCOFIPS"].astype(int).astype(str).str.zfill(5)
    c = nri.set_index("county_fips").join(dmg).join(mit)
    c[["damage", "deaths", "flood_events", "mitigation", "projects"]] = \
        c[["damage", "deaths", "flood_events", "mitigation", "projects"]].fillna(0)
    c["nri_flood_eal_buildings"] = c["IFLD_EALB"].fillna(0) + c["CFLD_EALB"].fillna(0)
    c["vulnerability"] = pd.qcut(c["SOVI_SCORE"], 5, labels=QUINTILES)
    c["has_project"] = c["mitigation"] > 0
    c["cents_per_dollar_damage"] = 100 * c["mitigation"] / c["damage"].replace(0, np.nan)
    c["county_size"] = pd.qcut(c["POPULATION"], 4, labels=["Smallest 25%", "Small-mid", "Mid-large", "Largest 25%"])
    c["priority"] = ((c["damage"] >= c["damage"].quantile(0.75))
                     & c["vulnerability"].isin(["High", "Very high"])
                     & (c["mitigation"] < 0.01 * c["damage"]))
    return c.reset_index()


def access_gap(c):
    """Among counties with recorded flood damage: share with >=1 project, recorded vs expected
    for counties in the same state and the same damage-size quartile."""
    s = c[c["damage"] > 0].copy()
    s["state"] = s["county_fips"].str[:2]
    s["dmg_band"] = pd.qcut(np.log10(s["damage"].clip(lower=1)), 4, labels=False)
    s["expected"] = s.groupby(["state", "dmg_band"])["has_project"].transform("mean")
    g = s.groupby("vulnerability", observed=True).agg(counties=("has_project", "size"),
                                                     recorded=("has_project", "mean"),
                                                     expected=("expected", "mean"))
    g[["recorded", "expected"]] *= 100
    g["recorded_over_expected"] = g["recorded"] / g["expected"]
    return g.reset_index()


# ----------------------------------------------------------------------------- charts
def fig_access(g, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(g))[::-1]
    for yi, r in zip(y, g.itertuples()):
        ax.plot([r.expected, r.recorded], [yi, yi], color=GRID, linewidth=3, zorder=1)
        ax.scatter([r.expected], [yi], s=70, color="white", edgecolor=MUTED, linewidth=1.5, zorder=2)
        ax.scatter([r.recorded], [yi], s=80, color=SHADES[list(g["vulnerability"]).index(r.vulnerability)],
                   zorder=3)
        ax.annotate(f"{r.recorded:.0f}%", (r.recorded, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.annotate(f"{r.recorded_over_expected:.2f}x expected", (max(r.recorded, r.expected), yi),
                    xytext=(14, -4), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [f"{v} vulnerability" for v in g["vulnerability"]])
    ax.set_xlim(15, 72)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_xlabel(f"Share of flood-damaged counties with at least one funded FEMA flood-mitigation project, FY{START}-{END}",
                  color=INK_2, fontsize=9)
    style(ax, grid="x")
    ax.scatter([], [], s=80, color=SHADES[2], label="Recorded")
    ax.scatter([], [], s=70, color="white", edgecolor=MUTED, linewidth=1.5,
               label="Expected for counties in the same state with similar flood damage")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    vh = g.iloc[-1]
    title(fig, f"The most vulnerable counties are {100 * (1 - vh.recorded_over_expected):.0f}% less likely "
               f"to get flood-mitigation money",
          f"Counties with recorded flood damage {START}-{END} (n={int(g['counties'].sum()):,}), grouped into fifths by "
          f"FEMA NRI Social Vulnerability.",
          note="Mitigation = FEMA Hazard Mitigation Assistance flood projects (buyouts, elevation, drainage, levees, FMA/SRL/RFC). "
               "Excludes state, local and Army Corps projects.")
    save(fig, path, top=0.78, bottom=0.12)


def fig_capacity(c, path):
    s = c[c["damage"] > 0]
    t = s.pivot_table(index="county_size", columns="vulnerability", values="has_project", aggfunc="mean",
                      observed=True) * 100
    n = s.pivot_table(index="county_size", columns="vulnerability", values="has_project", aggfunc="size",
                      observed=True)
    t = t.iloc[::-1]
    n = n.iloc[::-1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(t.values, cmap=cmap, vmin=0, vmax=85, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.0f}%\n(n={int(n.values[i, j])})", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 45 else INK)
    ax.set_xticks(range(t.shape[1]), [f"{v}\nvulnerability" for v in t.columns], fontsize=9, color=INK_2)
    ax.set_yticks(range(t.shape[0]), [f"{v} by population" for v in t.index], fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, t.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, t.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    rate = s.groupby("county_size", observed=True)["has_project"].mean() * 100
    small, large = rate["Smallest 25%"], rate["Largest 25%"]
    title(fig, f"Small counties rarely get funded: {small:.0f}% vs {large:.0f}% of the largest",
          f"Share of flood-damaged counties with at least one funded FEMA flood-mitigation project, FY{START}-{END}. "
          "Darker = more likely.",
          note="Rows = county population quartiles; columns = NRI Social Vulnerability fifths. "
               "Applying for FEMA grants takes staff, engineering studies and matching funds.")
    save(fig, path, top=0.83, bottom=0.1)


def fig_mismatch(c, path):
    s = c[c["damage"] > 1e4].copy()
    floor = 3e3
    rng = np.random.default_rng(0)
    jitter = floor * 10 ** rng.uniform(-0.25, 0.2, len(s))
    s["y"] = s["mitigation"].where(s["mitigation"] > 0, jitter)
    fig, ax = plt.subplots(figsize=(10, 6))
    other = s[~s["priority"]]
    pr = s[s["priority"]]
    ax.scatter(other["damage"], other["y"], s=10, color=BLUE, alpha=0.3, linewidths=0, label="Other counties")
    ax.scatter(pr["damage"], pr["y"], s=26, color=ORANGE, alpha=0.9, linewidths=0,
               label=f"Priority: top-25% damage, high vulnerability, <1¢ per $1 ({len(pr)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 3e10)
    ax.set_ylim(1.5e3, 3e9)
    ax.axhspan(1.5e3, 5e3, color="#f4f3ef", zorder=0)
    ax.text(1.2e4, 5.6e3, "No FEMA flood-mitigation project (band below)", fontsize=8, color=INK_2, va="bottom")
    xs = np.array([1e4, 3e10])
    ax.plot(xs, xs * 0.07, color=MUTED, linestyle="--", linewidth=1)
    ax.text(2e9, 2e9 * 0.07 * 1.6, "7¢ per $1 (national average)", fontsize=8, color=MUTED, rotation=33,
            ha="center")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: money(v) if v >= 1e4 else "")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(f"Recorded flood property damage, {START}-{END}", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Federal flood-mitigation money, FY{START}-{END}", color=INK_2, fontsize=9)
    style(ax, grid=None)
    ax.grid(True, color=GRID, linewidth=0.6)
    funded = pr[pr["mitigation"] > 0].nlargest(3, "damage")
    for r in funded.itertuples():
        ax.annotate(f"{r.COUNTY}, {r.STATEABBRV}", (r.damage, r.y), xytext=(6, -3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    zero = pr[pr["mitigation"] == 0].nlargest(6, "damage")
    lines = [f"{r.COUNTY}, {r.STATEABBRV}: {money(r.damage)}" + (f", {int(r.deaths)} deaths" if r.deaths >= 10 else "")
             for r in zero.itertuples()]
    ax.text(1.02, 0.02, "Largest priority counties with\nno FEMA flood-mitigation project:\n\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=8, color=INK, va="bottom", ha="left", linespacing=1.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    title(fig, f"{len(pr)} hard-hit, vulnerable counties received almost no flood-mitigation money",
          "Each dot is a county. Orange = recorded damage in the top 25%, High/Very high social vulnerability, "
          "and under 1¢ of FEMA mitigation per $1 of damage.",
          note="Aug 2026 dollars. Damage = NOAA Storm Events (Flood + Flash Flood). Mitigation = FEMA HMA flood projects only.")
    save(fig, path, top=0.86, bottom=0.1)


# ----------------------------------------------------------------------------- findings
def write_findings(out, projects, c, g):
    total_d, total_m = c["damage"].sum(), c["mitigation"].sum()
    pr = c[c["priority"]].sort_values("damage", ascending=False)
    by_type = projects.groupby("type_group")["federal"].sum().sort_values(ascending=False)
    small = c[(c["damage"] > 0) & (c["county_size"] == "Smallest 25%")]["has_project"].mean() * 100
    large = c[(c["damage"] > 0) & (c["county_size"] == "Largest 25%")]["has_project"].mean() * 100
    rows = "\n".join(f"| {r.COUNTY}, {r.STATEABBRV} | {money(r.damage)} | {int(r.deaths)} | {money(r.mitigation)} | "
                     f"{r.SOVI_SCORE:.0f} | {int(r.POPULATION):,} |" for r in pr.head(25).itertuples())
    text = f"""# Federal flood-mitigation money vs flood damage and vulnerability ({START}-{END})

Generated by `scripts/flood_mitigation.py`. Aug 2026 dollars. Do not edit by hand.

## Headline

- **{len(projects):,}** funded FEMA flood-mitigation projects (FY{START}-{END}) with a county: **{money(total_m)}** federal.
  Recorded flood property damage over the same years: **{money(total_d)}**. That is about **{100 * total_m / total_d:.0f}¢ of mitigation per $1 of recorded damage**.
- Spending by type: """ + "; ".join(f"{k} {money(v)}" for k, v in by_type.items()) + f""".
- **Access gap:** among counties with flood damage, the most socially vulnerable fifth got at least one project **{g.recorded.iloc[-1]:.0f}%** of the time,
  vs **{g.recorded.iloc[0]:.0f}-{g.recorded.iloc[1]:.0f}%** for the least vulnerable. Compared with counties in the same state and with similar
  flood damage, that is **{g.recorded_over_expected.iloc[-1]:.2f}x** expected.
- **Capacity:** {small:.0f}% of the smallest counties (by population) got a project, vs {large:.0f}% of the largest. Within the same size group,
  vulnerability matters much less, so the gap mostly reflects which counties can apply (staff, studies, matching funds).
- **{len(pr)} priority counties**: top-25% flood damage, High/Very high vulnerability, under 1¢ of mitigation per $1 of damage.

## Access by vulnerability (counties with recorded flood damage)

| Vulnerability | Counties | With a project | Expected (same state, similar damage) | Recorded / expected |
|---|---|---|---|---|
""" + "\n".join(f"| {r.vulnerability} | {r.counties:,} | {r.recorded:.0f}% | {r.expected:.0f}% | {r.recorded_over_expected:.2f} |"
                  for r in g.itertuples()) + f"""

## Priority counties (top 25 by recorded damage)

| County | Recorded flood damage | Direct deaths | FEMA flood mitigation | Vulnerability score | Population |
|---|---|---|---|---|---|
{rows}

Full list: `tables/priority_counties.csv`.

## Policy suggestions the data supports

1. **Fund capacity, not only projects.** Small and vulnerable counties rarely apply. Pre-application technical assistance
   (grant writers, engineering studies) and lower local cost-share for high-vulnerability counties target the gap directly.
2. **Use a priority list like this one** (high damage + high vulnerability + little mitigation) to direct outreach before the next flood.
3. **Do not rely on recorded damage alone** to allocate money: NOAA damage is under-recorded in most counties (see
   `outputs/flood_vulnerability/FINDINGS.md`), which would push money away from places with weak reporting.

## Caveats

- Only FEMA Hazard Mitigation Assistance. State, local, Army Corps (USACE) and private flood projects are not included,
  so "no project" means no FEMA HMA flood project, not no protection at all.
- **Linking counties.** NOAA and HMA both use state FIPS + county FIPS. Projects that cannot be placed in one county are left out:
  {LINK.get('statewide', 0):,} statewide/multi-county projects (county code 0) and {LINK.get('connecticut', 0):,} Connecticut projects
  (NRI uses Connecticut's 2022 planning regions, which do not map one-to-one to the old counties NOAA and HMA use).
  Multi-county projects with a listed county are assigned to that county.
- **Projects are selected by what the money does, not by disaster type.** A home elevation after a hurricane counts as flood
  mitigation; generators or wind retrofits after a flood do not. {LINK.get('no_disaster_number', 0):,} of the flood projects have no
  disaster number (pre-disaster programs such as FMA, PDM, BRIC), so filtering on incident type would have dropped them.
- HMA v4 was frozen on 2026-08-31; pending projects are excluded.
- Projects can lag the floods they respond to; FY and flood years are compared over the same {END - START + 1}-year window.
- Associations, not causes.
"""
    with open(os.path.join(out, "FINDINGS.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hma", default="HazardMitigationAssistanceProjects.csv")
    ap.add_argument("--events", default="outputs/flood_severity/flood_events.csv.gz")
    ap.add_argument("--nri", default="NRI_Table_Counties.csv")
    ap.add_argument("--cpi", default="CPIAUCSL.csv")
    ap.add_argument("--out", default="outputs/flood_mitigation")
    args = ap.parse_args()
    global START, END
    yrs = pd.read_csv(args.events, usecols=["YEAR"])["YEAR"]
    START, END = int(yrs.min()), int(yrs.max())
    fig_dir, tab_dir = os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    projects = load_projects(os.path.expanduser(args.hma), args.cpi)
    c = county_table(projects, args.events, args.nri)
    g = access_gap(c)

    cols = ["county_fips", "COUNTY", "STATEABBRV", "POPULATION", "SOVI_SCORE", "vulnerability", "county_size",
            "damage", "deaths", "flood_events", "mitigation", "projects", "cents_per_dollar_damage",
            "nri_flood_eal_buildings", "priority"]
    c[cols].round(2).to_csv(os.path.join(tab_dir, "county_mitigation.csv"), index=False)
    c[c["priority"]].sort_values("damage", ascending=False)[cols].round(2).to_csv(
        os.path.join(tab_dir, "priority_counties.csv"), index=False)
    g.round(3).to_csv(os.path.join(tab_dir, "access_by_vulnerability.csv"), index=False)

    fig_access(g, os.path.join(fig_dir, "fig1_access_gap.png"))
    fig_capacity(c, os.path.join(fig_dir, "fig2_county_size_vs_vulnerability.png"))
    fig_mismatch(c, os.path.join(fig_dir, "fig3_damage_vs_mitigation.png"))
    write_findings(args.out, projects, c, g)
    print(f"Done. {int(c['priority'].sum())} priority counties. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
