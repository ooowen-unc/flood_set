"""
County maps of flood property damage and flood level (Flood + Flash Flood; period taken from the events file).

Saves PNG images by default (small, easy to put in slides and the README):
    map1_damage_level.png    county colour = average yearly property damage, in six levels
    map2_extreme_floods.png  county colour = how many L5 Extreme floods (16+ counties) hit the county
    map4_mitigation_gap.png  county colour = priority counties vs counties that got FEMA flood-mitigation money
                             (only if outputs/flood_mitigation/tables/county_mitigation.csv exists;
                              run scripts/flood_mitigation.py first)
With --format html (or both) it also writes interactive HTML versions (zoom, hover for details),
including map3_damage_by_year.html with a year slider. HTML files are large, so they are not kept in git.

Flood level = counties hit by one flood episode (same definition as flood_damage_levels.py):
    L1 Local 1 | L2 Small 2-3 | L3 Medium 4-7 | L4 Large 8-15 | L5 Extreme 16+
Darker blue always means more damage or more extreme floods.

Needs:  pip3 install -U plotly kaleido pandas   (kaleido writes the PNGs; it uses the Chrome you already have)
        internet access the first run (downloads US county shapes, then caches them)
Usage (from the repo root):
        python3 scripts/flood_maps.py                  # PNGs
        python3 scripts/flood_maps.py --format html    # interactive HTML maps
"""

import argparse
import json
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

COUNTIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
LEVEL_BINS = [1, 2, 4, 8, 16, np.inf]
LEVELS = ["L1 Local (1 county)", "L2 Small (2-3)", "L3 Medium (4-7)", "L4 Large (8-15)", "L5 Extreme (16+)"]
DMG_BINS = [-1, 0, 1e4, 1e5, 1e6, 1e7, np.inf]
DMG_LEVELS = ["$0 recorded", "< $10K a year", "$10K-100K", "$100K-1M", "$1M-10M", "$10M+ a year"]
EXT_BINS = [-1, 0, 1, 3, 6, 10, np.inf]
EXT_LEVELS = ["None", "1", "2-3", "4-6", "7-10", "11+"]
SIX = ["#e4e3df", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BLUES = [[0.0, "#eef4fc"], [0.25, "#b7d3f6"], [0.5, "#5598e7"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
DOLLARS = "Aug 2026 dollars"
PERIOD, YEARS = "2007-2025", 19   # reset from the events file in prepare()
FORMAT = "png"                    # set from --format in main()


def money(v):
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


# ----------------------------------------------------------------------------- data
def prepare(events_path, nri_path=None):
    """Return (county table, county x year table). Pure pandas, no plotting."""
    global PERIOD, YEARS
    e = pd.read_csv(events_path, dtype={"county_fips": str})
    PERIOD, YEARS = f"{e['YEAR'].min()}-{e['YEAR'].max()}", e["YEAR"].max() - e["YEAR"].min() + 1
    e = e.dropna(subset=["county_fips"])            # forecast-zone records have no county
    e["county_fips"] = e["county_fips"].str.zfill(5)
    e["level_num"] = pd.cut(e["episode_counties"], LEVEL_BINS, right=False, labels=False) + 1

    ep = e.groupby(["county_fips", "EPISODE_ID"])["episode_counties"].first().reset_index()
    extreme = ep[ep["episode_counties"] >= 16].groupby("county_fips").size()

    c = e.groupby("county_fips").agg(
        state=("STATE", "first"), county=("CZ_NAME", "first"),
        events=("EVENT_ID", "size"), damage=("damage_property", "sum"),
        deaths=("DEATHS_DIRECT", "sum"), worst_level=("level_num", "max"))
    if nri_path and os.path.exists(nri_path):
        nri = pd.read_csv(nri_path, dtype=str, engine="python", on_bad_lines="skip")
        nri.columns = nri.columns.str.strip().str.upper()
        nri = nri[["STCOFIPS", "COUNTY", "STATEABBRV", "POPULATION", "SOVI_RATNG"]].dropna(subset=["STCOFIPS"])
        nri["POPULATION"] = pd.to_numeric(nri["POPULATION"], errors="coerce")
        nri["county_fips"] = pd.to_numeric(nri["STCOFIPS"], errors="coerce").dropna().astype(int).astype(str).str.zfill(5)
        nri = nri.dropna(subset=["county_fips"])
        c = c.join(nri.set_index("county_fips")[["COUNTY", "STATEABBRV", "POPULATION", "SOVI_RATNG"]])
        c["name"] = c["COUNTY"].fillna(c["county"].str.title()) + ", " + c["STATEABBRV"].fillna(c["state"].str.title())
        c["vulnerability"] = c["SOVI_RATNG"].fillna("n/a")
    else:
        c["name"] = c["county"].str.title() + ", " + c["state"].str.title()
        c["vulnerability"] = "n/a"
    c["worst_level"] = c["worst_level"].astype(int)
    c["extreme_floods"] = extreme.reindex(c.index, fill_value=0).astype(int)
    c["extreme_label"] = pd.cut(c["extreme_floods"], EXT_BINS, labels=EXT_LEVELS).astype(str)
    c["damage_per_year"] = c["damage"] / YEARS
    c["damage_level"] = pd.cut(c["damage_per_year"], DMG_BINS, labels=DMG_LEVELS).astype(str)
    c["level_label"] = c["worst_level"].map(dict(enumerate(LEVELS, 1)))
    c["log_damage"] = np.log10(c["damage"].clip(lower=1e3))
    c["damage_text"] = c["damage"].map(money)
    c = c.reset_index()

    y = e.groupby(["county_fips", "YEAR"]).agg(damage=("damage_property", "sum"),
                                               events=("EVENT_ID", "size")).reset_index()
    y = y[y["damage"] > 0].copy()
    y["log_damage"] = np.log10(y["damage"].clip(lower=1e3))
    y["damage_text"] = y["damage"].map(money)
    y = y.merge(c[["county_fips", "name"]], on="county_fips", how="left")
    return c, y


def load_counties(cache):
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    print(f"Downloading county shapes from {COUNTIES_URL} ...")
    try:
        raw = urllib.request.urlopen(COUNTIES_URL, timeout=60).read()
    except urllib.error.URLError as err:
        if isinstance(getattr(err, "reason", None), ssl.SSLError):
            raise SystemExit("SSL certificate error. On macOS run: /Applications/Python*/Install\\ Certificates.command, "
                             "or download the URL above in a browser and save it as " + cache)
        raise
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        fh.write(raw)
    return json.loads(raw)


def save(fig, out, html_only=False):
    """Write fig as PNG and/or HTML depending on --format. out has no extension."""
    if FORMAT in ("html", "both"):
        fig.write_html(out + ".html", include_plotlyjs="cdn")
        print(f"  wrote {out}.html")
    if FORMAT in ("png", "both") and not html_only:
        fig.write_image(out + ".png", width=1500, height=900, scale=2)
        print(f"  wrote {out}.png")


# ----------------------------------------------------------------------------- maps
def log_colorbar(lo=3, hi=10):
    ticks = list(range(lo, hi + 1))
    return dict(title=dict(text="Property damage", side="right"), tickvals=ticks,
                ticktext=[money(10 ** t) for t in ticks], thickness=14, len=0.7)


def layout(fig, title, subtitle):
    fig.update_geos(scope="usa", showlakes=False, bgcolor="white")
    fig.update_traces(marker_line_width=0.2, marker_line_color="white")
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:#52514e'>{subtitle}</span>",
                   x=0.02, xanchor="left"),
        font=dict(family="Helvetica, Arial, sans-serif", color="#0b0b0b"),
        margin=dict(l=10, r=10, t=90, b=10), paper_bgcolor="white", height=640)
    return fig


def map_total(px, c, geo, out):
    fig = px.choropleth(
        c, geojson=geo, locations="county_fips", color="damage_level",
        color_discrete_map=dict(zip(DMG_LEVELS, SIX)), category_orders={"damage_level": DMG_LEVELS[::-1]},
        scope="usa", hover_name="name",
        hover_data={"county_fips": False, "damage_level": False, "damage_text": True, "events": True,
                    "deaths": True, "extreme_floods": True, "vulnerability": True},
        labels={"damage_level": "Yearly damage", "damage_text": f"Total damage {PERIOD}",
                "events": "Flood events", "deaths": "Direct deaths",
                "extreme_floods": "L5 extreme floods", "vulnerability": "Social vulnerability (NRI)"})
    fig.update_layout(legend=dict(title="Average property<br>damage per year", x=1.0, y=0.5))
    top = c.nlargest(3, "damage")
    n10 = int((c["damage_per_year"] >= 1e7).sum())
    layout(fig, f"{n10} counties average $10M+ a year in flood damage",
           f"Flood + Flash Flood property damage per year, {PERIOD} ({DOLLARS}). Darker = more damage. "
           f"Top: " + ", ".join(f"{r.name} {money(r.damage)}" for r in top.itertuples()) + ".")
    save(fig, out)


def map_level(px, c, geo, out):
    fig = px.choropleth(
        c, geojson=geo, locations="county_fips", color="extreme_label",
        color_discrete_map=dict(zip(EXT_LEVELS, SIX)), category_orders={"extreme_label": EXT_LEVELS[::-1]},
        scope="usa", hover_name="name",
        hover_data={"county_fips": False, "extreme_label": False, "extreme_floods": True,
                    "damage_text": True, "events": True, "vulnerability": True},
        labels={"extreme_floods": "L5 extreme floods", "damage_text": f"Total damage {PERIOD}",
                "events": "Flood events", "vulnerability": "Social vulnerability (NRI)"})
    fig.update_layout(legend=dict(title="L5 extreme floods<br>(16+ counties)", x=1.0, y=0.5))
    hit = int((c["extreme_floods"] > 0).sum())
    many = int((c["extreme_floods"] >= 7).sum())
    layout(fig, f"{many} counties were caught in 7 or more extreme floods",
           f"Number of L5 Extreme flood episodes (one flood hitting 16+ counties) per county, {PERIOD}. "
           f"{hit:,} of {len(c):,} counties were hit at least once. Darker = more often.")
    save(fig, out)


def map_years(px, y, geo, out):
    y = y.sort_values("YEAR")
    fig = px.choropleth(
        y, geojson=geo, locations="county_fips", color="log_damage", animation_frame="YEAR",
        color_continuous_scale=BLUES, range_color=(3, 10), scope="usa",
        hover_name="name",
        hover_data={"county_fips": False, "log_damage": False, "damage_text": True, "events": True},
        labels={"damage_text": "Property damage", "events": "Flood events", "YEAR": "Year"})
    fig.update_layout(coloraxis_colorbar=log_colorbar())
    layout(fig, "Flood property damage, year by year",
           f"Press play or drag the slider. County colour = that year's Flood + Flash Flood damage ({DOLLARS}); "
           f"blank = no recorded damage.")
    save(fig, out, html_only=True)   # an animation cannot be a PNG; see the GIF


GAP_LEVELS = ["Priority: high damage, high vulnerability, <1¢ mitigation per $1",
              "Got FEMA flood-mitigation money", "Flood damage, no FEMA mitigation project",
              "No recorded flood damage"]
GAP_COLORS = ["#eb6834", "#2a78d6", "#b7d3f6", "#e4e3df"]


def gap_table(path):
    m = pd.read_csv(path, dtype={"county_fips": str})
    m["county_fips"] = m["county_fips"].str.zfill(5)
    m["group"] = np.select([m["priority"].astype(bool), m["mitigation"] > 0, m["damage"] > 0],
                           GAP_LEVELS[:3], default=GAP_LEVELS[3])
    m["name"] = m["COUNTY"] + ", " + m["STATEABBRV"]
    m["damage_text"] = m["damage"].map(money)
    m["mitigation_text"] = m["mitigation"].map(money)
    m["cents"] = (m["cents_per_dollar_damage"].fillna(0)).round(1)
    return m


def map_gap(px, m, geo, out):
    fig = px.choropleth(
        m, geojson=geo, locations="county_fips", color="group",
        color_discrete_map=dict(zip(GAP_LEVELS, GAP_COLORS)), category_orders={"group": GAP_LEVELS},
        scope="usa", hover_name="name",
        hover_data={"county_fips": False, "group": False, "damage_text": True, "mitigation_text": True,
                    "cents": True, "vulnerability": True, "deaths": True},
        labels={"damage_text": f"Flood damage {PERIOD}", "mitigation_text": "FEMA flood mitigation",
                "cents": "Cents per $1 of damage", "vulnerability": "Social vulnerability", "deaths": "Direct deaths"})
    fig.update_layout(legend=dict(title="", orientation="h", x=0, y=-0.02))
    n = int((m["group"] == GAP_LEVELS[0]).sum())
    layout(fig, f"{n} hard-hit, vulnerable counties got almost no flood-mitigation money",
           f"Orange = top-25% flood damage {PERIOD}, High/Very high social vulnerability (NRI), and under 1¢ of FEMA "
           "flood mitigation per $1 of damage. Blue = got FEMA flood-mitigation money.")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="outputs/flood_severity/flood_events.csv.gz")
    ap.add_argument("--nri", default="NRI_Table_Counties.csv")
    ap.add_argument("--out", default="outputs/flood_maps")
    ap.add_argument("--geo-cache", default="data/geo/geojson-counties-fips.json")
    ap.add_argument("--mitigation", default="outputs/flood_mitigation/tables/county_mitigation.csv")
    ap.add_argument("--format", choices=("png", "html", "both"), default="png")
    args = ap.parse_args()
    global FORMAT
    FORMAT = args.format

    c, y = prepare(args.events, args.nri)
    os.makedirs(args.out, exist_ok=True)
    c.to_csv(os.path.join(args.out, "county_flood_summary.csv"), index=False)
    print(f"{len(c):,} counties with flood events; {len(y):,} county-years with damage")

    import plotly.express as px  # imported here so the data step can run without plotly
    geo = load_counties(args.geo_cache)
    map_total(px, c, geo, os.path.join(args.out, "map1_damage_level"))
    map_level(px, c, geo, os.path.join(args.out, "map2_extreme_floods"))
    map_years(px, y, geo, os.path.join(args.out, "map3_damage_by_year"))
    if os.path.exists(args.mitigation):
        map_gap(px, gap_table(args.mitigation), geo, os.path.join(args.out, "map4_mitigation_gap"))
    else:
        print(f"Skipped map4: {args.mitigation} not found (run scripts/flood_mitigation.py first)")
    print(f"Done. Maps are in {args.out}/")


if __name__ == "__main__":
    main()
