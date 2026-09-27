"""
Flood property damage by flood level, seen from several angles (Flood + Flash Flood; period taken from the events file).

Flood level = how many counties one flood episode hit (EPISODE_ID), the strongest severity
signal in flood_damage_severity.py:
    L1 Local 1 county | L2 Small 2-3 | L3 Medium 4-7 | L4 Large 8-15 | L5 Extreme 16+
Darker blue always means a more severe level (or a bigger damage bucket).

Charts (no plain bar charts):
    fig1  two donuts: share of flood episodes vs share of property damage, by level
    fig2  100% stacked area: each year's damage split by level
    fig3  heatmap: top states x level, cell = property damage
    fig4  100% stacked bars: flood cause x damage size of each event
    fig5  concentration curve: share of damage vs share of events (costliest first)

Input: outputs/flood_severity/flood_events.csv.gz (from flood_damage_severity.py).
Usage (from the repo root):
    python scripts/flood_damage_levels.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd

LEVELS = ["L1 Local\n1 county", "L2 Small\n2-3", "L3 Medium\n4-7", "L4 Large\n8-15", "L5 Extreme\n16+"]
LEVEL_SHORT = ["L1 Local", "L2 Small", "L3 Medium", "L4 Large", "L5 Extreme"]
LEVEL_BINS = [1, 2, 4, 8, 16, np.inf]
# ordinal blue ramp: lightest step still visible on white, darkest = most severe
SHADES = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
NO_DAMAGE = "#dcdbd5"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
DOLLARS = "Aug 2026 dollars"
PERIOD = "2007-2025"   # reset from the events file in load()


def money(v):
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M" if v >= 1e8 else f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def title(fig, text, sub, note=None):
    fig.suptitle(text, x=0.02, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, sub, ha="left", fontsize=9.5, color=INK_2)
    if note:
        fig.text(0.02, 0.01, note, ha="left", fontsize=7.5, color=MUTED)


def save(fig, path, top=0.88, bottom=0.06):
    fig.subplots_adjust(top=top, bottom=bottom)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load(path):
    global PERIOD
    e = pd.read_csv(path)
    PERIOD = f"{e['YEAR'].min()}-{e['YEAR'].max()}"
    e["level"] = pd.cut(e["episode_counties"], LEVEL_BINS, right=False, labels=LEVEL_SHORT)
    ep = e.groupby("EPISODE_ID").agg(level=("level", "first"), damage=("damage_property", "sum"),
                                     deaths=("DEATHS_DIRECT", "sum"), year=("YEAR", "min"))
    return e, ep


# ----------------------------------------------------------------------------- fig1 donuts
def fig_donuts(ep, path):
    g = ep.groupby("level", observed=True).agg(episodes=("damage", "size"), damage=("damage", "sum"),
                                               deaths=("deaths", "sum"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1, 1]})
    specs = [("episodes", "Flood\nepisodes", f"{int(g['episodes'].sum()):,}"),
             ("damage", "Property\ndamage", money(g["damage"].sum())),
             ("deaths", "Direct\ndeaths", f"{int(g['deaths'].sum()):,}")]
    for ax, (col, label, total) in zip(axes, specs):
        share = g[col] / g[col].sum() * 100
        wedges, _ = ax.pie(g[col], colors=SHADES, startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
        for w, s in zip(wedges, share):
            if s < 3:
                continue
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            r = 0.81
            ax.text(r * np.cos(ang), r * np.sin(ang), f"{s:.0f}%", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="white" if w.get_facecolor()[0] < 0.5 else INK)
        ax.text(0, 0.1, label, ha="center", va="center", fontsize=10, fontweight="bold", color=INK,
                linespacing=1.1)
        ax.text(0, -0.2, total, ha="center", va="center", fontsize=9, color=INK_2)
        ax.set_aspect("equal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SHADES]
    fig.legend(handles, [l.replace("\n", " · ") for l in LEVELS], loc="lower center", ncol=5,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02), title="Flood level (counties hit by one episode)",
               title_fontsize=9)
    big = g.iloc[-1]
    title(fig, f"Extreme floods are {big['episodes'] / g['episodes'].sum() * 100:.1f}% of episodes "
               f"but {big['damage'] / g['damage'].sum() * 100:.0f}% of the damage",
          f"Flood + Flash Flood, {PERIOD}. Same five shades in every ring; darker = more counties hit. "
          f"Property damage in {DOLLARS}.")
    save(fig, path, top=0.84, bottom=0.14)
    return g


# ----------------------------------------------------------------------------- fig2 area
def fig_area(ep, path):
    t = ep.pivot_table(index="year", columns="level", values="damage", aggfunc="sum", observed=True).fillna(0)
    share = t.div(t.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(share.index, [share[c] for c in LEVEL_SHORT], colors=SHADES, edgecolor="white", linewidth=0.8)
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_xticks(range(share.index.min(), share.index.max() + 1, 2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_ylabel("Share of that year's flood damage", color=INK_2, fontsize=9)
    # label each band at its thickest year, directly on the chart
    cum = share[LEVEL_SHORT].cumsum(axis=1)
    for i, c in enumerate(LEVEL_SHORT):
        yr = share[c].idxmax()
        lo = cum.loc[yr].iloc[i - 1] if i else 0
        mid = lo + share.loc[yr, c] / 2
        if share.loc[yr, c] > 8:
            ax.text(yr, mid, c, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if i >= 2 else INK)
    tot = t.sum(axis=1)
    costliest = ", ".join(f"{yr} ({money(tot[yr])})" for yr in tot.nlargest(3).index)
    l5 = share["L5 Extreme"]
    title(fig, f"In most years, the few extreme floods carry the largest share of damage",
          f"Each year's property damage split by flood level (100% = that year's total). "
          f"L5 share: median {l5.median():.0f}%, {int((l5 > 50).sum())} of {len(l5)} years above 50%.",
          note=f"Flood + Flash Flood episodes, {DOLLARS}. Costliest years: {costliest}.")
    save(fig, path, top=0.83, bottom=0.1)
    return share


# ----------------------------------------------------------------------------- fig3 heatmap
def fig_heatmap(e, path, n_states=15):
    tot = e.groupby("STATE")["damage_property"].sum().nlargest(n_states)
    x = e[e["STATE"].isin(tot.index)]
    h = x.pivot_table(index="STATE", columns="level", values="damage_property", aggfunc="sum",
                      observed=True).reindex(index=tot.index, columns=LEVEL_SHORT).fillna(0)
    cmap = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(10, 7.2))
    vals = h.values.astype(float)
    im = ax.imshow(np.where(vals > 0, vals, np.nan), cmap=cmap, norm=LogNorm(vmin=1e6, vmax=vals.max()),
                   aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = money(v) if v > 0 else "-"
            dark = v > 3e8
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if dark else INK)
    ax.set_xticks(range(5), LEVELS, fontsize=9, color=INK_2)
    ax.set_yticks(range(len(h)), [f"{s.title()}  ({money(tot[s])})" for s in h.index], fontsize=9, color=INK_2)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5), minor=True)
    ax.set_yticks(np.arange(-0.5, len(h)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    tx = h.loc["TEXAS", "L5 Extreme"] if "TEXAS" in h.index else np.nan
    top_cell = h.stack().idxmax()
    title(fig, f"Texas alone lost {money(tx)} to extreme floods; elsewhere damage spreads across levels",
          f"Property damage by state and flood level, top {n_states} states by total damage "
          f"(total in brackets), {PERIOD}. Darker = more damage (log scale).",
          note=f"Flood + Flash Flood, {DOLLARS}. Level = counties hit by the whole episode, which can cross state lines.")
    save(fig, path, top=0.82, bottom=0.05)
    return h


# ----------------------------------------------------------------------------- fig4 stacked bars
DMG_BINS = [-1, 0, 1e4, 1e5, 1e6, 1e7, np.inf]
DMG_LABELS = ["$0 recorded", "< $10K", "$10K-100K", "$100K-1M", "$1M-10M", "$10M+"]


def fig_cause(e, path):
    x = e[e["damage_reported"]].copy()   # blank = no estimate; left out of damage-size shares
    x["bucket"] = pd.cut(x["damage_property"], DMG_BINS, labels=DMG_LABELS)
    t = pd.crosstab(x["FLOOD_CAUSE"], x["bucket"], normalize="index").reindex(columns=DMG_LABELS).fillna(0) * 100
    n = x["FLOOD_CAUSE"].value_counts()
    big = t["$1M-10M"] + t["$10M+"]
    t = t.loc[big.sort_values().index]
    colors = [NO_DAMAGE] + SHADES
    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(t))
    y = np.arange(len(t))
    for col, c in zip(DMG_LABELS, colors):
        ax.barh(y, t[col], left=left, color=c, height=0.66, edgecolor="white", linewidth=1.5, label=col)
        for yi, l, w in zip(y, left, t[col]):
            if w >= 6:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if c in SHADES[2:] else INK)
        left += t[col].values
    ax.set_yticks(y, [f"{c}  (n={n[c]:,})" for c in t.index], fontsize=9, color=INK_2)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(ncol=6, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.07),
              title="Property damage of each event", title_fontsize=8.5)
    first, second = big.sort_values(ascending=False).index[:2]
    rain = big.get("Heavy Rain", np.nan)
    short = {"Dam / Levee Break": "dam or levee failures", "Heavy Rain / Tropical System": "tropical systems"}
    title(fig, f"{short.get(first, first).capitalize()} and {short.get(second, second)} most often cause $1M+ floods",
          f"$1M+ events: {big[first]:.0f}% of {short.get(first, first)}, {big[second]:.0f}% of "
          f"{short.get(second, second)}, {rain:.0f}% of rain floods. Each row = 100% of events with that cause.",
          note=f"Flood + Flash Flood events, {PERIOD}, {DOLLARS}. Grey = estimate of $0. Events with a blank damage field (no estimate) are left out.")
    save(fig, path, top=0.83, bottom=0.2)
    return t


# ----------------------------------------------------------------------------- fig5 concentration
def fig_curve(e, path):
    d = np.sort(e.loc[e["damage_reported"], "damage_property"].values)[::-1]
    cum = np.cumsum(d) / d.sum() * 100
    pct = np.arange(1, len(d) + 1) / len(d) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(pct, cum, color=SHADES[0], alpha=0.35, linewidth=0)
    ax.plot(pct, cum, color=SHADES[3], linewidth=2)
    ax.plot(pct, pct, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1.2, 14, "If every flood did\nequal damage", fontsize=8, color=MUTED, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 102)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    for q, label in [(0.1, "0.1%"), (1, "1%"), (10, "10%")]:
        k = max(int(len(d) * q / 100), 1)
        v = cum[k - 1]
        ax.scatter([q], [v], color=SHADES[4], zorder=3, s=30)
        ax.annotate(f"Costliest {label} of events\n= {v:.0f}% of damage", (q, v), xytext=(10, -26),
                    textcoords="offset points", fontsize=8.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of flood events, costliest first (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("Cumulative share of property damage", color=INK_2, fontsize=9)
    k1 = max(int(len(d) * 0.01), 1)
    title(fig, f"A handful of floods do almost all the damage",
          f"{len(d):,} Flood + Flash Flood events with a damage estimate, {PERIOD}. The costliest 1% ({k1:,} events) "
          f"account for {cum[k1 - 1]:.0f}% of all recorded property damage.")
    save(fig, path, top=0.83, bottom=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="outputs/flood_severity/flood_events.csv.gz")
    ap.add_argument("--out", default="outputs/flood_levels")
    args = ap.parse_args()
    fig_dir, tab_dir = os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    e, ep = load(args.events)
    fig_donuts(ep, os.path.join(fig_dir, "fig1_donuts_by_level.png")).round(0).to_csv(
        os.path.join(tab_dir, "by_level.csv"))
    fig_area(ep, os.path.join(fig_dir, "fig2_damage_share_by_year.png")).round(1).to_csv(
        os.path.join(tab_dir, "damage_share_by_year_level.csv"))
    fig_heatmap(e, os.path.join(fig_dir, "fig3_state_level_heatmap.png")).round(0).to_csv(
        os.path.join(tab_dir, "state_by_level.csv"))
    fig_cause(e, os.path.join(fig_dir, "fig4_cause_damage_mix.png")).round(1).to_csv(
        os.path.join(tab_dir, "cause_by_damage_bucket.csv"))
    fig_curve(e, os.path.join(fig_dir, "fig5_concentration_curve.png"))
    print(f"Done. {len(e):,} events, {len(ep):,} episodes. Figures in {fig_dir}")


if __name__ == "__main__":
    main()
