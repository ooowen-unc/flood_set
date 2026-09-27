"""One feature definition for both training and scenario prediction."""
import numpy as np
import pandas as pd

from config import BACKGROUND, KEYS, LABELS, PRIOR, REQUIRED, TARGET, TYPES


def read_data(path):
    frame = pd.read_csv(path, dtype={"county_fips": str}, usecols=list(REQUIRED))
    if frame.duplicated(list(KEYS)).any():
        raise ValueError(f"Repeated county/year in {path}")
    numeric = set(REQUIRED) - {"county_fips", "county_name", "nri_match_status", "nri_nri_ver"}
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if np.isinf(frame[list(numeric)].to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite numeric value in {path}")
    if frame["analysis_year"].isna().any() or not frame["analysis_year"].between(1999, 2025).all():
        raise ValueError("Expected annual observations in 1999-2025")
    return frame


def fit_impact_thresholds(train):
    """Project-specific loss tiers, fitted to TRAIN rows only, in 2025 CPI USD."""
    loss = train["flood_material_loss_adjusted_usd"].dropna()
    if loss.empty or (loss < 0).any():
        raise ValueError("No usable nonnegative training losses for impact tiers")
    loss = loss[loss > 0]
    if loss.empty:
        raise ValueError("Positive training losses are required for impact thresholds")
    thresholds = loss.quantile([1 / 3, 2 / 3]).tolist()
    if thresholds[0] >= thresholds[1]:
        raise ValueError("Training loss distribution cannot support three distinct tiers")
    return thresholds


def impact_levels(frame, thresholds):
    loss = frame["flood_material_loss_adjusted_usd"]
    level = pd.Series(np.searchsorted(thresholds, loss.to_numpy(), side="right") + 1, index=frame.index, dtype=float)
    level[loss.isna()] = np.nan
    injuries = frame[["flood_injuries_direct", "flood_injuries_indirect"]].sum(axis=1, min_count=1)
    deaths = frame[["flood_deaths_direct", "flood_deaths_indirect"]].sum(axis=1, min_count=1)
    level.loc[injuries > 0] = np.maximum(level.loc[injuries > 0].fillna(2), 2)
    level.loc[deaths > 0] = 3
    return level


def features(frame, thresholds, *, scenario_type=None, scenario_level=None, reference_year=None):
    out = pd.DataFrame(index=frame.index)
    out["impact_level"] = impact_levels(frame, thresholds) if scenario_level is None else scenario_level
    out["analysis_year"] = frame["analysis_year"] if reference_year is None else reference_year
    for name in TYPES:
        out[name] = (frame[name] > 0).astype(float)
    if scenario_type is not None:
        out[TYPES[0]] = float(scenario_type == "Flash Flood")
        out[TYPES[1]] = float(scenario_type == "Flood")
    for name in BACKGROUND:
        values = frame[name]
        if name in BACKGROUND[:4]:
            if (values.dropna() < 0).any():
                raise ValueError(f"Negative size/exposure feature: {name}")
            values = np.log1p(values)
        out[name] = values
    for name in (*PRIOR, "hma_physical_flood_closed_before_year_records"):
        values = frame[name]
        if (values.dropna() < 0).any():
            raise ValueError(f"Negative prior-project count: {name}")
        out[name] = np.log1p(values)
    # County identifiers, contemporaneous losses and PA/HMA outcome fields do
    # not enter X. Loss/casualties only construct the user-visible impact tier.
    return out


def task_rows(frame):
    supported = (frame[list(TYPES)] > 0).any(axis=1)
    funded = supported & frame[TARGET].notna() & (frame[TARGET] > 0)
    recorded_measures = supported & (frame[list(LABELS)] > 0).any(axis=1)
    return funded, recorded_measures


def split_validation(frame, seed, selection_fraction=.5):
    """Split by county: first portion for fitting/selection, second for calibration."""
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be between zero and one")
    groups = np.array(sorted(frame["county_fips"].unique()))
    np.random.default_rng(seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError("Validation requires at least two counties")
    count = max(1, min(len(groups) - 1, int(len(groups) * selection_fraction)))
    selected = set(groups[:count])
    selection = frame["county_fips"].isin(selected)
    return frame.loc[selection].copy(), frame.loc[~selection].copy()
