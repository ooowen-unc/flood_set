import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS, BACKGROUND, MEASURES, MEASURE_NAMES, PRIOR, SOURCE
from data import features, read_data
from models import regression_logs, strategy_scores, top_two

LEVELS = {"light": 1, "moderate": 2, "severe": 3}
LEVEL_ALIASES = {"轻": "light", "中": "moderate", "重": "severe"}


def finite_value(value):
    return None if pd.isna(value) else float(value)


def predict(model_path, source, county_fips, flood_type, impact_level, year=2025):
    if not re.fullmatch(r"\d{5}", county_fips):
        raise ValueError("county-fips must be a five-character county code, including leading zeros")
    if flood_type not in {"Flood", "Flash Flood"} or impact_level not in {1, 2, 3}:
        raise ValueError("Supported types: Flood, Flash Flood; impact levels: 1, 2, 3")
    bundle = joblib.load(model_path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model artifact version")
    if not bundle["reference_year_range"][0] <= year <= bundle["reference_year_range"][1]:
        raise ValueError("Reference year must be within trained data years 1999-2025")
    frame = read_data(source)
    county = frame.loc[(frame.county_fips == county_fips) & (frame.analysis_year <= year)]
    if county.empty:
        raise ValueError("No recorded county context at/before this year; choose a county present in the annual table")
    county = county.sort_values("analysis_year").tail(1).copy()
    row = county.iloc[0]
    background_year = int(row["analysis_year"])
    warnings = []
    if background_year < year:
        warnings.append(f"Prior-measure context comes from {background_year}; later closures may be missing.")
    if row["nri_match_status"] != "matched":
        warnings.append("Historical county code has no NRI match; missing background uses training imputations.")
    if Path(source).resolve() != Path(bundle["source"]).resolve():
        warnings.append("County background source differs from the source used for training.")
    x = features(county, bundle["impact_thresholds"], scenario_type=flood_type,
                 scenario_level=impact_level, reference_year=year)
    if list(x.columns) != bundle["feature_names"]:
        raise ValueError("Feature schema differs from the model artifact")
    dollars = np.expm1(regression_logs(bundle["funding_model"], x)[0])
    if not np.isfinite(dollars).all():
        raise ValueError("Funding prediction outside finite numeric range")
    scores = strategy_scores(bundle["strategy_model"], x)
    recommendations = [
        {"measure": MEASURES[i], "name": MEASURE_NAMES[MEASURES[i]],
         "ranking_score": round(float(scores[0, i]), 4)} for i in top_two(scores)[0]
    ]
    return {
        "input": {"county_fips": county_fips, "flood_type": flood_type,
                  "impact_level": impact_level, "reference_year": year},
        "county_background": {
            "county_name": None if pd.isna(row["county_name"]) else row["county_name"],
            "nri_match_status": row["nri_match_status"],
            "nri_snapshot_version": None if pd.isna(row["nri_nri_ver"]) else row["nri_nri_ver"],
            "population_reference": "NRI static population snapshot; not a historical annual population estimate",
            "static_values": {name: finite_value(row[name]) for name in BACKGROUND},
            "prior_measure_context_year": background_year,
            "prior_closed_projects_by_measure": {measure: finite_value(row[field]) for measure, field in zip(MEASURES, PRIOR)},
            "prior_physical_flood_project_records": finite_value(row["hma_physical_flood_closed_before_year_records"]),
            "prior_context_definition": "Closure year strictly BEFORE context year; project coverage counts, not verified facilities; activity counts are non-additive",
        },
        "pa_cumulative_assistance_reference": {
            "currency": "USD", "price_basis": "nominal declaration-year cohort dollars",
            "lower_usd": round(float(dollars[0]), 2),
            "upper_usd": round(float(dollars[2]), 2), "nominal_coverage_target": bundle["coverage"],
            "empirical_test_coverage": bundle["test_interval_coverage"], "algorithm": bundle["funding_model"]["name"],
            "target_definition": "Recorded positive federal PA obligations for exact Flood incident declaration-year cohorts; cumulative snapshot, not annual payments or all-agency aid",
        },
        "recommended_measures": recommendations,
        "measure_algorithm": bundle["strategy_model"]["name"],
        "impact_definition": {"training_positive_loss_thresholds_2025_cpi_usd": bundle["impact_thresholds"],
                              "meaning": "Project-specific county/year impact tiers, not official flood depth/severity grades"},
        "interpretation": "Annual historical policy association; scores rank recorded choices, not effectiveness or guaranteed funding eligibility",
        "warnings": warnings,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "flood_policy.joblib")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--county-fips", required=True)
    parser.add_argument("--flood-type", choices=("Flood", "Flash Flood"), required=True)
    parser.add_argument("--impact-level", type=lambda value: LEVEL_ALIASES.get(value, value),
                        choices=tuple(LEVELS), required=True)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    result = predict(args.model, args.source, args.county_fips, args.flood_type, LEVELS[args.impact_level], args.year)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
