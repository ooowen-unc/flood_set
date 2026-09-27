import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import ARTIFACTS, LABELS, MEASURES, SPLITS, TARGET
from data import features, fit_impact_thresholds, read_data, split_validation, task_rows
from diagnostics import by_impact, measure_curves, model_description
from models import (CLASSIFIERS, REGRESSORS, fit_funding, funding_metrics,
                    regression_logs, strategy_candidate, strategy_metrics, strategy_scores, top_two)
from stability import evaluate_stability


def require_size(frame, minimum, label):
    if len(frame) < minimum:
        raise ValueError(f"{label}: {len(frame)} usable rows; at least {minimum} required")


def train(split_dir, output_dir, *, seed=42, jobs=2, coverage=.8,
          regressors=REGRESSORS, classifiers=CLASSIFIERS, cv_folds=3, overwrite=False):
    if not .5 <= coverage < 1 or jobs == 0:
        raise ValueError("coverage must be in [0.5, 1); jobs cannot be zero")
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    artifact, report_path = output_dir / "flood_policy.joblib", output_dir / "training_report.json"
    curves_path = output_dir / "measure_test_curves.json"
    funding_csv = output_dir / "funding_test_predictions.csv"
    measures_csv = output_dir / "measure_test_predictions.csv"
    if cv_folds != 0 and cv_folds < 2:
        raise ValueError("cv-folds must be zero (disabled) or at least two")
    if any(p.exists() for p in (artifact, report_path, curves_path, funding_csv, measures_csv)) and not overwrite:
        raise ValueError("Model outputs exist; use --overwrite to replace model outputs")
    meta = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    frames = {name: read_data(split_dir / (name + ".csv")) for name in ("train", "validation", "test")}
    for name, frame in frames.items():
        if len(frame) != meta["splits"][name]["rows"]:
            raise ValueError(f"Split metadata row mismatch: {name}")
    keys = {name: set(zip(frame.county_fips, frame.analysis_year)) for name, frame in frames.items()}
    counties = {name: set(frame.county_fips) for name, frame in frames.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if keys[a] & keys[b] or (meta["unit"] == "county" and counties[a] & counties[b]):
            raise ValueError(f"Split overlap: {a}/{b}")
    thresholds = fit_impact_thresholds(frames["train"])
    x = {name: features(frame, thresholds) for name, frame in frames.items()}
    masks = {name: task_rows(frame) for name, frame in frames.items()}
    selection_frame, calibration_frame = split_validation(frames["validation"], seed + 1)
    selection_mask = task_rows(selection_frame)[0]
    calibration_mask = task_rows(calibration_frame)[0]
    funding_train = frames["train"].loc[masks["train"][0]]
    funding_selection = selection_frame.loc[selection_mask]
    funding_calibration = calibration_frame.loc[calibration_mask]
    funding_test = frames["test"].loc[masks["test"][0]]
    for frame, minimum, label in ((funding_train, 50, "funding train"), (funding_selection, 10, "funding selection"),
                                  (funding_calibration, 10, "funding calibration"), (funding_test, 10, "funding test")):
        require_size(frame, minimum, label)
    rx_train = x["train"].loc[funding_train.index]
    rx_calibration = x["validation"].loc[funding_calibration.index]
    rx_test = x["test"].loc[funding_test.index]
    ry_train, ry_selection, ry_calibration, ry_test = [np.log1p(frame[TARGET].to_numpy(dtype=float))
                                                     for frame in (funding_train, funding_selection, funding_calibration, funding_test)]
    # Preliminary calibration is inside TRAIN, so final calibration labels have
    # no influence on model selection. Refit selected algorithm on full TRAIN.
    preliminary_fit, preliminary_calibration = split_validation(frames["train"], seed + 2, .8)
    preliminary_thresholds = fit_impact_thresholds(preliminary_fit)
    preliminary_frames = [preliminary_fit, preliminary_calibration, selection_frame]
    preliminary_x, preliminary_y = [], []
    for frame in preliminary_frames:
        mask = task_rows(frame)[0]
        require_size(frame.loc[mask], 10, "funding preliminary portion")
        preliminary_x.append(features(frame, preliminary_thresholds).loc[mask])
        preliminary_y.append(np.log1p(frame.loc[mask, TARGET].to_numpy(dtype=float)))
    funding_comparison, best_funding_name, best_score = {}, None, float("inf")
    for name in ("median_baseline", *regressors):
        print(f"Selecting funding with calibrated intervals: {name}", flush=True)
        candidate, _ = fit_funding(name, preliminary_x[0], preliminary_y[0],
                                   preliminary_x[1], preliminary_y[1], seed=seed, jobs=jobs, coverage=coverage)
        metrics = funding_metrics(preliminary_y[2], regression_logs(candidate, preliminary_x[2]), coverage)
        funding_comparison[name] = metrics
        if metrics["mean_interval_score_log1p"] < best_score:
            best_score, best_funding_name = metrics["mean_interval_score_log1p"], name
    funding_candidates, funding_times = {}, {}
    for name in ("median_baseline", *regressors):
        print(f"Final fitting funding: {name}", flush=True)
        candidate, funding_times[name] = fit_funding(name, rx_train, ry_train, rx_calibration, ry_calibration,
                                                    seed=seed, jobs=jobs, coverage=coverage)
        funding_candidates[name] = candidate
    best_funding = funding_candidates[best_funding_name]
    # Independent final calibration never reselects algorithms. Full-TRAIN
    # refits of all candidates are retained only for descriptive comparisons.
    funding_details = {}
    for name, candidate in funding_candidates.items():
        predictions = {"train": regression_logs(candidate, rx_train),
                       "calibration": regression_logs(candidate, rx_calibration),
                       "test": regression_logs(candidate, rx_test)}
        funding_details[name] = {
            "description": model_description(candidate, rx_train.columns, funding_times[name]),
            "train": funding_metrics(ry_train, predictions["train"], coverage),
            "selection_calibrated": funding_comparison[name],
            "calibration": funding_metrics(ry_calibration, predictions["calibration"], coverage),
            "test": funding_metrics(ry_test, predictions["test"], coverage),
            "test_by_impact_level": by_impact(funding_metrics, ry_test, predictions["test"],
                                              rx_test["impact_level"], coverage=coverage),
        }
    funding_calibration_metrics = funding_details[best_funding["name"]]["calibration"]

    sx, sy = {}, {}
    for name, frame in frames.items():
        selected = frame.loc[masks[name][1]]
        require_size(selected, 50 if name == "train" else 10, "strategy " + name)
        sx[name] = x[name].loc[selected.index]
        sy[name] = (selected[list(LABELS)].to_numpy() > 0).astype(int)
    for i, measure in enumerate(MEASURES):
        counts = np.bincount(sy["train"][:, i], minlength=2)
        if min(counts) < 5:
            raise ValueError(f"Too few positive/negative TRAIN labels for {measure}: {counts.tolist()}")
    baseline = {"name": "frequency_baseline", "frequencies": sy["train"].mean(axis=0)}
    strategy_candidates, strategy_times = {baseline["name"]: baseline}, {baseline["name"]: 0.0}
    strategy_comparison = {baseline["name"]: strategy_metrics(sy["validation"], strategy_scores(baseline, sx["validation"]))}
    best_strategy = baseline
    initial = strategy_comparison[baseline["name"]]
    best_rank = (initial["recall_at_2"], initial["macro_average_precision"] or 0)
    for name in classifiers:
        print(f"Fitting measures: {name}", flush=True)
        candidate = strategy_candidate(name, seed, jobs)
        started = perf_counter()
        candidate["estimator"].fit(sx["train"], sy["train"])
        strategy_times[name] = perf_counter() - started
        strategy_candidates[name] = candidate
        metrics = strategy_metrics(sy["validation"], strategy_scores(candidate, sx["validation"]))
        strategy_comparison[name] = metrics
        rank = (metrics["recall_at_2"], metrics["macro_average_precision"] or 0)
        if rank > best_rank:
            best_rank, best_strategy = rank, candidate
    # Both choices are fixed by validation rules. TEST describes
    # every candidate but never changes these choices or hyperparameters.
    strategy_details, curve_data = {}, {}
    for name, candidate in strategy_candidates.items():
        train_scores = strategy_scores(candidate, sx["train"])
        test_scores = strategy_scores(candidate, sx["test"])
        strategy_details[name] = {
            "description": model_description(candidate, sx["train"].columns, strategy_times[name]),
            "train": strategy_metrics(sy["train"], train_scores),
            "validation": strategy_comparison[name],
            "test": strategy_metrics(sy["test"], test_scores),
            "test_by_impact_level": by_impact(strategy_metrics, sy["test"], test_scores, sx["test"]["impact_level"]),
        }
        curve_data[name] = measure_curves(sy["test"], test_scores)
    funding_test_metrics = funding_details[best_funding["name"]]["test"]
    strategy_test_metrics = strategy_details[best_strategy["name"]]["test"]
    stability = evaluate_stability(pd.concat([frames["train"], frames["validation"]], ignore_index=True),
                                   folds=cv_folds, regressors=regressors, classifiers=classifiers,
                                   seed=seed + 100, jobs=jobs, coverage=coverage) if cv_folds else None
    report = {
        "funding": {"target": TARGET, "conditional_on": "positive recorded Flood-incident PA federal obligations",
                    "selected": best_funding["name"], "selection_metric": "mean_interval_score_log1p",
                    "selection_protocol": "TRAIN county split 80/20 fitting/preliminary calibration; calibrated intervals scored on selection counties; full TRAIN refit and independent final calibration after algorithm freezes",
                    "preliminary_impact_thresholds_2025_cpi_usd": preliminary_thresholds,
                    "selection_comparison": funding_comparison, "nominal_interval_coverage": coverage,
                    "calibration": funding_calibration_metrics, "test": funding_test_metrics,
                    "test_median_baseline": funding_details["median_baseline"]["test"],
                    "all_models": funding_details},
        "measures": {"labels": list(MEASURES), "selected": best_strategy["name"],
                     "conditional_on": "at least one recorded first-approved measure among the four classes",
                     "selection_metric": "recall_at_2, then macro_average_precision",
                     "validation_comparison": strategy_comparison, "test": strategy_test_metrics,
                     "test_frequency_baseline": strategy_details["frequency_baseline"]["test"],
                     "all_models": strategy_details, "test_curves_file": str(curves_path)},
        "impact": {"loss_thresholds_2025_cpi_usd": thresholds,
                   "rule": "Positive TRAIN loss terciles; zero loss is tier 1; any recorded injury raises minimum tier to 2; any death sets tier 3; not an official hazard grade"},
        "features": list(x["train"].columns), "seed": seed, "jobs": jobs,
        "source": meta["source"], "source_sha256": meta["source_sha256"], "split_unit": meta["unit"],
        "split_seed": meta["seed"],
        "split_sizes": {name: {key: values[key] for key in ("rows", "counties")}
                        for name, values in meta["splits"].items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "report_schema_version": 3,
        "stability": stability,
        "prediction_files": {"funding_test": str(funding_csv), "measures_test": str(measures_csv)},
        "sample_sizes": {"funding_train": len(ry_train), "funding_selection": len(ry_selection),
                         "funding_calibration": len(ry_calibration), "funding_test": len(ry_test),
                         "funding_preliminary_fit": len(preliminary_y[0]),
                         "funding_preliminary_calibration": len(preliminary_y[1]),
                         "measures": {name: len(labels) for name, labels in sy.items()}},
        "metric_definitions": {
            "classification_decision": "Exactly top 2 scores, stable tie-breaking in configured measure order; NOT a 0.5 threshold",
            "f1_samples": "Mean per-row 2*hits/(2+number of recorded labels); not harmonic mean of already averaged precision/recall",
            "f1_macro_micro_weighted": "Macro: equal class weights; micro: pooled counts; weighted: positive class supports",
            "confusion_matrix": "Per class [[TN, FP], [FN, TP]], from top-2 binary decisions",
            "roc_and_pr": "Use continuous scores; AP/ROC null for classes without both outcomes; macro excludes undefined classes",
            "mape": "Percent scale, evaluated on positive funding labels only; can be large for small actual amounts",
            "r2": "May be negative; null for one-row/constant-target groups; point predictions, not interval quality",
            "mean_interval_score_nominal_usd": "Mean upper-lower + 2/(1-coverage) times distance of truth outside interval; lower is better",
            "mean_interval_score_log1p": "Same proper interval score on log1p USD scale; selects calibrated candidates using endpoints only",
            "selection_calibrated": "Preliminary models, calibrated within TRAIN and evaluated on independent selection counties before full TRAIN refit",
            "train_metrics": "In-sample diagnostics; intervals use held-out calibration padding, so training coverage is not a held-out guarantee",
            "test_comparison": "All candidates evaluated after choices freeze; do not repeatedly optimize using this TEST report",
            "model_explanations": "Coefficients in transformed/scaled feature space; tree importance is impurity-based; neither is causal",
        },
        "notes": ["No unmatched funding amount is filled with zero; negative adjustments are not positive funding targets.",
                  "Missing project rows are not negative government-action labels; four-label selection is conditional on recorded measures.",
                  "All funding candidates are preliminarily calibrated before selection; final calibration uses separate validation counties after selection freezes; test labels never tune or select.",
                  "Calibration targets empirical row coverage; county/year dependence and drift prevent a guaranteed coverage claim.",
                  "Current NRI/PA/HMA snapshots and random splits do not establish future forecasting or causal effectiveness.",
                  "Dollar values remain nominal declaration-year cohort snapshots, not actual payments or optimal budgets.",
                  "A single-type query is a scenario against county/year mixed-type training contexts, not event attribution.",
                  "Baselines remain eligible selections when more complex algorithms do not improve validation scores."]}
    bundle = {"format_version": 1, "funding_model": best_funding, "strategy_model": best_strategy,
              "impact_thresholds": thresholds, "feature_names": list(x["train"].columns),
              "coverage": coverage, "source": meta["source"], "source_sha256": meta["source_sha256"],
              "reference_year_range": [1999, 2025], "versions": report["versions"],
              "test_interval_coverage": funding_test_metrics["empirical_interval_coverage"]}
    # Check JSON before writing outputs; no NaN metrics are silently persisted.
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    curves_json = json.dumps({"split": "test", "selected": best_strategy["name"],
                              "decision": "top_two", "models": curve_data,
                              "threshold_note": "null ROC initial threshold means infinity; null PR terminal threshold means no cutoff"},
                             ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    report_path.write_text(report_json + "\n", encoding="utf-8")
    curves_path.write_text(curves_json + "\n", encoding="utf-8")
    funding_predictions = funding_test[["county_fips", "analysis_year"]].copy()
    funding_predictions["impact_level"] = rx_test["impact_level"]
    funding_predictions["actual_usd"] = funding_test[TARGET]
    dollars = np.expm1(regression_logs(best_funding, rx_test))
    funding_predictions["lower_usd"], funding_predictions["upper_usd"] = dollars[:, 0], dollars[:, 2]
    funding_predictions["covered"] = funding_predictions.actual_usd.between(
        funding_predictions.lower_usd, funding_predictions.upper_usd)
    funding_predictions.to_csv(funding_csv, index=False, float_format="%.2f")
    measures_predictions = frames["test"].loc[masks["test"][1], ["county_fips", "analysis_year"]].copy()
    measures_predictions["impact_level"] = sx["test"]["impact_level"]
    scores = strategy_scores(best_strategy, sx["test"])
    for index, name in enumerate(MEASURES):
        measures_predictions["recorded_" + name] = sy["test"][:, index]
        measures_predictions["score_" + name] = scores[:, index]
    for rank in (0, 1):
        measures_predictions[f"recommended_{rank + 1}"] = [MEASURES[i] for i in top_two(scores)[:, rank]]
    measures_predictions.to_csv(measures_csv, index=False, float_format="%.6f")
    print(json.dumps({"selected_funding": best_funding["name"], "selected_measures": best_strategy["name"],
                      "artifact": str(artifact), "report": str(report_path), "curves": str(curves_path),
                      "funding_predictions": str(funding_csv), "measure_predictions": str(measures_csv)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--coverage", type=float, default=.8)
    parser.add_argument("--cv-folds", type=int, default=3, help="Development-only county CV; 0 disables stability diagnostics")
    parser.add_argument("--regressors", nargs="+", choices=REGRESSORS, default=REGRESSORS)
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(args.split_dir, args.output_dir, seed=args.seed, jobs=args.jobs, coverage=args.coverage,
          regressors=args.regressors, classifiers=args.classifiers, cv_folds=args.cv_folds, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
