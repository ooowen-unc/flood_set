import numpy as np

from config import LABELS, TARGET
from data import features, fit_impact_thresholds, split_validation, task_rows
from models import (fit_funding, funding_metrics, regression_logs, strategy_candidate,
                    strategy_metrics, strategy_scores)


def summarize_folds(folds, task, names, metrics):
    result = {}
    for name in names:
        result[name] = {}
        for metric in metrics:
            values = [fold[task][name][metric] for fold in folds]
            values = [v for v in values if v is not None]
            result[name][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "defined_folds": len(values),
            }
    return result


def evaluate_stability(development, *, folds, regressors, classifiers, seed, jobs, coverage):
    groups = np.array(sorted(development.county_fips.unique()))
    if folds < 2 or folds > len(groups):
        raise ValueError("CV requires 2 or more folds and at least one county per fold")
    np.random.default_rng(seed).shuffle(groups)
    funding_names = ("median_baseline", *regressors)
    measure_names = ("frequency_baseline", *classifiers)
    results = []
    for index, held_out in enumerate(np.array_split(groups, folds), start=1):
        print(f"County CV: fold {index}/{folds}", flush=True)
        validation = development.loc[development.county_fips.isin(held_out)]
        training = development.loc[~development.county_fips.isin(held_out)]
        fitting, calibration = split_validation(training, seed + index, .8)
        thresholds = fit_impact_thresholds(fitting)
        # The held-out fold contributes neither thresholds nor preprocessing.
        fx, cx, vx = [features(frame, thresholds) for frame in (fitting, calibration, validation)]
        fm, cm, vm = [task_rows(frame) for frame in (fitting, calibration, validation)]
        fy, cy, vy = [np.log1p(frame.loc[mask[0], TARGET].to_numpy(dtype=float))
                      for frame, mask in ((fitting, fm), (calibration, cm), (validation, vm))]
        if min(len(fy), len(cy), len(vy)) < 10:
            raise ValueError("Too few funding labels in a CV fitting/calibration/held-out portion")
        funding_results = {}
        for name in funding_names:
            candidate, _ = fit_funding(name, fx.loc[fm[0]], fy, cx.loc[cm[0]], cy,
                                       seed=seed, jobs=jobs, coverage=coverage)
            funding_results[name] = funding_metrics(vy, regression_logs(candidate, vx.loc[vm[0]]), coverage)
        tx = features(training, thresholds)
        tm = task_rows(training)[1]
        ty = (training.loc[tm, list(LABELS)].to_numpy() > 0).astype(int)
        sy = (validation.loc[vm[1], list(LABELS)].to_numpy() > 0).astype(int)
        if not len(sy) or np.any(ty.sum(axis=0) == 0) or np.any(ty.sum(axis=0) == len(ty)):
            raise ValueError("CV strategy training needs both outcomes for all four measures")
        baseline = {"name": "frequency_baseline", "frequencies": ty.mean(axis=0)}
        measure_results = {"frequency_baseline": strategy_metrics(sy, strategy_scores(baseline, vx.loc[vm[1]]))}
        for name in classifiers:
            candidate = strategy_candidate(name, seed, jobs)
            candidate["estimator"].fit(tx.loc[tm], ty)
            measure_results[name] = strategy_metrics(sy, strategy_scores(candidate, vx.loc[vm[1]]))
        results.append({"fold": index, "fitting_counties": fitting.county_fips.nunique(),
                        "calibration_counties": calibration.county_fips.nunique(),
                        "held_out_counties": len(held_out), "impact_thresholds": thresholds,
                        "funding": funding_results, "measures": measure_results})
    return {
        "folds": folds, "unit": "county", "development_rows": len(development),
        "development_counties": len(groups), "seed": seed,
        "role": "Fixed-candidate stability diagnostics only; does not select or tune the final models",
        "std_definition": "Sample standard deviation across folds, not a confidence interval",
        "funding": summarize_folds(results, "funding", funding_names,
                                   ("empirical_interval_coverage", "median_interval_width_nominal_usd",
                                    "mean_interval_score_nominal_usd", "mean_interval_score_log1p")),
        "measures": summarize_folds(results, "measures", measure_names,
                                    ("recall_at_2", "f1_macro", "macro_average_precision", "macro_roc_auc")),
        "fold_results": results,
    }
