"""Small sklearn factories and shared scoring; no work at import time."""
import math
from time import perf_counter

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diagnostics import funding_metrics, strategy_metrics, top_two

REGRESSORS = ("elastic_net", "random_forest", "quantile_gbdt")
CLASSIFIERS = ("logistic", "random_forest")


def pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline([*steps, ("estimator", estimator)])


def regression_candidate(name, seed, jobs, coverage):
    if name == "median_baseline":
        estimators = [pipeline(DummyRegressor(strategy="median"))]
    elif name == "elastic_net":
        estimators = [pipeline(ElasticNet(alpha=.01, l1_ratio=.25, max_iter=5000, random_state=seed), scale=True)]
    elif name == "random_forest":
        estimators = [pipeline(RandomForestRegressor(n_estimators=200, max_depth=12,
                      min_samples_leaf=5, max_features=.8, n_jobs=jobs, random_state=seed))]
    elif name == "quantile_gbdt":
        tail = (1 - coverage) / 2
        estimators = [pipeline(GradientBoostingRegressor(loss="quantile", alpha=q,
                      n_estimators=150, learning_rate=.05, max_depth=2, min_samples_leaf=15,
                      subsample=.85, random_state=seed)) for q in (tail, .5, 1 - tail)]
    else:
        raise ValueError(f"Unknown regressor: {name}")
    return {"name": name, "estimators": estimators}


def regression_logs(candidate, x):
    """Columns: lower, typical center, upper, all on log1p USD scale."""
    predictions = np.column_stack([est.predict(x) for est in candidate["estimators"]])
    if predictions.shape[1] == 1:
        predictions = np.repeat(predictions, 3, axis=1)
    predictions = np.sort(predictions, axis=1)  # Resolve independently fitted quantile crossings.
    padding = candidate.get("log_interval_padding", 0.0)
    predictions[:, 0] -= padding
    predictions[:, 2] += padding
    return np.maximum(predictions, 0)


def interval_padding(y_log, raw_predictions, coverage):
    """Held-out residual/CQR padding. Dependent annual rows limit formal guarantees."""
    errors = np.maximum.reduce((raw_predictions[:, 0] - y_log,
                                y_log - raw_predictions[:, 2], np.zeros(len(y_log))))
    k = math.ceil((len(errors) + 1) * coverage)
    if not 1 <= k <= len(errors):
        raise ValueError("Too few calibration rows for the requested coverage")
    return float(np.partition(errors, k - 1)[k - 1])


def fit_funding(name, x_train, y_train, x_calibration, y_calibration, *, seed, jobs, coverage):
    """Fit and calibrate one fixed candidate using disjoint fitting/calibration rows."""
    candidate = regression_candidate(name, seed, jobs, coverage)
    started = perf_counter()
    for estimator in candidate["estimators"]:
        estimator.fit(x_train, y_train)
    elapsed = perf_counter() - started
    candidate["log_interval_padding"] = interval_padding(
        y_calibration, regression_logs(candidate, x_calibration), coverage)
    return candidate, elapsed


def strategy_candidate(name, seed, jobs):
    if name == "logistic":
        estimator = LogisticRegression(C=1, max_iter=1000, random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5,
                                           max_features="sqrt", n_jobs=jobs, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    # Four independent labels; do not nest parallel jobs at both levels.
    return {"name": name, "estimator": pipeline(MultiOutputClassifier(estimator), scale=name == "logistic")}


def strategy_scores(candidate, x):
    if candidate["name"] == "frequency_baseline":
        return np.tile(candidate["frequencies"], (len(x), 1))
    estimator = candidate["estimator"]
    probabilities = estimator.predict_proba(x)
    classes = estimator.named_steps["estimator"].classes_
    return np.column_stack([values[:, int(np.flatnonzero(labels == 1)[0])]
                            for labels, values in zip(classes, probabilities)])
