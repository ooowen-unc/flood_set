"""Evaluation and model descriptions. No fitting, threshold tuning or selection."""
import math

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_pinball_loss, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)

from config import MEASURES


def top_two(scores):
    return np.argsort(-scores, axis=1, kind="stable")[:, :2]


def _funding_arrays(y_log, predictions, coverage):
    y_log, predictions = np.asarray(y_log, dtype=float), np.asarray(predictions, dtype=float)
    if y_log.ndim != 1 or not len(y_log) or predictions.shape != (len(y_log), 3):
        raise ValueError("Expected nonempty funding target and N x 3 interval predictions")
    if not np.isfinite(y_log).all() or not np.isfinite(predictions).all() or np.any(y_log <= 0):
        raise ValueError("Funding diagnostics require finite positive log1p targets and finite predictions")
    if np.any(predictions < 0) or np.any(np.diff(predictions, axis=1) < 0) or not 0 < coverage < 1:
        raise ValueError("Funding predictions must be ordered/nonnegative; coverage must be in (0, 1)")
    return y_log, predictions


def _r2(y, predicted):
    return float(r2_score(y, predicted)) if len(y) > 1 and np.ptp(y) > 0 else None


def funding_metrics(y_log, predictions, coverage):
    y_log, predictions = _funding_arrays(y_log, predictions, coverage)
    y, dollars = np.expm1(y_log), np.expm1(predictions)
    residual = dollars[:, 1] - y
    absolute = np.abs(residual)
    log_residual = predictions[:, 1] - y_log
    alpha = 1 - coverage
    lower, upper = dollars[:, 0], dollars[:, 2]
    widths = upper - lower
    covered = (y >= lower) & (y <= upper)
    interval_score = widths + 2 / alpha * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    log_lower, log_upper = predictions[:, 0], predictions[:, 2]
    log_interval_score = log_upper - log_lower + 2 / alpha * (
        np.maximum(log_lower - y_log, 0) + np.maximum(y_log - log_upper, 0))
    quantiles = {}
    for i, (name, q) in enumerate(zip(("lower", "center", "upper"), (alpha / 2, .5, 1 - alpha / 2))):
        quantiles[name] = {"quantile": q,
                           "pinball_log1p": float(mean_pinball_loss(y_log, predictions[:, i], alpha=q)),
                           "pinball_nominal_usd": float(mean_pinball_loss(y, dollars[:, i], alpha=q))}
    return {
        "rows": len(y_log), "mae_log1p": float(mean_absolute_error(y_log, predictions[:, 1])),
        "rmse_log1p": float(np.sqrt(np.mean(log_residual ** 2))), "r2_log1p": _r2(y_log, predictions[:, 1]),
        "mae_nominal_usd": float(np.mean(absolute)), "rmse_nominal_usd": float(np.sqrt(np.mean(residual ** 2))),
        "r2_nominal_usd": _r2(y, dollars[:, 1]), "median_absolute_error_nominal_usd": float(np.median(absolute)),
        "mape_percent": float(np.mean(absolute / y) * 100),
        "median_absolute_percentage_error": float(np.median(absolute / y) * 100),
        "smape_percent": float(np.mean(2 * absolute / (y + dollars[:, 1])) * 100),
        "mean_bias_nominal_usd": float(np.mean(residual)),
        "mean_pinball_log1p": float(np.mean([v["pinball_log1p"] for v in quantiles.values()])),
        "mean_pinball_nominal_usd": float(np.mean([v["pinball_nominal_usd"] for v in quantiles.values()])),
        "quantile_losses": quantiles,
        "nominal_interval_coverage": coverage, "empirical_interval_coverage": float(np.mean(covered)),
        "coverage_gap_percentage_points": float((np.mean(covered) - coverage) * 100),
        "below_lower_fraction": float(np.mean(y < lower)), "above_upper_fraction": float(np.mean(y > upper)),
        "mean_interval_width_nominal_usd": float(np.mean(widths)),
        "median_interval_width_nominal_usd": float(np.median(widths)),
        "mean_width_over_mean_actual": float(np.mean(widths) / np.mean(y)),
        "mean_interval_score_nominal_usd": float(np.mean(interval_score)),
        "mean_interval_score_log1p": float(np.mean(log_interval_score)),
        "actual_amount_summary": {"minimum": float(np.min(y)), "median": float(np.median(y)),
                                  "mean": float(np.mean(y)), "maximum": float(np.max(y))},
    }


def _strategy_arrays(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 2 or not len(labels) or labels.shape[1] != len(MEASURES) or scores.shape != labels.shape:
        raise ValueError("Expected nonempty N x 4 labels and matching scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or np.any(labels.sum(axis=1) == 0):
        raise ValueError("Strategy diagnostics require binary recorded labels and finite scores")
    predicted = np.zeros_like(labels, dtype=int)
    np.put_along_axis(predicted, top_two(scores), 1, axis=1)
    return labels.astype(int), scores, predicted


def _auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def _average(values):
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def strategy_metrics(labels, scores):
    labels, scores, predicted = _strategy_arrays(labels, scores)
    hits = (labels * predicted).sum(axis=1)
    support = labels.sum(axis=1)
    per_measure, aps, aucs = {}, {}, {}
    for i, name in enumerate(MEASURES):
        truth, guess = labels[:, i], predicted[:, i]
        tp, fp = int(np.sum((truth == 1) & (guess == 1))), int(np.sum((truth == 0) & (guess == 1)))
        fn, tn = int(np.sum((truth == 1) & (guess == 0))), int(np.sum((truth == 0) & (guess == 0)))
        aps[name] = float(average_precision_score(truth, scores[:, i])) if len(np.unique(truth)) == 2 else None
        aucs[name] = _auc(truth, scores[:, i])
        per_measure[name] = {
            "positive_support": tp + fn, "negative_support": tn + fp, "predicted_positive": tp + fp,
            "prevalence": float(np.mean(truth)), "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "specificity": tn / (tn + fp) if tn + fp else None, "label_accuracy": (tp + tn) / len(truth),
            "average_precision": aps[name], "roc_auc": aucs[name],
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
    valid_auc = [(per_measure[m]["positive_support"], v) for m, v in aucs.items() if v is not None]
    result = {
        "rows": len(labels), "recall_at_2": float(np.mean(hits / support)),
        "precision_at_2": float(np.mean(hits / 2)), "any_hit_at_2": float(np.mean(hits > 0)),
        "exact_match_accuracy": float(np.mean(np.all(labels == predicted, axis=1))),
        "hamming_loss": float(np.mean(labels != predicted)), "mean_recorded_measures": float(np.mean(support)),
        "macro_average_precision": _average(aps.values()), "per_measure_average_precision": aps,
        "micro_average_precision": float(average_precision_score(labels.ravel(), scores.ravel()))
            if len(np.unique(labels)) == 2 else None,
        "macro_roc_auc": _average(aucs.values()), "micro_roc_auc": _auc(labels.ravel(), scores.ravel()),
        "weighted_roc_auc": sum(weight * value for weight, value in valid_auc) / sum(weight for weight, _ in valid_auc)
            if valid_auc else None,
        "per_measure_roc_auc": aucs, "per_measure": per_measure,
    }
    for average in ("samples", "micro", "macro", "weighted"):
        result["precision_" + average] = float(precision_score(labels, predicted, average=average, zero_division=0))
        result["recall_" + average] = float(recall_score(labels, predicted, average=average, zero_division=0))
        result["f1_" + average] = float(f1_score(labels, predicted, average=average, zero_division=0))
    return result


def measure_curves(labels, scores):
    """Complete TEST curve coordinates. ROC infinity/PR terminal thresholds are null."""
    labels, scores, _ = _strategy_arrays(labels, scores)
    curves = {}
    for i, measure in enumerate(MEASURES):
        y, score = labels[:, i], scores[:, i]
        if len(np.unique(y)) != 2:
            curves[measure] = {"defined": False, "reason": "both positive and negative labels are required", "pr": [], "roc": []}
            continue
        precision, recall, pr_thresholds = precision_recall_curve(y, score)
        fpr, tpr, roc_thresholds = roc_curve(y, score, drop_intermediate=False)
        curves[measure] = {
            "defined": True, "positive_support": int(y.sum()), "negative_support": int(len(y) - y.sum()),
            "prevalence": float(np.mean(y)), "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "pr": [{"precision": float(p), "recall": float(r),
                    "threshold": float(pr_thresholds[j]) if j < len(pr_thresholds) else None}
                   for j, (p, r) in enumerate(zip(precision, recall))],
            "roc": [{"false_positive_rate": float(f), "true_positive_rate": float(t),
                     "threshold": float(s) if math.isfinite(s) else None}
                    for f, t, s in zip(fpr, tpr, roc_thresholds)],
        }
    return curves


def by_impact(metric_function, truth, predictions, levels, **kwargs):
    levels = np.asarray(levels)
    return {str(level): {"rows": int(np.sum(levels == level)),
                         "metrics": metric_function(truth[levels == level], predictions[levels == level], **kwargs)
                         if np.any(levels == level) else None} for level in (1, 2, 3)}


def _parameter(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _parameter(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_parameter(v) for v in value]
    return str(value)


def model_description(candidate, feature_names, fit_seconds):
    description = {"name": candidate["name"], "fit_seconds": float(fit_seconds), "input_features": list(feature_names)}
    if candidate["name"] == "frequency_baseline":
        description["training_label_prevalence"] = dict(zip(MEASURES, candidate["frequencies"].tolist()))
        return description
    pipelines = candidate.get("estimators", [candidate.get("estimator")])
    description["heads"] = {}
    for index, pipe in enumerate(pipelines):
        names = pipe.named_steps["imputer"].get_feature_names_out(feature_names).tolist()
        estimator = pipe.named_steps["estimator"]
        learners = list(zip(MEASURES, estimator.estimators_)) if "estimators" not in candidate else [
            (("lower", "center", "upper")[index] if len(pipelines) == 3 else "center", estimator)]
        for head, learner in learners:
            details = {"estimator": type(learner).__name__, "parameters": _parameter(learner.get_params(deep=False)),
                       "transformed_features": names,
                       "scaled": "scaler" in pipe.named_steps}
            if hasattr(learner, "coef_"):
                details["coefficients"] = dict(zip(names, learner.coef_.reshape(-1).tolist()))
                details["intercept"] = np.asarray(learner.intercept_).reshape(-1).tolist()
            if hasattr(learner, "feature_importances_"):
                details["impurity_feature_importance"] = dict(zip(names, learner.feature_importances_.tolist()))
            if hasattr(learner, "constant_"):
                details["constant_log1p"] = np.asarray(learner.constant_).reshape(-1).tolist()
            description["heads"][head] = details
    if "log_interval_padding" in candidate:
        description["log_interval_padding"] = float(candidate["log_interval_padding"])
    return description
