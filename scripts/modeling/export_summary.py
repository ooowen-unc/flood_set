import argparse
import json
from pathlib import Path

from config import ARTIFACTS, MEASURE_NAMES, SOURCE
from predict import predict


def pct(value):
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def number(value, spec=".2f"):
    return format(value, spec) if value is not None else "N/A"


def local_link(label, path):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def algorithm_parameters(description):
    heads = description.get("heads", {})
    if not heads:
        return "training_label_prevalence=" + json.dumps(description.get("training_label_prevalence"))
    params = next(iter(heads.values()))["parameters"]
    keys = ("n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "subsample", "alpha", "l1_ratio", "C", "strategy")
    return ", ".join(f"{key}={params[key]}" for key in keys if key in params) or "N/A"


def export_summary(output_dir, source=SOURCE, *, county_fips="37155", flood_type="Flash Flood", impact_level=3, year=2025):
    output_dir = Path(output_dir).resolve()
    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    result = predict(output_dir / "flood_policy.joblib", source, county_fips, flood_type, impact_level, year)
    funding, measures, sizes = report["funding"], report["measures"], report["sample_sizes"]
    ft, mt = funding["test"], measures["test"]
    names = [funding["selected"], measures["selected"]]
    fd = funding["all_models"][names[0]]["description"]
    md = measures["all_models"][names[1]]["description"]
    lines = [
        "# Flood Policy Model Data", "",
        "## Dataset", "",
        "| Field | Value |", "| --- | --- |",
        f"| County-year records | {sum(s['rows'] for s in report['split_sizes'].values()):,} |",
        f"| Split seed | {report['split_seed']} |",
        f"| Features | {len(report['features'])} |",
        f"| Funding target | `{funding['target']}` |",
        f"| Nominal interval coverage | {pct(funding['nominal_interval_coverage'])} |", "",
        "| Split | Rows | Counties |", "| --- | ---: | ---: |",
    ]
    for split, size in report["split_sizes"].items():
        lines.append(f"| {split} | {size['rows']:,} | {size['counties']:,} |")
    lines += ["", "## Labeled Samples", "",
              "| Task | Subset | Samples |", "| --- | --- | ---: |"]
    for key, count in sizes.items():
        if key.startswith("funding_"):
            lines.append(f"| Funding | {key.removeprefix('funding_')} | {count:,} |")
    for split, count in sizes["measures"].items():
        lines.append(f"| Measures | {split} | {count:,} |")
    lines += ["", "## Selected Models", "",
              "| Task | Algorithm | Parameters |", "| --- | --- | --- |",
              f"| Funding | {names[0]} | {algorithm_parameters(fd)} |",
              f"| Measures | {names[1]} | {algorithm_parameters(md)} |", "",
              "## Test Metrics", "",
              "| Task | Metric | Selected model | Baseline |", "| --- | --- | ---: | ---: |"]
    metrics = [
        ("Funding", "Interval coverage", "empirical_interval_coverage", ft, funding["test_median_baseline"], pct),
        ("Funding", "Median interval width (nominal USD)", "median_interval_width_nominal_usd", ft, funding["test_median_baseline"], number),
        ("Funding", "Mean interval score (nominal USD)", "mean_interval_score_nominal_usd", ft, funding["test_median_baseline"], number),
        ("Measures", "Recall@2", "recall_at_2", mt, measures["test_frequency_baseline"], pct),
        ("Measures", "Any hit@2", "any_hit_at_2", mt, measures["test_frequency_baseline"], pct),
        ("Measures", "Macro F1", "f1_macro", mt, measures["test_frequency_baseline"], lambda v: number(v, ".3f")),
        ("Measures", "Macro AP", "macro_average_precision", mt, measures["test_frequency_baseline"], lambda v: number(v, ".3f")),
        ("Measures", "Macro ROC-AUC", "macro_roc_auc", mt, measures["test_frequency_baseline"], lambda v: number(v, ".3f")),
    ]
    for task, label, key, selected, baseline, formatter in metrics:
        lines.append(f"| {task} | {label} | {formatter(selected[key])} | {formatter(baseline[key])} |")
    lines += ["", "## Per-Measure Test Metrics", "",
              "| Measure | Positive samples | F1 | AP | ROC-AUC |", "| --- | ---: | ---: | ---: | ---: |"]
    for measure, row in mt["per_measure"].items():
        lines.append(f"| {MEASURE_NAMES[measure]} | {row['positive_support']} | {number(row['f1'], '.3f')} | {number(row['average_precision'], '.3f')} | {number(row['roc_auc'], '.3f')} |")
    stability = report.get("stability")
    if stability:
        lines += ["", "## Cross-Validation", "",
                  "| Field | Value |", "| --- | --- |",
                  f"| Development counties | {stability['development_counties']:,} |",
                  f"| Folds | {stability['folds']} |",
                  f"| Unit | {stability['unit']} |", "",
                  "| Task | Metric | Fold mean | Fold sample standard deviation |", "| --- | --- | ---: | ---: |"]
        for task, name in zip(("funding", "measures"), names):
            for metric, values in stability[task][name].items():
                lines.append(f"| {task.title()} | {metric} | {number(values['mean'], '.6f')} | {number(values['std'], '.6f')} |")
    background, interval = result["county_background"], result["pa_cumulative_assistance_reference"]
    lines += ["", "## Example Prediction", "",
              "| Field | Value |", "| --- | --- |",
              f"| County | {background['county_name']} |",
              f"| County FIPS | {county_fips} |",
              f"| Flood type | {flood_type} |",
              f"| Impact level | {impact_level} |",
              f"| Reference year | {year} |",
              f"| Prior-measure context year | {background['prior_measure_context_year']} |"]
    for field, value in background["static_values"].items():
        lines.append(f"| {field} | {number(value)} |")
    lines += [f"| Funding lower bound (nominal USD) | {number(interval['lower_usd'], ',.2f')} |",
              f"| Funding upper bound (nominal USD) | {number(interval['upper_usd'], ',.2f')} |", "",
              "| Prior measure | Closed project records |", "| --- | ---: |"]
    for measure, count in background["prior_closed_projects_by_measure"].items():
        lines.append(f"| {MEASURE_NAMES[measure]} | {number(count, '.0f')} |")
    lines += ["", "| Rank | Recommended measure | Ranking score |", "| ---: | --- | ---: |"]
    for index, measure in enumerate(result["recommended_measures"], start=1):
        lines.append(f"| {index} | {measure['name']} | {measure['ranking_score']:.4f} |")
    lines += ["", "## Data Files", "",
              "| Artifact | File |", "| --- | --- |"]
    for label, filename in (
        ("Model", "flood_policy.joblib"),
        ("Training report", "training_report.json"),
        ("Funding test predictions", "funding_test_predictions.csv"),
        ("Measure test predictions", "measure_test_predictions.csv"),
        ("Measure test curves", "measure_test_curves.json"),
        ("Example prediction", "example_prediction.json"),
    ):
        lines.append(f"| {label} | {local_link(filename, output_dir / filename)} |")
    lines.append("")
    example_path, summary_path = output_dir / "example_prediction.json", output_dir / "model_summary.md"
    example_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": str(summary_path), "example": str(example_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--county-fips", default="37155")
    parser.add_argument("--flood-type", choices=("Flood", "Flash Flood"), default="Flash Flood")
    parser.add_argument("--impact-level", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    print(json.dumps(export_summary(args.output_dir, args.source, county_fips=args.county_fips,
                                   flood_type=args.flood_type, impact_level=args.impact_level, year=args.year), indent=2))


if __name__ == "__main__":
    main()
