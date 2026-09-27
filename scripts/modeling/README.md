# Flood Policy Model Data

## Dataset

| Field | Value |
| --- | --- |
| Source | `data/analyzed_data/flood_policy_annual/flood_policy_county_year_1999_2025.csv` |
| Years | 1999-2025 |
| County-year records | 36,450 |
| Source columns | 261 |
| Model features | 17 |
| Split unit | County |
| Requested train / validation / test fractions | 70% / 15% / 15% |
| Split seed | 42 |
| Nominal funding interval coverage | 80% |

## Samples

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| All rows | 25,507 | 5,366 | 5,577 |
| Counties | 2,214 | 474 | 475 |
| Funding samples | 1,347 | 254 | 262 |
| Measure samples | 1,642 | 369 | 352 |

| Funding subset | Samples |
| --- | ---: |
| Preliminary fitting | 1,060 |
| Preliminary calibration | 287 |
| Validation selection | 127 |
| Final calibration | 127 |

## Models

| Task | Selected algorithm | Parameters | Target |
| --- | --- | --- | --- |
| Funding | `elastic_net` | alpha=0.01, l1_ratio=0.25 | `pa_flood_incident_federal_share_obligated_nominal_usd` |
| Measures | `logistic` | l1_ratio=0.0, C=1 | Acquisition, drainage, elevation, flood control |

| Task | Candidate algorithms |
| --- | --- |
| Funding | `median_baseline`, `elastic_net`, `random_forest`, `quantile_gbdt` |
| Measures | `frequency_baseline`, `logistic`, `random_forest` |

## Model Files

| Data | File |
| --- | --- |
| Splits | `data/modeling/splits/train.csv`, `validation.csv`, `test.csv` |
| Split metadata | [split_metadata.json](../../data/modeling/splits/split_metadata.json) |
| Model | [flood_policy.joblib](../../data/modeling/models/flood_policy.joblib) |
| Results | [model_summary.md](../../data/modeling/models/model_summary.md) |
| Full training report | [training_report.json](../../data/modeling/models/training_report.json) |
| Funding test predictions | [funding_test_predictions.csv](../../data/modeling/models/funding_test_predictions.csv) |
| Measure test predictions | [measure_test_predictions.csv](../../data/modeling/models/measure_test_predictions.csv) |
| Measure test curves | [measure_test_curves.json](../../data/modeling/models/measure_test_curves.json) |
| Example prediction | [example_prediction.json](../../data/modeling/models/example_prediction.json) |
