# Flood Policy Model Data

## Dataset

| Field | Value |
| --- | --- |
| County-year records | 36,450 |
| Split seed | 42 |
| Features | 17 |
| Funding target | `pa_flood_incident_federal_share_obligated_nominal_usd` |
| Nominal interval coverage | 80.00% |

| Split | Rows | Counties |
| --- | ---: | ---: |
| train | 25,507 | 2,214 |
| validation | 5,366 | 474 |
| test | 5,577 | 475 |

## Labeled Samples

| Task | Subset | Samples |
| --- | --- | ---: |
| Funding | train | 1,347 |
| Funding | selection | 127 |
| Funding | calibration | 127 |
| Funding | test | 262 |
| Funding | preliminary_fit | 1,060 |
| Funding | preliminary_calibration | 287 |
| Measures | train | 1,642 |
| Measures | validation | 369 |
| Measures | test | 352 |

## Selected Models

| Task | Algorithm | Parameters |
| --- | --- | --- |
| Funding | elastic_net | alpha=0.01, l1_ratio=0.25 |
| Measures | logistic | l1_ratio=0.0, C=1 |

## Test Metrics

| Task | Metric | Selected model | Baseline |
| --- | --- | ---: | ---: |
| Funding | Interval coverage | 78.63% | 79.39% |
| Funding | Median interval width (nominal USD) | 2238862.55 | 3501411.96 |
| Funding | Mean interval score (nominal USD) | 6348586.73 | 7387640.47 |
| Measures | Recall@2 | 84.47% | 81.87% |
| Measures | Any hit@2 | 87.78% | 86.93% |
| Measures | Macro F1 | 0.473 | 0.304 |
| Measures | Macro AP | 0.500 | 0.286 |
| Measures | Macro ROC-AUC | 0.727 | 0.500 |

## Per-Measure Test Metrics

| Measure | Positive samples | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Property acquisition and floodplain retreat | 211 | 0.800 | 0.822 | 0.791 |
| Drainage and stormwater management | 107 | 0.491 | 0.615 | 0.791 |
| Building elevation | 48 | 0.290 | 0.234 | 0.606 |
| Flood control infrastructure | 36 | 0.310 | 0.329 | 0.720 |

## Cross-Validation

| Field | Value |
| --- | --- |
| Development counties | 2,688 |
| Folds | 3 |
| Unit | county |

| Task | Metric | Fold mean | Fold sample standard deviation |
| --- | --- | ---: | ---: |
| Funding | empirical_interval_coverage | 0.838641 | 0.024652 |
| Funding | median_interval_width_nominal_usd | 3233769.305151 | 484322.296908 |
| Funding | mean_interval_score_nominal_usd | 14152547.499109 | 2694538.505515 |
| Funding | mean_interval_score_log1p | 5.828274 | 0.062560 |
| Measures | recall_at_2 | 0.823227 | 0.014609 |
| Measures | f1_macro | 0.442849 | 0.017166 |
| Measures | macro_average_precision | 0.515064 | 0.042709 |
| Measures | macro_roc_auc | 0.725067 | 0.037607 |

## Example Prediction

| Field | Value |
| --- | --- |
| County | Robeson |
| County FIPS | 37155 |
| Flood type | Flash Flood |
| Impact level | 3 |
| Reference year | 2025 |
| Prior-measure context year | 2024 |
| nri_population | 116414.00 |
| nri_buildvalue | 20676381604.00 |
| nri_agrivalue | 496647982.00 |
| nri_area | 958.84 |
| nri_sovi_score | 98.35 |
| nri_resl_score | 14.09 |
| nri_ifld_risks | 86.39 |
| nri_ifld_afreq | 0.71 |
| Funding lower bound (nominal USD) | 261,497.59 |
| Funding upper bound (nominal USD) | 8,722,002.28 |

| Prior measure | Closed project records |
| --- | ---: |
| Property acquisition and floodplain retreat | 3 |
| Drainage and stormwater management | 0 |
| Building elevation | 0 |
| Flood control infrastructure | 0 |

| Rank | Recommended measure | Ranking score |
| ---: | --- | ---: |
| 1 | Property acquisition and floodplain retreat | 0.6996 |
| 2 | Drainage and stormwater management | 0.3536 |

## Data Files

| Artifact | File |
| --- | --- |
| Model | [flood_policy.joblib](<E:/Programs/GitHub/26datachallenge/data/modeling/models/flood_policy.joblib>) |
| Training report | [training_report.json](<E:/Programs/GitHub/26datachallenge/data/modeling/models/training_report.json>) |
| Funding test predictions | [funding_test_predictions.csv](<E:/Programs/GitHub/26datachallenge/data/modeling/models/funding_test_predictions.csv>) |
| Measure test predictions | [measure_test_predictions.csv](<E:/Programs/GitHub/26datachallenge/data/modeling/models/measure_test_predictions.csv>) |
| Measure test curves | [measure_test_curves.json](<E:/Programs/GitHub/26datachallenge/data/modeling/models/measure_test_curves.json>) |
| Example prediction | [example_prediction.json](<E:/Programs/GitHub/26datachallenge/data/modeling/models/example_prediction.json>) |
