"""Export a concise final model document and one reproducible scenario query."""
import argparse
import json
from pathlib import Path

from config import ARTIFACTS, MEASURE_NAMES, SOURCE
from predict import predict

FUNDING_NAMES = {"median_baseline": "中位数基线", "elastic_net": "弹性网络",
                 "random_forest": "随机森林回归", "quantile_gbdt": "分位数梯度提升树"}
MEASURE_NAMES_ALGORITHM = {"frequency_baseline": "历史频率基线", "logistic": "多标签逻辑回归",
                           "random_forest": "多标签随机森林"}


def pct(value):
    return f"{value * 100:.2f}%"


def number(value, spec=".2f"):
    return format(value, spec) if value is not None else "未匹配"


def local_link(label, path):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def algorithm_parameters(description):
    heads = description.get("heads", {})
    if not heads:
        return "训练集四类措施出现频率"
    params = next(iter(heads.values()))["parameters"]
    keys = ("n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "subsample", "alpha", "l1_ratio", "C")
    return "，".join(f"{key}={params[key]}" for key in keys if key in params) or "训练集金额中位数"


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
        "# 洪水灾后援助与防治措施模型：最终结果", "",
        "输入洪水类型（Flood / Flash Flood）、县 FIPS、影响等级（1–3），参考年份默认2025。"
        "查询县人口、社会脆弱性、韧性和既有措施背景，输出PA累计援助额上下界，并从收购、排水、建筑抬升、防洪工程中推荐两项。", "",
        "## 数据与训练方法", "",
        f"使用1999–2025年的{sum(s['rows'] for s in report['split_sizes'].values()):,}条县年度记录。按县随机分成约70%训练、15%验证、15%测试，种子{report['split_seed']}；同一县不跨集合。"
        "两个任务分别使用标签有效的样本，资金未记录值不填零，措施仅在至少记录四类之一的年度学习历史选择。", "",
        f"- 有效资金样本：训练{sizes['funding_train']}，验证选优{sizes['funding_selection']}，独立校准{sizes['funding_calibration']}，测试{sizes['funding_test']}。",
        f"- 有效措施样本：训练{sizes['measures']['train']}，验证{sizes['measures']['validation']}，测试{sizes['measures']['test']}。",
        f"- 使用{len(report['features'])}个特征，包括等级、年份、洪水类型、县背景和此前结项项目计数。规模特征取log1p；中位数填补缺失值并加入缺失指示，线性模型标准化。预处理仅在相应训练部分拟合。",
        f"- 资金比较中位数基线、弹性网络、随机森林、分位数梯度提升树，在log1p(金额)上拟合。候选均先在训练集内部按县分出独立数据校准区间，再以验证集的log1p区间评分选优；选定后用完整训练集重训，并使用另一些验证县完成最终校准。目标覆盖率{pct(funding['nominal_interval_coverage'])}。",
        "- 措施比较历史频率基线、多标签逻辑回归、多标签随机森林。四类措施采用独立二分类器，按分数选择前两项；验证集按Recall@2、再按宏平均AP选优。固定参数比较，测试集仅作最终评估。", "",
        "## 最终算法与测试结果", "",
        f"**资金：{FUNDING_NAMES[names[0]]}（{names[0]}）**。{algorithm_parameters(fd)}。",
        f"**措施：{MEASURE_NAMES_ALGORITHM[names[1]]}（{names[1]}）**。{algorithm_parameters(md)}。", "",
        "| 任务 | 指标 | 最终模型 | 基线 |", "| --- | --- | ---: | ---: |",
        f"| 资金 | 区间覆盖率 | {pct(ft['empirical_interval_coverage'])} | {pct(funding['test_median_baseline']['empirical_interval_coverage'])} |",
        f"| 资金 | 区间宽度中位数（万美元） | {ft['median_interval_width_nominal_usd']/10000:.2f} | {funding['test_median_baseline']['median_interval_width_nominal_usd']/10000:.2f} |",
        f"| 资金 | 美元区间评分（万美元，越低越好） | {ft['mean_interval_score_nominal_usd']/10000:.2f} | {funding['test_median_baseline']['mean_interval_score_nominal_usd']/10000:.2f} |",
        f"| 措施 | Recall@2 | {pct(mt['recall_at_2'])} | {pct(measures['test_frequency_baseline']['recall_at_2'])} |",
        f"| 措施 | 至少命中一项 | {pct(mt['any_hit_at_2'])} | {pct(measures['test_frequency_baseline']['any_hit_at_2'])} |",
        f"| 措施 | 宏平均F1 | {mt['f1_macro']:.3f} | {measures['test_frequency_baseline']['f1_macro']:.3f} |",
        f"| 措施 | 宏平均AP / ROC-AUC | {mt['macro_average_precision']:.3f} / {mt['macro_roc_auc']:.3f} | {measures['test_frequency_baseline']['macro_average_precision']:.3f} / {measures['test_frequency_baseline']['macro_roc_auc']:.3f} |", "",
        "| 措施 | 测试正例数 | F1 | AP | ROC-AUC |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for measure, row in mt["per_measure"].items():
        lines.append(f"| {MEASURE_NAMES[measure]} | {row['positive_support']} | {row['f1']:.3f} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} |")
    stability = report.get("stability")
    if stability:
        fc, mc = stability["funding"][names[0]], stability["measures"][names[1]]
        lines += ["", "## 稳定性", "",
                  f"训练与验证集合共{stability['development_counties']}个县，进行{stability['folds']}折县级交叉验证，每折单独学习等级阈值和预处理，并保留独立资金校准县。以下为折均值±折间标准差，不是置信区间。交叉验证不使用最终测试集，也不重新选择算法。", "",
                  f"- 资金覆盖率：{pct(fc['empirical_interval_coverage']['mean'])} ± {fc['empirical_interval_coverage']['std']*100:.2f}个百分点；区间宽度中位数的折均值：{fc['median_interval_width_nominal_usd']['mean']/10000:.2f}万美元。",
                  f"- 措施Recall@2：{pct(mc['recall_at_2']['mean'])} ± {mc['recall_at_2']['std']*100:.2f}个百分点；宏平均F1：{mc['f1_macro']['mean']:.3f} ± {mc['f1_macro']['std']:.3f}。"]
    background, interval = result["county_background"], result["pa_cumulative_assistance_reference"]
    values = background["static_values"]
    lines += ["", "## 查询输出示例", "",
              f"{background['county_name']}县（{county_fips}），{flood_type}，影响等级{impact_level}，参考年{year}。",
              f"人口{number(values['nri_population'], ',.0f')}；社会脆弱性分数{number(values['nri_sovi_score'])}；韧性分数{number(values['nri_resl_score'])}。既有措施背景取{background['prior_measure_context_year']}年，计数为此前结项项目覆盖记录。",
              "既有措施记录：" + "，".join(f"{MEASURE_NAMES[name]} {number(count, '.0f')}" for name, count in background['prior_closed_projects_by_measure'].items()) + "。", "",
              f"**累计援助额参考范围：{interval['lower_usd']:,.2f}–{interval['upper_usd']:,.2f}美元。**", ""]
    for index, measure in enumerate(result["recommended_measures"], start=1):
        lines.append(f"{index}. {measure['name']}，排序分数{measure['ranking_score']:.4f}。")
    lines += ["", "## 使用范围与产物", "",
              "适合历史政策关联查询和项目原型。资金是明确Flood类PA项目的联邦累计承诺快照，按灾害声明年归集，保留名义美元；不是年度支付或所有机构援助。"
              "措施得分表示历史选择排序，不代表效果或采用概率。NRI为静态快照；影响等级为训练损失分位数与伤亡构造的内部等级。随机县测试和交叉验证不证明未来预测或因果效果。",
              "", "- " + local_link("模型文件", output_dir / "flood_policy.joblib"),
              "- " + local_link("完整指标、算法参数和稳定性报告", output_dir / "training_report.json"),
              "- " + local_link("资金测试预测表", output_dir / "funding_test_predictions.csv") + "（实际金额、上下界、是否覆盖）",
              "- " + local_link("措施测试预测表", output_dir / "measure_test_predictions.csv") + "（历史标签、四类得分、两项推荐）",
              "- " + local_link("完整示例查询JSON", output_dir / "example_prediction.json"),
              "", "## 运行", "", "```powershell",
              "python scripts/modeling/train.py --overwrite",
              "python scripts/modeling/export_summary.py",
              'python scripts/modeling/predict.py --county-fips 37155 --flood-type "Flash Flood" --impact-level severe',
              "```", ""]
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
