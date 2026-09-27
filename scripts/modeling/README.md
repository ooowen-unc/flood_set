# 洪水政策模型

目标：输入洪水类型、县 FIPS、影响等级，展示县背景，输出 PA 灾后累计援助额参考范围，并从收购、排水、建筑抬升、防洪工程中推荐两项。

## 项目现状和执行步骤

现有清洗、CPI 和年度合并脚本继续保留。本目录只读取已经生成的 `data/analyzed_data/flood_policy_annual/flood_policy_county_year_1999_2025.csv`，不重新合并、不增加数据集、不修改原始文件。

年度主表为 1999–2025 年的 36,450 个县年度样本、261 个字段。全量数据先统一切分；训练时才分别筛出援助金额标签和措施标签可用的子集，不要求两个任务的标签在同一行同时存在。

1. **切分**：`split_data.py` 仅使用 Python 标准库，保存全部原始行和列。默认随机按县分组，约 70%/15%/15%，种子 42。同一县的全部年度属于同一集合；比例是县比例，行比例可能略有差别。
2. **特征和等级**：`data.py` 统一训练与查询的输入。影响等级阈值只使用训练集正值洪水损失的三分位数，单位为 2025 年年均 CPI 美元。低于第一阈值为轻，第一至第二阈值为中，达到第二阈值为重；零损失为轻，伤者至少为中，有死亡为重。这是项目内部的县年度影响等级，不是官方洪水等级。未来用户选等级时直接提供 1/2/3，不要求再次输入真实损失。
3. **比较、选优和校准**：`train.py` 训练少量固定参数的模型，不进行大规模搜索。资金候选均先校准再选优；选定后完整训练集重训，独立最终校准。默认同时完成训练/验证集合内三折县级稳定性检查。仅验证集选优，最终只保存两个任务各自选中的模型，但报告描述所有候选模型。
4. **查询**：`predict.py` 读取模型与县背景，返回 JSON。只支持 `Flood`、`Flash Flood`；年份默认 2025，可以指定 1999–2025 的参考年。背景取指定县不晚于参考年的最近记录，若较旧会提示。

## 模型和标签

| 任务 | 候选算法 | 标签与规则 |
| --- | --- | --- |
| PA 金额 | 历史中位数基线、弹性网络、随机森林、分位数梯度提升树 | `pa_flood_incident_federal_share_obligated_nominal_usd`，仅明确 Flood 类的已记录正值；不把未匹配或空白填零 |
| 四类措施 | 历史频率基线、多标签逻辑回归、多标签随机森林 | 当年首次批准的四类措施计数转为 0/1；仅在至少明确记录其中一项的县年度训练，其他年度不当作“没有行动” |

模型输入只含影响等级、两种洪水类型标记、参考年、人口/资产/面积、社会脆弱性、韧性、内陆洪水风险/频率和此前年度结项的措施背景。规模与数量特征使用 `log1p`；缺失值填补、缺失指示和线性模型缩放都在训练集的模型流水线中拟合。县编号、同年 PA/HMA 拨款和项目状态不作为输入；实际损失和伤亡只用于构造等级。

**金额**：训练 `log1p(金额)`。分位数树内部拟合低、中、高三个分位数；弹性网络和随机森林先拟合点预测。初步选优时，训练县按80%/20%分为拟合与校准部分，每个候选均使用残差/CQR规则形成区间，再在独立验证选优县上比较 `mean_interval_score_log1p`。该评分只依赖上下界，兼顾区间宽度和漏出惩罚。选择固定后，全部候选用完整训练集重训，并在另外的验证校准县上完成最终校准；不根据最终校准或测试表现重新选择。默认目标覆盖率80%。查询JSON和预测CSV仅输出上下界，不输出中心金额；内部点预测仍用于训练报告中的误差诊断。

**措施**：四个独立二分类器组成多标签模型，不要求只能有一个正确标签。每次查询仅输出排序前两项。验证集按 Recall@2、再按宏平均 AP 选择。所有 F1、精确率、召回率和混淆矩阵都使用实际前两项决策，不使用 0.5 阈值。PR/ROC 使用连续分数。分数是历史选择排序分数，不是已验证效果或校准后的政策采用概率。

基线可以被选中；不因算法复杂就强行使用它。测试集描述所有冻结候选，但不参与参数、阈值和模型选择。资金只在选优后用完整训练集重训，不把验证县加入最终拟合。

**稳定性**：训练与验证集合合并为开发集合，进行三折县级交叉验证，各折单独拟合等级阈值和预处理；资金每折训练县再留20%作校准，措施使用该折完整训练县。所有候选参数固定，记录折均值及样本标准差，不据此重选算法或调参。标准差不是置信区间。此检查衡量历史县级泛化稳定性，不是按年份向未来预测的验证。

## 输出指标和模型描述

`training_report.json`（schema 3）的 `funding.all_models` 与 `measures.all_models` 提供每个候选的模型参数、训练耗时、输入及变换后特征、线性系数或树特征重要性，以及训练、验证/选优、校准和测试指标。资金选优指标位于 `selection_calibrated`，对应初步拟合及校准后的模型；其余资金指标对应完整训练集重训后的模型。`stability` 保存三折结果与汇总。原有等级分组仅保留描述，不进行专项优化。`funding.test`、`measures.test` 等选中模型字段继续保留。

| 任务 | 报告指标 |
| --- | --- |
| 资金中心预测 | 原美元及 log1p 尺度的 MAE、RMSE、R²，绝对误差中位数，MAPE、SMAPE、偏差和实际金额分布 |
| 资金区间 | 目标及实测覆盖率、覆盖差距、上下界漏出比例、区间均值/中位宽度、美元和 log1p 区间评分，各分位数的美元和 log1p 分位数损失 |
| 措施推荐 | Recall@2、Precision@2、至少一项命中率；samples/micro/macro/weighted 的 Precision、Recall、F1；完全匹配率及 Hamming loss |
| 措施排序与类别 | 各类 AP、ROC-AUC，宏/微平均 AP 和宏/微/加权 ROC-AUC；每类样本数、精确率、召回率、F1、特异度及 `[[TN,FP],[FN,TP]]` 混淆矩阵 |

`measure_test_curves.json` 保存所有措施候选在测试集上的四类 PR 和 ROC 的完整坐标及阈值，可独立绘图，不需要重新预测。ROC 起始无穷阈值、PR 末尾无对应阈值记为 `null`。某类只有一种真实结果时，其 AP/ROC 记为 `null` 并说明原因；常量目标或单样本组的 R² 也记为 `null`。

F1(samples) 是每个县年度 F1 的平均，不是对已经平均的 Precision@2 和 Recall@2 再求调和平均。固定输出两项而历史只记一项时，额外推荐会降低精确率/F1；指标衡量历史一致性，不直接判断措施效果。R² 可为负、MAPE 会被小金额放大。训练指标是样本内描述，不能代替测试成绩。系数及重要性也不代表因果效应。

## 运行

使用 Python **3.11 或更高版本**，从项目根目录执行。可以在自己的虚拟环境安装依赖；切分无需第三方包。

```powershell
# 数据已切好；仅需重建时执行这一行，替换已有切分文件
python scripts/modeling/split_data.py --overwrite

# 安装模型依赖
python -m pip install -r scripts/modeling/requirements.txt

# 训练、比较、校准并评估两个任务
python scripts/modeling/train.py --overwrite

# 导出最终汇总文档及完整示例查询JSON
python scripts/modeling/export_summary.py

# 训练成功后查询；Robeson County, NC，重度山洪情景
python scripts/modeling/predict.py --county-fips 37155 --flood-type "Flash Flood" --impact-level severe
```

首次生成切分文件时可省略 `--overwrite`；脚本默认保护已有文件。

可选用法：

```powershell
# 减少候选：仅训练分位数树和逻辑回归（仍包含两个基线）
python scripts/modeling/train.py --regressors quantile_gbdt --classifiers logistic --jobs 2

# 已有模型需要重训，并要求 90% 名义区间覆盖率
python scripts/modeling/train.py --coverage 0.9 --overwrite

# 中文等级也支持：轻、中、重
python scripts/modeling/predict.py --county-fips 37155 --flood-type Flood --impact-level 重 --year 2025

# 可选：纯随机行切分，写到另一目录。该方式会使同一县跨集合，评估可能更乐观。
python scripts/modeling/split_data.py --unit row --output-dir data/modeling/row_splits
python scripts/modeling/train.py --split-dir data/modeling/row_splits --output-dir data/modeling/row_models
```

默认切分保存到 `data/modeling/splits/`：`train.csv`、`validation.csv`、`test.csv`、`split_metadata.json`。默认训练产物位于 `data/modeling/models/`：`flood_policy.joblib`、`training_report.json`、`measure_test_curves.json`、`funding_test_predictions.csv` 和 `measure_test_predictions.csv`。汇总脚本生成 `model_summary.md` 与 `example_prediction.json`。资金CSV有262条测试预测，措施CSV有352条测试预测（当前默认切分）；资金仅保留实际金额与上下界，措施保存真实标签、四类分数和前两项推荐。修改代码不会自动更新已有报告。重新生成时运行：

```powershell
python scripts/modeling/train.py --overwrite
python scripts/modeling/export_summary.py
```

如需曲线图片，可单独安装可选绘图依赖并运行；只读取新曲线 JSON，不训练模型：

```powershell
python -m pip install matplotlib
python scripts/modeling/plot_metrics.py
```

输出 `data/modeling/models/plots/measure_test_pr.png` 和 `measure_test_roc.png`，每张图四个子图，各自比较全部措施候选。仅需指标及曲线坐标时无需安装 matplotlib。

运行指标、县级隔离和区间选优口径测试；仅校准测试拟合极小中位数基线：

```powershell
python -m unittest discover -s scripts/modeling -p "test_*.py" -v
```

## 解释边界

- 资金目标是 PA 中明确 Flood 类项目的联邦累计承诺快照，归入灾害声明年。它不是实际年度支付、所有政府机构援助总额或最优预算。政府金额保留名义值，参考年是模型输入。
- NRI 是当前静态快照，不能视为各历史年的真实人口、脆弱性和韧性。输出会注明版本。
- 既有措施使用严格早于背景年份的结项记录。它是项目覆盖背景，不是每个灾害点的防护验证；一个项目的多个活动和跨县覆盖不能简单相加。
- 原始年度中可以同时有山洪和内陆洪水。单类型查询是年度情景参考，不把同一笔年度资金复制成多个独立灾害训练样本。
- 默认按县随机切分降低县身份重复，但不是未来年份预测验证，不能消除共有 FEMA 灾害和多县项目的所有相关性。
- 残差校准追求经验覆盖；同县年度相关、静态快照和未来分布变化限制理论覆盖保证。查看测试报告中的真实覆盖和宽度，不只看名义 80%。
- 按县随机测试是历史关联评估；新增诊断不会改变这一验证口径。测试报告不应反复用于选参数。

算法参数和接口参考：[分位数梯度提升树](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)、[随机森林回归](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)、[多标签分类](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)。

指标接口参考：[F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)、[PR](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)、[ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)。
