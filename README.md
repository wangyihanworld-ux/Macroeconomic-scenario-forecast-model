# 宏观经济情景与经营预测模型

[![tests](https://github.com/wangyihanworld-ux/Macroeconomic-scenario-forecast-model/actions/workflows/tests.yml/badge.svg)](https://github.com/wangyihanworld-ux/Macroeconomic-scenario-forecast-model/actions/workflows/tests.yml)

这是面向经济分析、财务 BP 和 FP&A 求职场景的作品集项目四。项目使用完全合成的数据，把 GDP 增长率、通胀率、失业率、消费者信心和汇率指数连接到月度经营收入，建立可解释的多元回归、滚动样本外回测及压力/基准/乐观情景预测。

> 项目中的经济体、企业、月份、指标和金额均为程序生成的虚构数据，不代表真实公司业绩或官方宏观统计。

## 业务问题

- 宏观指标变化与经营收入有什么统计关联？
- 模型在未参与估计的月份上误差多大？
- 压力、基准和乐观假设下，下一年度收入区间如何变化？
- 财务 BP 应如何把模型结果转化为计划讨论，而不是把相关性误说成因果关系？

## 已实现

- 固定随机种子生成 60 个月合成历史数据和 36 条未来情景假设；
- 校验字段、日期、数值、重复月份、非正收入和重复情景键；
- 用最小二乘法估计五项宏观驱动及季节项；
- 输出 R²、调整后 R²、拟合值、残差与驱动系数；
- 采用前 36 个月起步的扩展窗口，对后 24 个月进行样本外滚动回测；
- 计算 MAE、RMSE 和 MAPE，并与季节朴素预测进行同窗口基准比较；
- 输出五项经济驱动的 VIF，共线性风险不再被高 R² 掩盖；
- 一键生成合成输入工作簿和含 8 张工作表的 Excel 管理报告；
- 18 项自动化测试覆盖数据契约、模型勾稽、回测、基准比较、VIF、情景方向和报告可重开。

## 一键运行

```powershell
python -m pip install -e .
python -m macro_forecast.demo
```

也可以使用安装后的命令：

```powershell
macro-forecast-demo --output-dir .\demo_output
```

输出文件：

- `合成宏观与经营输入.xlsx`
- `宏观经济情景与经营预测报告.xlsx`

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 方法边界

- 回归系数表示控制其他变量后的样本内统计关联，不等同于因果效应；
- 合成数据的拟合度和误差不能作为真实预测准确率承诺；
- 情景预测回答“假设这样变化会怎样”，不提供发生概率；
- 多重共线性、结构突变、滞后效应和外生冲击仍需在真实落地时进一步检验。

详细定义见 [数据字典与建模口径](docs/data-dictionary.md)，实施过程见 [项目计划](docs/project-plan.md)。

