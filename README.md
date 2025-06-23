# 运筹优化项目合集

## 项目列表

### 1. 多地区资源分配优化（DW分解与遗传算法）
路径：[`column_generation_GA/`](./column_generation_GA)

- 建立多地区多产品混合整数规划模型，在容量与供应限制下最大化总期望销量
- 使用 Gurobi 实现 Dantzig-Wolfe 分解，并设计启发式遗传算法进行对比
- 分析不同求解方法在准确性与效率上的差异

### 2. 数据驱动库存优化（M5零售预测 + Prescriptive优化框架复现）
路径：[`predictive_prescriptions/`](./predictive_prescriptions)

- 复现 Bertsimas 提出的预测-优化一体化框架（From Predictive to Prescriptive Analytics）
- 使用非参数预测方法（随机森林、kNN等）+ 样本加权机制 + Gurobi 求解非线性目标
- 分析模型的 Prescriptiveness 指标

所使用的数据集下载链接：https://www.kaggle.com/competitions/m5-forecasting-accuracy
整合生成的数据放置于data文件夹下
