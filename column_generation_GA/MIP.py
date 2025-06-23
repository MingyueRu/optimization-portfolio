import gurobipy as gp
from gurobipy import GRB
import numpy as np
from my_parameters import ProblemParameters

def min_expectation(prob_matr, y_matr):
    min_exp = {}
    max_alloc = max(np.max(S), np.max(K))  # 最大分配量上限
    
    for j in range(d):
        for loc in range(R):
            # 动态步长（高需求区域更密集）
            step = max(1, int(lambdas[j, loc] / 10))
            x_vals = list(range(0, max_alloc + 1, step))
            
            # 向量化计算期望
            y_vals = [
                np.sum(prob_matr[j, loc] * np.minimum(z, y_matr[j, loc]))
                for z in x_vals
            ]
            
            min_exp[(j, loc)] = (x_vals, y_vals)
    return min_exp

# 参数初始化
params = ProblemParameters(d=20, r=10)
d = params.d
R = params.r
lambdas = params.lambdas
S = params.S
K = params.K
prob_matr, y_matr = params.compute_d_prob()

# 计算PWL分段点
min_expectation = min_expectation(prob_matr, y_matr)

# 模型构建
m = gp.Model("Media_Allocation")
z = m.addVars(d, R, vtype=GRB.INTEGER, lb=0, ub=int(np.max(K)), name="z")
aux = m.addVars(d, R, name="aux")

# 添加PWL约束
for j in range(d):
    for loc in range(R):
        x_vals, y_vals = min_expectation[(j, loc)]
        m.addGenConstrPWL(z[j, loc], aux[j, loc], x_vals, y_vals, f"pwl_{j}_{loc}")

# 目标与约束
m.setObjective(aux.sum(), GRB.MAXIMIZE)
for j in range(d):
    m.addConstr(z.sum(j, '*') <= S[j], f"supply_{j}")
for loc in range(R):
    m.addConstr(z.sum('*', loc) <= K[loc], f"cap_{loc}")

# 求解与输出
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f"最优期望销量: {m.objVal:.2f}")
    for j in range(d):
        for loc in range(R):
            if z[j, loc].X > 1e-6:
                print(f"产品 {j} -> 地区 {loc}: {int(z[j, loc].X)} 单位")
else:
    print(f"求解失败，状态码: {m.status}")

