import numpy as np
import math

class ProblemParameters:
    def __init__(self, d=100, r=20, demand_scale=15, seed=123):
        """
        改进版参数生成器：
        1. 地区容量与当地总需求正相关
        2. 产品供应量与其总需求匹配
        3. 确保基础可行性
        """
        np.random.seed(seed)
        self.d = d
        self.r = r
        
        # 1. 生成需求参数（确保最小需求为1）
        self.lambdas = np.maximum(
            np.random.poisson(demand_scale, (d, r)).astype(float) * 
            np.random.choice([0.5, 0.8, 1, 1.5, 2], (d, r)),
            1.0  # 最小需求为1
        )
        
        # 2. 计算各地区的总需求（用于容量生成）
        region_demands = np.sum(self.lambdas, axis=0)  # 各地区的总需求
        
        # 3. 供应量 = 产品总需求 * (1.1~1.3) + 缓冲量
        self.S = np.maximum(
            (np.sum(self.lambdas, axis=1) * np.random.uniform(1.1, 1.3)).astype(int),
            np.max(self.lambdas, axis=1).astype(int) + 3  # 至少满足最大地区需求+3
        ).astype(int)
        
        # 4. 地区容量 = 该地区总需求 * (1.2~1.5) + 缓冲量
        # self.K = np.maximum(
        #     (region_demands * np.random.uniform(1.2, 1.5)).astype(int),
        #     np.sum(self.lambdas > 5, axis=0) * z_max  # 高需求地区至少能容纳所有高需求产品
        # )
        self.K = (region_demands * np.random.uniform(1.2, 1.5)).astype(int)
    
        
    
    def compute_d_prob(params):
        """
        计算每个产品-地区组合的需求概率（泊松分布）
        参数:
            params: ProblemParameters 实例
            
        返回:
            prob_matrix: 形状 (d, r)，每个元素是概率列表
            y_matrix: 形状 (d, r)，每个元素是y值列表
        """
        d, r = params.d, params.r
        lambdas = params.lambdas  # 形状 (d, r)
        
        y_matrix = np.empty((d, r), dtype=object)       
        prob_matrix = np.empty((d, r), dtype=object)
        
        for i in range(d):
            for j in range(r):
                lambda_ = lambdas[i, j]
                y_max = int(max(lambda_ + 3*math.sqrt(lambda_), 10))  # 修正1

                # 生成y值列表
                y_list = list(range(y_max + 1))
                y_matrix[i, j] = y_list
                
                # 计算概率列表
                prob_list = [math.exp(-lambda_) * (lambda_**k) / math.factorial(k) 
                            for k in y_list]  # 修正2和3
                prob_matrix[i, j] = prob_list
                if [i,j]==[0,0]:
                    print(lambda_)
                    print(y_max)
                    print(y_list[:])
                    print(prob_list[:])
        return prob_matrix, y_matrix
    
