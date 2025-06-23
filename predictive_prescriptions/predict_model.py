from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import numpy as np

# 用KNN方法计算得到有权重的订购量值，并记录相应的id，需要每期输入

def knn_weighted_sales_by_id(train_data, test_data, feature_cols, k=5):
    """
    仅在相同id的产品历史数据中寻找最近邻
    """
    results = []
    
    # 按id分组处理
    for id_val, id_group in train_data.groupby('id'):
        # 获取该id对应的测试数据
        test_samples = test_data[test_data['id'] == id_val]
        if len(test_samples) == 0:
            continue
            
        # 训练该id的KNN模型
        X_train = id_group[feature_cols].values
        nn = NearestNeighbors(n_neighbors=min(k, len(id_group)), metric='euclidean')
        nn.fit(X_train)
        
        # 处理该id的每个测试样本
        X_test = test_samples[feature_cols].values
        distances, indices = nn.kneighbors(X_test)
        
        for i, (_, test_row) in enumerate(test_samples.iterrows()):
            neighbors_data = []
            for idx in indices[i]:
                neighbors_data.append({
                    'weekly_sales': id_group.iloc[idx]['weekly_sales'],
                    'weight': 1/k
                })
            #输出的结构长这个样子，需要获取到id因为最后的最优解要知道对于哪种产品采购多少
            results.append({
                'test_id': id_val,
                'test_week': test_row['week'],
                'neighbors': neighbors_data
            })
    
    return results

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
def epanechnikov_kernel(u):
    """Epanechnikov核函数（标准形式）"""
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def kde_weighted_sales_by_id(train_data, test_data, feature_cols, bandwidth=1.0):
    """
    使用Epanechnikov核密度估计计算相同ID产品的样本权重
    
    参数：
    train_data  : 训练数据集（包含id, feature_cols, weekly_sales）
    test_data   : 测试数据集（同结构）
    feature_cols: 用于建模的特征列
    bandwidth   : 核带宽（控制平滑程度）
    
    返回：
    results : 结构与决策树/KNN函数一致，权重由核密度估计计算
    """
    results = []
    
    # 按id分组处理
    for id_val, id_group in train_data.groupby('id'):
        test_samples = test_data[test_data['id'] == id_val]
        if len(test_samples) == 0:
            continue

        # 准备数据
        X_train = id_group[feature_cols].values
        y_train = id_group['weekly_sales'].values
        
        # 标准化特征（使带宽对不同特征尺度敏感度一致）
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train_norm = (X_train - X_mean) / (X_std + 1e-8)
        
        # 处理每个测试样本
        for _, test_row in test_samples.iterrows():
            x_test = test_row[feature_cols].values.reshape(1, -1)
            x_test_norm = (x_test - X_mean) / (X_std + 1e-8)
            
            # 确保输入是二维数组
            if x_test_norm.ndim == 1:
                x_test_norm = x_test_norm.reshape(1, -1)
            if X_train_norm.ndim == 1:
                X_train_norm = X_train_norm.reshape(-1, 1)
            # print(x_test_norm)
            # print(X_train_norm)
            x_test_norm = x_test_norm.astype(float)  # 尝试转换为 float
            X_train_norm = X_train_norm.astype(float)
            # 计算测试点到所有训练点的标准化距离
            distances = cdist(x_test_norm, X_train_norm, metric='euclidean').flatten()
            u = distances / bandwidth  # 标准化距离
            
            # 应用Epanechnikov核函数计算权重
            weights = epanechnikov_kernel(u)
            if weights.sum() == 0:
                weights = np.ones_like(weights) / len(weights)  # 避免除零
            else:
                weights = weights / weights.sum()  # 归一化
            
            # 构建结果（保持与决策树/KNN相同结构）
            neighbors_data = [{
                'weekly_sales': y_train[i],
                'weight': weights[i]
            } for i in range(len(y_train)) if weights[i] > 0]
            
            results.append({
                'test_id': id_val,
                'test_week': test_row['week'],
                'neighbors': neighbors_data
            })
    
    return results
# # 核函数方法
# def kernel_weighted_sales_by_id(train_data, test_data, feature_cols, 
#                               kernel_type='naive', 
#                               adaptive_bandwidths=None):
#     """
#     基于核方法的权重计算（支持公式13和14）
    
#     参数：
#     train_data  : 训练数据集（包含id, feature_cols, weekly_sales）
#     test_data   : 测试数据集（同结构）
#     feature_cols: 用于距离计算的特征列
#     kernel_type : 核函数类型 ['naive', 'epanechnikov', 'tricubic']
#     bandwidth   : 固定带宽（公式13的h_N）
#     adaptive_bandwidths : 各样本的自适应带宽数组（公式14的h_i）
    
#     返回：
#     results : 结构同随机森林版本，格式如下：
#         [{
#             'test_id': id,
#             'test_week': week,
#             'neighbors': [{'weekly_sales': y, 'weight': w}, ...]
#         }, ...]
#     """
    
#     # 核函数定义
#     kernels = {
#         'naive': lambda u: (np.linalg.norm(u, axis=1) <= 1).astype(float),
#         'epanechnikov': lambda u: (1 - np.linalg.norm(u, axis=1)**2) * (np.linalg.norm(u, axis=1) <= 1),
#         'tricubic': lambda u: (1 - np.linalg.norm(u, axis=1)**3)**3 * (np.linalg.norm(u, axis=1) <= 1)
#     }
#     K = kernels[kernel_type]
    
#     results = []
#     for id_val, id_group in train_data.groupby('id'):
#         test_samples = test_data[test_data['id'] == id_val]
#         if len(test_samples) == 0:
#             continue
            
#         # 准备数据
#         X_train = id_group[feature_cols].values
#         y_train = id_group['weekly_sales'].values
#         X_train = X_train.astype(np.float64)  # 或 float32
        
#         # 处理每个测试样本
#         for _, test_row in test_samples.iterrows():
#             x_test = test_row[feature_cols].values.reshape(1, -1)
#             x_test = x_test.astype(np.float64)
#             # 计算距离矩阵 (1, n_train)
#                 # 公式13：固定带宽
#             distances = cdist(x_test, X_train, 'euclidean') 
#             bandwidth = np.percentile(distances, 30)  # 覆盖70%的样本
#             distances=distances/bandwidth
            
#             # 计算权重分子
#             weights = K(distances.T).flatten()  # (n_train,)
            
#             # 归一化
#             weights /= (weights.sum() + 1e-8)
            
#             # 构建结果（结构同随机森林版本）
#             neighbors_data = [{
#                 'weekly_sales': y_train[i],
#                 'weight': weights[i]
#             } for i in np.where(weights > 1e-6)[0]]
            
#             results.append({
#                 'test_id': id_val,
#                 'test_week': test_row['week'],
#                 'neighbors': neighbors_data
#             })
    
#     return results



def optimized_rf_weights(train_data, test_data, feature_cols, 
                        n_estimators=100, max_depth=None, 
                        min_samples_leaf=1):
    """
    优化版本（假设每个ID在测试数据中只有1个样本）
    """
    results = []
    for id_val, id_group in train_data.groupby('id'):
        # 直接获取唯一测试样本（无需循环）
        test_sample = test_data[test_data['id'] == id_val]
        if len(test_sample) == 0:
            continue

        # 训练随机森林
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
        X_train = id_group[feature_cols].values
        y_train = id_group['weekly_sales'].values
        rf.fit(X_train, y_train)

        # 计算测试样本的叶节点（形状: (n_estimators,)）
        test_leaves = np.array([tree.apply(test_sample[feature_cols].values)[0] 
                         for tree in rf.estimators_])

        # 预计算训练数据的叶节点（形状: (n_estimators, n_train_samples)）
        leaf_ids = np.vstack([tree.apply(X_train) for tree in rf.estimators_])

        # 计算权重
        weights = np.zeros(len(X_train))
        for t in range(n_estimators):
            same_leaf = (leaf_ids[t] == test_leaves[t]).astype(float)
            same_leaf_sum = same_leaf.sum()
            if same_leaf_sum > 0:
                weights += same_leaf / (n_estimators * same_leaf_sum)

        # 收集邻居
        neighbors = [{
            'weekly_sales': y_train[j],
            'weight': weights[j]
        } for j in np.flatnonzero(weights)]

        results.append({
            'test_id': id_val,
            'test_week': test_sample.iloc[0]['week'],
            'neighbors': neighbors
        })
    return results


# 决策树方法
def dtree_weighted_sales_by_id(train_data, test_data, feature_cols, max_depth=None, min_samples_leaf=5):
    """
    使用决策树计算相同ID产品的样本权重
    
    参数：
    train_data  : 训练数据集（包含id, feature_cols, weekly_sales）
    test_data   : 测试数据集（同结构）
    feature_cols: 用于建模的特征列
    max_depth   : 树的最大深度（None表示不限制）
    min_samples_leaf : 叶节点最小样本数
    
    返回：
    results : 结构同KNN函数，但权重由决策树相似度计算
    """
    results = []
    
    # 按id分组处理
    for id_val, id_group in train_data.groupby('id'):
        test_samples = test_data[test_data['id'] == id_val]
        if len(test_samples) == 0:
            continue

        # 准备数据
        X_train = id_group[feature_cols].values
        y_train = id_group['weekly_sales'].values
        
        # 训练决策树模型
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        tree.fit(X_train, y_train)
        
        # 获取训练样本的叶节点索引（作为相似度分组依据）
        leaf_ids = tree.apply(X_train)
        
        # 处理每个测试样本
        for _, test_row in test_samples.iterrows():
            x_test = test_row[feature_cols].values.reshape(1, -1)
            test_leaf = tree.apply(x_test)[0]
            
            # 计算权重（同叶节点的样本权重均等）
            same_leaf_mask = (leaf_ids == test_leaf)
            n_neighbors = same_leaf_mask.sum()
            
            if n_neighbors == 0:
                # 如果没有相同叶节点的样本，使用最近叶节点的样本
                distances = tree.decision_path(X_train).toarray()[-1]  # 到测试样本的路径距离
                closest_idx = np.argmin(distances)
                same_leaf_mask = (leaf_ids == leaf_ids[closest_idx])
                n_neighbors = same_leaf_mask.sum()
            
            weights = np.where(same_leaf_mask, 1/n_neighbors, 0)
            
            # 构建结果（保持与KNN相同结构）
            neighbors_data = [{
                'weekly_sales': y_train[i],
                'weight': weights[i]
            } for i in np.where(same_leaf_mask)[0]]
            
            results.append({
                'test_id': id_val,
                'test_week': test_row['week'],
                'neighbors': neighbors_data
            })
    
    return results





def saa_weighted_sales_by_id(train_data, test_data):
    """
    基于历史频率的权重计算（结构同KNN版本）
    
    参数：
    train_data : 训练数据集（需包含'id'和'weekly_sales'列）
    test_data : 测试数据集（需包含'id'和'week'列）
    
    返回：
    results : 与knn_weighted_sales_by_id相同结构的列表
        [{
            'test_id': id,
            'test_week': week,
            'neighbors': [{'weekly_sales': y, 'weight': freq}, ...]
        }, ...]
    """
    results = []
    
    # 按id分组处理
    for id_val, id_group in train_data.groupby('id'):

        # 获取该id对应的测试数据
        test_samples = test_data[test_data['id'] == id_val]
        if len(test_samples) == 0:
            continue
            
        # 计算历史销量频率分布
        sales_counts = id_group['weekly_sales'].value_counts(normalize=True) 
        # total_samples = len(id_group)
        
        
        # 构建邻居数据（所有历史出现过的销量值+频率权重）
        neighbors_data = [{
            'weekly_sales': y,
            'weight': count  # 频率作为权重
        } for y, count in sales_counts.items()]
        
        # 为每个测试样本添加相同的历史分布（按week记录）
        for _, test_row in test_samples.iterrows():
            results.append({
                'test_id': id_val,
                'test_week': test_row['week'],
                'neighbors': neighbors_data.copy()  # 避免引用问题
            })
    
    return results

