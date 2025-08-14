import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv('result1000.csv')

# 定义深度范围
depth_ranges = [(2, 9), (9, 16), (16, 23), (23, np.inf)]
depth_labels = ['0-2米', '2-4米', '4-6米', '>6米']
models = ['stumpf', 'log-linear', 'random_forest_prediction', 'random_forest_depth']

# 打印每个深度范围的点数
print("各深度范围的点数：")
for depth_range, label in zip(depth_ranges, depth_labels):
    mask = (data['depth'] >= depth_range[0]) & (data['depth'] < depth_range[1])
    print(f"{label}: {mask.sum()}点")

# 打印总的点数
total_points = len(data)
print(f"总的数据点数：{total_points}点")

# 计算并打印每个模型的总体 RMSE 和每个深度范围内的 RMSE
for model in models:
    # 计算总体 RMSE
    total_rmse = np.sqrt(mean_squared_error(data['depth'], data[model]))
    print(f"\n{model} 模型的总体 RMSE: {total_rmse:.2f}")

    print(f"{model} 模型的深度范围 RMSE：", end=' ')
    rmse_list = []
    for depth_range, label in zip(depth_ranges, depth_labels):
        mask = (data['depth'] >= depth_range[0]) & (data['depth'] < depth_range[1])
        rmse = np.sqrt(mean_squared_error(data.loc[mask, 'depth'], data.loc[mask, model]))
        rmse_list.append(f"{label}: {rmse:.2f}")
    # 打印每个深度范围的 RMSE
    print(', '.join(rmse_list))