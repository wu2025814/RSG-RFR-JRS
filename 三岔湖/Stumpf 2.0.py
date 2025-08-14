import numpy as np
import pandas as pd
from scipy.stats import linregress

# 从 testdata2.csv 文件中读取数据
image_data = pd.read_csv('testdata2.csv', low_memory=False)

# 确保 'Band_1' 和 'Band_2' 列是数值类型
image_data['Band_1'] = pd.to_numeric(image_data['Band_1'], errors='coerce')
image_data['Band_2'] = pd.to_numeric(image_data['Band_2'], errors='coerce')

# 删除缺失值
image_data = image_data.dropna(subset=['Band_1', 'Band_2', 'depth'])

# 提取 r2 和 r3 列的数据
r2_data = image_data['Band_1'].values
r3_data = image_data['Band_2'].values

# 提取 depth 列的数据作为目标值
dep_data = image_data['depth'].values

# Stumpf method for SDB (Blue/Green)
a = np.log(1000 * np.pi * r2_data)  # blue
b = np.log(1000 * np.pi * r3_data)  # green
c = a / b
Cbg = c

# 清除Cbg中的0值
Cbg[Cbg == 0] = np.nan

# 去除Cbg中的NaN值
valid_indices = ~np.isnan(Cbg)
Cbg = Cbg[valid_indices]
dep_data = dep_data[valid_indices]

# 构建特征矩阵
C11 = np.column_stack((np.ones_like(Cbg), Cbg))

# 线性回归拟合
reg = linregress(C11[:, 1], dep_data)

# 提取回归系数
a0 = reg.intercept
a1 = reg.slope

# 计算预测值
ZSbg = a0 + a1 * Cbg

# 计算相关性
R = np.corrcoef(dep_data, Cbg)[0, 1]
RR2 = R**2

print('R2 (blue-green) =', RR2)
print('SDB_Stumpf(bg) =', round(a1, 2), ' x pSDB_bg +', round(a0, 2))

# 读取新数据
new_data = pd.read_csv('image3.csv', low_memory=False)

# 确保 'Band_1' 和 'Band_2' 列是数值类型
new_data['Band_1'] = pd.to_numeric(new_data['Band_1'], errors='coerce')
new_data['Band_2'] = pd.to_numeric(new_data['Band_2'], errors='coerce')

# 删除缺失值
new_data = new_data.dropna(subset=['Band_1', 'Band_2'])

# 提取 r2 和 r3 列的数据
r2_data_new = new_data['Band_1'].values
r3_data_new = new_data['Band_2'].values

# Stumpf method for SDB (Blue/Green)
a_new = np.log(1000 * np.pi * r2_data_new)  # blue
b_new = np.log(1000 * np.pi * r3_data_new)  # green
c_new = a_new / b_new
Cbg_new = c_new

# 去除Cbg_new中的0值
Cbg_new[Cbg_new == 0] = np.nan
valid_indices_new = ~np.isnan(Cbg_new)
Cbg_new = Cbg_new[valid_indices_new]

# 使用之前训练好的模型参数进行预测
predicted_depth = a0 + a1 * Cbg_new

# 加载现有的CSV文件
existing_data = pd.read_csv('image3.csv', low_memory=False)

# 确保索引对齐
existing_data = existing_data.iloc[valid_indices_new].reset_index(drop=True)

# 将预测出来的浅水深度数据添加为新列
existing_data['stumpf'] = predicted_depth

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result1.csv', index=False)