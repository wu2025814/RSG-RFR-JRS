import pandas as pd
import numpy as np
from scipy.stats import linregress

# 1. 数据读取与预处理
# 读取数据
image_data = pd.read_csv('train_set2000.csv', low_memory=False)

# 转换 Band_1 和 Band_2 为数值类型
image_data['Band_1'] = pd.to_numeric(image_data['Band_1'], errors='coerce')
image_data['Band_2'] = pd.to_numeric(image_data['Band_2'], errors='coerce')

# 删除包含缺失值的行
image_data = image_data.dropna(subset=['Band_1', 'Band_2', 'depth'])

# 2. 提取数据
r2_data = image_data['Band_1'].values
r3_data = image_data['Band_2'].values
dep_data = image_data['depth'].values

# 3. Stumpf 方法计算
# 计算蓝光和绿光的比值
a = np.log(1000 * np.pi * np.abs(r2_data))  # 蓝光
b = np.log(1000 * np.pi * np.abs(r3_data))  # 绿光
c = a / b
Cbg = c

# 处理比值中的无效值
Cbg[Cbg == 0] = np.nan
valid_indices = ~np.isnan(Cbg)
Cbg = Cbg[valid_indices]
dep_data = dep_data[valid_indices]

# 4. 线性回归拟合
C11 = np.column_stack((np.ones_like(Cbg), Cbg))  # 创建特征矩阵
reg = linregress(C11[:, 1], dep_data)  # 线性回归拟合

# 提取回归系数
a0 = reg.intercept
a1 = reg.slope

# 计算预测值
ZSbg = a0 + a1 * Cbg

# 计算相关系数和决定系数
R = np.corrcoef(dep_data, Cbg)[0, 1]
RR2 = R**2
print('R2 (blue-green) =', RR2)
print('SDB_Stumpf(bg) =', round(a1, 2), ' x pSDB_bg +', round(a0, 2))

# 5. 新数据预测
# 读取新数据文件
new_data = pd.read_csv('pred_set2000.csv', low_memory=False)
new_data['Band_1'] = pd.to_numeric(new_data['Band_1'], errors='coerce')
new_data['Band_2'] = pd.to_numeric(new_data['Band_2'], errors='coerce')
new_data = new_data.dropna(subset=['Band_1', 'Band_2'])

# 计算新数据的 Cbg_new
r2_data_new = new_data['Band_1'].values
r3_data_new = new_data['Band_2'].values
a_new = np.log(1000 * np.pi * np.abs(r2_data_new))  # 蓝光
b_new = np.log(1000 * np.pi * np.abs(r3_data_new))  # 绿光
c_new = a_new / b_new
Cbg_new = c_new

# 处理新数据中的无效值
Cbg_new[Cbg_new == 0] = np.nan
valid_indices_new = ~np.isnan(Cbg_new)
Cbg_new = Cbg_new[valid_indices_new]

# 使用训练好的模型对新数据进行预测
predicted_depth = a0 + a1 * Cbg_new

# 6. 结果保存
# 确保索引对齐并且重置索引
existing_data = pd.read_csv('pred_set2000.csv', low_memory=False)
# 在这里使用 valid_indices_new 选取有效数据
valid_rows = new_data[valid_indices_new].reset_index(drop=True)

# 将预测结果添加为新列
valid_rows['stumpf'] = predicted_depth

print(len(valid_rows))

# 将更新后的数据保存到 CSV 文件中
valid_rows.to_csv('result2000.csv', index=False)

print("预测结果已保存到 result2000.csv")
