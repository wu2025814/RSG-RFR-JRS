import pandas as pd

# 读取CSV文件
file_name = 'result1000.csv'  # CSV文件名
df = pd.read_csv(file_name)  # 假设文件名为result3000.csv

# 设置固定的区间范围
min_depth = 0  # 区间最小值
max_depth = 10  # 区间最大值

# 将区间平均分成四份
num_intervals = 4  # 固定为4个区间
interval_size = (max_depth - min_depth) / num_intervals  # 每个区间的大小

# 定义区间边界
bins = [min_depth + (i * interval_size) for i in range(num_intervals + 1)]

# 使用pandas的cut函数将数据分到各个区间，并统计每个区间的频次
df['depth_group'] = pd.cut(df['depth'], bins=bins, include_lowest=True)  # include_lowest=True确保包含最小值
depth_counts = df['depth_group'].value_counts().sort_index()

# 输出每个区间的点数统计
print("各区间的点数统计：")
print(depth_counts)

# 计算总点数
total_points = df.shape[0]  # 获取总点数
print("\n总点数：", total_points)

# 如果需要统计超出范围的点数
out_of_range_points = df[(df['depth'] < min_depth) | (df['depth'] > max_depth)].shape[0]
print("超出范围的点数：", out_of_range_points)