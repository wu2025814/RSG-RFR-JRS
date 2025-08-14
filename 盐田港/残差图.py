import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置支持英文的字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 载入数据
df = pd.read_csv('result0.csv')

# 检查每列是否存在NaN值
print(df.isnull().sum())
df.dropna(subset=['depth', 'stumpf', 'log-linear', 'random_forest_prediction', 'random_forest_depth'], inplace=True)

# 计算每个模型的残差误差
df['residual_stumpf'] = df['depth'] - df['stumpf']
df['residual_log-linear'] = df['depth'] - df['log-linear']
df['residual_Random_Forest'] = df['depth'] - df['random_forest_prediction']
df['residual_RSG-RFR'] = df['depth'] - df['random_forest_depth']

print(df.head())

# 定义直方图的bins范围，使其在-10到10之间，并且更多，以减小柱子宽度
bins_range = np.linspace(-10, 10, 100)  # 增加bins数量

# 绘制残差误差的直方图
plt.figure(figsize=(6, 6))

# 绘制直方图，透明度设置为1，并加灰色边框
plt.hist(df['residual_stumpf'], bins=bins_range, alpha=1, color='#90C2E7', edgecolor='grey', linewidth=0.3, label='Stumpf Model')
plt.hist(df['residual_log-linear'], bins=bins_range, alpha=1, color='#DACC3E', edgecolor='grey', linewidth=0.3, label='Log-Linear Model')
plt.hist(df['residual_Random_Forest'], bins=bins_range, alpha=1, color='#307473', edgecolor='grey', linewidth=0.3, label='Random Forest Model')
plt.hist(df['residual_RSG-RFR'], bins=bins_range, alpha=1, color='#DD614A', edgecolor='grey', linewidth=0.3, label='RSG-RFR Model')

# 设置x轴和y轴的限制范围
plt.xlim(-10, 10)
#plt.ylim(0, 30)
# 获取Y轴的最大值来设置刻度
max_pixel_count = max(
    np.histogram(df['residual_stumpf'], bins=bins_range)[0].max(),
    np.histogram(df['residual_log-linear'], bins=bins_range)[0].max(),
    np.histogram(df['residual_Random_Forest'], bins=bins_range)[0].max(),
    np.histogram(df['residual_RSG-RFR'], bins=bins_range)[0].max(),
)

plt.xticks(np.arange(-10, 15, 5))
# 设置Y轴的刻度间隔为50
#plt.yticks(np.arange(0, max_pixel_count + 1, 5))
plt.yticks(np.arange(0, 150, 30))
# 设置x轴和y轴的限制范围
plt.ylim(0, 150)
# 添加标签和标题，设置字体大小为14
plt.xlabel('Residual error(m)', fontsize=14)
plt.ylabel('Pixel count', fontsize=14)
plt.legend(loc='upper left')  # 将图例放在左上角

# 移除网格线
plt.grid(False)



plt.savefig("comparison_models1124.jpg", dpi=600)