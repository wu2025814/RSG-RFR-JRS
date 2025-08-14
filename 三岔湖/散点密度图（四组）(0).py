import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
import pandas as pd

# 显示中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置英文为新罗马字体
plt.rcParams['font.serif'] = 'Arial'

def get_regression_line(real, pred, data_range=(0, 30)):
    # 拟合（若换MK，自行操作）最小二乘
    def slope(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b
    k, b = slope(real, pred)
    regression_line = []
    for a in range(data_range[0], data_range[1]+1):
        regression_line.append((k * a) + b)
    return regression_line

# 读取数据
data = pd.read_csv('result1.csv')

# 提取数据
real1 = data['depth'].values
pred1 = data['stumpf'].values
real2 = data['depth'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列
pred2 = data['log-linear'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列
real3 = data['depth'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列
pred3 = data['random_forest_prediction'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列
real4 = data['depth'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列
pred4 = data['random_forest_depth'].values  # 这里示例使用相同的数据，实际情况应更改为不同的数据列

# 计算概率密度分布
xy1 = np.vstack([real1, pred1])
z1 = gaussian_kde(xy1)(xy1)

xy2 = np.vstack([real2, pred2])
z2 = gaussian_kde(xy2)(xy2)

xy3 = np.vstack([real3, pred3])
z3 = gaussian_kde(xy3)(xy3)

xy4 = np.vstack([real4, pred4])
z4 = gaussian_kde(xy4)(xy4)

# 设置统一的刻度和限制
xmin, xmax, ymin, ymax = 0, 30, 0, 30

# 创建画布和子图
fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(16, 4))

# 绘制第一组图像
axs[0].scatter(real1, pred1, marker='o', c=z1*100, edgecolors=None, s=10,
                label='LST', cmap='Spectral_r')
axs[0].plot([0, 30], [0, 30], 'k--', lw=1)  # 画的1:1线
regression_line = get_regression_line(real1, pred1, data_range=(0, 30))
axs[0].plot(regression_line, 'r-', lw=1.)      # 预测与实测数据之间的回归线
axs[0].set_xlabel('In-situ depth(m)', fontsize=14, fontname='Arial')
axs[0].set_ylabel('SDB Stumpf(m)', fontsize=14, fontname='Arial')
axs[0].set_title('Stumpf Model', fontsize=14, fontname='Arial')
axs[0].set_xlim(0, 30)
axs[0].set_ylim(0, 30)

# 计算统计指标和添加文本标注
MSE1 = mean_squared_error(real1, pred1)
RMSE1 = np.sqrt(MSE1)
MAE1 = mean_absolute_error(real1, pred1)
MRE1 = np.mean(np.abs((real1 - pred1) / real1))
R2_1 = r2_score(real1, pred1)

axs[0].text(2, 28, f'R$^2$={R2_1:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[0].text(2, 26, f'MAE={MAE1:.2f}m', ha='left', va='top', family='Arial', fontsize=12)
axs[0].text(2, 24, f'MRE={MRE1:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[0].text(2, 22, f'RMSE={RMSE1:.2f}m', ha='left', va='top', family='Arial', fontsize=12)


# 绘制第二组图像
axs[1].scatter(real2, pred2, marker='o', c=z2*100, edgecolors=None, s=10,
                label='LST', cmap='Spectral_r')
axs[1].plot([0, 30], [0, 30], 'k--', lw=1)  # 画的1:1线
regression_line = get_regression_line(real2, pred2, data_range=(0, 30))
axs[1].plot(regression_line, 'r-', lw=1.)      # 预测与实测数据之间的回归线
axs[1].set_xlabel('In-situ depth(m)', fontsize=14, fontname='Arial')
axs[1].set_ylabel('SDB Log-Linear(m)', fontsize=14, fontname='Arial')
axs[1].set_title('Log-Linear Model', fontsize=14, fontname='Arial')
axs[1].set_xlim(0, 30)
axs[1].set_ylim(0, 30)

# 计算统计指标和添加文本标注
MSE2 = mean_squared_error(real2, pred2)
RMSE2 = np.sqrt(MSE2)
MAE2 = mean_absolute_error(real2, pred2)
MRE2 = np.mean(np.abs((real2 - pred2) / real2))
R2_2 = r2_score(real2, pred2)

axs[1].text(2, 28, f'R$^2$={R2_2:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[1].text(2, 26, f'MAE={MAE2:.2f}m', ha='left', va='top', family='Arial', fontsize=12)
axs[1].text(2, 24, f'MRE={MRE2:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[1].text(2, 22, f'RMSE={RMSE2:.2f}m', ha='left', va='top', family='Arial', fontsize=12)


# 绘制第三组图像
axs[2].scatter(real3, pred3, marker='o', c=z3*100, edgecolors=None, s=10,
                label='LST', cmap='Spectral_r')
axs[2].plot([0, 30], [0, 30], 'k--', lw=1)  # 画的1:1线
regression_line = get_regression_line(real3, pred3, data_range=(0, 30))
axs[2].plot(regression_line, 'r-', lw=1.)      # 预测与实测数据之间的回归线
axs[2].set_xlabel('In-situ depth(m)', fontsize=14, fontname='Arial')
axs[2].set_ylabel('SDB Neural Network(m)', fontsize=14, fontname='Arial')
axs[2].set_title('Random_Forest Model', fontsize=14, fontname='Arial')
axs[2].set_xlim(0, 30)
axs[2].set_ylim(0, 30)

# 计算统计指标和添加文本标注
MSE3 = mean_squared_error(real3, pred3)
RMSE3 = np.sqrt(MSE3)
MAE3 = mean_absolute_error(real3, pred3)
MRE3 = np.mean(np.abs((real3 - pred3) / real3))
R2_3 = r2_score(real3, pred3)

axs[2].text(2, 28, f'R$^2$={R2_3:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[2].text(2, 26, f'MAE={MAE3:.2f}m', ha='left', va='top', family='Arial', fontsize=12)
axs[2].text(2, 24, f'MRE={MRE3:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[2].text(2, 22, f'RMSE={RMSE3:.2f}m', ha='left', va='top', family='Arial', fontsize=12)

# 绘制第四组图像
axs[3].scatter(real4, pred4, marker='o', c=z4*100, edgecolors=None, s=10,
                label='LST', cmap='Spectral_r')
axs[3].plot([0, 30], [0, 30], 'k--', lw=1)  # 画的1:1线
regression_line = get_regression_line(real4, pred4, data_range=(0, 30))
axs[3].plot(regression_line, 'r-', lw=1.)      # 预测与实测数据之间的回归线
axs[3].set_xlabel('In-situ depth(m)', fontsize=14, fontname='Arial')
axs[3].set_ylabel('SDB RF_Lon./Lat.(m)', fontsize=14, fontname='Arial')
axs[3].set_title('RSG-RFR Model', fontsize=14, fontname='Arial')
axs[3].set_xlim(0, 30)
axs[3].set_ylim(0, 30)

# 计算统计指标和添加文本标注
MSE4 = mean_squared_error(real4, pred4)
RMSE4 = np.sqrt(MSE4)
MAE4 = mean_absolute_error(real4, pred4)
MRE4 = np.mean(np.abs((real4 - pred4) / real4))
R2_4 = r2_score(real4, pred4)

axs[3].text(2, 28, f'R$^2$={R2_4:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[3].text(2, 26, f'MAE={MAE4:.2f}m', ha='left', va='top', family='Arial', fontsize=12)
axs[3].text(2, 24, f'MRE={MRE4:.2f}', ha='left', va='top', family='Arial', fontsize=12)
axs[3].text(2, 22, f'RMSE={RMSE4:.2f}m', ha='left', va='top', family='Arial', fontsize=12)

# 统一设置XY轴刻度和限制
for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(xmin, xmax+1, 3))  # 例如，每3单位一个刻度
    ax.set_yticks(np.arange(ymin, ymax+1, 3))

# 绘制色标
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=0.5), cmap="Spectral_r"),
                    ax=axs, orientation='vertical', aspect=30, pad=0.02)
cbar.set_label('Point Density(dl)', fontsize=14, fontname='Arial')

# 保存图像为jpg格式，dpi设置为600
plt.savefig("comparison_models.jpg", dpi=600)