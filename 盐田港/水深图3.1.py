import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
# 设置全局字体大小
plt.rcParams['font.size'] = 14

# 定义自定义颜色映射
color = [(0, 0, 1), (0.3, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]  # 蓝色到红色
colors = color[::-1]
cmap = LinearSegmentedColormap.from_list('CustomMap', colors, N=256)

# 定义绘制地图的函数
def draw_map(ax, data, column, title):
    LONLIMS = [104.22, 104.30]
    LATLIMS = [30.22, 30.34]
    
    # 创建地图投影
    ax.set_extent([LONLIMS[0], LONLIMS[1], LATLIMS[0], LATLIMS[1]], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, weight='bold')
    
    # 添加地理特征
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='white')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='lightblue')
    
    # 绘制经纬度刻度线
    parallels = np.linspace(LATLIMS[0], LATLIMS[1], 5)
    meridians = np.linspace(LONLIMS[0], LONLIMS[1], 5)
    
    # 只保留左边和下边的经纬度标签
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False  # 不显示顶部标签
    gl.right_labels = False  # 不显示右侧标签
    
    # 处理并绘制数据
    lon = data['Longitude']
    lat = data['Latitude']
    depth = data[column]
    ax.scatter(lon, lat, c=depth, cmap=cmap, s=1, transform=ccrs.PlateCarree(), vmin=10, vmax=20)  # 将点的大小调整为1

# 读取预测水深数据
data = pd.read_csv('result-.csv')
# 为了使点更密集，可以减少采样间隔
data = data.iloc[::50, :]  # 采样间隔从100调整为50

# 创建画布并设置GridSpec
fig = plt.figure(figsize=(12, 3))  # 将整体图片大小缩小
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])

# 创建四个子图用于绘制地图
axes = [fig.add_subplot(gs[i], projection=ccrs.PlateCarree()) for i in range(4)]

# 绘制每个模型的地图
for i, (col, title) in enumerate([
    ('stumpf', 'Stumpf Model'),
    ('log-linear', 'Log Linear Model'),
    ('random_forest_prediction', 'Random Forest Model'),
    ('random_forest_depth', 'RSG-RFR Model')
]):
    draw_map(axes[i], data, col, title)

# 添加共用的色标
cbar_ax = fig.add_subplot(gs[4])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=10, vmax=20))
sm._A = []
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Depth (m)')

# 设置色标的刻度为10到20米，每5米一个刻度
cbar.set_ticks(np.arange(10, 21, 5))
cbar.ax.set_yticklabels(np.arange(10, 21, 5), fontsize=12)

# 调整布局以适应子图和色标的大小
plt.tight_layout()

plt.show()