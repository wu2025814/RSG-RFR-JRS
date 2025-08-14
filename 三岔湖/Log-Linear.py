import numpy as np
import pandas as pd

# 读取数据文件
data = pd.read_csv('testdata.csv', dtype=float, on_bad_lines='warn', na_values=['#NAME?', 'NaN', ''])

# 提取特征 and 目标变量
XI = data['Band_1']
XJ = data['Band_2']
XK = data['Band_3']
XL = data['Band_4']
XM = data['Band_5']
XN = data['Band_6']
XO = data['Band_7']
XP = data['Band_8']
dep = data['depth']

# 处理 NaN 值
XI.replace([0, np.inf, -np.inf], np.nan, inplace=True)
XI.dropna(inplace=True)

# 使用有效索引来整齐其他波段和深度数据
valid_index = XI.index
XJ = XJ[valid_index]
XK = XK[valid_index]
XL = XL[valid_index]
XM = XM[valid_index]
XN = XN[valid_index]
XO = XO[valid_index]
XP = XP[valid_index]
dep = dep[valid_index]

# 准备预测数据并进行最小值减法
def preprocess_column(column):
    adjusted = column - np.min(column) + 1e-5  # 避免对数变换中的零值
    return np.log(adjusted)

# 应用处理
XI_transformed = preprocess_column(XI)
XJ_transformed = preprocess_column(XJ)
XK_transformed = preprocess_column(XK)
XL_transformed = preprocess_column(XL)
XM_transformed = preprocess_column(XM)
XN_transformed = preprocess_column(XN)
XO_transformed = preprocess_column(XO)
XP_transformed = preprocess_column(XP)

# 创建线性回归模型并拟合
X = np.column_stack((XI_transformed, XJ_transformed, XK_transformed, XL_transformed,
                     XM_transformed, XN_transformed, XO_transformed, XP_transformed))
y = dep.values.reshape(-1, 1)
XX = np.column_stack((np.ones(X.shape[0]), X))  # 创建常数项数组
reg = np.linalg.lstsq(XX, y, rcond=None)[0]

# 准备预测数据
predict_data = pd.read_csv('image1.csv', dtype=float, on_bad_lines='warn', na_values=['#NAME?', 'NaN', ''])

# 处理预测数据中的波段
b_columns = [f'Band_{i}' for i in range(1, 9)]  # Band_1 到 Band_8
b_columns_transformed = []

for b in b_columns:
    if b in predict_data.columns:  # 确保列存在
        transformed = preprocess_column(predict_data[b])
        b_columns_transformed.append(transformed)

# 将处理后的数据堆叠为一个特征矩阵，用于预测
X_transformed = np.column_stack(b_columns_transformed)

# 确保只选择用于训练模型的特征，这里假设用了8个特征加一个常数项
XX_transformed = np.column_stack((np.ones(X_transformed.shape[0]), X_transformed))

# 使用之前训练好的模型进行深度预测
predicted_depths = XX_transformed.dot(reg)
#predicted_depths /= 1000  # 根据需要调整单位

# 加载现有的CSV文件并确保长度匹配
existing_data = pd.read_csv('result.csv', dtype=float, on_bad_lines='warn', na_values=['#NAME?', 'NaN', ''])

# 检查与调整长度
num_existing = len(existing_data)
num_predicted = len(predicted_depths)

if num_existing == num_predicted:
    existing_data['log-linear'] = predicted_depths
else:
    print(f"Length mismatch: existing data length = {num_existing}, predicted depths length = {num_predicted}")
    
    # 处理长度不匹配的方法
    # 用NaN填充较短的列
    if num_existing < num_predicted:
        # 用 NaN 填充 existing_data
        existing_data = pd.concat([existing_data, pd.DataFrame({"log-linear": [np.nan] * (num_predicted - num_existing)})], ignore_index=True)
        existing_data['log-linear'] = predicted_depths
    else:
        # 截取 predicted_depths 以匹配 existing_data
        existing_data['log-linear'] = predicted_depths[:num_existing]

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result.csv', index=False)

# 可选：画图部分（若需要的话，可以取消注释）
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

# # 加载经纬度和水深数据
# lat = np.loadtxt('lat.txt')
# lon = np.loadtxt('lon.txt')
# ZLm = np.loadtxt('ZLm.txt')
#
# # 创建Miller投影的地图
# plt.figure(figsize=(10, 8))
# m = Basemap(projection='mill', lon_0=0)
#
# # 绘制地图和水深数据
# x, y = m(lon, lat)
# pcm = m.pcolormesh(x, y, ZLm, cmap='jet', shading='flat')
# m.colorbar(pcm, location='right', label='Water Depth (m)')
#
# # 添加地图特征
# m.drawcoastlines()
# m.drawcountries()
# m.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])
#
# # 添加标题
# plt.title('Lyzenga Model Predicted Water Depth')
#
# # 显示图形
# 
