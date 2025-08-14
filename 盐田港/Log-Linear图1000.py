import numpy as np
import pandas as pd

# 读取数据文件，并将无效值设置为 NaN
data = pd.read_csv('train_set3000.csv', na_values=['#NAME?', ''])

# 提取特征和目标变量
XI = pd.to_numeric(data['Band_1'], errors='coerce')
XJ = pd.to_numeric(data['Band_2'], errors='coerce')
XK = pd.to_numeric(data['Band_3'], errors='coerce')
XL = pd.to_numeric(data['Band_4'], errors='coerce')
XM = pd.to_numeric(data['Band_5'], errors='coerce')
XN = pd.to_numeric(data['Band_6'], errors='coerce')
XO = pd.to_numeric(data['Band_7'], errors='coerce')
XP = pd.to_numeric(data['Band_8'], errors='coerce')
dep = pd.to_numeric(data['depth'], errors='coerce')

# 处理NaN值
XI.replace([0, np.inf, -np.inf], np.nan, inplace=True)
XI.dropna(inplace=True)

# 根据 XI 的索引来整理其他特征和目标变量
XJ = XJ[XI.index]
XK = XK[XI.index]
XL = XL[XI.index]
XM = XM[XI.index]
XN = XN[XI.index]
XO = XO[XI.index]
XP = XP[XI.index]
dep = dep[XI.index]

# 准备预测数据并进行最小值减法和对数变换
XI = XI - np.min(XI) + 1e-5
XJ = XJ - np.min(XJ) + 1e-5
XK = XK - np.min(XK) + 1e-5
XL = XL - np.min(XL) + 1e-5
XM = XM - np.min(XM) + 1e-5
XN = XN - np.min(XN) + 1e-5
XO = XO - np.min(XO) + 1e-5
XP = XP - np.min(XP) + 1e-5

# 创建线性回归模型并拟合
X = np.column_stack((XI, XJ, XK, XL, XM, XN, XO, XP))
y = dep.values.reshape(-1, 1)

# 确保创建的常数项数组形状正确
XX = np.column_stack((np.ones(XI.shape[0]), X))
reg = np.linalg.lstsq(XX, y, rcond=None)[0]

# 准备预测数据
predict_data = pd.read_csv('pred_set1000.csv', na_values=['#NAME?', ''])

# 定义数据预处理和对数变换的函数
def preprocess_and_log_transform(column):
    adjusted = pd.to_numeric(column, errors='coerce') - np.min(column) + 1e-5  # 避免对数变换中的零值
    return np.log(adjusted)

# 对 Band_1 到 Band_8 应用处理
b_columns = [f'Band_{i}' for i in range(1, 9)]
b_columns_transformed = [preprocess_and_log_transform(predict_data[b]) for b in b_columns]

# 将处理后的数据堆叠为一个特征矩阵，用于预测
X_transformed = np.column_stack(b_columns_transformed)

# 确保只选择用于训练模型的特征，这里我们假设用了 8 个特征加一个常数项
XX_transformed = np.column_stack((np.ones(X_transformed.shape[0]), X_transformed))

# 使用之前训练好的模型进行深度预测
predicted_depths = XX_transformed.dot(reg)
#predicted_depths = predicted_depths / 1000  # 根据需要调整单位

# 加载现有的 CSV 文件
existing_data = pd.read_csv('result1001.csv')

# 确保预测结果与现有数据长度一致
if len(existing_data) == len(predicted_depths):
    existing_data['log-linear'] = predicted_depths
else:
    print(f"Length mismatch: existing data length = {len(existing_data)}, predictions length = {len(predicted_depths)}")
    
    # 根据需要进行处理，例如截取或填充
    if len(existing_data) > len(predicted_depths):
        existing_data = existing_data.iloc[:len(predicted_depths)]  # 截取 existing_data
        existing_data['log-linear'] = predicted_depths
    else:
        predicted_depths = predicted_depths[:len(existing_data)]  # 截取 predictions
        existing_data['log-linear'] = predicted_depths

# 将更新后的 DataFrame 保存回 CSV 文件中
existing_data.to_csv('result1001.csv', index=False)
