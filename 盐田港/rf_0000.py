# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 01:30:47 2024

@author: wuzho
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import os

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def clean_data(df):
    # 逐列检查数据，将无法转换为浮点数的值替换为 NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 删除包含 NaN 的行
    df.dropna(inplace=True)
    return df

# 读取数据
data = pd.read_csv("train_set.csv")

# 清理数据
data = clean_data(data)

# 打乱数据
data_shuffled = data.sample(frac=1, random_state=42)

# 分离特征和目标
X = data_shuffled.drop(columns=['Longitude', 'Latitude', 'depth'])
y = data_shuffled['depth']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# 构建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算平均绝对误差
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 保存模型
joblib.dump(model, "random_forest_model.joblib")

# 读取需要预测的新数据
new_data = pd.read_csv("pred_set.csv", low_memory=False)

# 清理新数据
new_data = clean_data(new_data)

# 检查新数据是否包含所需的列
if not all(col in new_data.columns for col in ['Longitude', 'Latitude', 'depth']):
    raise ValueError("The input file must contain 'Longitude', 'Latitude', and 'depth' columns.")

# 分离特征并进行标准化
X_new = new_data.drop(columns=['Longitude', 'Latitude', 'depth'])
X_new_scaled = scaler.transform(X_new)

# 加载模型进行预测
loaded_model = joblib.load("random_forest_model.joblib")
predictions = loaded_model.predict(X_new_scaled)

# 检查 result1.csv 是否存在
if os.path.exists('result.csv'):
    existing_data = pd.read_csv('result.csv')
else:
    # 如果文件不存在，创建一个空的 DataFrame
    existing_data = pd.DataFrame()

# 确保 existing_data 有足够的行来匹配 predictions
if len(existing_data) < len(predictions):
    # 创建一个新的 DataFrame 来填充额外的行
    additional_rows = len(predictions) - len(existing_data)
    additional_data = pd.DataFrame(index=range(additional_rows))
    # 合并现有的 DataFrame 和新的 DataFrame
    existing_data = pd.concat([existing_data, additional_data], ignore_index=True)
elif len(existing_data) > len(predictions):
    # 如果 existing_data 的行数多于 predictions，则截断 existing_data
    existing_data = existing_data.iloc[:len(predictions)]

# 将预测出来的浅水深度数据添加为新列
existing_data['random_forest_prediction'] = predictions

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result.csv', index=False)

print("预测结果已保存到 result.csv 文件中。")