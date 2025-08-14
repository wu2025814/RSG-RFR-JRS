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

# 读取数据  
data = pd.read_csv("xunlian.csv", na_values=['#NAME?'])  

# 打乱数据  
data_shuffled = data.sample(frac=1, random_state=42)  

# 分离特征和目标  
X = data_shuffled.drop(columns=['Longitude', 'Latitude', 'depth'])  # 正确地去掉了'Longitude', 'Latitude'和'depth'  
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
print(f"Mean Absolute Error: {mae:.4f}")  

# 保存模型  
joblib.dump(model, "random_forest_model.joblib")  

# 读取需要预测的新数据  
new_data = pd.read_csv("yanzheng.csv", na_values=['#NAME?'])  

# 删除可能存在的无效数据行
new_data.dropna(inplace=True)

# 分离特征并进行标准化  
X_new = new_data.drop(columns=['Longitude', 'Latitude', 'depth'], errors='ignore')  # 使用 errors='ignore' 以防列不存在  
X_new_scaled = scaler.transform(X_new)  

# 加载模型进行预测  
loaded_model = joblib.load("random_forest_model.joblib")  
predictions = loaded_model.predict(X_new_scaled)  

# 加载现有的CSV文件  
existing_data = pd.read_csv('result0001.csv')  

# 确保预测结果与现有数据长度一致
if len(existing_data) == len(predictions):
    existing_data['random_forest_prediction'] = predictions
else:
    print(f"Length mismatch: existing data length = {len(existing_data)}, predictions length = {len(predictions)}")
    
    # 根据需要进行处理，例如截取或填充
    if len(existing_data) > len(predictions):
        existing_data = existing_data.iloc[:len(predictions)]  # 截取 existing_data
        existing_data['random_forest_prediction'] = predictions
    else:
        predictions = predictions[:len(existing_data)]  # 截取 predictions
        existing_data['random_forest_prediction'] = predictions

# 将更新后的DataFrame保存回CSV文件中  
existing_data.to_csv('result0001.csv', index=False)
