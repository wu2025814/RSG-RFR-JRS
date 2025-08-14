#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 19:48:17 2025

@author: wuzhenghan
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取两个CSV文件
file_path1 = "yanzheng.csv"
file_path2 = "xunlian.csv"

# 分别读取两个文件
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# 将两个数据框连接起来
df = pd.concat([df1, df2], ignore_index=True)

# 去除包含NaN值的行
df_clean = df.dropna()

# 检查清洗后数据
print(f"原始数据量: {len(df)}，清洗后数据量: {len(df_clean)}")
print("------------------------------------------")

# 划分数据集（假设最后一列Band_10为预测目标）
X = df_clean.iloc[:, :-1]  # 特征：所有列除了最后一列
y = df_clean.iloc[:, -1]   # 目标：最后一列Band_10

# 按8:2比例随机分割
X_train, X_pred, y_train, y_pred = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42
)

# 合并特征和标签保存（可选）
train_set = pd.concat([X_train, y_train], axis=1)
pred_set = pd.concat([X_pred, y_pred], axis=1)

# 保存结果
train_set.to_csv("train_set3000.csv", index=False)
pred_set.to_csv("pred_set1000.csv", index=False)

print("处理完成！")
print(f"训练集样本数: {len(train_set)}")
print(f"预测集样本数: {len(pred_set)}")
print("------------------------------------------")
print("训练集前5行：")
print(train_set.head())