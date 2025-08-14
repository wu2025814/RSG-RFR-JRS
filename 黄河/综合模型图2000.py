import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# 显示中文
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置英文为新罗马字体（如果系统中没有Arial，可以选择其他字体）
plt.rcParams['font.serif'] = 'Arial'

# 加载数据（请确保文件路径和文件名正确）
train_data = pd.read_csv("train_set2000.csv", low_memory=False)
test_data = pd.read_csv("pred_set2000.csv", low_memory=False)

# 确保必要的列为数值型，并处理可能的字符串
for band in ['Band_1', 'Band_2', 'Band_3', 'Band_4']:
    train_data[band] = pd.to_numeric(train_data[band], errors='coerce')
    test_data[band] = pd.to_numeric(test_data[band], errors='coerce')

def engineer_features(df):
    # 基本特征
    df['NDWI'] = (df['Band_3'] - df['Band_4']) / (df['Band_3'] + df['Band_4'])
    df['Band_ratio_1_2'] = df['Band_1'] / df['Band_2']
    df['Band_ratio_3_4'] = df['Band_3'] / df['Band_4']
    
    # 地理特征
    df['Lon_Lat_interaction'] = df['Longitude'] * df['Latitude']
    df['Distance_from_origin'] = np.sqrt(df['Longitude']**2 + df['Latitude']**2)
    
    # 波段组合
    df['Band_sum'] = df['Band_1'] + df['Band_2'] + df['Band_3'] + df['Band_4']
    df['Band_mean'] = df['Band_sum'] / 4
    df['Band_std'] = df[['Band_1', 'Band_2', 'Band_3', 'Band_4']].std(axis=1)
    
    # 非线性变换
    for band in ['Band_1', 'Band_2', 'Band_3', 'Band_4']:
        df[band] = pd.to_numeric(df[band], errors='coerce')  # 确保为数值型
        df[f'{band}_log'] = np.log1p(np.abs(df[band]))
        df[f'{band}_squared'] = df[band] ** 2
    
    return df

# 特征工程
X = engineer_features(train_data.drop(['depth'], axis=1))
y = train_data['depth']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
# 适应和转换训练集
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))

# 转换测试集
X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

# 定义随机森林模型
rf_model = RandomForestRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 使用随机搜索进行超参数调优
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, 
                                   n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# 获取最佳模型
best_rf_model = random_search.best_estimator_

# 在测试集上评估模型
y_pred = best_rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 特征重要性分析
importances = permutation_importance(best_rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances.importances_mean})
feature_imp = feature_imp.sort_values('importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
colors = sns.color_palette("hsv", len(feature_imp.head(20)))
sns.barplot(x='importance', y='feature', data=feature_imp.head(20), palette=colors)
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig("新模型2001.jpg", dpi=600)
plt.show()

# 在完整测试集上进行预测
X_test_full = engineer_features(test_data.drop(['depth'], axis=1))
X_test_full_scaled = scaler.transform(X_test_full)
y_pred_full = best_rf_model.predict(X_test_full_scaled)

 # 计算评估指标
y_true_full = test_data['depth']  # 真实深度
mae_full = mean_absolute_error(y_true_full, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y_true_full, y_pred_full))
r2_full = r2_score(y_true_full, y_pred_full)

print("Full Test Set Random Forest Model Performance:")
print(f"MAE: {mae_full:.2f}")
print(f"RMSE: {rmse_full:.2f}")
print(f"R2 Score: {r2_full:.2f}")

# 读取现有的 result2.csv 文件（确保文件存在）
result2_data = pd.read_csv('result2001.csv')

# 确保 result2.csv 的行数与预测结果的数量相同
if len(result2_data) != len(y_pred_full):
    print(y_pred_full.size)
    print(result2_data.size)
    raise ValueError("预测结果的数量与 result2001.csv 中的行数不匹配")

# 将预测结果添加为新列
result2_data['random_forest_depth'] = y_pred_full

# 保存更新后的 result2.csv 文件
result2_data.to_csv('result2001.csv', index=False)
print("Predictions added to result2001.csv as 'random_forest_depth' column")

# 输出最佳参数
print("\nBest Random Forest Parameters:")
for param, value in best_rf_model.get_params().items():
    print(f"{param}: {value}")