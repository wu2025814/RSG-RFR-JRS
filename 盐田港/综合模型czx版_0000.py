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

# 设置英文为新罗马字体
plt.rcParams['font.serif'] = 'Arial'

# 加载数据，确保指定 dtype 或使用 low_memory=False
train_data = pd.read_csv("train_set.csv", low_memory=False)
test_data = pd.read_csv("pred_set.csv", low_memory=False)

# 确保必要的列为数值型，并处理可能的字符串
for band in ['Band_1', 'Band_2', 'Band_3', 'Band_4']:
    train_data[band] = pd.to_numeric(train_data[band], errors='coerce')
    test_data[band] = pd.to_numeric(test_data[band], errors='coerce')

# 特征工程函数
def engineer_features(df):
    # 基本特征
    # 计算 NDWI 并处理可能存在的分母为零的情况
    df['NDWI'] = np.where((df['Band_3'] + df['Band_4']) != 0, 
                         (df['Band_3'] - df['Band_4']) / (df['Band_3'] + df['Band_4']),
                         np.nan)  # 处理为 NaN

    for band in ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_8', 'Band_10']:
        df[band] = pd.to_numeric(df[band], errors='coerce')  # 确保为数值型
        
        # 对绝对值进行对数计算
        df[f'{band}_log'] = np.log1p(np.abs(df[band]))  # 计算取绝对值后的对数
        df[f'{band}_squared'] = df[band] ** 2  # 计算平方
        
    return df

# 特征工程
X = engineer_features(train_data.drop(['depth'], axis=1))
y = train_data['depth']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 实例化标准化器
scaler = StandardScaler()

# 标准化特征，确保再一次处理数据清洁
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

# 可视化特征重要性，使用不同的颜色
plt.figure(figsize=(12, 8))
colors = sns.color_palette("hsv", len(feature_imp.head(20)))
sns.barplot(x='importance', y='feature', data=feature_imp.head(20), palette=colors)
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig("新模型0000.jpg", dpi=600)
plt.show()

# 在完整测试集上进行预测
X_test_full = engineer_features(test_data.drop(['depth'], axis=1))
X_test_full_scaled = scaler.transform(X_test_full.select_dtypes(include=[np.number]))
y_pred_full = best_rf_model.predict(X_test_full_scaled)

# 读取现有的 result1.csv 文件
result2_data = pd.read_csv('result.csv', low_memory=False)

# 确保 result2.csv 的行数与预测结果的数量相同
if len(result2_data) != len(y_pred_full):
    print(y_pred_full.size)
    raise ValueError("预测结果的数量与 result.csv 中的行数不匹配")

# 将预测结果添加为新列
result2_data['random_forest_depth'] = y_pred_full

# 保存更新后的 result1.csv 文件
result2_data.to_csv('result.csv', index=False)
print("Predictions added to result.csv as 'random_forest_depth' column")

# 输出最佳参数
print("\nBest Random Forest Parameters:")
for param, value in best_rf_model.get_params().items():
    print(f"{param}: {value}")
