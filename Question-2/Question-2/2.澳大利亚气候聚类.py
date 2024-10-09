import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 打印进度
def print_progress(message):
    print(f"正在进行: {message}...")


# 读取数据
print_progress("读取数据")
file_path = 'processed_data.csv'
data = pd.read_csv(file_path)

features = ['AverTemp', 'Rainfall', 'Humidity3pm', 'Pressure9am']

# 按地区分组
grouped = data.groupby('Location')


# 定义一个函数来进行趋势拟合并返回拟合结果
def fit_trend(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    trend = np.polyval(coefficients, x)
    return trend


# 存储所有地区的拟合结果
fitted_trends = {}

# 对每个地区的数据进行拟合
for location, group in grouped:
    x = np.arange(len(group))  # 假设 x 是时间序列或其他连续变量
    trends = {}

    for feature in features:
        y = group[feature].values

        # 进行趋势拟合，这里选择二次多项式拟合
        degree = 3
        trend = fit_trend(x, y, degree)

        # 将拟合的趋势线添加到字典中
        trends[f'Trend_{feature}'] = trend

    # 将拟合的趋势线添加到字典中
    fitted_trends[location] = trends


# 创建 DataFrame 存储拟合趋势线和地区信息
trends_df = pd.DataFrame({
    'Location': list(fitted_trends.keys()),
    'Trend_Temp': [np.mean(data[f'Trend_AverTemp']) for data in fitted_trends.values()],  # 将拟合趋势线转换为标量值
    'Trend_Rainfall': [np.mean(data[f'Trend_Rainfall']) for data in fitted_trends.values()],
    'Trend_Humidity3pm': [np.mean(data[f'Trend_Humidity3pm']) for data in fitted_trends.values()],
    'Trend_Pressure9am': [np.mean(data[f'Trend_Pressure9am']) for data in fitted_trends.values()],
})

# 提取拟合趋势线作为聚类特征
X = trends_df[['Trend_Temp', 'Trend_Rainfall','Trend_Humidity3pm',
               'Trend_Pressure9am']].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 聚类数量
k = 4  # 假设聚为 4 类

# K-means 聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

# 将聚类结果添加到数据中
trends_df['Cluster'] = clusters

# 将聚类结果合并回原始数据
data_with_clusters = pd.merge(data, trends_df[['Location', 'Cluster']], on='Location', how='left')

# 输出每个聚类的地区列表
clusters_dict = {}
for cluster in range(k):
    cluster_locations = trends_df[trends_df['Cluster'] == cluster]['Location'].tolist()
    clusters_dict[f'Cluster {cluster + 1}'] = cluster_locations

# 打印每个聚类的地区列表
for cluster, locations in clusters_dict.items():
    print(f"\n聚类 {cluster}:")
    for loc in locations:
        print(loc)

'''
# 输出含聚类数据文件
output_file = 'processed_data_with_clusters.csv'
data_with_clusters.to_csv(output_file, index=False)
'''
