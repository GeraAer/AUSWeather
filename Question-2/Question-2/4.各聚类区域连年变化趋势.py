import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 假设你的数据已经包含了年份（Year）、月份（Month）、日平均温度（AverTemp）等特征，还有聚类特征（Cluster）
# 读取CSV文件，假设第一列是日期，第二列是每日平均温度
data = pd.read_csv('processed_data_with_clusters.csv')

# 季节划分
seasons = {
    'Spring': [9, 10, 11],  # 9-11月为春季
    'Summer': [12, 1, 2],   # 12-2月为夏季
    'Autumn': [3, 4, 5],    # 3-5月为秋季
    'Winter': [6, 7, 8]     # 6-8月为冬季
}

# 计算每个季节的平均温度
def get_season(month):
    for season, months in seasons.items():
        if month in months:
            return season

# 创建一个图形窗口
fig, axs = plt.subplots(4, 1, figsize=(12, 18))

# 存储每个聚类的回归结果
regression_results = {}

# 遍历每个聚类进行线性回归分析和绘图
for cluster in range(4):  # 假设有四个聚类
    cluster_data = data[data['Cluster'] == cluster]

    # 计算该聚类的全年平均温度
    annual_mean_temp = cluster_data.groupby('Year')['AverTemp'].mean().reset_index()
    slope_annual, intercept_annual, r_value_annual, p_value_annual, std_err_annual = \
        linregress(annual_mean_temp['Year'], annual_mean_temp['AverTemp'])

    # 计算该聚类的每个季节平均温度
    seasonal_mean_temp = cluster_data.groupby(['Year', 'Month']).agg({'AverTemp': 'mean'}).reset_index()
    seasonal_mean_temp['Season'] = seasonal_mean_temp['Month'].apply(get_season)
    seasonal_mean_temp = seasonal_mean_temp.groupby(['Year', 'Season'])['AverTemp'].mean().reset_index()

    regression_results[cluster] = {
        'annual': {
            'slope': slope_annual,
            'r_squared': r_value_annual ** 2
        },
        'seasonal': {}
    }

    # 遍历每个季节进行线性回归分析和绘图
    for i, (label, temp_data) in enumerate([('Annual', annual_mean_temp), ('Spring', 'Spring'), ('Summer', 'Summer'),
                                            ('Autumn', 'Autumn'), ('Winter', 'Winter')]):
        if label == 'Annual':
            years = annual_mean_temp['Year']
            mean_temps = annual_mean_temp['AverTemp']
        else:
            season_data = seasonal_mean_temp[seasonal_mean_temp['Season'] == label]
            years = season_data['Year']
            mean_temps = season_data['AverTemp']

        # 进行线性回归分析
        slope, intercept, r_value, p_value, std_err = linregress(years, mean_temps)
        regression_results[cluster]['seasonal'][label] = {
            'slope': slope,
            'r_squared': r_value ** 2
        }

        # 计算预测值
        predicted_temps = slope * years + intercept

        # 绘制图像
        axs[cluster].scatter(years, mean_temps, label=f'Cluster {cluster + 1} - {label} Actual Mean Temp', alpha=0.8)
        axs[cluster].plot(years, predicted_temps, label=f'Cluster {cluster + 1} - {label} Linear Regression',
                          color='red', linestyle='--')
        axs[cluster].set_title(f'Cluster {cluster + 1} - Linear Regression of {label} Mean Temperature')
        axs[cluster].set_xlabel('Year')
        axs[cluster].set_ylabel('Mean Temperature (℃)')
        axs[cluster].legend()
        axs[cluster].grid(True)

        # 输出气候倾向率和R-squared值
        print(f"Cluster {cluster + 1} - {label} Mean Temperature: {mean_temps.mean():.2f} ℃")
        print(f"Cluster {cluster + 1} - {label} Climate Trend Rate (℃/year): {slope:.3f}")
        print(f"Cluster {cluster + 1} - {label} R-squared: {r_value ** 2:.3f}")
        print()

plt.tight_layout()
plt.show()
