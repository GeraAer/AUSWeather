import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设你的数据已经包含了年份（Year）、月份（Month）、日平均温度（AverTemp）等特征
# 读取CSV文件，假设第一列是日期，第二列是每日平均温度
data = pd.read_csv('processed_data.csv')

# 季节划分
seasons = {
    '春季': [9, 10, 11],  # 9-11月为春季
    '夏季': [12, 1, 2],   # 12-2月为夏季
    '秋季': [3, 4, 5],    # 3-5月为秋季
    '冬季': [6, 7, 8]     # 6-8月为冬季
}

# 获取季节函数
def get_season(month):
    for season, months in seasons.items():
        if month in months:
            return season

# 计算全年的平均温度
annual_mean_temp = data.groupby('Year')['AverTemp'].mean().reset_index()

# 计算每个季节的平均温度
data['季节'] = data['Month'].apply(get_season)
seasonal_mean_temp = data.groupby(['Year', '季节'])['AverTemp'].mean().reset_index()

# 遍历全年和每个季节进行线性回归分析和绘图
for label, temp_data in [('全年', annual_mean_temp), ('春季', '春季'), ('夏季', '夏季'), ('秋季', '秋季'), ('冬季', '冬季')]:
    if label == '全年':
        years = annual_mean_temp['Year']
        mean_temps = annual_mean_temp['AverTemp']
    else:
        season_data = seasonal_mean_temp[seasonal_mean_temp['季节'] == label]
        years = season_data['Year']
        mean_temps = season_data['AverTemp']

    # 进行线性回归分析
    slope, intercept, r_value, p_value, std_err = linregress(years, mean_temps)

    # 计算预测值
    predicted_temps = slope * years + intercept

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.scatter(years, mean_temps, label=f'{label} 实际平均温度', color='blue')
    plt.plot(years, predicted_temps, label=f'{label} 线性回归', color='red', linestyle='--')
    plt.title(f'{label} 平均温度的线性回归')
    plt.xlabel('年份')
    plt.ylabel('平均温度 (℃)')
    plt.legend()
    plt.grid(True)

    # 输出气候倾向率和R-squared值
    print(f"{label} 平均温度: {mean_temps.mean():.2f} ℃")
    print(f"{label} 气候倾向率 (℃/年): {slope:.3f}")
    print(f"{label} R平方: {r_value ** 2:.3f}")
    print()

    plt.show()