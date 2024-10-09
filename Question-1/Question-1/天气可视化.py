import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def print_progress(message):
    print(f"正在进行: {message}...")


# 读取数据
print_progress("读取数据")
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)
# 打印进度
# 额外分析：10年内气候变化规律
print_progress("额外分析：10年内气候变化规律")
data_full = pd.read_csv(file_path)
data_full['Date'] = pd.to_datetime(data_full['Date'])
data_full.set_index('Date', inplace=True)

# 只选择数值型列进行resample操作
numeric_data = data_full.select_dtypes(include=[np.number])

# 按月计算平均值
climate_trend = numeric_data.resample('M').mean()

# 可视化气候变化趋势
print_progress("可视化气候变化趋势")
plt.figure(figsize=(12, 6))
plt.plot(climate_trend.index, climate_trend['MinTemp'], label='平均最低温度')
plt.plot(climate_trend.index, climate_trend['MaxTemp'], label='平均最高温度')
plt.xlabel('年份')
plt.ylabel('温度 (°C)')
plt.title('澳大利亚10年内气候变化趋势')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(climate_trend.index, climate_trend['Rainfall'], label='月降雨量')
plt.xlabel('年份')
plt.ylabel('降雨量 (mm)')
plt.title('澳大利亚10年内降雨量变化趋势')
plt.legend()
plt.show()
