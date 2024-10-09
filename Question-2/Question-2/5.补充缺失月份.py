import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 打印进度
def print_progress(message):
    print(f"正在进行: {message}...")


# 读取数据
print_progress("读取数据")
file_path = 'processed_data_with_clusters.csv'
data = pd.read_csv(file_path)

# 创建日期
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

# 从日期中提取年月日特征
print_progress("从日期中提取年月日特征")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# 确保包含所有的月份
print_progress("检查并填补缺失的月份数据")
full_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='MS')
full_months = {(date.year, date.month) for date in full_range}

existing_months = {(row['Year'], row['Month']) for _, row in data.iterrows()}

missing_months = full_months - existing_months
print(f"缺失的所有月份: {sorted(missing_months)}")

for year, month in sorted(missing_months):
    print(f"缺失的月份: {year}-{month}")
    prev_year_data = data[(data['Year'] == year - 1) & (data['Month'] == month)]
    next_year_data = data[(data['Year'] == year + 1) & (data['Month'] == month)]

    if not prev_year_data.empty and not next_year_data.empty:
        prev_year_avg = prev_year_data.select_dtypes(include=[np.number]).mean()
        next_year_avg = next_year_data.select_dtypes(include=[np.number]).mean()
        avg_values = (prev_year_avg + next_year_avg) / 2
        new_row = avg_values.to_dict()
        new_row['Year'] = year
        new_row['Month'] = month
        new_row['Day'] = 1  # 设置为月份的第一天
        new_row['Date'] = pd.Timestamp(year=year, month=month, day=1)
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

# 对聚类特征进行四舍五入处理
data['Cluster'] = data['Cluster'].round()

data = data.sort_values(by='Date').reset_index(drop=True)


# 生成新的处理后文件
data.to_csv('processed_data_with_month-filled.csv', index=False)

print("数据预处理完成，已生成新文件：processed_data_with_month-filled.csv")
