import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 打印进度
def print_progress(message):
    print(f"正在进行: {message}...")


# 使用移动均值和众数填充缺失
def fill_data(data, window_size=5):
    # 处理数值类型的列
    for column in data.select_dtypes(include=[np.number]).columns:
        # 使用滑动窗口均值填充数值类型的列
        data[column] = data[column].fillna(data[column].rolling(window=window_size, min_periods=1).mean())
    for column in data.select_dtypes(include=[object]).columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    return data


# 读取数据
print_progress("读取数据")
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# 删去数据缺失过多特征
print_progress("数据清洗与预处理")
data = data.dropna(axis=1, thresh=int(0.8 * len(data)))

# 删去数据缺失过多的样本
data = data.dropna(axis=0, thresh=data.shape[1] - 2)

# 异常值处理
print_progress("处理异常值")
numeric_columns = data.select_dtypes(include=[np.number]).columns
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1


def remove_outliers(df, columns, Q1, Q3, IQR):
    for column in columns:
        df = df[~((df[column] < (Q1[column] - 1.5 * IQR[column])) | (df[column] > (Q3[column] + 1.5 * IQR[column])))]
    return df


data = remove_outliers(data, numeric_columns, Q1, Q3, IQR)

# 填补缺失值 移动均值法和众数法
data = fill_data(data, window_size=5)

# 从日期中提取年月日特征
print_progress("从日期中提取年月日特征")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data = data.drop('Date', axis=1)

# 加入新特征
data['Temp_Difference1'] = data['MaxTemp'] - data['MinTemp']
data['Temp_Difference2'] = data['Temp3pm'] - data['Temp9am']
data['Pressure_Difference'] = data['Pressure3pm'] - data['Pressure9am']
data['Humidity_Difference'] = data['Humidity3pm'] - data['Humidity9am']
data['AverTemp'] = (data['MinTemp'] + data['MaxTemp'])/2
data = data.dropna()

'''
# 生成数据文件
data.to_csv('processed_data.csv', index=False)
'''