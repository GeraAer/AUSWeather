import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 打印进度
def print_progress(message):
    print(f"正在进行: {message}...")

# 读取数据
print_progress("读取数据")
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# 数据清洗与预处理
print_progress("数据清洗与预处理")

# 删除包含过多缺失值的列与日期这一无关列
data = data.dropna(axis=1, thresh=int(0.8 * len(data)))
data = data.drop(data.columns[0], axis=1)

# 填补缺失值
for column in data.select_dtypes(include=[np.number]).columns:
    data[column] = data[column].fillna(data[column].mean())

for column in data.select_dtypes(include=[object]).columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# 将分类变量转换为数字
print_progress("将分类变量转换为数字")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 添加新特征:
print_progress("添加新特征")
data['Rainfall_3days'] = data['Rainfall'].rolling(window=3).sum().shift(1)
data['MinTemp_3days'] = data['MinTemp'].rolling(window=3).mean().shift(1)
data['MaxTemp_3days'] = data['MaxTemp'].rolling(window=3).mean().shift(1)
data['Temp_Difference1'] = data['MaxTemp'] - data['MinTemp']
data['Temp_Difference2'] = data['Temp3pm'] - data['Temp9am']
data['Pressure_Difference'] = data['Pressure3pm'] - data['Pressure9am']
data['Humidity_Difference'] = data['Humidity3pm'] - data['Humidity9am']

data = data.dropna()

# 定义特征和目标变量
print_progress("定义特征和目标变量")
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# 数据标准化
print_progress("数据标准化")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算皮尔逊相关系数
print_progress("计算皮尔逊相关系数")
corr_with_target = data.corr()['RainTomorrow'].abs().sort_values(ascending=False)

# 使用互信息法选择特征
print_progress("使用互信息法选择特征")
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_scores = pd.Series(mi_scores, name="互信息分数", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# 计算皮尔逊相关系数
print_progress("计算皮尔逊相关系数")
corr_matrix = data.corr()
# 绘制皮尔逊相关系数热力图
plt.figure(figsize=(16, 12))
plt.title('皮尔逊相关系数热力图')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.show()

# 打印前十个信息熵值和皮尔逊系数值
print("\n前十个特征的信息熵值：")
print(mi_scores.head(10))

print("\n前十个特征的皮尔逊相关系数值：")
print(corr_with_target.head(10))

# 绘制皮尔逊相关系数条形图
plt.figure(figsize=(12, 6))
corr_with_target.plot(kind='bar', color='skyblue')
plt.title("皮尔逊相关系数与目标变量之间的关系")
plt.ylabel("相关系数绝对值")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 绘制互信息分数条形图
plt.figure(figsize=(12, 6))
mi_scores.plot(kind='bar', color='lightgreen')
plt.title("互信息分数与目标变量之间的关系")
plt.ylabel("互信息分数")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
