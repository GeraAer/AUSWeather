import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

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
# 删除包含过多缺失值的列
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

# 添加新特征：过去几天的降雨量和温度变化趋势
print_progress("添加新特征")
data['Rainfall_3days'] = data['Rainfall'].rolling(window=3).sum().shift(1)
data['MinTemp_3days'] = data['MinTemp'].rolling(window=3).mean().shift(1)
data['MaxTemp_3days'] = data['MaxTemp'].rolling(window=3).mean().shift(1)
data = data.dropna()

# 定义特征和目标变量
print_progress("定义特征和目标变量")
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# 数据标准化
print_progress("数据标准化")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集、验证集和测试集
print_progress("划分训练集、验证集和测试集")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义单个模型
print_progress("定义单个模型")
logistic = LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=10000)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1, max_depth=4)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, alpha=0.01, solver='adam', random_state=42)

# 定义投票分类器
print_progress("定义投票分类器")
voting_clf = VotingClassifier(estimators=[
    ('logistic', logistic),
    ('random_forest', random_forest),
    ('xgboost', xgboost),
    ('mlp', mlp)
], voting='soft')  # 使用软投票

# 训练投票分类器
print_progress("训练投票分类器")
voting_clf.fit(X_train, y_train)

# 在验证集上进行预测并评估
print_progress("在验证集上进行预测并评估")
y_val_pred = voting_clf.predict(X_val)
print("验证集结果：")
print(classification_report(y_val, y_val_pred, target_names=['否', '是'], digits=4))
print("验证集准确率：", accuracy_score(y_val, y_val_pred))

# 在测试集上进行最终评估
print_progress("在测试集上进行最终评估")
y_test_pred = voting_clf.predict(X_test)
print("测试集结果：")
print(classification_report(y_test, y_test_pred, target_names=['否', '是'], digits=4))
print("测试集准确率：", accuracy_score(y_test, y_test_pred))