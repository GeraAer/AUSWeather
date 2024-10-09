import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打印进度
def print_progress(message):
    print(f"正在进行: {message}...")

# 读取数据
print_progress("读取数据")
file_path = 'weatherAUS.csv'
data = pd.read_csv(file_path)

# 数据清洗与预处理
print_progress("数据清洗与预处理")
data = data.dropna(axis=1, thresh=int(0.8 * len(data)))

# 填补缺失值
print_progress("使用K近邻填补缺失值")
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data.select_dtypes(include=[np.number]))
data[data.select_dtypes(include=[np.number]).columns] = data_imputed

for column in data.select_dtypes(include=[object]).columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# 将分类变量转换为数字
print_progress("将分类变量转换为数字")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 从日期中提取年月日特征
print_progress("从日期中提取年月日特征")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data = data.drop('Date', axis=1)

# 使用互信息法选择特征
print_progress("使用互信息法选择特征")
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']
selector = SelectKBest(mutual_info_classif, k=10)
selector.fit(X, y)
mutual_info_features = X.columns[selector.get_support()].tolist()
print("使用互信息法选择的特征：", mutual_info_features)

# 使用随机森林评估特征重要性
print_progress("使用随机森林评估特征重要性")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("随机森林评估的特征重要性：")
print(feature_importances.head(10))

# 选择最终的特征：使用互信息法和随机森林法得出的前10个特征的并集
final_features = list(set(mutual_info_features + feature_importances.head(10).index.tolist()))
print("最终选择的特征：", final_features)

# 添加新特征：过去几天的降雨量和温度变化趋势
# 前几天的降雨量和温度变化趋势
data['Rainfall_3days'] = data['Rainfall'].rolling(window=3).sum().shift(1)
data['MinTemp_3days'] = data['MinTemp'].rolling(window=3).mean().shift(1)
data['MaxTemp_3days'] = data['MaxTemp'].rolling(window=3).mean().shift(1)

# 温度骤变（最大温度和最小温度的差值）
data['TempChange'] = data['MaxTemp'] - data['MinTemp']
data['HumidityDiff'] = data['Humidity3pm'] - data['Humidity9am']


# 气压骤变（前一天和当天气压的差值）
data['PressureChange'] = data['Pressure9am'] - data['Pressure3pm']

# 确保新的特征没有缺失值
data = data.dropna()

# 更新特征集
final_features.extend(['Rainfall_3days', 'MinTemp_3days', 'MaxTemp_3days', 'TempChange', 'PressureChange','HumidityDiff'])


# 定义特征和目标变量
print_progress("定义特征和目标变量")
X = data[final_features]
y = data['RainTomorrow']

# 数据标准化
print_progress("数据标准化")
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 划分训练集、验证集和测试集
print_progress("划分训练集、验证集和测试集")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# 数据平衡：仅对训练集使用Borderline-SMOTE
print_progress("处理数据不平衡问题")
smote = BorderlineSMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 将数据重塑为LSTM输入格式
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 构建LSTM模型
print_progress("构建LSTM模型")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 提前停止回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
print_progress("训练模型")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 在测试集上进行最终评估
print_progress("在测试集上进行最终评估")
y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int)
print("测试集结果：")
print(classification_report(y_test, y_test_pred, target_names=['否', '是'], digits=4))
print("测试集准确率：", accuracy_score(y_test, y_test_pred))

# 混淆矩阵
print("混淆矩阵：")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# 计算漏判概率（假阴性率）
fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
print("漏判概率（假阴性率）：", fn_rate)

# 可视化训练过程
print_progress("可视化训练过程")
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.title('训练过程中的损失变化')
plt.legend()
plt.savefig('training_loss_lstm.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.title('训练过程中的准确率变化')
plt.legend()
plt.savefig('training_accuracy_lstm.png')
plt.show()