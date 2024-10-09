import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体以确保图表中可以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 计算年均温度的函数
def calculate_annual_avg_temperature(daily_temp_data):
    year = pd.DatetimeIndex(daily_temp_data.index).year
    annual_avg_temp = daily_temp_data.groupby(year).mean()
    return annual_avg_temp.index, annual_avg_temp.values


# Mann-Kendall 突变检验的函数（记录所有可能的突变点）
def mann_kendall_seasonal_test(data, alpha=0.05):
    n = len(data)
    # Step 1: 计算 S_k 值
    S = np.zeros(n)
    for k in range(1, n):
        for i in range(k):
            if data[i] < data[k]:
                S[k] += 1
            elif data[i] > data[k]:
                S[k] -= 1

    # Step 2: 计算 U_Fk 和 U_Bk 统计量
    U_F = np.zeros(n)
    U_B = np.zeros(n)
    for k in range(2, n):
        U_F[k] = S[k] - np.mean(S[1:k + 1])
        U_B[k] = S[k] - np.mean(S[1:k + 1])

    # Step 3: 计算临界值
    '''
    z = 1.96  # 对于 alpha = 0.05
    '''
    z = 4  # 降低阈值
    # Step 4: 找到所有超过临界值的突变点
    change_points = []
    for k in range(2, n):
        if abs(U_F[k]) > z:
            change_points.append(k)

    return U_F, change_points


# 获取季节的函数
def get_season(month):
    if month in [9, 10, 11]:
        return '春季'
    elif month in [12, 1, 2]:
        return '夏季'
    elif month in [3, 4, 5]:
        return '秋季'
    elif month in [6, 7, 8]:
        return '冬季'
    else:
        return None


# 读取 CSV 文件中的数据
file_path = 'processed_data_with_month-filled.csv'  # 替换为您的包含季节和聚类信息的 CSV 文件路径
data = pd.read_csv(file_path)

# 创建日期时间索引
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.set_index('Date', inplace=True)  # 将日期列设置为索引

# 获取四季的月份范围
seasons = {
    '春季': [9, 10, 11],
    '夏季': [12, 1, 2],
    '秋季': [3, 4, 5],
    '冬季': [6, 7, 8]
}

# 获取聚类的列表
clusters = data['Cluster'].unique()

# 执行全年年均温度的 Mann-Kendall 突变检验并可视化
annual_data = data[['AverTemp']]
year_annual, annual_avg_temp = calculate_annual_avg_temperature(annual_data)
U_F_annual, change_points_annual = mann_kendall_seasonal_test(annual_avg_temp)

# 可视化全年年均温度的结果
plt.figure(figsize=(10, 6))
plt.plot(year_annual, annual_avg_temp, marker='o', linestyle='-', color='b', label='全年年均温度数据')  # 绘制全年年均温度数据
for cp in change_points_annual:
    trend = "上升" if U_F_annual[cp] > 0 else "下降"
    plt.plot(year_annual[cp], annual_avg_temp[cp], marker='o', markersize=10, color='r',
             label=f'全年年均温度{trend}突变点')  # 标记全年检测到的突变点及趋势
plt.title('全年年均温度数据的Mann-Kendall突变检验结果')  # 图表标题
plt.xlabel('年份')  # x轴标签
plt.ylabel('年均温度（摄氏度）')  # y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线
plt.show()

if change_points_annual:
    print(f"全年检测到的年均温度突变点的年份为 {year_annual[change_points_annual]}，对应的趋势为:")
    for cp in change_points_annual:
        trend = "上升" if U_F_annual[cp] > 0 else "下降"
        print(f"年份 {year_annual[cp]}，趋势为 {trend}")
else:
    print("全年未检测到显著的年均温度突变点")

# 执行四季年均温度的 Mann-Kendall 突变检验并可视化
for season_name, month_range in seasons.items():
    seasonal_data = data[data.index.month.isin(month_range)][['AverTemp']]
    year_seasonal, seasonal_avg_temp = calculate_annual_avg_temperature(seasonal_data)
    U_F_seasonal, change_points_seasonal = mann_kendall_seasonal_test(seasonal_avg_temp)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(year_seasonal, seasonal_avg_temp, marker='o', linestyle='-', color='b',
             label=f'{season_name}年均温度数据')  # 绘制四季年均温度数据
    for cp in change_points_seasonal:
        trend = "上升" if U_F_seasonal[cp] > 0 else "下降"
        plt.plot(year_seasonal[cp], seasonal_avg_temp[cp], marker='o', markersize=10, color='r',
                 label=f'{season_name}年均温度{trend}突变点')  # 标记检测到的突变点及趋势
    plt.title(f'{season_name}年均温度数据的Mann-Kendall突变检验结果')  # 图表标题
    plt.xlabel('年份')  # x轴标签
    plt.ylabel('年均温度（摄氏度）')  # y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.show()

    if change_points_seasonal:
        print(f"{season_name}检测到的年均温度突变点的年份为 {year_seasonal[change_points_seasonal]}，对应的趋势为:")
        for cp in change_points_seasonal:
            trend = "上升" if U_F_seasonal[cp] > 0 else "下降"
            print(f"年份 {year_seasonal[cp]}，趋势为 {trend}")
    else:
        print(f"{season_name}未检测到显著的年均温度突变点")

# 执行四个聚类的年均温度的 Mann-Kendall 突变检验并可视化
for cluster in clusters:
    cluster_data = data[data['Cluster'] == cluster][['AverTemp']]
    year_cluster, cluster_avg_temp = calculate_annual_avg_temperature(cluster_data)
    U_F_cluster, change_points_cluster = mann_kendall_seasonal_test(cluster_avg_temp)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(year_cluster, cluster_avg_temp, marker='o', linestyle='-', color='b',
             label=f'聚类 {cluster} 的年均温度数据')  # 绘制聚类的年均温度数据
    for cp in change_points_cluster:
        trend = "上升" if U_F_cluster[cp] > 0 else "下降"
        plt.plot(year_cluster[cp], cluster_avg_temp[cp], marker='o', markersize=10, color='r',
                 label=f'聚类 {cluster} 年均温度{trend}突变点')  # 标记检测到的突变点及趋势
    plt.title(f'聚类 {cluster} 的年均温度数据的Mann-Kendall突变检验结果')  # 图表标题
    plt.xlabel('年份')  # x轴标签
    plt.ylabel('年均温度（摄氏度）')  # y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.show()

    if change_points_cluster:
        print(f"聚类 {cluster} 检测到的年均温度突变点的年份为 {year_cluster[change_points_cluster]}，对应的趋势为:")
        for cp in change_points_cluster:
            trend = "上升" if U_F_cluster[cp] > 0 else "下降"
            print(f"年份 {year_cluster[cp]}，趋势为 {trend}")
    else:
        print(f"聚类 {cluster} 未检测到显著的年均温度突变点")

    # 执行四季年均温度的 Mann-Kendall 突变检验并可视化
    for season_name, month_range in seasons.items():
        seasonal_data_cluster = cluster_data[cluster_data.index.month.isin(month_range)]
        year_seasonal_cluster, seasonal_avg_temp_cluster = calculate_annual_avg_temperature(seasonal_data_cluster)
        U_F_seasonal_cluster, change_points_seasonal_cluster = mann_kendall_seasonal_test(seasonal_avg_temp_cluster)

        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.plot(year_seasonal_cluster, seasonal_avg_temp_cluster, marker='o', linestyle='-', color='b',
                 label=f'{season_name}年均温度数据')  # 绘制四季年均温度数据
        for cp in change_points_seasonal_cluster:
            trend = "上升" if U_F_seasonal_cluster[cp] > 0 else "下降"
            plt.plot(year_seasonal_cluster[cp], seasonal_avg_temp_cluster[cp], marker='o', markersize=10, color='r',
                     label=f'{season_name}年均温度{trend}突变点')  # 标记检测到的突变点及趋势
        plt.title(f'{season_name}年均温度数据的Mann-Kendall突变检验结果')  # 图表标题
        plt.xlabel('年份')  # x轴标签
        plt.ylabel('年均温度（摄氏度）')  # y轴标签
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格线
        plt.show()

        if change_points_seasonal_cluster:
            print(
                f"聚类 {cluster} {season_name}检测到的年均温度突变点的年份为 {year_seasonal_cluster[change_points_seasonal_cluster]}，对应的趋势为:")
            for cp in change_points_seasonal_cluster:
                trend = "上升" if U_F_seasonal_cluster[cp] > 0 else "下降"
                print(f"年份 {year_seasonal_cluster[cp]}，趋势为 {trend}")
        else:
            print(f"聚类 {cluster} {season_name}未检测到显著的年均温度突变点")
