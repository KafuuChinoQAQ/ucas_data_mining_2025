import pandas as pd
import chinese_calendar as cc
import numpy as np
import json

# 计算WOE编码的函数
def calc_woe(df, feature, target):
    # 统计各类别的好坏样本数
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    woe_dict = {}
    iv = 0
    for v in sorted(df[feature].unique()):
        good = ((df[feature] == v) & (df[target] == 0)).sum()
        bad = ((df[feature] == v) & (df[target] == 1)).sum()
        # 避免分母为0，做平滑处理
        good = good if good > 0 else 0.5
        bad = bad if bad > 0 else 0.5
        rate_good = good / total_good
        rate_bad = bad / total_bad
        woe = np.log(rate_good / rate_bad)
        woe_dict[v] = woe
        iv += (rate_good - rate_bad) * woe
    print(f"IV for {feature}: {iv:.4f}")  # 打印IV值
    return woe_dict

# 判断是否为工作日的函数
def is_workday(dt):
    # 只取日期部分
    date = dt.date()
    return cc.is_workday(date)



# 读取数据
df = pd.read_csv('train.csv')
# 删除多余的列
df = df.drop(['browser', 'ip_type', 'op_month','os_type'], axis=1)
# 确保op_datetime为datetime类型
df['op_datetime'] = pd.to_datetime(df['op_datetime'])

# 从op_datetime生成衍生列

# 添加access_frequency列
df = df.sort_values(by=['ip_transform', 'op_datetime'])
freq_count = {}
last_time = {}
access_frequency = []
for idx, row in df.iterrows():
    ip = row['ip_transform']
    now = row['op_datetime']
    last = last_time.get(ip, None)
    count = freq_count.get(ip, 0)
    if last is not None:
        interval = (now - last).total_seconds()
        if interval < 30:
            count += 1
        else:
            count = max(0, count - 1)
    else:
        count = 0
    freq_count[ip] = count
    last_time[ip] = now
    access_frequency.append(1 if count >= 6 else 0)
df['access_frequency'] = access_frequency

# 添加access_worktime列
df['hour'] = df['op_datetime'].dt.hour
df['access_worktime'] = ((df['hour'] >= 8) & (df['hour'] < 20)).astype(int)
df = df.drop(['hour'], axis=1)

# 添加is_workday列
df['is_workday'] = df['op_datetime'].apply(is_workday)

# 删除op_datetime列
df = df.drop(['op_datetime'], axis=1)

# 计算is_risk的总比例
risk_ratio = df['is_risk'].mean()
print(f"Total risk ratio: {risk_ratio:.4f}")

# 创建新的df_1用于写入
# 先写入id列
df_1 = pd.DataFrame({'id': df['id']})


woe_maps = {}  # 用于保存所有特征的woe映射
# 计算除了id和target=is_risk外的所有列的WOE编码并保存到df1中 然后将woe映射写入woe_maps
for col in df.columns:
    # 排除不需要处理的列 首先当然有'id'和'target=is_risk'
    if col not in ['id', 'is_risk']:
        # 输出当前列名
        print(f"Processing column: {col}")
        # 统一类型为字符串，避免排序时报错
        df[col] = df[col].astype(str)
        # 计算WOE编码并写入df_11中
        woe_dict = calc_woe(df, col, 'is_risk')
        df_1[col] = df[col].map(woe_dict)
        # 保存当前列的WOE映射
        woe_maps[col] = woe_dict
        # 输出当前列的唯一值数量
        print(f"Unique values for {col}: {df_1[col].nunique()}")
        print("\n")

# 最后写入is_risk列
df_1['is_risk'] = df['is_risk']

# 保存结果到processed_data.csv
df_1.to_csv('processed_data.csv', index=False)

# 保存WOE映射到JSON文件
with open('woe_maps.json', 'w', encoding='utf-8') as f:
    json.dump(woe_maps, f, ensure_ascii=False, indent=4)