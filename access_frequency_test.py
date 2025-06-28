import pandas as pd
import numpy as np
from tqdm import tqdm

def calc_iv(df, feature, target):
    """计算单个特征的IV值"""
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    iv = 0
    for v in df[feature].unique():
        good = ((df[feature] == v) & (df[target] == 0)).sum()
        bad = ((df[feature] == v) & (df[target] == 1)).sum()
        good = good if good > 0 else 0.5
        bad = bad if bad > 0 else 0.5
        rate_good = good / total_good
        rate_bad = bad / total_bad
        woe = np.log(rate_good / rate_bad)
        iv += (rate_good - rate_bad) * woe
    return iv

# 读取数据
df = pd.read_csv('train.csv')
df['op_datetime'] = pd.to_datetime(df['op_datetime'])

# 备份原df
df_origin = df.copy()
best_iv = -1
best_x1 = None
best_x2 = None
best_access_frequency = None

# 定义不同的时间间隔（秒）和阈值
window_options = [30, 60, 120, 180, 300, 600, 900, 1800, 3600]  # x1 单位：秒
threshold_options = [2, 3, 4, 5, 6, 7]  # x2

for x1 in tqdm(window_options, desc="窗口遍历"):
    for x2 in threshold_options:
        df = df_origin.copy()
        df = df.sort_values(by=['ip_transform', 'op_datetime'])
        freq_count = {}
        last_time = {}
        access_frequency = []
        for idx, row in df.iterrows():
            ip = row['ip_transform']
            now = row['op_datetime']
            # 获取上一次访问时间和当前频繁计数
            last = last_time.get(ip, None)
            count = freq_count.get(ip, 0)
            if last is not None:
                interval = (now - last).total_seconds()
                if interval < x1:
                    count += 1
                else:
                    count = max(0, count - 1)
            else:
                count = 0
            freq_count[ip] = count
            last_time[ip] = now
            access_frequency.append(1 if count >= x2 else 0)
        df['access_frequency'] = access_frequency
        iv = calc_iv(df, 'access_frequency', 'is_risk')
        print(f"时间间隔: {x1}s, 阈值: {x2}, IV: {iv:.4f}")
        if iv > best_iv:
            best_iv = iv
            best_x1 = x1
            best_x2 = x2
            best_access_frequency = df['access_frequency'].copy()

print(f"\n最佳时间间隔: {best_x1}s, 最佳阈值: {best_x2}, 最大IV: {best_iv:.4f}")

# 用最佳参数生成access_frequency列并保存
df = df_origin.copy()
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
        if interval < best_x1:
            count += 1
        else:
            count = max(0, count - 1)
    else:
        count = 0
    freq_count[ip] = count
    last_time[ip] = now
    access_frequency.append(1 if count >= best_x2 else 0)
df['access_frequency'] = access_frequency

df.to_csv('train_with_access_frequency.csv', index=False)
print("已保存带access_frequency的新数据文件：train_with_access_frequency.csv")