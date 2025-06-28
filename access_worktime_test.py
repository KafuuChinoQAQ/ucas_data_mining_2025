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

# 测试不同的时间区间
best_iv = -1
best_x1 = None
best_x2 = None
best_access_worktime = None

# 以小时为单位遍历所有可能的时间区间
hour_options = list(range(0, 24))
for x1 in tqdm(hour_options, desc="起始小时遍历"):
    for x2 in range(x1 + 1, 25):  # 保证x2 > x1
        df['hour'] = df['op_datetime'].dt.hour
        df['access_worktime'] = ((df['hour'] >= x1) & (df['hour'] < x2)).astype(int)
        iv = calc_iv(df, 'access_worktime', 'is_risk')
        print(f"工作时间区间: {x1}:00-{x2}:00, IV: {iv:.4f}")
        if iv > best_iv:
            best_iv = iv
            best_x1 = x1
            best_x2 = x2
            best_access_worktime = df['access_worktime'].copy()

print(f"\n最佳工作时间区间: {best_x1}:00-{best_x2}:00, 最大IV: {best_iv:.4f}")

# 用最佳参数生成access_worktime列并保存
df['hour'] = df['op_datetime'].dt.hour
df['access_worktime'] = ((df['hour'] >= best_x1) & (df['hour'] < best_x2)).astype(int)
df = df.drop(['hour'], axis=1)
df.to_csv('train_with_access_worktime.csv', index=False)
print("已保存带access_worktime的新数据集到'train_with_access_worktime.csv'")