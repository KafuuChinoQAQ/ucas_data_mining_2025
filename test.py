import lightgbm as lgb
import pandas as pd
import json
import chinese_calendar as cc

# 判断是否为工作日的函数
def is_workday(dt):
    # 只取日期部分
    date = dt.date()
    return cc.is_workday(date)

# 加载模型
model = lgb.Booster(model_file='lgbm_woe_model.txt')

# 加载woe映射表
with open('woe_maps.json', 'r', encoding='utf-8') as f:
    woe_maps = json.load(f)

# 读取数据
df = pd.read_csv('evaluation_public.csv')
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

# 应用woe编码并将其另存为到evaluation_woe.csv
for col in woe_maps:
    if col in df.columns:
        df[col] = df[col].astype(str).map(woe_maps[col]).fillna(0).astype(float)
# 保存WOE编码后的数据
df.to_csv('evaluation_woe.csv', index=False)

# 取特征
X = df.drop(['id'], axis=1)

# 预测
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.2).astype(int)

# 保存结果
df['pred_prob'] = y_pred_prob
df['is_risk'] = y_pred
# 将输出按id排序
df = df.sort_values(by='id')
df[['id','is_risk']].to_csv('predict.csv', index=False)

# 输出预测的is_risk比例
risk_ratio = df['is_risk'].mean()
print(f"Predicted risk ratio: {risk_ratio:.4f}")