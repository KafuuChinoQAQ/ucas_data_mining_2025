import lightgbm as lgb
import pandas as pd
import json

# 加载模型
model = lgb.Booster(model_file='lgbm_woe_model.txt')

# 加载woe映射表
with open('woe_maps.json', 'r', encoding='utf-8') as f:
    woe_maps = json.load(f)

# 加载数据
df = pd.read_csv('evaluation_public.csv')
# 删除多余的列
df = df.drop(['browser', 'ip_type', 'op_month','os_type'], axis=1)
# 确保op_datetime为datetime类型并转为is_workday
df['op_datetime'] = pd.to_datetime(df['op_datetime'])
def is_workday(dt):
    # 只取日期部分
    date = dt.date()
    return date.weekday() < 5  # 0-4表示工作日
df['is_workday'] = df['op_datetime'].apply(is_workday)
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
y_pred = (y_pred_prob > 0.16).astype(int)

# 保存结果
df['pred_prob'] = y_pred_prob
df['is_risk'] = y_pred
df[['id', 'is_risk']].to_csv('predict.csv', index=False)

# 输出预测的is_risk比例
risk_ratio = df['is_risk'].mean()
print(f"Predicted risk ratio: {risk_ratio:.4f}")