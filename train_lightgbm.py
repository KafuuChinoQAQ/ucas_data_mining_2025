import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取WOE编码后的数据
df = pd.read_csv('processed_data.csv')

# 特征列（排除id和is_risk）
features = [col for col in df.columns if col not in ['id', 'is_risk']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['is_risk'], test_size=0.2, random_state=42, stratify=df['is_risk']
)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# 训练模型
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 预测与评估
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {auc:.4f}")

# 保存模型
model.save_model('lgbm_woe_model.txt')