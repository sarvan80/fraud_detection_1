import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

folder = r'c:\Users\thiag\Downloads\Schwab-Fraud detection'

users = pd.read_csv(f'{folder}\\users_10000.csv')
logins = pd.read_csv(f'{folder}\\login_events_10000.csv')
transactions = pd.read_csv(f'{folder}\\transactions_10000.csv')

# Clean date columns
for df, col in [(logins, 'event_time'), (users, 'signup_date'), (transactions, 'txn_time')]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Merge user-level context into login rows
logins = logins.merge(users, on='user_id', how='left')

# Aggregate transaction features by user
user_txn = transactions.groupby('user_id').agg(
    txn_count=('txn_id', 'count'),
    avg_txn_amount=('txn_amount', 'mean'),
    max_txn_amount=('txn_amount', 'max'),
    txn_fraud_count=('is_fraud', 'sum'),
    high_risk_txn_rate=('txn_risk_score', lambda s: (s > 80).mean()),
    merchant_risk_txn_count=('merchant_risk_flag', 'sum'),
    beneficiary_mismatch_count=('beneficiary_country_mismatch', 'sum')
).reset_index()

logins = logins.merge(user_txn, on='user_id', how='left')

# Sort by user and event time for sequential features
logins = logins.sort_values(['user_id', 'event_time']).reset_index(drop=True)

# User history features from prior login events
logins['user_login_count'] = logins.groupby('user_id').cumcount() + 1
logins['user_prev_fraud_count'] = logins.groupby('user_id')['is_fraud'].transform(lambda s: s.shift(1).fillna(0).cumsum())


def prev_expanding_mean(s):
    vals = s.shift(1).expanding().mean().bfill()
    return vals.to_numpy()

logins['user_prev_risk_mean'] = logins.groupby('user_id')['login_risk_score'].transform(prev_expanding_mean)
logins['user_prev_failed_mean'] = logins.groupby('user_id')['failed_login_count_24h'].transform(prev_expanding_mean)
logins['user_prev_new_device_rate'] = logins.groupby('user_id')['new_device_flag'].transform(prev_expanding_mean)
logins['user_prev_proxy_rate'] = logins.groupby('user_id')['is_proxy'].transform(prev_expanding_mean)

# Add a recent risk window feature based on prior 7-day behavior
for col in ['login_risk_score', 'failed_login_count_24h', 'new_device_flag', 'is_proxy', 'is_vpn', 'impossible_travel_flag', 'is_fraud']:
    # This will be used in windows of recent activity by user
    pass

# Derive travel and behavior features based on login geography
logins['geo_distance_km'] = np.sqrt((logins['ip_lat'] - logins['home_lat']) ** 2 + (logins['ip_lon'] - logins['home_lon']) ** 2) * 111.32
logins['distance_from_home_km'] = np.round(logins['geo_distance_km'], 2)
logins['velocity_flag'] = (logins['distance_from_home_km'] > 2000).astype(int)
logins['travel_risk_score'] = (
    5 * logins['impossible_travel_flag'] +
    3 * logins['is_proxy'] +
    2 * logins['is_vpn'] +
    2 * logins['new_device_flag'] +
    2 * logins['velocity_flag']
)

# Fill missing values
for col in ['txn_count', 'avg_txn_amount', 'max_txn_amount', 'txn_fraud_count', 'high_risk_txn_rate', 'merchant_risk_txn_count', 'beneficiary_mismatch_count']:
    logins[col] = logins[col].fillna(0)

# Normalize some numerical features
logins['avg_txn_amount'] = logins['avg_txn_amount'].fillna(0)
logins['max_txn_amount'] = logins['max_txn_amount'].fillna(0)
logins['failed_login_count_24h'] = logins['failed_login_count_24h'].fillna(0)
logins['user_prev_risk_mean'] = logins['user_prev_risk_mean'].fillna(logins['login_risk_score'])
logins['user_prev_failed_mean'] = logins['user_prev_failed_mean'].fillna(0)
logins['user_prev_new_device_rate'] = logins['user_prev_new_device_rate'].fillna(0)
logins['user_prev_proxy_rate'] = logins['user_prev_proxy_rate'].fillna(0)

# Final feature set for model training
feature_columns = [
    'account_age_days',
    'has_mfa',
    'login_risk_score',
    'device_risk_score',
    'session_risk_score',
    'failed_login_count_24h',
    'new_device_flag',
    'is_proxy',
    'is_vpn',
    'impossible_travel_flag',
    'mfa_approved',
    'typing_velocity',
    'mouse_velocity',
    'click_count',
    'keystroke_count',
    'distance_from_home_km',
    'travel_risk_score',
    'txn_count',
    'avg_txn_amount',
    'max_txn_amount',
    'txn_fraud_count',
    'high_risk_txn_rate',
    'merchant_risk_txn_count',
    'beneficiary_mismatch_count',
    'user_login_count',
    'user_prev_fraud_count',
    'user_prev_risk_mean',
    'user_prev_failed_mean',
    'user_prev_new_device_rate',
    'user_prev_proxy_rate'
]

# Handle categorical columns
train_df = logins.copy()
train_df['risk_tier'] = train_df['risk_tier'].fillna('low')
train_df['has_mfa'] = train_df['has_mfa'].astype(int)

# One-hot encode categorical fields
train_df = pd.get_dummies(train_df, columns=['risk_tier'], prefix='risk_tier', drop_first=False)

# Keep target and feature columns
model_df = train_df[['is_fraud'] + feature_columns + [c for c in train_df.columns if c.startswith('risk_tier_')]]

# Ensure no missing values
for c in model_df.columns:
    model_df[c] = model_df[c].fillna(0)

# Target is already integer 0/1
X = model_df.drop(columns=['is_fraud'])
y = model_df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_final = X_train.copy()
train_final['is_fraud'] = y_train.values

test_final = X_test.copy()
test_final['is_fraud'] = y_test.values

train_path = f'{folder}\\train_login_model.csv'
test_path = f'{folder}\\test_login_model.csv'
model_path = f'{folder}\\model_ready_login.csv'

train_final.to_csv(train_path, index=False)
test_final.to_csv(test_path, index=False)
model_df.to_csv(model_path, index=False)

print('Saved model_ready_login.csv')
print('Saved train_login_model.csv')
print('Saved test_login_model.csv')
print('Train fraud rate:', y_train.mean())
print('Test fraud rate:', y_test.mean())
print('Overall fraud rate:', y.mean())
