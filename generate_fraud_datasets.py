import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

folder = r'c:\Users\thiag\Downloads\Schwab-Fraud detection'
N = 10_000

# ----------------------------
# Users dataset
# ----------------------------
users = []
for i in range(1, N + 1):
    country = random.choice(['US', 'CA', 'GB', 'IN', 'DE', 'AU', 'FR', 'BR'])
    city = random.choice(['New York', 'Toronto', 'London', 'Bengaluru', 'Berlin', 'Sydney', 'Paris', 'Sao Paulo'])
    signup = datetime(2019, 1, 1) + timedelta(days=random.randint(0, 2000))
    users.append({
        'user_id': f'U{i:05d}',
        'email_domain': random.choice(['gmail.com', 'outlook.com', 'yahoo.com', 'company.com']),
        'account_age_days': random.randint(30, 3000),
        'country': country,
        'home_city': city,
        'home_lat': round(random.uniform(20, 60), 4),
        'home_lon': round(random.uniform(-150, 50), 4),
        'signup_date': signup.strftime('%Y-%m-%d'),
        'has_mfa': random.choice([True, False]),
        'risk_tier': random.choices(['low', 'medium', 'high'], weights=[70, 22, 8], k=1)[0],
    })
user_df = pd.DataFrame(users)

# ----------------------------
# Login events dataset
# ----------------------------
login_rows = []
for i in range(1, N + 1):
    user = users[i - 1]
    user_id = user['user_id']
    event_time = datetime(2025, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    base_ip = f'192.168.{random.randint(1, 250)}.{random.randint(1, 250)}'
    scenario = random.choices(
        ['normal', 'credential_stuffing', 'impossible_travel', 'new_device', 'mfa_fatigue', 'proxy', 'session_hijack'],
        weights=[97, 1.2, 0.8, 0.7, 0.6, 0.4, 0.3],
        k=1
    )[0]

    if scenario == 'normal':
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = random.choice([True, True, True, False])
        failed_logins = random.randint(0, 3)
        device_risk = random.randint(0, 40)
        session_risk = random.randint(0, 30)
        login_result = 'success' if random.random() > 0.08 else 'failed'
    elif scenario == 'credential_stuffing':
        is_proxy = random.choice([True, False])
        is_vpn = random.choice([True, False])
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = False if random.random() < 0.8 else True
        failed_logins = random.randint(8, 25)
        device_risk = random.randint(35, 75)
        session_risk = random.randint(30, 70)
        login_result = 'failed'
    elif scenario == 'impossible_travel':
        is_proxy = False
        is_vpn = random.choice([False, True])
        new_device_flag = False
        is_impossible_travel = True
        mfa_approved = random.choice([True, False])
        failed_logins = random.randint(0, 5)
        device_risk = random.randint(20, 55)
        session_risk = random.randint(50, 82)
        login_result = random.choice(['success', 'failed'])
    elif scenario == 'new_device':
        is_proxy = False
        is_vpn = False
        new_device_flag = True
        is_impossible_travel = False
        mfa_approved = random.choice([False, True])
        failed_logins = random.randint(1, 7)
        device_risk = random.randint(55, 90)
        session_risk = random.randint(35, 75)
        login_result = random.choice(['success', 'failed'])
    elif scenario == 'mfa_fatigue':
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = False
        failed_logins = random.randint(0, 6)
        device_risk = random.randint(25, 60)
        session_risk = random.randint(58, 90)
        login_result = 'failed'
    elif scenario == 'proxy':
        is_proxy = True
        is_vpn = random.choice([False, True])
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = random.choice([True, False])
        failed_logins = random.randint(0, 4)
        device_risk = random.randint(30, 68)
        session_risk = random.randint(40, 80)
        login_result = random.choice(['success', 'failed'])
    else:
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = True
        failed_logins = random.randint(0, 3)
        device_risk = random.randint(30, 70)
        session_risk = random.randint(70, 96)
        login_result = random.choice(['success', 'failed'])

    login_risk = min(99, max(0, int((device_risk * 0.45) + (session_risk * 0.55) + (failed_logins * 2.5))))
    if is_proxy:
        login_risk += 10
    if is_vpn:
        login_risk += 8
    if new_device_flag:
        login_risk += 15
    if is_impossible_travel:
        login_risk += 18
    if not mfa_approved and scenario in ['mfa_fatigue', 'credential_stuffing', 'new_device']:
        login_risk += 12
    login_risk = max(0, min(99, login_risk))

    scenario_base = {
        'normal': 0.0025,
        'credential_stuffing': 0.52,
        'impossible_travel': 0.35,
        'new_device': 0.42,
        'mfa_fatigue': 0.54,
        'proxy': 0.32,
        'session_hijack': 0.60,
    }
    is_fraud = 1 if random.random() < scenario_base.get(scenario, 0.0025) else 0
    if login_result == 'success' and random.random() < 0.004:
        is_fraud = 1
    if is_impossible_travel and random.random() < 0.05:
        is_fraud = 1

    fraud_type = None
    if is_fraud == 1:
        if scenario == 'credential_stuffing':
            fraud_type = 'credential_stuffing'
        elif scenario == 'impossible_travel':
            fraud_type = 'impossible_travel'
        elif scenario == 'new_device':
            fraud_type = 'device_spoofing'
        elif scenario == 'mfa_fatigue':
            fraud_type = 'mfa_fatigue'
        elif scenario == 'proxy':
            fraud_type = 'distributed_proxy'
        elif scenario == 'session_hijack':
            fraud_type = 'session_hijack'
        else:
            fraud_type = 'unknown_anomaly'

    login_rows.append({
        'login_id': f'L{i:05d}',
        'user_id': user_id,
        'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
        'device_id': f'D{random.randint(1000, 9999)}',
        'device_fingerprint': f'fp_{random.randint(100000, 999999)}',
        'ip_address': base_ip,
        'ip_country': random.choice(['US', 'CA', 'GB', 'US', 'IN', 'DE', 'AU']),
        'ip_region': random.choice(['NA', 'EU', 'APAC', 'LATAM']),
        'ip_lat': round(random.uniform(-55, 75), 4),
        'ip_lon': round(random.uniform(-180, 180), 4),
        'user_agent': random.choice(['Mozilla/5.0', 'Chrome/124', 'Safari/17', 'Edge/124']),
        'browser_name': random.choice(['Chrome', 'Safari', 'Firefox', 'Edge']),
        'browser_version': f'{random.randint(100, 130)}.0',
        'os_name': random.choice(['Windows', 'macOS', 'Linux', 'iOS', 'Android']),
        'os_version': f'{random.randint(10, 15)}',
        'country': user['country'],
        'home_city': user['home_city'],
        'new_device_flag': int(new_device_flag),
        'is_proxy': int(is_proxy),
        'is_vpn': int(is_vpn),
        'impossible_travel_flag': int(is_impossible_travel),
        'failed_login_count_24h': failed_logins,
        'mfa_required': int(random.choice([True, True, True, False])),
        'mfa_approved': int(mfa_approved),
        'login_result': login_result,
        'session_risk_score': session_risk,
        'device_risk_score': device_risk,
        'login_risk_score': login_risk,
        'is_fraud': int(is_fraud),
        'fraud_type': fraud_type,
        'fraud_score': round(login_risk / 100, 2),
        'typing_velocity': round(random.uniform(20, 120), 2),
        'mouse_velocity': round(random.uniform(30, 180), 2),
        'click_count': random.randint(0, 40),
        'keystroke_count': random.randint(0, 300),
    })
login_df = pd.DataFrame(login_rows)

# ----------------------------
# Transaction dataset
# ----------------------------
transaction_rows = []
for i in range(1, N + 1):
    user = users[i - 1]
    user_id = user['user_id']
    login = login_df.iloc[i - 1]
    amount = round(random.uniform(5, 6000), 2)
    if login['is_fraud'] == 1:
        amount = round(random.uniform(100, 25000), 2)
    txn_time = datetime(2025, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    transaction_rows.append({
        'txn_id': f'TXN{i:05d}',
        'user_id': user_id,
        'txn_time': txn_time.strftime('%Y-%m-%d %H:%M:%S'),
        'device_id': login['device_id'],
        'device_fingerprint': login['device_fingerprint'],
        'ip_address': login['ip_address'],
        'txn_amount': amount,
        'merchant_country': random.choice(['US', 'CA', 'GB', 'IN', 'DE', 'FR']),
        'payment_method': random.choice(['ACH', 'Debit', 'Wire', 'Card']),
        'mfa_required': int(random.choice([True, True, False])),
        'mfa_approved': int(random.choice([True, True, False])),
        'txn_risk_score': min(99, max(10, int(login['login_risk_score'] + random.randint(0, 25)))),
        'is_fraud': int(login['is_fraud']),
        'fraud_type': login['fraud_type'],
        'merchant_risk_flag': int(random.random() < 0.12),
        'beneficiary_country_mismatch': int(random.random() < 0.08),
    })
transaction_df = pd.DataFrame(transaction_rows)

# Save
user_df.to_csv(f'{folder}\\users_10000.csv', index=False)
login_df.to_csv(f'{folder}\\login_events_10000.csv', index=False)
transaction_df.to_csv(f'{folder}\\transactions_10000.csv', index=False)

print('Generated files: users_10000.csv, login_events_10000.csv, transactions_10000.csv')
print('Fraud rate in login events:', login_df['is_fraud'].mean())
