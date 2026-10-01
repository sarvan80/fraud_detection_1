from pathlib import Path

import pandas as pd


FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection\reference")
CSV_PATH = FOLDER / "data_dictionary.csv"
MD_PATH = FOLDER / "data_dictionary.md"


# Each row documents one attribute across the raw source files and the
# engineered model-ready datasets produced by the pipeline scripts in the
# project root (generate_fraud_datasets.py, build_model_ready_dataset.py,
# train_fraud_model_v2.py, train_precision_recall_models.py,
# train_deep_learning_models.py).
RECORDS = [
    # ---------------------------------------------------------------
    # users_10000.csv
    # ---------------------------------------------------------------
    {"dataset": "users_10000.csv", "attribute": "user_id", "data_type": "string",
     "description": "Unique identifier for a customer account, formatted as U##### and used as the join key across all datasets.",
     "example_values": "U00001, U00002", "source_or_derivation": "Generated sequentially in generate_fraud_datasets.py"},
    {"dataset": "users_10000.csv", "attribute": "email_domain", "data_type": "categorical (string)",
     "description": "Domain portion of the account's registered email address, used as a weak identity/reputation signal.",
     "example_values": "gmail.com, outlook.com, yahoo.com, company.com", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "account_age_days", "data_type": "integer",
     "description": "Number of days since the account was created; older accounts are generally lower risk.",
     "example_values": "30-3000", "source_or_derivation": "Randomly sampled uniform range"},
    {"dataset": "users_10000.csv", "attribute": "country", "data_type": "categorical (string, ISO-like code)",
     "description": "Country associated with the customer's registered profile.",
     "example_values": "US, CA, GB, IN, DE, AU, FR, BR", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "home_city", "data_type": "categorical (string)",
     "description": "City associated with the customer's registered home address; used to derive travel/geo-distance features.",
     "example_values": "New York, Toronto, London, Bengaluru, Berlin, Sydney, Paris, Sao Paulo", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "home_lat", "data_type": "float (decimal degrees)",
     "description": "Latitude of the customer's registered home location, used to compute impossible-travel and geo-distance features.",
     "example_values": "20.0-60.0", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "home_lon", "data_type": "float (decimal degrees)",
     "description": "Longitude of the customer's registered home location, used to compute impossible-travel and geo-distance features.",
     "example_values": "-150.0-50.0", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "signup_date", "data_type": "date (YYYY-MM-DD)",
     "description": "Date the account was created/registered.",
     "example_values": "2019-01-01 to ~2024-06", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "has_mfa", "data_type": "boolean",
     "description": "Whether the account has multi-factor authentication enabled; accounts without MFA are more vulnerable to ATO.",
     "example_values": "True, False", "source_or_derivation": "Randomly sampled"},
    {"dataset": "users_10000.csv", "attribute": "risk_tier", "data_type": "categorical (string)",
     "description": "Business-assigned baseline risk classification for the account, weighted toward 'low'.",
     "example_values": "low (70%), medium (22%), high (8%)", "source_or_derivation": "Weighted random sample"},

    # ---------------------------------------------------------------
    # login_events_10000.csv
    # ---------------------------------------------------------------
    {"dataset": "login_events_10000.csv", "attribute": "login_id", "data_type": "string",
     "description": "Unique identifier for a single login/authentication event.",
     "example_values": "L00001, L00002", "source_or_derivation": "Generated sequentially"},
    {"dataset": "login_events_10000.csv", "attribute": "user_id", "data_type": "string",
     "description": "Foreign key to users_10000.csv identifying which account performed the login.",
     "example_values": "U00001", "source_or_derivation": "Copied from users table"},
    {"dataset": "login_events_10000.csv", "attribute": "event_time", "data_type": "datetime",
     "description": "Timestamp when the login attempt occurred.",
     "example_values": "2025-09-27 02:00:00", "source_or_derivation": "Randomly sampled across 2025"},
    {"dataset": "login_events_10000.csv", "attribute": "device_id", "data_type": "string",
     "description": "Identifier for the physical/virtual device used for the login attempt.",
     "example_values": "D1000-D9999", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "device_fingerprint", "data_type": "string (hash-like)",
     "description": "Composite fingerprint derived from device/browser attributes; used to recognize returning devices.",
     "example_values": "fp_343031", "source_or_derivation": "Randomly generated"},
    {"dataset": "login_events_10000.csv", "attribute": "ip_address", "data_type": "string (IPv4)",
     "description": "Source IP address of the login attempt.",
     "example_values": "192.168.168.203", "source_or_derivation": "Randomly generated (private-range placeholder)"},
    {"dataset": "login_events_10000.csv", "attribute": "ip_country", "data_type": "categorical (string)",
     "description": "Country associated with the login IP address via geolocation lookup.",
     "example_values": "US, CA, GB, IN, DE, AU", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "ip_region", "data_type": "categorical (string)",
     "description": "Broad geographic region associated with the login IP address.",
     "example_values": "NA, EU, APAC, LATAM", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "ip_lat", "data_type": "float (decimal degrees)",
     "description": "Latitude of the login IP's geolocation; compared against home_lat to detect impossible travel.",
     "example_values": "-55 to 75", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "ip_lon", "data_type": "float (decimal degrees)",
     "description": "Longitude of the login IP's geolocation; compared against home_lon to detect impossible travel.",
     "example_values": "-180 to 180", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "user_agent", "data_type": "categorical (string)",
     "description": "Raw user-agent string reported by the client/browser.",
     "example_values": "Mozilla/5.0, Chrome/124, Safari/17, Edge/124", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "browser_name", "data_type": "categorical (string)",
     "description": "Parsed browser family used for the login.",
     "example_values": "Chrome, Safari, Firefox, Edge", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "browser_version", "data_type": "string (version)",
     "description": "Version number of the browser used for the login.",
     "example_values": "100.0-130.0", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "os_name", "data_type": "categorical (string)",
     "description": "Operating system of the device used for the login.",
     "example_values": "Windows, macOS, Linux, iOS, Android", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "os_version", "data_type": "string (version)",
     "description": "Operating system version.",
     "example_values": "10-15", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "country", "data_type": "categorical (string)",
     "description": "Registered account country, copied from the users table for convenience.",
     "example_values": "US, CA, GB", "source_or_derivation": "Copied from users table"},
    {"dataset": "login_events_10000.csv", "attribute": "home_city", "data_type": "categorical (string)",
     "description": "Registered account home city, copied from the users table for convenience.",
     "example_values": "New York, Toronto", "source_or_derivation": "Copied from users table"},
    {"dataset": "login_events_10000.csv", "attribute": "new_device_flag", "data_type": "boolean (0/1)",
     "description": "Indicates the login originated from a device not previously associated with the account (ATO scenario: device spoofing/new device).",
     "example_values": "0, 1", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "is_proxy", "data_type": "boolean (0/1)",
     "description": "Indicates the connection was routed through a proxy server (ATO scenario: distributed proxy).",
     "example_values": "0, 1", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "is_vpn", "data_type": "boolean (0/1)",
     "description": "Indicates the connection was routed through a VPN, which can mask true geolocation.",
     "example_values": "0, 1", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "impossible_travel_flag", "data_type": "boolean (0/1)",
     "description": "Indicates two logins occurred farther apart geographically than physically possible in the elapsed time (ATO scenario: impossible travel).",
     "example_values": "0, 1", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "failed_login_count_24h", "data_type": "integer",
     "description": "Number of failed login attempts for the account/IP in the prior 24 hours; elevated for credential-stuffing scenarios.",
     "example_values": "0-25", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "mfa_required", "data_type": "boolean (0/1)",
     "description": "Whether multi-factor authentication was required for this login attempt.",
     "example_values": "0, 1", "source_or_derivation": "Randomly sampled, weighted"},
    {"dataset": "login_events_10000.csv", "attribute": "mfa_approved", "data_type": "boolean (0/1)",
     "description": "Whether the MFA challenge was approved by the user; repeated denials indicate MFA-fatigue attacks.",
     "example_values": "0, 1", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "login_result", "data_type": "categorical (string)",
     "description": "Outcome of the login attempt.",
     "example_values": "success, failed", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "session_risk_score", "data_type": "integer (0-99)",
     "description": "Composite risk score for the session behavior surrounding the login, higher indicates riskier session context.",
     "example_values": "0-96", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "device_risk_score", "data_type": "integer (0-90)",
     "description": "Composite risk score for the device used in the login, higher indicates a riskier/less-trusted device.",
     "example_values": "0-90", "source_or_derivation": "Scenario-driven simulation"},
    {"dataset": "login_events_10000.csv", "attribute": "login_risk_score", "data_type": "integer (0-99)",
     "description": "Blended overall risk score for the login event, combining device risk, session risk, failed logins, proxy/VPN, new device, and impossible-travel signals.",
     "example_values": "0-99", "source_or_derivation": "Computed in generate_fraud_datasets.py"},
    {"dataset": "login_events_10000.csv", "attribute": "is_fraud", "data_type": "boolean (0/1) - TARGET",
     "description": "Ground-truth fraud label for this login event; the primary modeling target across all training scripts.",
     "example_values": "0, 1 (~2.4% positive rate)", "source_or_derivation": "Scenario-driven simulation with probabilistic rules"},
    {"dataset": "login_events_10000.csv", "attribute": "fraud_type", "data_type": "categorical (string, nullable)",
     "description": "Named ATO fraud scenario associated with a fraudulent event; null/blank for legitimate events.",
     "example_values": "credential_stuffing, impossible_travel, device_spoofing, mfa_fatigue, distributed_proxy, session_hijack, unknown_anomaly",
     "source_or_derivation": "Assigned based on simulated scenario"},
    {"dataset": "login_events_10000.csv", "attribute": "fraud_score", "data_type": "float (0.0-0.99)",
     "description": "login_risk_score expressed as a 0-1 probability-like value (login_risk_score / 100).",
     "example_values": "0.29, 0.99", "source_or_derivation": "Derived: login_risk_score / 100"},
    {"dataset": "login_events_10000.csv", "attribute": "typing_velocity", "data_type": "float",
     "description": "Behavioral biometric signal: characters typed per second during the session; unusually high/low values may indicate automation.",
     "example_values": "20.0-120.0", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "mouse_velocity", "data_type": "float",
     "description": "Behavioral biometric signal: pixels-per-second of mouse movement; extreme values may indicate automation or remote control.",
     "example_values": "30.0-180.0", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "click_count", "data_type": "integer",
     "description": "Number of mouse clicks recorded during the session.",
     "example_values": "0-40", "source_or_derivation": "Randomly sampled"},
    {"dataset": "login_events_10000.csv", "attribute": "keystroke_count", "data_type": "integer",
     "description": "Number of keystrokes recorded during the session.",
     "example_values": "0-300", "source_or_derivation": "Randomly sampled"},

    # ---------------------------------------------------------------
    # transactions_10000.csv
    # ---------------------------------------------------------------
    {"dataset": "transactions_10000.csv", "attribute": "txn_id", "data_type": "string",
     "description": "Unique identifier for a financial transaction.",
     "example_values": "TXN00001", "source_or_derivation": "Generated sequentially"},
    {"dataset": "transactions_10000.csv", "attribute": "user_id", "data_type": "string",
     "description": "Foreign key to users_10000.csv / login_events_10000.csv identifying the account performing the transaction (1:1 with a login event by row position in this synthetic dataset).",
     "example_values": "U00001", "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "txn_time", "data_type": "datetime",
     "description": "Timestamp when the transaction occurred.",
     "example_values": "2025-02-20 02:10:00", "source_or_derivation": "Randomly sampled across 2025"},
    {"dataset": "transactions_10000.csv", "attribute": "device_id", "data_type": "string",
     "description": "Device used to initiate the transaction; copied from the associated login event.",
     "example_values": "D8009", "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "device_fingerprint", "data_type": "string (hash-like)",
     "description": "Device fingerprint used to initiate the transaction; copied from the associated login event.",
     "example_values": "fp_343031", "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "ip_address", "data_type": "string (IPv4)",
     "description": "Source IP address of the transaction; copied from the associated login event.",
     "example_values": "192.168.168.203", "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "txn_amount", "data_type": "float (currency)",
     "description": "Monetary value of the transaction; fraudulent transactions are simulated with a higher amount range.",
     "example_values": "5.00-25000.00", "source_or_derivation": "Randomly sampled, elevated range if is_fraud=1"},
    {"dataset": "transactions_10000.csv", "attribute": "merchant_country", "data_type": "categorical (string)",
     "description": "Country of the merchant or transaction counterparty.",
     "example_values": "US, CA, GB, IN, DE, FR", "source_or_derivation": "Randomly sampled"},
    {"dataset": "transactions_10000.csv", "attribute": "payment_method", "data_type": "categorical (string)",
     "description": "Payment rail used for the transaction.",
     "example_values": "ACH, Debit, Wire, Card", "source_or_derivation": "Randomly sampled"},
    {"dataset": "transactions_10000.csv", "attribute": "mfa_required", "data_type": "boolean (0/1)",
     "description": "Whether MFA was required to authorize this transaction.",
     "example_values": "0, 1", "source_or_derivation": "Randomly sampled"},
    {"dataset": "transactions_10000.csv", "attribute": "mfa_approved", "data_type": "boolean (0/1)",
     "description": "Whether the MFA challenge for this transaction was approved.",
     "example_values": "0, 1", "source_or_derivation": "Randomly sampled"},
    {"dataset": "transactions_10000.csv", "attribute": "txn_risk_score", "data_type": "integer (10-99)",
     "description": "Composite risk score for the transaction, derived from the associated login's risk score plus additional random variance.",
     "example_values": "10-99", "source_or_derivation": "Derived: login_risk_score + random offset"},
    {"dataset": "transactions_10000.csv", "attribute": "is_fraud", "data_type": "boolean (0/1) - TARGET",
     "description": "Ground-truth fraud label for the transaction; identical to the paired login event's label in this synthetic dataset.",
     "example_values": "0, 1", "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "fraud_type", "data_type": "categorical (string, nullable)",
     "description": "Named ATO fraud scenario associated with the transaction; copied from the paired login event.",
     "example_values": "credential_stuffing, impossible_travel, device_spoofing, mfa_fatigue, distributed_proxy, session_hijack",
     "source_or_derivation": "Copied from paired login event"},
    {"dataset": "transactions_10000.csv", "attribute": "merchant_risk_flag", "data_type": "boolean (0/1)",
     "description": "Indicates the merchant is on a risk watch-list or has an elevated risk profile.",
     "example_values": "0, 1 (~12% positive rate)", "source_or_derivation": "Randomly sampled"},
    {"dataset": "transactions_10000.csv", "attribute": "beneficiary_country_mismatch", "data_type": "boolean (0/1)",
     "description": "Indicates the beneficiary/recipient country differs from the expected account country, a common money-movement fraud indicator.",
     "example_values": "0, 1 (~8% positive rate)", "source_or_derivation": "Randomly sampled"},

    # ---------------------------------------------------------------
    # Engineered / model-ready attributes
    # (model_ready_login.csv, train_login_model.csv, test_login_model.csv)
    # ---------------------------------------------------------------
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "distance_from_home_km", "data_type": "float",
     "description": "Great-circle-approximate distance between the login IP geolocation and the account's registered home location, in kilometers.",
     "example_values": "0-30000+", "source_or_derivation": "Derived in build_model_ready_dataset.py: sqrt((ip_lat-home_lat)^2+(ip_lon-home_lon)^2) * 111.32"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "velocity_flag", "data_type": "boolean (0/1)",
     "description": "Flags a login as a high-velocity/implausible travel event when distance_from_home_km exceeds 2000 km.",
     "example_values": "0, 1", "source_or_derivation": "Derived: distance_from_home_km > 2000"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "travel_risk_score", "data_type": "integer",
     "description": "Weighted composite of travel/network risk indicators (impossible travel, proxy, VPN, new device, velocity flag).",
     "example_values": "0-14", "source_or_derivation": "Derived: 5*impossible_travel_flag + 3*is_proxy + 2*is_vpn + 2*new_device_flag + 2*velocity_flag"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "txn_count", "data_type": "integer",
     "description": "Total number of transactions associated with the user (aggregated from transactions_10000.csv). Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0-1", "source_or_derivation": "Aggregated via groupby('user_id') in build_model_ready_dataset.py"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "avg_txn_amount", "data_type": "float",
     "description": "Average transaction amount for the user. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0-25000", "source_or_derivation": "Aggregated mean of txn_amount by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "max_txn_amount", "data_type": "float",
     "description": "Maximum transaction amount for the user. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0-25000", "source_or_derivation": "Aggregated max of txn_amount by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "txn_fraud_count", "data_type": "integer",
     "description": "Count of fraudulent transactions for the user. Excluded from modeling in later scripts due to direct target leakage.",
     "example_values": "0-1", "source_or_derivation": "Aggregated sum of is_fraud by user_id (transactions table)"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "high_risk_txn_rate", "data_type": "float (0-1)",
     "description": "Share of the user's transactions with txn_risk_score above 80. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0.0-1.0", "source_or_derivation": "Aggregated mean of (txn_risk_score > 80) by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "merchant_risk_txn_count", "data_type": "integer",
     "description": "Count of the user's transactions flagged with merchant_risk_flag. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0-1", "source_or_derivation": "Aggregated sum of merchant_risk_flag by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "beneficiary_mismatch_count", "data_type": "integer",
     "description": "Count of the user's transactions with beneficiary_country_mismatch. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0-1", "source_or_derivation": "Aggregated sum of beneficiary_country_mismatch by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_login_count", "data_type": "integer",
     "description": "Running count of how many login events this user has had up to and including the current row, ordered by event_time.",
     "example_values": "1, 2, 3...", "source_or_derivation": "Derived: groupby('user_id').cumcount() + 1"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_prev_fraud_count", "data_type": "integer",
     "description": "Cumulative count of prior fraudulent logins for this user before the current event. Excluded from modeling in later scripts due to target-leakage risk.",
     "example_values": "0, 1, 2...", "source_or_derivation": "Derived: cumulative sum of shifted is_fraud by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_prev_risk_mean", "data_type": "float",
     "description": "Expanding average of login_risk_score across the user's prior login events (backfilled for the first event).",
     "example_values": "0-99", "source_or_derivation": "Derived: shift(1).expanding().mean().bfill() by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_prev_failed_mean", "data_type": "float",
     "description": "Expanding average of failed_login_count_24h across the user's prior login events.",
     "example_values": "0-25", "source_or_derivation": "Derived: shift(1).expanding().mean().bfill() by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_prev_new_device_rate", "data_type": "float (0-1)",
     "description": "Expanding average of new_device_flag across the user's prior login events, i.e. historical rate of new-device logins.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: shift(1).expanding().mean().bfill() by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "user_prev_proxy_rate", "data_type": "float (0-1)",
     "description": "Expanding average of is_proxy across the user's prior login events, i.e. historical rate of proxy usage.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: shift(1).expanding().mean().bfill() by user_id"},
    {"dataset": "model_ready_login.csv / train_login_model.csv / test_login_model.csv", "attribute": "risk_tier_low / risk_tier_medium / risk_tier_high", "data_type": "boolean (one-hot)",
     "description": "One-hot encoded indicators for the user's risk_tier category from the users table.",
     "example_values": "True/False", "source_or_derivation": "Derived: pd.get_dummies(risk_tier)"},

    # ---------------------------------------------------------------
    # Engineered attributes added in train_fraud_model_v2.py and reused by
    # train_precision_recall_models.py / train_deep_learning_models.py
    # ---------------------------------------------------------------
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "risk_signal_count", "data_type": "integer (0-5)",
     "description": "Count of simultaneously active risk indicators: new device, proxy, VPN, impossible travel, and MFA not approved.",
     "example_values": "0-5", "source_or_derivation": "Derived in train_fraud_model_v2.py: sum of new_device_flag + is_proxy + is_vpn + impossible_travel_flag + (mfa_approved==0)"},
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "authentication_pressure", "data_type": "float",
     "description": "Composite score representing authentication strain, combining failed-login volume with session and device risk.",
     "example_values": "0-300+", "source_or_derivation": "Derived: failed_login_count_24h*2 + session_risk_score + device_risk_score"},
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "risk_score_gap", "data_type": "float",
     "description": "Difference between the current login_risk_score and the user's historical average risk (user_prev_risk_mean); positive values indicate an unusual spike in risk.",
     "example_values": "-99 to 99", "source_or_derivation": "Derived: login_risk_score - user_prev_risk_mean"},
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "behavior_irregularity", "data_type": "float",
     "description": "Ratio of mouse velocity to typing velocity; unusual ratios may indicate non-human or scripted interaction patterns.",
     "example_values": "0.2-9.0", "source_or_derivation": "Derived: mouse_velocity / (typing_velocity + 1.0)"},
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "interaction_density", "data_type": "float",
     "description": "Total UI interactions (clicks + keystrokes) normalized by the user's login history length; flags unusually sparse or dense sessions relative to tenure.",
     "example_values": "0-300+", "source_or_derivation": "Derived: (click_count + keystroke_count) / (user_login_count + 1.0)"},
    {"dataset": "v2 modeling scripts (in-memory feature set)", "attribute": "network_travel_interaction", "data_type": "float",
     "description": "Interaction term amplifying geographic distance when a proxy or VPN is also present, capturing compounded network+location risk.",
     "example_values": "0-90000+", "source_or_derivation": "Derived: distance_from_home_km * (1 + is_proxy + is_vpn)"},

    # ---------------------------------------------------------------
    # Hybrid / alternate-approach derived signals
    # ---------------------------------------------------------------
    {"dataset": "alternate_approach hybrid model outputs", "attribute": "cluster_id", "data_type": "integer (0-3)",
     "description": "K-Means cluster assignment (4 clusters) from the PCA-reduced feature space, used as an unsupervised behavioral pseudo-class.",
     "example_values": "0, 1, 2, 3", "source_or_derivation": "Derived in run_hybrid_security_model.py via PCA + KMeans"},
    {"dataset": "alternate_approach hybrid model outputs", "attribute": "cluster_risk", "data_type": "float (0-1)",
     "description": "Normalized fraud-rate lift of the assigned cluster relative to the global training fraud rate, capped at 1.0.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: (cluster fraud_rate / global fraud_rate) clipped to [0,4] / 4"},
    {"dataset": "alternate_approach hybrid model outputs", "attribute": "anomaly_score", "data_type": "float (0-1)",
     "description": "Normalized Isolation Forest anomaly score (fit on legitimate-only training data); higher values indicate greater deviation from normal behavior.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: percentile-clipped -score_samples() from IsolationForest"},
    {"dataset": "alternate_approach hybrid model outputs", "attribute": "rule_probability", "data_type": "float (0-1)",
     "description": "Predicted fraud probability from the interpretable decision-tree rule-extraction model.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: DecisionTreeClassifier.predict_proba()"},
    {"dataset": "alternate_approach hybrid model outputs", "attribute": "hybrid_probability", "data_type": "float (0-1)",
     "description": "Blended decision-intelligence score combining rule probability, anomaly score, and cluster risk.",
     "example_values": "0.0-1.0", "source_or_derivation": "Derived: 0.65*rule_probability + 0.20*anomaly_score + 0.15*cluster_risk"},

    # ---------------------------------------------------------------
    # Model output / scoring columns (common across scored_test_*.csv files)
    # ---------------------------------------------------------------
    {"dataset": "scored_test_*.csv output files", "attribute": "fraud_probability / hybrid_probability / <model_name>_probability", "data_type": "float (0-1)",
     "description": "Predicted probability that the record is fraudulent, produced by the corresponding trained model.",
     "example_values": "0.0-1.0", "source_or_derivation": "Model predict_proba() output"},
    {"dataset": "scored_test_*.csv output files", "attribute": "predicted_fraud / predicted_fraud_v2 / <model_name>_prediction", "data_type": "boolean (0/1)",
     "description": "Binary fraud decision after applying the model's tuned operating threshold to its probability score.",
     "example_values": "0, 1", "source_or_derivation": "Derived: probability >= selected_threshold"},
]


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Data Dictionary",
        "",
        "This dictionary documents every attribute across the raw synthetic datasets "
        "(`users_10000.csv`, `login_events_10000.csv`, `transactions_10000.csv`), the "
        "model-ready feature tables (`model_ready_login.csv`, `train_login_model.csv`, "
        "`test_login_model.csv`), the engineered features added by the version-2 and "
        "precision/recall/deep-learning training scripts, the hybrid/alternate-approach "
        "signature signals, and the common scoring output columns.",
        "",
        "Source scripts: `generate_fraud_datasets.py`, `build_model_ready_dataset.py`, "
        "`train_fraud_model.py`, `train_fraud_model_v2.py`, "
        "`train_precision_recall_models.py`, `train_deep_learning_models.py`, and "
        "`alternate_approach/run_hybrid_security_model.py`.",
        "",
    ]
    for dataset, group in df.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Attribute | Data Type | Description | Example Values | Source / Derivation |")
        lines.append("|---|---|---|---|---|")
        for _, row in group.iterrows():
            lines.append(
                f"| {row['attribute']} | {row['data_type']} | {row['description']} "
                f"| {row['example_values']} | {row['source_or_derivation']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    df = pd.DataFrame(RECORDS)
    df.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(build_markdown(df), encoding="utf-8")
    print(f"Created {CSV_PATH} with {len(df)} attribute entries.")
    print(f"Created {MD_PATH}")


if __name__ == "__main__":
    main()
