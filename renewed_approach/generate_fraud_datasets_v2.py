from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
N = 10_000

COUNTRIES = ["US", "CA", "GB", "IN", "DE", "AU", "FR", "BR"]
CITY_OPTIONS = ["New York", "Toronto", "London", "Bengaluru", "Berlin", "Sydney", "Paris", "Sao Paulo"]
REGION_BY_COUNTRY = {
    "US": "NA",
    "CA": "NA",
    "GB": "EU",
    "IN": "APAC",
    "DE": "EU",
    "AU": "APAC",
    "FR": "EU",
    "BR": "LATAM",
}
LOCALE_BY_COUNTRY = {
    "US": ("en-US", -5),
    "CA": ("en-CA", -5),
    "GB": ("en-GB", 0),
    "IN": ("en-IN", 5.5),
    "DE": ("de-DE", 1),
    "AU": ("en-AU", 10),
    "FR": ("fr-FR", 1),
    "BR": ("pt-BR", -3),
}
COUNTRY_CENTERS = {
    "US": (39.5, -98.3),
    "CA": (56.1, -106.3),
    "GB": (54.5, -3.4),
    "IN": (22.9, 78.9),
    "DE": (51.2, 10.4),
    "AU": (-25.2, 133.8),
    "FR": (46.2, 2.2),
    "BR": (-14.2, -51.9),
}
OS_OPTIONS = ["Windows", "macOS", "Linux", "iOS", "Android"]
BROWSER_BY_OS = {
    "Windows": ["Chrome", "Edge", "Firefox"],
    "macOS": ["Chrome", "Safari", "Firefox"],
    "Linux": ["Chrome", "Firefox"],
    "iOS": ["Safari", "Chrome"],
    "Android": ["Chrome", "Firefox"],
}
USER_AGENT_BY_BROWSER = {
    "Chrome": "Chrome/124",
    "Safari": "Safari/17",
    "Firefox": "Firefox/126",
    "Edge": "Edge/124",
}
GPU_BY_OS = {
    "Windows": ["NVIDIA GTX 1650", "Intel Iris Xe", "AMD Radeon RX 6600"],
    "macOS": ["Apple M2 GPU", "Apple M3 GPU"],
    "Linux": ["Mesa Intel UHD", "NVIDIA Tesla T4", "AMD Radeon Pro"],
    "iOS": ["Apple A16 GPU", "Apple A17 GPU"],
    "Android": ["Adreno 730", "Mali-G710", "Adreno 740"],
}
PASSWORD_RESET_CHANNELS = ["email_link", "SMS_OTP", "security_questions", "support_assisted"]
MFA_METHODS = ["none", "SMS_OTP", "app_OTP", "push", "hardware_key"]
PAYMENT_METHODS = ["ACH", "Debit", "Wire", "Card"]
ANONYMIZER_TYPES = ["none", "proxy", "VPN", "Tor", "smart_DNS"]
ISP_TYPES = ["consumer", "business", "mobile", "hosting", "anonymizer_heavy"]
SCENARIO_WEIGHTS = [97, 1.2, 0.8, 0.7, 0.6, 0.4, 0.3]
SCENARIOS = [
    "normal",
    "credential_stuffing",
    "impossible_travel",
    "new_device",
    "mfa_fatigue",
    "proxy",
    "session_hijack",
]
SCENARIO_BASE = {
    "normal": 0.0025,
    "credential_stuffing": 0.52,
    "impossible_travel": 0.35,
    "new_device": 0.42,
    "mfa_fatigue": 0.54,
    "proxy": 0.32,
    "session_hijack": 0.60,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def chance(probability: float) -> int:
    return int(random.random() < clamp(probability, 0.0, 1.0))


def bounded_normal(mean: float, stddev: float, low: float, high: float, digits: int = 2) -> float:
    return round(clamp(random.gauss(mean, stddev), low, high), digits)


def poisson_int(lam: float, high: int) -> int:
    return int(min(np.random.poisson(max(lam, 0.01)), high))


def exp_recency(days_since: float) -> float:
    return round(math.exp(-days_since / 30.0), 4)


def country_coordinates(country: str, home_lat: float | None = None, home_lon: float | None = None, far: bool = False) -> tuple[float, float]:
    lat_center, lon_center = COUNTRY_CENTERS[country]
    if far or home_lat is None or home_lon is None:
        return round(random.gauss(lat_center, 5.5), 4), round(random.gauss(lon_center, 8.0), 4)
    return round(home_lat + random.uniform(-4.5, 4.5), 4), round(home_lon + random.uniform(-5.0, 5.0), 4)


def geo_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.32


def random_hash(prefix: str, width: int = 6) -> str:
    return f"{prefix}_{random.randint(10 ** (width - 1), (10 ** width) - 1)}"


def scenario_profile(scenario: str) -> dict[str, int]:
    return {
        "device_attack": int(scenario in {"new_device", "session_hijack"}),
        "identity_attack": int(scenario in {"credential_stuffing", "mfa_fatigue", "new_device", "session_hijack"}),
        "network_attack": int(scenario in {"credential_stuffing", "impossible_travel", "proxy", "session_hijack"}),
        "automation_attack": int(scenario in {"credential_stuffing", "proxy"}),
        "takeover_attack": int(scenario in {"new_device", "mfa_fatigue", "session_hijack"}),
    }


users = []
profiles = []
for i in range(1, N + 1):
    country = random.choice(COUNTRIES)
    city = random.choice(CITY_OPTIONS)
    signup = datetime(2019, 1, 1) + timedelta(days=random.randint(0, 2000))
    email_domain = random.choice(["gmail.com", "outlook.com", "yahoo.com", "company.com"])
    preferred_os = random.choices(OS_OPTIONS, weights=[32, 18, 8, 18, 24], k=1)[0]
    preferred_browser = random.choice(BROWSER_BY_OS[preferred_os])
    locale, timezone_offset = LOCALE_BY_COUNTRY[country]
    usual_asn = f"AS{random.randint(1000, 9999)}"
    asn_type = random.choices(ISP_TYPES, weights=[58, 10, 18, 10, 4], k=1)[0]
    usual_ja3 = random_hash("ja3")
    usual_tcp = random_hash("tcp")
    usual_gpu = random.choice(GPU_BY_OS[preferred_os])
    usual_channel = random.choices(["web", "mobile_app", "desktop_app"], weights=[46, 42, 12], k=1)[0]

    users.append({
        "user_id": f"U{i:05d}",
        "email_domain": email_domain,
        "account_age_days": random.randint(30, 3000),
        "country": country,
        "home_city": city,
        "home_lat": round(random.uniform(20, 60), 4),
        "home_lon": round(random.uniform(-150, 50), 4),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "has_mfa": random.choice([True, False]),
        "risk_tier": random.choices(["low", "medium", "high"], weights=[70, 22, 8], k=1)[0],
    })
    profiles.append({
        "preferred_os": preferred_os,
        "preferred_browser": preferred_browser,
        "usual_user_agent": USER_AGENT_BY_BROWSER[preferred_browser],
        "locale": locale,
        "timezone_offset": timezone_offset,
        "usual_asn": usual_asn,
        "usual_asn_type": asn_type,
        "usual_ja3": usual_ja3,
        "usual_tcp": usual_tcp,
        "usual_gpu": usual_gpu,
        "usual_channel": usual_channel,
        "usual_resolver": f"resolver-{random.randint(1, 30)}",
    })

user_df = pd.DataFrame(users)
login_rows = []
user_enrichment_rows = []
transaction_rows = []

for i in range(1, N + 1):
    user = users[i - 1]
    profile = profiles[i - 1]
    user_id = user["user_id"]
    event_time = datetime(2025, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    base_ip = f"192.168.{random.randint(1, 250)}.{random.randint(1, 250)}"
    scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    sp = scenario_profile(scenario)

    if scenario == "normal":
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = random.choice([True, True, True, False])
        failed_logins = random.randint(0, 3)
        device_risk = random.randint(0, 40)
        session_risk = random.randint(0, 30)
        login_result = "success" if random.random() > 0.08 else "failed"
    elif scenario == "credential_stuffing":
        is_proxy = random.choice([True, False])
        is_vpn = random.choice([True, False])
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = False if random.random() < 0.8 else True
        failed_logins = random.randint(8, 25)
        device_risk = random.randint(35, 75)
        session_risk = random.randint(30, 70)
        login_result = "failed"
    elif scenario == "impossible_travel":
        is_proxy = False
        is_vpn = random.choice([False, True])
        new_device_flag = False
        is_impossible_travel = True
        mfa_approved = random.choice([True, False])
        failed_logins = random.randint(0, 5)
        device_risk = random.randint(20, 55)
        session_risk = random.randint(50, 82)
        login_result = random.choice(["success", "failed"])
    elif scenario == "new_device":
        is_proxy = False
        is_vpn = False
        new_device_flag = True
        is_impossible_travel = False
        mfa_approved = random.choice([False, True])
        failed_logins = random.randint(1, 7)
        device_risk = random.randint(55, 90)
        session_risk = random.randint(35, 75)
        login_result = random.choice(["success", "failed"])
    elif scenario == "mfa_fatigue":
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = False
        failed_logins = random.randint(0, 6)
        device_risk = random.randint(25, 60)
        session_risk = random.randint(58, 90)
        login_result = "failed"
    elif scenario == "proxy":
        is_proxy = True
        is_vpn = random.choice([False, True])
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = random.choice([True, False])
        failed_logins = random.randint(0, 4)
        device_risk = random.randint(30, 68)
        session_risk = random.randint(40, 80)
        login_result = random.choice(["success", "failed"])
    else:
        is_proxy = False
        is_vpn = False
        new_device_flag = False
        is_impossible_travel = False
        mfa_approved = True
        failed_logins = random.randint(0, 3)
        device_risk = random.randint(30, 70)
        session_risk = random.randint(70, 96)
        login_result = random.choice(["success", "failed"])

    login_risk = min(99, max(0, int((device_risk * 0.45) + (session_risk * 0.55) + (failed_logins * 2.5))))
    if is_proxy:
        login_risk += 10
    if is_vpn:
        login_risk += 8
    if new_device_flag:
        login_risk += 15
    if is_impossible_travel:
        login_risk += 18
    if not mfa_approved and scenario in ["mfa_fatigue", "credential_stuffing", "new_device"]:
        login_risk += 12
    login_risk = max(0, min(99, login_risk))

    is_fraud = 1 if random.random() < SCENARIO_BASE.get(scenario, 0.0025) else 0
    if login_result == "success" and random.random() < 0.004:
        is_fraud = 1
    if is_impossible_travel and random.random() < 0.05:
        is_fraud = 1

    fraud_type = None
    if is_fraud == 1:
        fraud_type = {
            "credential_stuffing": "credential_stuffing",
            "impossible_travel": "impossible_travel",
            "new_device": "device_spoofing",
            "mfa_fatigue": "mfa_fatigue",
            "proxy": "distributed_proxy",
            "session_hijack": "session_hijack",
        }.get(scenario, "unknown_anomaly")

    same_country_probability = {
        "normal": 0.88,
        "credential_stuffing": 0.52,
        "impossible_travel": 0.0,
        "new_device": 0.65,
        "mfa_fatigue": 0.80,
        "proxy": 0.35,
        "session_hijack": 0.45,
    }[scenario]
    if random.random() < same_country_probability:
        ip_country = user["country"]
        ip_lat, ip_lon = country_coordinates(ip_country, user["home_lat"], user["home_lon"], far=False)
    else:
        country_choices = [c for c in COUNTRIES if c != user["country"]]
        ip_country = random.choice(country_choices)
        ip_lat, ip_lon = country_coordinates(ip_country, far=True)
    ip_region = REGION_BY_COUNTRY[ip_country]
    distance_km = geo_distance_km(ip_lat, ip_lon, user["home_lat"], user["home_lon"])
    velocity_flag = int(distance_km > 2000)

    preferred_os = profile["preferred_os"]
    preferred_browser = profile["preferred_browser"]
    current_os = preferred_os if scenario not in {"new_device", "session_hijack"} or random.random() < 0.62 else random.choice(OS_OPTIONS)
    current_browser = preferred_browser if scenario not in {"new_device", "proxy", "session_hijack"} or random.random() < 0.60 else random.choice(BROWSER_BY_OS[current_os])
    current_user_agent = USER_AGENT_BY_BROWSER[current_browser]
    if scenario in {"proxy", "session_hijack"} and random.random() < 0.45:
        current_user_agent = random.choice(list(USER_AGENT_BY_BROWSER.values()))
    browser_version = f"{random.randint(100, 130)}.0"
    os_version = str(random.randint(10, 15))
    gpu = profile["usual_gpu"] if current_os == preferred_os and random.random() < 0.70 else random.choice(GPU_BY_OS[current_os])
    locale_novelty_for_user = chance(0.01 + 0.25 * (ip_country != user["country"]) + 0.20 * sp["device_attack"])
    device_locale = profile["locale"] if not locale_novelty_for_user else random.choice([value[0] for value in LOCALE_BY_COUNTRY.values()])

    is_new_asn_for_user = chance(0.02 + 0.48 * sp["network_attack"] + 0.25 * new_device_flag)
    current_asn_type = profile["usual_asn_type"] if not is_new_asn_for_user else random.choices(ISP_TYPES, weights=[20, 8, 12, 38, 22], k=1)[0]
    asn_for_ip = profile["usual_asn"] if not is_new_asn_for_user else f"AS{random.randint(1000, 9999)}"
    is_hosting_provider_ip = int(current_asn_type in {"hosting", "anonymizer_heavy"}) or chance(0.03 + 0.20 * sp["automation_attack"])
    is_tor_exit_node = chance(0.002 + 0.15 * int(scenario == "proxy") + 0.04 * int(scenario == "credential_stuffing"))
    anonymizer_type = "none"
    if is_tor_exit_node:
        anonymizer_type = "Tor"
    elif is_proxy:
        anonymizer_type = random.choice(["proxy", "smart_DNS"])
    elif is_vpn:
        anonymizer_type = "VPN"

    ip_reputation_score = round(clamp(
        random.gauss(18, 9)
        + 27 * sp["network_attack"]
        + 22 * sp["automation_attack"]
        + 10 * is_hosting_provider_ip
        + 12 * is_tor_exit_node,
        0,
        100,
    ), 2)
    ip_malicious_seen_recently_flag = chance(0.01 + 0.18 * sp["network_attack"] + 0.10 * sp["automation_attack"])
    ip_blacklist_count_30d = poisson_int(0.1 + 1.6 * sp["network_attack"] + 0.7 * is_hosting_provider_ip, 12)
    if ip_reputation_score < 25:
        ip_abuse_confidence_bucket = "low"
    elif ip_reputation_score < 50:
        ip_abuse_confidence_bucket = "medium"
    elif ip_reputation_score < 75:
        ip_abuse_confidence_bucket = "high"
    else:
        ip_abuse_confidence_bucket = "severe"
    new_country_for_user_flag = int(ip_country != user["country"])
    country_switch_velocity_7d = poisson_int(0.1 + 1.2 * sp["network_attack"] + 0.7 * int(is_vpn), 8)
    geo_velocity_risk_score = round(clamp(
        35 * is_impossible_travel
        + 25 * new_country_for_user_flag
        + min(distance_km / 120.0, 35)
        + random.uniform(0, 6),
        0,
        100,
    ), 2)
    residential_to_hosting_asn_switch_flag = int(profile["usual_asn_type"] == "consumer" and current_asn_type in {"hosting", "anonymizer_heavy"})
    isp_risk_tier = current_asn_type
    asn_account_concentration_30d = poisson_int(1.0 + 10.0 * is_hosting_provider_ip + 4.0 * sp["network_attack"], 60)
    anonymized_ip_persistence_7d = poisson_int(0.1 + 3.2 * int(anonymizer_type != "none") + 1.0 * sp["network_attack"], 20)
    ja3_hash = profile["usual_ja3"] if random.random() < (0.92 - 0.25 * sp["network_attack"]) else random_hash("ja3")
    ja3_novelty_for_user = int(ja3_hash != profile["usual_ja3"])
    tcp_stack_fingerprint = profile["usual_tcp"] if random.random() < (0.90 - 0.22 * sp["network_attack"]) else random_hash("tcp")
    network_fingerprint_mismatch_flag = chance(0.01 + 0.24 * sp["network_attack"] + 0.18 * sp["automation_attack"])
    tls_cipher_anomaly_score = round(clamp(
        random.gauss(14, 8)
        + 34 * sp["network_attack"]
        + 20 * sp["automation_attack"]
        + 10 * network_fingerprint_mismatch_flag,
        0,
        100,
    ), 2)
    ip_accounts_count_24h = poisson_int(1.2 + 18.0 * sp["automation_attack"] + 7.0 * is_hosting_provider_ip, 150)
    ip_new_account_burst_24h = poisson_int(0.1 + 4.0 * sp["automation_attack"] + 1.5 * sp["network_attack"], 40)
    ip_device_count_24h = poisson_int(1.0 + 16.0 * sp["automation_attack"] + 6.5 * is_hosting_provider_ip, 120)
    ip_household_consistency_score = round(clamp(
        random.gauss(82, 8)
        - 30 * new_country_for_user_flag
        - 18 * is_hosting_provider_ip
        - 12 * int(anonymizer_type != "none"),
        0,
        100,
    ), 2)
    session_rtt_outlier_flag = chance(0.03 + 0.16 * sp["network_attack"] + 0.10 * int(anonymizer_type != "none"))
    dns_resolver_change_flag = chance(0.02 + 0.18 * sp["network_attack"] + 0.12 * is_vpn)
    request_burst_rate_5m = round(clamp(
        random.gauss(5.0, 2.5)
        + 42 * int(scenario == "credential_stuffing")
        + 18 * int(scenario == "proxy")
        + 10 * int(scenario == "session_hijack"),
        0,
        120,
    ), 2)
    connection_reset_retry_ratio = round(clamp(
        random.gauss(0.08, 0.05)
        + 0.18 * sp["automation_attack"]
        + 0.10 * sp["network_attack"],
        0,
        1,
    ), 3)
    network_risk_score = round(clamp(
        0.40 * ip_reputation_score
        + 0.30 * geo_velocity_risk_score
        + 7.0 * int(anonymizer_type != "none")
        + 0.20 * ip_accounts_count_24h
        + 12.0 * network_fingerprint_mismatch_flag
        + random.uniform(0, 5),
        0,
        100,
    ), 2)
    proxy_travel_evasion_combo = int(is_impossible_travel and anonymizer_type != "none" and new_country_for_user_flag)

    device_id = f"D{random.randint(1000, 9999)}"
    device_fingerprint = random_hash("fp")
    new_devices_per_user_7d = poisson_int(0.1 + 1.8 * new_device_flag + 0.8 * sp["takeover_attack"], 10)
    device_days_since_last_seen = round(clamp(
        random.gauss(30, 20)
        + 240 * new_device_flag
        + 120 * int(scenario == "session_hijack")
        + 90 * int(scenario == "impossible_travel"),
        0,
        1500,
    ), 1)
    device_bound_after_mfa_flag = chance(0.72 if user["has_mfa"] else 0.18) if not new_device_flag else chance(0.18)
    new_device_to_sensitive_action_mins = round(clamp(
        random.gauss(360, 200)
        - 240 * new_device_flag
        - 120 * sp["takeover_attack"],
        3,
        1440,
    ), 1)
    fingerprint_population_rarity = round(clamp(
        random.gauss(22, 12)
        + 28 * sp["device_attack"]
        + 20 * sp["automation_attack"]
        + 10 * is_hosting_provider_ip,
        0,
        100,
    ), 2)
    fingerprint_similarity_to_fraud_devices = round(clamp(
        random.gauss(15, 10)
        + 25 * sp["automation_attack"]
        + 20 * sp["device_attack"]
        + 10 * int(scenario == "proxy"),
        0,
        100,
    ), 2)
    user_agent_consistency = chance(0.97 - 0.28 * sp["device_attack"] - 0.20 * sp["network_attack"])
    user_agent_change_for_device = chance(0.03 + 0.30 * sp["device_attack"] + 0.18 * sp["network_attack"])
    hardware_spec_consistency = chance(0.95 - 0.35 * sp["device_attack"] - 0.20 * sp["automation_attack"])
    os_version_downgrade_flag = chance(0.01 + 0.18 * sp["device_attack"])
    font_list_change_font_count_anomaly = round(clamp(
        random.gauss(10, 8)
        + 35 * sp["device_attack"]
        + 18 * sp["automation_attack"],
        0,
        100,
    ), 2)
    interaction_event_count_pre_submit = round(clamp(
        random.gauss(46, 13)
        - 32 * sp["automation_attack"]
        + 10 * int(scenario == "session_hijack"),
        0,
        120,
    ), 2)
    time_to_login_submit_ms = round(clamp(
        random.gauss(24000, 10000)
        - 18000 * int(scenario == "credential_stuffing")
        - 9000 * int(scenario == "proxy")
        - 5000 * int(scenario == "new_device")
        - 6000 * int(scenario == "session_hijack"),
        350,
        120000,
    ), 2)
    mouse_movement_entropy = round(clamp(
        random.gauss(0.72, 0.12)
        - 0.42 * int(scenario == "credential_stuffing")
        - 0.22 * int(scenario == "proxy")
        + 0.12 * int(scenario == "session_hijack"),
        0.02,
        1.20,
    ), 4)
    session_journey_anomaly = round(clamp(
        random.gauss(18, 10)
        + 28 * sp["takeover_attack"]
        + 16 * sp["network_attack"],
        0,
        100,
    ), 2)
    mfa_prompts_per_device_1h = poisson_int(0.2 + 8.0 * int(scenario == "mfa_fatigue") + 2.0 * int(scenario == "new_device"), 25)
    password_reset_from_new_device = int(new_device_flag and chance(0.05 + 0.45 * sp["identity_attack"]))
    device_spoofed = chance(0.01 + 0.30 * sp["device_attack"] + 0.10 * sp["network_attack"])
    device_emulator = chance(0.01 + 0.26 * sp["automation_attack"] + 0.08 * sp["device_attack"])
    device_tampered = chance(0.005 + 0.25 * int(scenario == "session_hijack") + 0.06 * sp["device_attack"])
    device_rooted_or_jailbroken = chance(0.03 + 0.10 * sp["takeover_attack"])
    device_risky_reputation = chance(0.02 + 0.24 * sp["network_attack"] + 0.10 * sp["device_attack"])
    suspicious_fingerprint = chance(0.02 + 0.25 * sp["device_attack"] + 0.14 * sp["automation_attack"])
    device_network_novelty_combo = int(new_device_flag and (is_new_asn_for_user or anonymizer_type != "none"))
    device_identity_confidence = round(clamp(
        92
        - 0.45 * fingerprint_population_rarity
        - 18 * new_device_flag
        - 10 * device_spoofed
        - 8 * suspicious_fingerprint
        - 8 * is_new_asn_for_user
        + 10 * device_bound_after_mfa_flag
        + random.uniform(-4, 4),
        0,
        100,
    ), 2)
    device_location_risk_score = round(clamp(
        26 * new_country_for_user_flag
        + 0.02 * distance_km
        + 7 * abs(profile["timezone_offset"] - LOCALE_BY_COUNTRY[ip_country][1])
        + 10 * is_new_asn_for_user
        + random.uniform(0, 6),
        0,
        100,
    ), 2)
    device_novelty_score = round(clamp(
        42 * new_device_flag
        + 0.18 * fingerprint_population_rarity
        + 14 * is_new_asn_for_user
        + 12 * new_country_for_user_flag
        + 9 * suspicious_fingerprint
        + random.uniform(0, 6),
        0,
        100,
    ), 2)
    bot_likelihood_score = round(clamp(
        0.28 * (100 - min(interaction_event_count_pre_submit * 1.6, 100))
        + 0.24 * (100 - min(time_to_login_submit_ms / 300.0, 100))
        + 0.22 * (100 - min(mouse_movement_entropy * 100, 100))
        + 0.18 * min(request_burst_rate_5m, 100)
        + 8 * device_emulator,
        0,
        100,
    ), 2)

    dormant_account_reactivation_flag = chance(0.02 + 0.18 * int(scenario in {"impossible_travel", "session_hijack"}))
    email_domain_type = "corporate" if user["email_domain"] == "company.com" else random.choices(["free", "disposable"], weights=[97, 3], k=1)[0]
    email_change_frequency_90d = poisson_int(0.1 + 1.3 * sp["identity_attack"], 8)
    phone_change_frequency_90d = poisson_int(0.1 + 1.0 * sp["identity_attack"] + 1.1 * int(scenario == "mfa_fatigue"), 8)
    identity_mismatch_flag = chance(0.01 + 0.20 * sp["takeover_attack"])
    account_ownership_change_flag = chance(0.005 + 0.08 * int(scenario == "session_hijack"))
    password_reset_freq_30d = poisson_int(0.2 + 1.8 * sp["identity_attack"] + 1.1 * int(scenario == "credential_stuffing"), 12)
    if scenario == "credential_stuffing":
        password_reset_channel = random.choices(PASSWORD_RESET_CHANNELS, weights=[28, 26, 26, 20], k=1)[0]
    elif scenario == "mfa_fatigue":
        password_reset_channel = random.choices(PASSWORD_RESET_CHANNELS, weights=[20, 36, 12, 32], k=1)[0]
    else:
        password_reset_channel = random.choices(PASSWORD_RESET_CHANNELS, weights=[52, 22, 8, 18], k=1)[0]
    security_question_change_flag = chance(0.01 + 0.12 * sp["identity_attack"])
    mfa_enrolled_flag = int(user["has_mfa"])
    if not mfa_enrolled_flag:
        mfa_method_type = "none"
    elif scenario == "mfa_fatigue":
        mfa_method_type = "push"
    elif chance(0.12 + 0.14 * int(scenario == "proxy")):
        mfa_method_type = "SMS_OTP"
    else:
        mfa_method_type = random.choices(["app_OTP", "push", "hardware_key", "SMS_OTP"], weights=[34, 34, 14, 18], k=1)[0]
    mfa_reset_frequency_90d = poisson_int(0.1 + 1.1 * sp["identity_attack"], 6)
    account_recovery_frequency_90d = poisson_int(0.1 + 1.6 * sp["identity_attack"], 8)
    beneficiary_or_linked_payee_change_flag = chance(0.03 + 0.18 * sp["takeover_attack"])
    credential_leak_match_flag = chance(0.02 + 0.30 * int(scenario == "credential_stuffing"))
    prior_ato_incident_flag = chance(0.02 + 0.04 * int(user["risk_tier"] == "high") + 0.08 * sp["takeover_attack"])
    is_dormant_reactivated_high_risk = int(dormant_account_reactivation_flag and user["account_age_days"] > 180)
    contact_change_count_7d = min(email_change_frequency_90d + phone_change_frequency_90d + poisson_int(0.1 + 0.4 * sp["identity_attack"], 4), 12)
    email_phone_change_same_session_flag = int(contact_change_count_7d >= 2 and chance(0.03 + 0.22 * sp["takeover_attack"]))
    disposable_email_risk_flag = int(email_domain_type == "disposable")
    kyc_identity_mismatch_recent_flag = int(identity_mismatch_flag and chance(0.65))
    ownership_change_days = random.randint(1, 60) if account_ownership_change_flag else random.randint(90, 365)
    ownership_change_recency_score = exp_recency(ownership_change_days) if account_ownership_change_flag else 0.0
    password_reset_velocity_7d_30d = round(password_reset_freq_30d * 0.55 + poisson_int(0.1 + 0.5 * sp["identity_attack"], 5), 2)
    security_question_reset_then_password_reset_flag = int(security_question_change_flag and password_reset_freq_30d > 0 and chance(0.35 + 0.25 * sp["identity_attack"]))
    mfa_downgrade_flag = chance(0.01 + 0.12 * int(mfa_method_type == "SMS_OTP") + 0.08 * int(scenario in {"proxy", "mfa_fatigue"}))
    mfa_disabled_recent_flag = int(mfa_enrolled_flag and chance(0.01 + 0.08 * int(scenario == "session_hijack")))
    no_mfa_high_tenure_flag = int((not mfa_enrolled_flag) and user["account_age_days"] > 180)
    recovery_failure_rate_30d = round(clamp(
        random.gauss(0.08, 0.07)
        + 0.28 * sp["identity_attack"]
        + 0.10 * int(scenario == "credential_stuffing"),
        0,
        1,
    ), 3)
    recovery_to_contact_change_flag = int(account_recovery_frequency_90d > 0 and contact_change_count_7d > 0 and chance(0.05 + 0.24 * sp["takeover_attack"]))
    beneficiary_change_after_recovery_flag = int(beneficiary_or_linked_payee_change_flag and account_recovery_frequency_90d > 0 and chance(0.06 + 0.26 * sp["takeover_attack"]))
    breach_exposure_recency_flag = int(credential_leak_match_flag and chance(0.62))
    cumulative_identity_risk_index = round(clamp(
        18 * disposable_email_risk_flag
        + 5.5 * contact_change_count_7d
        + 4.5 * password_reset_freq_30d
        + 6.0 * mfa_reset_frequency_90d
        + 22.0 * recovery_failure_rate_30d
        + 12.0 * breach_exposure_recency_flag
        + 12.0 * kyc_identity_mismatch_recent_flag
        + random.uniform(0, 5),
        0,
        100,
    ), 2)
    identity_event_cluster_score = min(
        int(password_reset_freq_30d > 0)
        + int(contact_change_count_7d > 0)
        + int(mfa_reset_frequency_90d > 0)
        + int(account_recovery_frequency_90d > 0)
        + int(beneficiary_or_linked_payee_change_flag > 0)
        + int(identity_mismatch_flag > 0),
        6,
    )
    password_reset_from_new_device_or_location = int(password_reset_freq_30d > 0 and (new_device_flag or new_country_for_user_flag or is_new_asn_for_user))
    login_after_password_reset_mins = round(clamp(
        random.gauss(720, 260)
        - 430 * sp["identity_attack"]
        - 90 * new_device_flag,
        2,
        2880,
    ), 1)
    failed_logins_before_reset_24h = min(failed_logins + poisson_int(0.2 + 4.5 * int(scenario == "credential_stuffing"), 25), 25)
    credential_leak_then_login_new_device_flag = int(breach_exposure_recency_flag and new_device_flag)
    mfa_enrollment_from_new_device_flag = int(mfa_reset_frequency_90d > 0 and new_device_flag and chance(0.12 + 0.25 * sp["takeover_attack"]))
    mfa_push_denied_then_approved_flag = int(scenario == "mfa_fatigue" and chance(0.58))
    mfa_prompt_count_1h = poisson_int(0.2 + 9.5 * int(scenario == "mfa_fatigue") + 1.0 * int(scenario == "new_device"), 25)
    phone_change_then_otp_login_flag = int(phone_change_frequency_90d > 0 and mfa_method_type == "SMS_OTP" and chance(0.04 + 0.24 * sp["identity_attack"]))
    sim_swap_recent_flag = chance(0.01 + 0.14 * int(mfa_method_type == "SMS_OTP") + 0.10 * int(scenario in {"proxy", "new_device"}))
    support_assisted_recovery_flag = int(password_reset_channel == "support_assisted")
    recovery_channel_mismatch_flag = int(account_recovery_frequency_90d > 0 and (contact_change_count_7d > 0 or not mfa_enrolled_flag) and chance(0.08 + 0.22 * sp["identity_attack"]))
    address_change_then_card_or_statement_request = chance(0.01 + 0.10 * int(scenario == "session_hijack"))
    linked_accounts_change_7d = poisson_int(0.05 + 0.8 * sp["takeover_attack"], 6)
    days_since_last_successful_login = round(clamp(
        random.gauss(11, 15)
        + 42 * dormant_account_reactivation_flag
        + 16 * int(scenario == "credential_stuffing"),
        0,
        365,
    ), 1)
    login_outside_usual_channel_flag = chance(0.03 + 0.18 * sp["device_attack"] + 0.10 * sp["network_attack"])
    identity_change_on_new_device_combo = int(identity_event_cluster_score >= 2 and (new_device_flag or is_new_asn_for_user))

    amount = round(random.uniform(5, 6000), 2)
    if is_fraud == 1:
        amount = round(random.uniform(100, 25000), 2)
    txn_time = event_time + timedelta(minutes=random.randint(1, 180))
    first_time_high_value_transfer_after_profile_change = int(
        amount > 6000
        and (contact_change_count_7d > 0 or beneficiary_or_linked_payee_change_flag)
        and chance(0.08 + 0.28 * sp["takeover_attack"])
    )
    new_payee_to_transfer_mins = round(clamp(
        random.gauss(880, 340)
        - 700 * sp["takeover_attack"]
        - 120 * beneficiary_or_linked_payee_change_flag,
        1,
        10080,
    ), 1)
    ato_sequence_score = int(
        new_device_flag
        + int(password_reset_freq_30d > 0 or account_recovery_frequency_90d > 0)
        + int(contact_change_count_7d > 0 or mfa_reset_frequency_90d > 0)
        + int(beneficiary_or_linked_payee_change_flag or first_time_high_value_transfer_after_profile_change)
    )
    network_identity_linkage_risk = round(clamp(
        18 * is_new_asn_for_user
        + 0.55 * request_burst_rate_5m
        + 10 * identity_event_cluster_score
        + 8 * identity_change_on_new_device_combo
        + random.uniform(0, 4),
        0,
        100,
    ), 2)

    login_rows.append({
        "login_id": f"L{i:05d}",
        "user_id": user_id,
        "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": device_id,
        "device_fingerprint": device_fingerprint,
        "ip_address": base_ip,
        "ip_country": ip_country,
        "ip_region": ip_region,
        "ip_lat": ip_lat,
        "ip_lon": ip_lon,
        "user_agent": current_user_agent,
        "browser_name": current_browser,
        "browser_version": browser_version,
        "os_name": current_os,
        "os_version": os_version,
        "country": user["country"],
        "home_city": user["home_city"],
        "new_device_flag": int(new_device_flag),
        "is_proxy": int(is_proxy),
        "is_vpn": int(is_vpn),
        "impossible_travel_flag": int(is_impossible_travel),
        "failed_login_count_24h": failed_logins,
        "mfa_required": int(random.choice([True, True, True, False])),
        "mfa_approved": int(mfa_approved),
        "login_result": login_result,
        "session_risk_score": session_risk,
        "device_risk_score": device_risk,
        "login_risk_score": login_risk,
        "is_fraud": int(is_fraud),
        "fraud_type": fraud_type,
        "fraud_score": round(login_risk / 100, 2),
        "typing_velocity": round(random.uniform(20, 120), 2),
        "mouse_velocity": round(random.uniform(30, 180), 2),
        "click_count": random.randint(0, 40),
        "keystroke_count": random.randint(0, 300),
        "device_novelty_score": device_novelty_score,
        "device_location_risk_score": device_location_risk_score,
        "device_identity_confidence": device_identity_confidence,
        "new_devices_per_user_7d": new_devices_per_user_7d,
        "device_days_since_last_seen": device_days_since_last_seen,
        "device_bound_after_mfa_flag": int(device_bound_after_mfa_flag),
        "new_device_to_sensitive_action_mins": new_device_to_sensitive_action_mins,
        "fingerprint_population_rarity": fingerprint_population_rarity,
        "fingerprint_similarity_to_fraud_devices": fingerprint_similarity_to_fraud_devices,
        "user_agent_consistency": int(user_agent_consistency),
        "user_agent_change_for_device": int(user_agent_change_for_device),
        "hardware_spec_consistency": int(hardware_spec_consistency),
        "os_version_downgrade_flag": int(os_version_downgrade_flag),
        "font_list_change_font_count_anomaly": font_list_change_font_count_anomaly,
        "locale_novelty_for_user": int(locale_novelty_for_user),
        "interaction_event_count_pre_submit": interaction_event_count_pre_submit,
        "time_to_login_submit_ms": time_to_login_submit_ms,
        "mouse_movement_entropy": mouse_movement_entropy,
        "session_journey_anomaly": session_journey_anomaly,
        "mfa_prompts_per_device_1h": mfa_prompts_per_device_1h,
        "password_reset_from_new_device": int(password_reset_from_new_device),
        "device_network_novelty_combo": int(device_network_novelty_combo),
        "bot_likelihood_score": bot_likelihood_score,
        "gpu": gpu,
        "device_spoofed": int(device_spoofed),
        "device_emulator": int(device_emulator),
        "device_tampered": int(device_tampered),
        "device_rooted_or_jailbroken": int(device_rooted_or_jailbroken),
        "device_risky_reputation": int(device_risky_reputation),
        "suspicious_fingerprint": int(suspicious_fingerprint),
        "ip_reputation_score": ip_reputation_score,
        "ip_malicious_seen_recently_flag": int(ip_malicious_seen_recently_flag),
        "ip_blacklist_count_30d": ip_blacklist_count_30d,
        "ip_abuse_confidence_bucket": ip_abuse_confidence_bucket,
        "is_hosting_provider_ip": int(is_hosting_provider_ip),
        "ip_geo_country": ip_country,
        "ip_geo_distance_from_usual_km": round(distance_km, 2),
        "new_country_for_user_flag": int(new_country_for_user_flag),
        "country_switch_velocity_7d": country_switch_velocity_7d,
        "geo_velocity_risk_score": geo_velocity_risk_score,
        "asn_for_ip": asn_for_ip,
        "is_new_asn_for_user": int(is_new_asn_for_user),
        "residential_to_hosting_asn_switch_flag": int(residential_to_hosting_asn_switch_flag),
        "isp_risk_tier": isp_risk_tier,
        "asn_account_concentration_30d": asn_account_concentration_30d,
        "is_tor_exit_node": int(is_tor_exit_node),
        "anonymizer_type": anonymizer_type,
        "anonymized_ip_persistence_7d": anonymized_ip_persistence_7d,
        "ja3_hash": ja3_hash,
        "ja3_novelty_for_user": int(ja3_novelty_for_user),
        "tcp_stack_fingerprint": tcp_stack_fingerprint,
        "network_fingerprint_mismatch_flag": int(network_fingerprint_mismatch_flag),
        "tls_cipher_anomaly_score": tls_cipher_anomaly_score,
        "ip_accounts_count_24h": ip_accounts_count_24h,
        "ip_new_account_burst_24h": ip_new_account_burst_24h,
        "ip_device_count_24h": ip_device_count_24h,
        "ip_household_consistency_score": ip_household_consistency_score,
        "session_rtt_outlier_flag": int(session_rtt_outlier_flag),
        "dns_resolver_change_flag": int(dns_resolver_change_flag),
        "request_burst_rate_5m": request_burst_rate_5m,
        "connection_reset_retry_ratio": connection_reset_retry_ratio,
        "network_risk_score": network_risk_score,
        "proxy_travel_evasion_combo": int(proxy_travel_evasion_combo),
        "network_identity_linkage_risk": network_identity_linkage_risk,
        "mfa_prompt_count_1h": mfa_prompt_count_1h,
        "failed_logins_before_reset_24h": failed_logins_before_reset_24h,
        "password_reset_from_new_device_or_location": int(password_reset_from_new_device_or_location),
        "login_after_password_reset_mins": login_after_password_reset_mins,
        "credential_leak_then_login_new_device_flag": int(credential_leak_then_login_new_device_flag),
        "mfa_enrollment_from_new_device_flag": int(mfa_enrollment_from_new_device_flag),
        "mfa_push_denied_then_approved_flag": int(mfa_push_denied_then_approved_flag),
        "phone_change_then_otp_login_flag": int(phone_change_then_otp_login_flag),
        "days_since_last_successful_login": days_since_last_successful_login,
        "login_outside_usual_channel_flag": int(login_outside_usual_channel_flag),
        "identity_change_on_new_device_combo": int(identity_change_on_new_device_combo),
        "ato_sequence_score": ato_sequence_score,
    })

    user_enrichment_rows.append({
        "user_id": user_id,
        "email_domain_type": email_domain_type,
        "dormant_account_reactivation_flag": int(dormant_account_reactivation_flag),
        "email_change_frequency_90d": email_change_frequency_90d,
        "phone_change_frequency_90d": phone_change_frequency_90d,
        "identity_mismatch_flag": int(identity_mismatch_flag),
        "account_ownership_change_flag": int(account_ownership_change_flag),
        "password_reset_freq_30d": password_reset_freq_30d,
        "password_reset_channel": password_reset_channel,
        "security_question_change_flag": int(security_question_change_flag),
        "mfa_enrolled_flag": int(mfa_enrolled_flag),
        "mfa_method_type": mfa_method_type,
        "mfa_reset_frequency_90d": mfa_reset_frequency_90d,
        "account_recovery_frequency_90d": account_recovery_frequency_90d,
        "beneficiary_or_linked_payee_change_flag": int(beneficiary_or_linked_payee_change_flag),
        "credential_leak_match_flag": int(credential_leak_match_flag),
        "prior_ato_incident_flag": int(prior_ato_incident_flag),
        "is_dormant_reactivated_high_risk": int(is_dormant_reactivated_high_risk),
        "contact_change_count_7d": contact_change_count_7d,
        "email_phone_change_same_session_flag": int(email_phone_change_same_session_flag),
        "disposable_email_risk_flag": int(disposable_email_risk_flag),
        "kyc_identity_mismatch_recent_flag": int(kyc_identity_mismatch_recent_flag),
        "ownership_change_recency_score": ownership_change_recency_score,
        "password_reset_velocity_7d_30d": password_reset_velocity_7d_30d,
        "security_question_reset_then_password_reset_flag": int(security_question_reset_then_password_reset_flag),
        "mfa_downgrade_flag": int(mfa_downgrade_flag),
        "mfa_disabled_recent_flag": int(mfa_disabled_recent_flag),
        "no_mfa_high_tenure_flag": int(no_mfa_high_tenure_flag),
        "recovery_failure_rate_30d": recovery_failure_rate_30d,
        "recovery_to_contact_change_flag": int(recovery_to_contact_change_flag),
        "breach_exposure_recency_flag": int(breach_exposure_recency_flag),
        "cumulative_identity_risk_index": cumulative_identity_risk_index,
        "identity_event_cluster_score": identity_event_cluster_score,
        "sim_swap_recent_flag": int(sim_swap_recent_flag),
        "support_assisted_recovery_flag": int(support_assisted_recovery_flag),
        "recovery_channel_mismatch_flag": int(recovery_channel_mismatch_flag),
        "linked_accounts_change_7d": linked_accounts_change_7d,
    })

    transaction_rows.append({
        "txn_id": f"TXN{i:05d}",
        "user_id": user_id,
        "txn_time": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": device_id,
        "device_fingerprint": device_fingerprint,
        "ip_address": base_ip,
        "txn_amount": amount,
        "merchant_country": random.choice(["US", "CA", "GB", "IN", "DE", "FR"]),
        "payment_method": random.choice(PAYMENT_METHODS),
        "mfa_required": int(random.choice([True, True, False])),
        "mfa_approved": int(random.choice([True, True, False])),
        "txn_risk_score": min(99, max(10, int(login_risk + random.randint(0, 25)))),
        "is_fraud": int(is_fraud),
        "fraud_type": fraud_type,
        "merchant_risk_flag": int(random.random() < 0.12),
        "beneficiary_country_mismatch": int(random.random() < 0.08),
        "beneficiary_change_after_recovery_flag": int(beneficiary_change_after_recovery_flag),
        "address_change_then_card_or_statement_request": int(address_change_then_card_or_statement_request),
        "first_time_high_value_transfer_after_profile_change": int(first_time_high_value_transfer_after_profile_change),
        "new_payee_to_transfer_mins": new_payee_to_transfer_mins,
    })

user_enrichment_df = pd.DataFrame(user_enrichment_rows)
login_df = pd.DataFrame(login_rows)
transaction_df = pd.DataFrame(transaction_rows)
user_df = user_df.merge(user_enrichment_df, on="user_id", how="left")

user_path = BASE_DIR / "users_v2.csv"
login_path = BASE_DIR / "login_events_v2.csv"
transaction_path = BASE_DIR / "transactions_v2.csv"
summary_path = BASE_DIR / "dataset_generation_summary_v2.json"

user_df.to_csv(user_path, index=False)
login_df.to_csv(login_path, index=False)
transaction_df.to_csv(transaction_path, index=False)

summary = {
    "seed": SEED,
    "rows": {
        "users_v2": int(len(user_df)),
        "login_events_v2": int(len(login_df)),
        "transactions_v2": int(len(transaction_df)),
    },
    "fraud_rate_login": float(login_df["is_fraud"].mean()),
    "fraud_rate_transaction": float(transaction_df["is_fraud"].mean()),
    "new_feature_columns": {
        "users_v2": sorted([c for c in user_df.columns if c not in pd.read_csv(Path(__file__).resolve().parents[1] / "users_10000.csv", nrows=1).columns]),
        "login_events_v2": sorted([c for c in login_df.columns if c not in pd.read_csv(Path(__file__).resolve().parents[1] / "login_events_10000.csv", nrows=1).columns]),
        "transactions_v2": sorted([c for c in transaction_df.columns if c not in pd.read_csv(Path(__file__).resolve().parents[1] / "transactions_10000.csv", nrows=1).columns]),
    },
}
summary_path.write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

print("Generated renewed datasets:")
print(f"- {user_path.name}")
print(f"- {login_path.name}")
print(f"- {transaction_path.name}")
print(f"Fraud rate in login events: {login_df['is_fraud'].mean():.4f}")
