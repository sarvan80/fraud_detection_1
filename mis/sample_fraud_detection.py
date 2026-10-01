import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def generate_sample_data(n_samples=1000):
    """Generates a simple synthetic dataset for fraud detection."""
    np.random.seed(42)

    data = {
        'amount': np.random.uniform(1, 1000, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'is_international': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'failed_attempts': np.random.randint(0, 5, n_samples),
        'is_fraud': 0
    }

    df = pd.DataFrame(data)

    # Create some fraud patterns
    # Pattern 1: High amount + international
    fraud_idx1 = df[(df['amount'] > 800) & (df['is_international'] == 1)].index
    # Pattern 2: High failed attempts
    fraud_idx2 = df[df['failed_attempts'] > 3].index

    fraud_indices = np.unique(np.concatenate([fraud_idx1, fraud_idx2]))
    df.loc[fraud_indices, 'is_fraud'] = 1

    # Add some noise to fraud (some legitimate transactions marked as fraud)
    noise_idx = df[df['is_fraud'] == 0].sample(frac=0.05).index
    df.loc[noise_idx, 'is_fraud'] = 1

    return df

def main():
    print("--- Generating Sample Data ---")
    df = generate_sample_data(1000)
    print(df.head())
    print("\nFraud distribution:\n", df['is_fraud'].value_counts())

    print("\n--- Preparing Data ---")
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Training Model (Random Forest) ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n--- Feature Importance ---")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances)

if __name__ == "__main__":
    main()

