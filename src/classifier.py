import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET_PATH = "data/processed/traffic_dataset.csv"
MODEL_PATH = "models/traffic_classifier.pkl"


def train_classifier():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:", df.shape)

    # Features used by the classifier
    features = [
        "protocol",
        "packet_count",
        "total_bytes",
        "duration",
        "average_packet_size",
        "packet_rate",
        "byte_rate",
        "average_iat"
    ]

    X = df[features]
    y = df["traffic_class"]

    print("\nFeatures:")
    print(features)

    print("\nClasses:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL RESULTS")
    print("=" * 50)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance)

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "accuracy": accuracy
        },
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_classifier()