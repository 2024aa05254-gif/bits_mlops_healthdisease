import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import get_pipeline, get_rf_pipeline
from src.mlflow_utils import log_experiment

# Paths
DATA_PATH = "data/raw/heart.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # Load & clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set MLflow experiment
    mlflow.set_experiment("Heart Disease Classification")

    # --- Logistic Regression ---
    lr_pipeline = get_pipeline()
    with mlflow.start_run(run_name="LogisticRegression_Pipeline"):
        lr_pipeline.fit(X_train, y_train)
        y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        log_experiment(
            model=lr_pipeline,
            params={"model": "LogisticRegression"},
            metrics={"roc_auc": roc_auc},
            model_name="logistic_model"
        )

        joblib.dump(lr_pipeline, os.path.join(MODEL_DIR, "logistic_model.pkl"))
        print(f"Logistic Regression model saved at {os.path.join(MODEL_DIR, 'logistic_model.pkl')}")

    # --- Random Forest ---
    rf_pipeline = get_rf_pipeline()
    with mlflow.start_run(run_name="RandomForest_Pipeline"):
        rf_pipeline.fit(X_train, y_train)
        y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        log_experiment(
            model=rf_pipeline,
            params={"model": "RandomForest"},
            metrics={"roc_auc": roc_auc},
            model_name="rf_model"
        )

        joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, "rf_model.pkl"))
        print(f"Random Forest model saved at {os.path.join(MODEL_DIR, 'rf_model.pkl')}")


if __name__ == "__main__":
    main()
