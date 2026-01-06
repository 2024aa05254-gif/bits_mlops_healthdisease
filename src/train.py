import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import get_pipeline
from src.mlflow_utils import log_experiment


DATA_PATH = "data/raw/heart.csv"
MODEL_PATH = "models/model.pkl"


def main():
    # Load & clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = get_pipeline()

    mlflow.set_experiment("Heart Disease Classification")

    with mlflow.start_run(run_name="LogisticRegression_Pipeline"):
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        log_experiment(
            model=pipeline,
            params={"model": "LogisticRegression"},
            metrics={"roc_auc": roc_auc},
            model_name="model"
        )

        # Save final model
        joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    main()
