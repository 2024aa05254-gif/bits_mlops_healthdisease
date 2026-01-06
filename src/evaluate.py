import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/raw/heart.csv"


def evaluate():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    return metrics


if __name__ == "__main__":
    print(evaluate())
