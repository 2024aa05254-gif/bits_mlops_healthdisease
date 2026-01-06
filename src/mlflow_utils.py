import mlflow
import mlflow.sklearn


def log_experiment(
    model,
    params: dict,
    metrics: dict,
    model_name: str
):
    """
    Log parameters, metrics and model to MLflow
    """
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, artifact_path=model_name)
