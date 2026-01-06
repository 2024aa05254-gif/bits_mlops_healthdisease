from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def get_pipeline():
    """
    Returns a sklearn pipeline for Logistic Regression
    with median imputation and standard scaling.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # handle NaNs
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    return pipeline
