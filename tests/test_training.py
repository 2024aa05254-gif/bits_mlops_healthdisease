import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.preprocess import get_preprocessing_pipeline

def test_model_pipeline_runs():
    numeric_features = ['age', 'trestbps']
    categorical_features = ['sex', 'cp']

    df = pd.DataFrame({
        'age': [45, 50, 60],
        'trestbps': [120, 130, 140],
        'sex': ['M', 'F', 'M'],
        'cp': ['typical', 'asymptomatic', 'non-anginal'],
        'target': [1, 0, 1]
    })

    X = df.drop('target', axis=1)
    y = df['target']

    pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline(numeric_features, categorical_features)),
        ('classifier', LogisticRegression(max_iter=100))
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
