import pandas as pd
from sklearn.pipeline import Pipeline
from src.preprocess import get_preprocessing_pipeline


def test_preprocessing_pipeline():
    numeric_features = ['age', 'trestbps']
    categorical_features = ['sex', 'cp']

    df = pd.DataFrame({
        'age': [45, 50, None],
        'trestbps': [120, None, 130],
        'sex': ['M', 'F', 'M'],
        'cp': ['typical', None, 'asymptomatic']
    })

    pipeline = get_preprocessing_pipeline(
        numeric_features, categorical_features)
    X_transformed = pipeline.fit_transform(df)

    # Check transformed output shape
    assert X_transformed.shape[0] == 3
