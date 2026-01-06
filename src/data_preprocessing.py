import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values
    """
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df
