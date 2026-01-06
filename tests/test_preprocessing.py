from src.data_preprocessing import load_data

def test_data_load():
    df = load_data("data/raw/heart.csv")
    assert df.shape[0] > 0
