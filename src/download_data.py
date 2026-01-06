import pandas as pd
import os

# URL of the Cleveland dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names
columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Read dataset
df = pd.read_csv(url, header=None, names=columns, na_values='?')

# Convert target to binary (0 = no disease, 1 = disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Create raw data folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save to CSV
df.to_csv("data/raw/heart.csv", index=False)

print("Dataset downloaded and saved to data/raw/heart.csv ✅")
