import pandas as pd

train = pd.read_csv("data/ml_training_data.csv")
print("=== ml_training_data.csv ===")
print(train.shape)
print(train.head())

inf = pd.read_csv("data/ml_inference_data_2026.csv")
print("\n=== ml_inference_data_2026.csv ===")
print(inf.shape)
print(inf.head())
