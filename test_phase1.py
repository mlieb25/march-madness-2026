import pandas as pd
train = pd.read_csv("data/ml_training_data.csv")
inf = pd.read_csv("data/ml_inference_data_2026.csv")
print("Training columns:")
print(train.columns.tolist())
print("\nInference columns:")
print(inf.columns.tolist())
