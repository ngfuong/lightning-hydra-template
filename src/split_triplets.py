import pandas as pd
from sklearn.model_selection import train_test_split

file = "/cm/shared/phuongln6/data/deepfashion/triplets.csv"
with open(file, 'r') as f:
    df = pd.read_csv(f)



print(f"Original_df at {file} have length {len(df)}")
train_size=0.7
seed=42
og_df = df.copy()

print(og_df.columns)
"""
# In the first step we will split the data in training and remaining dataset
df_train, df_val = train_test_split(og_df, train_size=0.8, random_state=42)

print("Original df: ", og_df.shape)
print("Train df", df_train.shape)
print("Val df", df_val.shape)
print(df_train.head())
print(df_val.head())
train_path = "/cm/shared/phuongln6/data/deepfashion/train_triplets.csv"
val_path = "/cm/shared/phuongln6/data/deepfashion/val_triplets.csv"
df_train.to_csv(train_path)
df_val.to_csv(val_path)
print(f"Csv files saved to {train_path} and {val_path}")
"""