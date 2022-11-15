import pandas as pd

file = "data/train_new_label.csv"
with open(file, 'r') as f:
    df = pd.read_csv(f)

print(len(df))

# seed = 1
# samples = df.sample(n=5000, random_state=1)
# print(len(samples))
# print(samples.head())
# out_path = "data/val_new_label_small.csv"
# samples.to_csv(out_path)
# print(f"File {file} sampled to smaller file saved at {out_path}")