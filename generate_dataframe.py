import pandas as pd

df = pd.read_csv("deepfashion2/val.csv", index_col=0)
groupby_df = df.groupby(by="pair_id")

labels = []
for group, frame in groupby_df:
    count = 0
    for index, row in frame.iterrows():
        if row["style"] == 0:
            label = f"{group}_no{count}"
            count += 1
        else:
            label = f"{group}_{row['style']}"
        labels.append(label)

df["label"] = labels

df.to_csv("deepfashion2/val_new_label.csv", index=False)
