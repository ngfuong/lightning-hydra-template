import pandas as pd

df = pd.read_csv("deepfashion2/val.csv", index_col=0)
df = df.loc[df["style"]]
labels = []
frame_lens = []
count = 0
for _, row in df.iterrows():
    if row["style"] == 0:
        label = f"{row['pair_id']}_no{count}"
        count += 1
    else:
        label = f"{row['pair_id']}_{row['style']}"
    labels.append(label)

# for group, frame in groupby_df:
#     count = 0
#     print(group, len(frame))
#     frame_lens.append(len(frame))
#     for index, row in frame.iterrows():
#         if row["style"] == 0:
#             label = f"{group}_no{count}"
#             count += 1
#         else:
#             label = f"{group}_{row['style']}"
#         labels.append(label)

df["label"] = labels
# print(max(frame_lens))
df.to_csv("deepfashion2/val_new_label.csv", index=False)
