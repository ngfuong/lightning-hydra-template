import os
import pandas as pd
import cv2
from PIL import Image
def pil_loader(path, bbox=None):
    """
    Function to read image and crop if have bounding box
    Args:
        path: Path to the image
        bbox: list bounding box consist of [xmin, ymin, xmax, ymax]

    Return:
        PIL image
    """
    IMG = Image.open(path)
    if bbox:
        IMG = IMG.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    return IMG.convert("RGB")

with open(
        os.path.join(f"val_query.json"),
        encoding="utf-8-sig",
    ) as f_input:
    df = pd.read_json(f_input)
print(df)
for _,row in df.iterrows():
    image_name = row['query_image_id']
    path = os.path.join('validation/image' ,f"{image_name:06d}.jpg")
    im = pil_loader(path, row["bbox"])
    im.save(f'test_query_demo/"{image_name:06d}.jpg')
