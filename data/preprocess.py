import os
import numpy as np
from PIL import Image


# folder_path = "hbku2019/imgs/imgs"
#
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         if file.lower().endswith('.jpg'):
#             file_path = os.path.join(root, file)
#
#             img = Image.open(file_path)
#             img_np  = np.array(img)
#             if len(img_np .shape) == 2:
#                 img_np_rgb = np.stack([img_np] * 3, axis=-1)
#
#                 img_rgb = Image.fromarray(img_np_rgb)
#                 img_rgb.save(file_path)
#                 print(f"Converted and saved: {file_path}")

import pandas as pd

df = pd.read_csv(os.path.join('hbku2019', 'labels/labels/labels_train.csv'), header=None)
img_paths = df.iloc[:, 0].tolist()
labels = df.iloc[:, 1:].to_numpy()
count = {i:0 for i in range(80)}

for ii in range(len(labels)):
    for i in range(len(labels[ii])):
        if labels[ii][i]==1:
            count[i]+=1
print(count)

