import os
import random
from typing import Any, Callable, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SampleDataset(Dataset):
    def __init__(self, root_dir, file_name):
        super(SampleDataset, self).__init__()

        self.root_dir = root_dir
        self.file_name = file_name
        self.samples = self.__read_xlsx()

    def __getitem__(self, index):
        samples = self.samples[index]
        # sample, target = samples[:-3],samples[-3:]

        return samples[1:], samples[0].long()

    def __len__(self):
        return len(self.samples)

    def __read_xlsx(self):
        f_pth = os.path.join(self.root_dir, self.file_name)
        # f_pth = os.path.join(root_dir, 'data.xlsx')
        df = pd.read_csv(f_pth,usecols=['body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price','make'])
        # ['body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price']
        # samples = torch.from_numpy(df.to_numpy()).float()
        # [make, body - style, wheel - base, engine - size, horsepower, peak - rpm, highway - mpg]
        # print(samples)
        # samples = df.to_dict(orient='list')


        samples = torch.from_numpy(df.to_numpy()).float()
        return samples


class CocoDataset(Dataset):
    def __init__(self, root, annFile,transform=None):
        super(CocoDataset, self).__init__()
        from pycocotools.coco import COCO
        self.root = root
        self.annFile = annFile
        self.coco = COCO(annFile)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cats_id_map = {self.cats[i]['id']:i for i in range(len(self.cats)) }
        self.cats_name = [cat['name'] for cat in self.cats]

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        # print(type(target))
        # print(target)
        all_cats = [self.cats_id_map[box['category_id']] for box in target]

        label = np.zeros(len(self.cats_name)).astype(np.float32)
        label[all_cats] = 1.0

        if self.transform is not None:
            image = self.transform(image)

        return image,label

    def __len__(self):
        return len(self.ids)




