import os
import random

import io
import pandas as pd
from PIL import Image
import pyarrow as pa
import torch
from torchvision.datasets import CocoDetection, CocoCaptions
from tqdm import tqdm

from data.embedding import BertEmbedding


class CocoDet(CocoDetection):
    def __init__(self, **kwargs):
        super(CocoDet, self).__init__(**kwargs)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_loaded = coco.loadAnns(ann_ids)
        ann = ann_loaded[0]
        target = coco.annToMask(ann)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# updating the get_item to pick a random annotation
class CocoCap(CocoCaptions):
    def __init__(self, cache_dir, split, **kwargs):
        super(CocoCap, self).__init__(**kwargs)
        self.embedder = BertEmbedding()
        self.cache_file = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = f'{cache_dir}/coco_{split}.feather'
        self._cache = False
        self._make_cache()

    def __getitem__(self, index: int):
        if self.cache_file:
            binary, target = self._get_cached_index(index)
        else:
            binary, target = self._get_index(index)
        img_bytes = io.BytesIO(binary)
        img_bytes.seek(0)
        img = Image.open(img_bytes).convert("RGB")
        target = torch.tensor(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _make_cache(self):
        if not self.cache_file:
            print('cache_dir not set. skip caching.')
            return
        if self.cache_file and os.path.exists(self.cache_file):
            print(f'cache exists: {self.cache_file}. skip caching.')
            self._load_cache()
            return

        print('cache cache not found.')
        print(f'creating cache: {self.cache_file}')
        transforms = self.transforms
        self.transforms = None

        data = []
        for index in tqdm(range(len(self.ids))):
            binary, target = self._get_index(index, process_all=False)
            data.append([binary, target])
        print(f'{len(data)} instances processed.')
        dataframe = pd.DataFrame(data, columns=["image", "target"])
        dataframe.to_feather(self.cache_file)

        self._load_cache()
        self.transforms = transforms

    def _get_cached_index(self, index):
        binary = self._cache["image"][index]
        if type(self._cache["target"][index]) == list:
            target = random.choice(self._cache["target"][index])
        else:
            target = self._cache["target"][index]
        return binary, target

    def _get_index(self, index, process_all=False):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if not process_all:
            rnd_cap = random.choice(anns)['caption']
            target = self.embedder.get_embedding(rnd_cap).numpy()
        else:
            target = [self.embedder.get_embedding(
                ann['caption']).numpy() for ann in anns]

        path = os.path.join(self.root,
                            coco.loadImgs(img_id)[0]['file_name'])
        with open(path, 'rb') as fp:
            binary = fp.read()

        return binary, target

    def _load_cache(self):
        print(f'load cache: {self.cache_file}')
        self._cache = pd.read_feather(self.cache_file)
