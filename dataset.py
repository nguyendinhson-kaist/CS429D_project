import os
from typing import List
import volumentations as vol
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from utils import box_blur

import numpy as np


def listdir(dname):
    fnames = list(
        chain(
            *[
                list(Path(dname).rglob("*." + ext))
                for ext in ["png", "jpg", "jpeg", "JPG"]
            ]
        )
    )
    return fnames


def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    """
    x: [B,C,H,W]
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        single_image = True

    x = (x * 0.5 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (x * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    if single_image:
        return images[0]
    return images


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, split: str, target_category: str=None, transform=None, max_num_images_per_cat=-1, label_offset=1
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transform = transform
        self.max_num_images_per_cat = max_num_images_per_cat
        self.label_offset = label_offset

        categories = ['chair', 'airplane', 'table']

        # assert target_categories
        if target_category:
            assert target_category in categories, f"Invalid categories: {target_category}"
            categories = [target_category]

        self.num_classes = len(categories)

        imgs, labels = [], []
        for idx, cat in enumerate(sorted(categories)):
            cat_imgs = np.load(Path.joinpath(Path(root), f"{cat}_voxels_{split}.npy"))
            imgs.append(cat_imgs)
            labels += [idx + label_offset] * len(cat_imgs) # label 0 is for null class.

        self.imgs = np.concatenate(imgs, axis=0)
        self.labels = labels

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        assert label >= self.label_offset
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class ShapeNetDataModule(object):
    def __init__(
        self,
        root: str = "data",
        target_categories: str=None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_num_images_per_cat: int = 1000,
        label_offset=1,
        transform=None
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hdf5_root = os.path.join(root, "hdf5_data")
        self.max_num_images_per_cat = max_num_images_per_cat
        self.label_offset = label_offset
        self.transform = transform
        self.target_categories = target_categories

        assert os.path.exists(self.hdf5_root), f"{self.hdf5_root} does not exist. Please download the dataset."

        self._set_dataset()

    def _set_dataset(self):
        # TODO: Implement transforms
        # if self.transform is None:
        #     self.transform = transforms.Compose(
        #         [
        #         ]
        #     )
        self.train_ds = ShapeNetDataset(
            self.hdf5_root,
            "train",
            self.target_categories,
            self.transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset
        )
        self.val_ds = ShapeNetDataset(
            self.hdf5_root,
            "val",
            self.target_categories,
            self.transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset
        )

        self.num_classes = self.train_ds.num_classes

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
    
class ShapeNetDataset2(torch.utils.data.Dataset):
    """
        Load each voxel data individually.
    """
    def __init__(
        self, root: str, split: str, target_category: str=None, transform=None, max_num_images_per_cat=-1,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transform = transform
        self.max_num_images_per_cat = max_num_images_per_cat

        categories = ['chair', 'airplane', 'table']
        self.label_dict = {'chair': 1, 'airplane': 2, 'table': 3}

        # assert target_categories
        if target_category:
            assert target_category in categories, f"Invalid categories: {target_category}"
            categories = [target_category]

        self.num_classes = len(categories)

        paths = []
        for cat in sorted(categories):
            cat_dir = os.path.join(root, split, cat)
            paths += glob(cat_dir + "/*.npy")[:max_num_images_per_cat]
            
        self.paths = paths

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(self.paths[idx])
        label = self.label_dict[path.split("/")[-1].split("_")[0]]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.paths)


class ShapeNetDataModule2(object):
    def __init__(
        self,
        root: str = "data",
        target_categories: str=None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_num_images_per_cat: int = -1,
        transform:str=None
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shapenet_root = os.path.join(root, "shapenet")
        self.max_num_images_per_cat = max_num_images_per_cat
        self._set_transform(transform)
        self.target_categories = target_categories

        assert os.path.exists(self.shapenet_root), f"{self.shapenet_root} does not exist. Please download the dataset."

        self._set_dataset()

    def _set_transform(self, transform_list:List[str]=None):
        '''
        Set train_transform and val_transform attributes.
        '''
        self.train_transform = None
        self.val_transform = None

        if transform_list is None:
            return

        assert len(transform_list) == 1, f"Currently flip or box but got {transform_list}"
        if transform_list[0] == 'flip':
            vol_aug = vol.Compose([vol.Flip(2, p=0.5)])
            def volumentation_transform(data):
                return vol_aug(image=data)['image']
            self.train_transform = volumentation_transform
        elif transform_list[0] == 'box':
            self.train_transform = box_blur

        # for tf in transform_list:
        #     tf = tf.lower()
        #     if tf == 'flip':
        #         to_compose.append(vol.Flip(2, p=0.5))
        #     # elif tf == 'rotate':
        #     #     to_compose.append(vol.Rotate((0, 0), (-15, 15), (0, 0)))
        #     # elif tf == 'scale':
        #     #     to_compose.append(vol.RandomScale([0.95, 1.05]))
        # composed = vol.Compose(to_compose)
        # def volumentation_transform(data):
        #     return composed(image=data)['image']
        
        # self.train_transform = volumentation_transform


    def _set_dataset(self):
        self.train_ds = ShapeNetDataset2(
            self.shapenet_root,
            "train",
            self.target_categories,
            self.train_transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
        )
        self.val_ds = ShapeNetDataset2(
            self.shapenet_root,
            "val",
            self.target_categories,
            self.val_transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
        )
        self.test_ds = ShapeNetDataset2(
            self.shapenet_root,
            "test",
            self.target_categories,
            self.val_transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
        )

        self.num_classes = self.train_ds.num_classes

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

if __name__ == "__main__":
    try:
        data_module = ShapeNetDataModule("data", 'airplane', 32, 4, -1, 1)
        print("DataModule is set up successfully.")
    except Exception as e:
        print(e)
        print("You may try to download the dataset first.")
        print("Refer to the README.md for instructions.")
        exit(1)
