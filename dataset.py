import os
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

from einops import rearrange


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

class Compressor:
    def compress(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def decompress(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class CubeDecimalCompressor(Compressor):
    def __init__(self, cube_size: int = 2, device='cuda'):
        self.cube_size = cube_size
        self.device = device

    def compress(self, data: np.ndarray) -> np.ndarray:
        # to tensor
        data = torch.tensor(data, device=self.device).float()

        # (H x W x D) -> (H//cube_size x W//cube_size x D//cube_size x cube_size x cube_size x cube_size)
        data = rearrange(data, '(h h1) (w w1) (d d1) -> h w d h1 w1 d1', h1=self.cube_size, w1=self.cube_size, d1=self.cube_size)
        data = data.flatten(start_dim=-3)

        # convert last dim from binary to decimal
        data = data @ (2. ** torch.arange(self.cube_size ** 3, device=self.device))

        # normalize to [-1, 1] (zero mean)
        mean = 2. ** (self.cube_size ** 3 - 1)
        data = (data - mean) / mean

        return data.cpu().numpy()
    
    def decompress(self, data: np.ndarray) -> np.ndarray:
        # to tensor
        data = torch.tensor(data, device=self.device).float()

        # denormalize
        mean = 2. ** (self.cube_size ** 3 - 1)
        data = data * mean + mean
        data = data.round()

        # convert decimal to binary
        data = data.unsqueeze(-1) // (2. ** torch.arange(self.cube_size ** 3, device=self.device))
        data = data % 2

        # (H//cube_size x W//cube_size x D//cube_size x cube_size x cube_size x cube_size) -> (H x W x D)
        data = rearrange(data, 'h w d (h1 w1 d1) -> (h h1) (w w1) (d d1)', h1=self.cube_size, w1=self.cube_size, d1=self.cube_size)

        return data.cpu().numpy()


def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    # """
    # x: [B,C,H,W]
    # """
    # if x.ndim == 3:
    #     x = x.unsqueeze(0)
    #     single_image = True

    # x = (x * 0.5 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    # images = (x * 255).round().astype("uint8")
    # images = [Image.fromarray(image) for image in images]
    # if single_image:
    #     return images[0]
    # return images

    # compressor = CubeDecimalCompressor()
    # voxel_grid = compressor.decompress(x.squeeze(0).cpu().numpy())

    # Get the coordinates of occupied voxels
    voxel_grid = x.squeeze(0).cpu().numpy()
    occupied_voxels = np.argwhere(voxel_grid >= 0.5)

    # Create a 3D plot
    fig = plt.figure()
    plt.tight_layout()

    ax = fig.add_subplot(111, projection='3d')

    # Plot occupied voxels as scatter points
    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 2], occupied_voxels[:, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set the limits for the axes
    ax.set_xlim([0, voxel_grid.shape[0]])
    ax.set_ylim([0, voxel_grid.shape[1]])
    ax.set_zlim([0, voxel_grid.shape[2]])
    
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move the buffer cursor to the beginning
    plt.close()
    # Convert the buffer into a Pillow Image
    img = Image.open(buf)
    return img


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
        self, 
        root: str, 
        split: str, 
        target_category: str=None, 
        transform=None, 
        max_num_images_per_cat=-1, 
        label_offset=1
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transform = transform
        self.max_num_images_per_cat = max_num_images_per_cat
        self.label_offset = label_offset

        categories = ['airplane', 'chair', 'table']

        # assert target_category
        if target_category:
            assert target_category in categories, f"Invalid categories: {target_category}"
            categories = [target_category]

        self.num_classes = len(categories)

        imgs, labels = [], []
        for idx, cat in enumerate(categories):
            # if self.split == "train":
            #     cat_imgs = np.load(Path.joinpath(Path(root), f"enc_{cat}_{split}.npy"))
            # else:
            #     cat_imgs = np.load(Path.joinpath(Path(root), f"{cat}_voxels_{split}.npy"))
            
            cat_imgs = np.load(Path.joinpath(Path(root), f"{cat}_voxels_{split}.npy"))

            if self.max_num_images_per_cat > 0:
                cat_imgs = cat_imgs[:self.max_num_images_per_cat]

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

        img = torch.tensor(img).float().unsqueeze(0)

        return img, label

    def __len__(self):
        return len(self.labels)


class ShapeNetDataModule(object):
    def __init__(
        self,
        root: str = "data",
        target_category: str=None,
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
        self.target_category = target_category

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
            self.target_category,
            self.transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset
        )
        self.val_ds = ShapeNetDataset(
            self.hdf5_root,
            "val",
            self.transform,
            self.target_category,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset,
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
    

if __name__ == "__main__":
    try:
        data_module = ShapeNetDataModule("data", 'airplane', 32, 4, -1, 1)
        print("DataModule is set up successfully.")
    except Exception as e:
        print(e)
        print("You may try to download the dataset first.")
        print("Refer to the README.md for instructions.")
        exit(1)
