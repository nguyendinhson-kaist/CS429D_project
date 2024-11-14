import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from CS492D_project.data.dataset import ShapeNetDataModule, get_data_iterator, tensor_to_pil_image
from models.autoencoder_2d import AutoencoderKL
from dotmap import DotMap
from model import DiffusionModule
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from lightning import Trainer
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        # self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)

def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    vae_config = OmegaConf.load(args.config)

    # ds_module = ShapeNetDataModule(
    #     "./data",
    #     target_categories=config.target_categories,
    #     batch_size=vae_config.data.batch_size,
    #     num_workers=vae_config.data.num_workers,
    #     max_num_images_per_cat=config.max_num_images_per_cat,
    # )
    ds_module = MNISTDataModule('./mnist')
    ds_module.prepare_data()
    ds_module.setup('fit')

    train_dl = ds_module.train_dataloader()
    val_dl = ds_module.val_dataloader()

    autoencoder = AutoencoderKL(ddconfig=vae_config.model.params.ddconfig,
                                kl_weight=vae_config.model.params.kl_weight, 
                                embed_dim=vae_config.model.params.embed_dim,
                                learning_rate=vae_config.model.learning_rate)
    autoencoder.to(config.device)
    autoencoder.train()

    name = f"train_vae_2d_{get_current_time()}"
    wandb_logger = WandbLogger(project="CS492D", name=name)
    checkpoint_callback = ModelCheckpoint(dirpath=f"logs/{name}", monitor="val/rec_loss", every_n_epochs=1)
    trainer = Trainer(callbacks=[checkpoint_callback])

    trainer = Trainer(
                logger=wandb_logger,
                default_root_dir="logs",
                callbacks=[checkpoint_callback],
                check_val_every_n_epoch=1,
                max_epochs=50,
                # limit_train_batches=2
                )
    
    trainer.fit(autoencoder, train_dl, val_dl)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_num_images_per_cat", type=int, default=1000)
    parser.add_argument("--target_categories", type=str, default=None)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    main(args)
