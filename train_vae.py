import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from data.dataset import ShapeNetDataModule2, get_data_iterator, tensor_to_pil_image
from models.autoencoder import AutoencoderKL
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
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb


matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    vae_config = OmegaConf.load(args.config)

    ds_module = ShapeNetDataModule2(
        "./data",
        target_categories=config.target_categories,
        batch_size=vae_config.data.batch_size,
        num_workers=vae_config.data.num_workers,
        max_num_images_per_cat=config.max_num_images_per_cat,
        transform=config.transform,
    )

    train_dl = ds_module.train_dataloader()
    val_dl = ds_module.val_dataloader()

    autoencoder = AutoencoderKL(ddconfig=vae_config.model.params.ddconfig,
                                disc_config=vae_config.model.params.disc_config,
                                kl_weight=vae_config.model.params.kl_weight, 
                                embed_dim=vae_config.model.params.embed_dim,
                                learning_rate=vae_config.model.learning_rate)
    autoencoder.to(config.device)
    autoencoder.train()

    suffix = f"_{config.exp_name}" if config.exp_name else ""
    name = f"train_vae_{get_current_time()}" + suffix
    wandb_logger = WandbLogger(project="CS492D", name=name, entity="CS492d_team20")
    checkpoint_callback = ModelCheckpoint(dirpath=f"logs/{name}", monitor="val/rec_loss", mode="min", filename="best-{epoch:02d}")
    every_checkpoint_callback = ModelCheckpoint(dirpath=f"logs/{name}", every_n_epochs=20, save_top_k=-1)
    
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
                logger=wandb_logger,
                default_root_dir=f"logs/{name}",
                callbacks=[checkpoint_callback, every_checkpoint_callback],
                check_val_every_n_epoch=1,
                max_epochs=200,
                log_every_n_steps=10,
                accumulate_grad_batches=config.accumulate_grad,
                # limit_train_batches=0.5,
                # limit_val_batches=0.1,
                # overfit_batches=1,
                )
    
    trainer.fit(autoencoder, train_dl, val_dl)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='', type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_num_images_per_cat", type=int, default=-1)
    parser.add_argument("--target_categories", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--transform", type=str, nargs="+", default=None)
    args = parser.parse_args()
    seed_everything(0)
    main(args)
