import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from data.dataset import ShapeNetLatentModule
from models.diffusion import Diffusion
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

    model_config = OmegaConf.load(args.config)

    ds_module = ShapeNetLatentModule(
        "./data",
        target_categories=config.target_categories,
        batch_size=model_config.data.batch_size,
        num_workers=model_config.data.num_workers,
    )

    train_dl = ds_module.train_dataloader()
    val_dl = ds_module.val_dataloader()

    var_scheduler = DDPMScheduler(
        num_train_timesteps=model_config.scheduler.num_train_timesteps,
        beta_1=model_config.scheduler.beta_1,
        beta_T=model_config.scheduler.beta_T,
        mode=model_config.scheduler.mode,
        sigma_type=model_config.scheduler.sigma_type,
    )

    diffusion_model = Diffusion(
        unet_config=model_config.model.params,
        vae_config=model_config.model.vae,
        learning_rate=model_config.model.learning_rate,
        var_scheduler=var_scheduler,
        ckpt_path=None,
        ignore_keys=["discriminator"],
        num_classes=ds_module.num_classes,
    )
    diffusion_model.to(config.device)
    diffusion_model.train()

    name = f"ldm_{get_current_time()}_{config.exp_name}"
    wandb_logger = WandbLogger(project="CS492D", name=name, entity="CS492d_team20")
    best_checkpoint_callback = ModelCheckpoint(dirpath=f"logs/{name}", monitor="val/loss", mode="min", filename="best-{epoch:02d}", save_top_k=3)
    every_checkpoint_callback = ModelCheckpoint(dirpath=f"logs/{name}", every_n_epochs=25, save_top_k=-1)

    trainer = Trainer(
                logger=wandb_logger,
                default_root_dir=f"logs/{name}",
                callbacks=[best_checkpoint_callback, every_checkpoint_callback],
                check_val_every_n_epoch=1,
                max_epochs=config.max_epochs,
                accumulate_grad_batches=config.accumulate_grad,
                log_every_n_steps=50,
                # overfit_batches=1,
                # limit_train_batches=1,
                # limit_val_batches=0,
                )
    
    trainer.fit(diffusion_model, train_dl, val_dl)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_categories", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()
    seed_everything(0)
    main(args)
