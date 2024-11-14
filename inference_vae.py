import argparse
import json
from datetime import datetime
from pathlib import Path
from dataset import ShapeNetDataModule2
from models.autoencoder import AutoencoderKL
from dotmap import DotMap
from pytorch_lightning import seed_everything
import numpy as np
from omegaconf import OmegaConf
import os
import yaml

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    vae_config = OmegaConf.load(args.config)
    save_dir = './vae_reconstruction/' + args.config.split("/")[-1].split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(vae_config, f"./{save_dir}/{args.config.split('/')[-1]}")

    assert args.ckpt is not None, "Please provide the path to the checkpoint."

    ds_module = ShapeNetDataModule2(
        "./data",
        target_categories=config.target_categories,
        batch_size=vae_config.data.batch_size,
        num_workers=vae_config.data.num_workers,
    )

    test_dl = ds_module.test_dataloader()
    val_dl = ds_module.val_dataloader()

    autoencoder = AutoencoderKL(ddconfig=vae_config.model.params.ddconfig,
                                disc_config=vae_config.model.params.disc_config,
                                kl_weight=vae_config.model.params.kl_weight, 
                                embed_dim=vae_config.model.params.embed_dim,
                                learning_rate=vae_config.model.learning_rate,
                                ckpt_path=args.ckpt,
                                ignore_keys=['discriminator'])
    autoencoder.to(config.device)
    
    rec_data = autoencoder.inference(test_dl, val_dl)
    np.save(f"./{save_dir}/rec_data.npy", rec_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_categories", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    seed_everything(0)
    main(args)