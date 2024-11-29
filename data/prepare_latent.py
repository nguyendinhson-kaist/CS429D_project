import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
from data.dataset import ShapeNetDataModule2, get_data_iterator, tensor_to_pil_image
from models.autoencoder import AutoencoderKL
from dotmap import DotMap
from pytorch_lightning import seed_everything
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    vae_config = OmegaConf.load(args.config)

    ds_module = ShapeNetDataModule2(
        "data/",
        batch_size=vae_config.data.batch_size,
        num_workers=vae_config.data.num_workers,
    )

    train_dl = ds_module.train_dataloader()
    val_dl = ds_module.val_dataloader()
    
    assert args.ckpt is not None, "Please provide the path to the checkpoint."
    autoencoder = AutoencoderKL(ddconfig=vae_config.model.params.ddconfig,
                                disc_config=vae_config.model.params.disc_config,
                                kl_weight=vae_config.model.params.kl_weight, 
                                embed_dim=vae_config.model.params.embed_dim,
                                learning_rate=vae_config.model.learning_rate,
                                ckpt_path=args.ckpt,
                                ignore_keys=['discriminator', 'optimizer'])
    autoencoder.to(config.device)
    autoencoder.eval()
    label_dict = {1: "chair", 2: "airplane", 3: "table"}
    
    os.makedirs("data/latent_data/train/chair", exist_ok=True)
    os.makedirs("data/latent_data/train/airplane", exist_ok=True)
    os.makedirs("data/latent_data/train/table", exist_ok=True)
    os.makedirs("data/latent_data/val/chair", exist_ok=True)
    os.makedirs("data/latent_data/val/airplane", exist_ok=True)
    os.makedirs("data/latent_data/val/table", exist_ok=True)

    with torch.no_grad():
        cnt = 1
        for batch in tqdm(train_dl):
            _, label = batch
            input = autoencoder.get_input(batch)
            posterior = autoencoder.encode(input)
            z = posterior.mode()
            z = z.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            for i in range(len(z)):
                np.save(f"data/latent_data/train/{label_dict[label[i]]}/{cnt}.npy", z[i])
                cnt += 1
        
        cnt = 1
        for batch in tqdm(val_dl):
            _, label = batch
            input = autoencoder.get_input(batch)
            posterior = autoencoder.encode(input)
            z = posterior.mode()
            z = z.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            for i in range(len(z)):
                np.save(f"data/latent_data/val/{label_dict[label[i]]}/{cnt}.npy", z[i])
                cnt += 1

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    seed_everything(0)
    main(args)