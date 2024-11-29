import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from data.dataset import ShapeNetLatentModule
from models.diffusion import Diffusion
from models.util import c2s, visualize_voxel
from dotmap import DotMap
from model import DiffusionModule
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from omegaconf import OmegaConf
import os
import numpy as np


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
    label_dict = ds_module.train_ds.label_dict

    var_scheduler = DDPMScheduler(
        num_train_timesteps=model_config.scheduler.num_train_timesteps,
        beta_1=model_config.scheduler.beta_1,
        beta_T=model_config.scheduler.beta_T,
        mode=model_config.scheduler.mode,
        sigma_type=model_config.scheduler.sigma_type,
    )

    model_config.model.vae.ckpt_path = None
    diffusion_model = Diffusion(
        unet_config=model_config.model.params,
        vae_config=model_config.model.vae,
        learning_rate=model_config.model.learning_rate,
        var_scheduler=var_scheduler,
        ignore_keys=["discriminator"],
        num_classes=ds_module.num_classes,
        ckpt_path=config.ckpt,
    )
    diffusion_model.to(config.device)
    diffusion_model.eval()

    if config.self_guidance > 0.0:
        bad_model = Diffusion(
            unet_config=model_config.model.params,
            vae_config=model_config.model.vae,
            learning_rate=model_config.model.learning_rate,
            var_scheduler=var_scheduler,
            ignore_keys=["discriminator"],
            num_classes=ds_module.num_classes,
            ckpt_path=config.bad_ckpt,
        )
        bad_model.to(config.device)
        bad_model.eval()
        diffusion_model.bad_network = bad_model.network
        del bad_model

    output_dir = 'output/' + config.output_dir
    if config.target_category is not None:
        output_dir += f"/{config.target_category}"
    os.makedirs(output_dir, exist_ok=True)

    # Sample 1000 data
    cnt = 0
    samples = []
    pbar = tqdm(total=1000)
    with torch.no_grad():
        while cnt < 1000:
            bs = min(1000 - cnt, config.batch_size)
            if model_config.model.params.use_cfg: # Train with cfg
                assert config.target_category is not None and config.target_category in label_dict
                target_label = torch.tensor([label_dict[config.target_category]] * bs).to(config.device)
                z0 = diffusion_model.sample(bs, class_label=target_label, guidance_scale=config.cfg, self_guidance=config.self_guidance)
            else: # Train without cfg
                z0 = diffusion_model.sample(bs, self_guidance=config.self_guidance)
            x0 = diffusion_model.vae.decode(z0)
            x0 = torch.where(c2s(x0) > 0.5, 1, 0).squeeze().cpu().numpy()
            samples.append(x0)
            if cnt % (bs*2) == 0:
                img = visualize_voxel(x0[0])
                img.save(output_dir + f"/sample_{cnt}.png")
            pbar.update(bs)
            cnt += bs
    pbar.close()

    samples = np.concatenate(samples, axis=0)
    np.save(output_dir + "/samples.npy", samples)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_category", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--self_guidance", type=float, default=0.0)
    parser.add_argument("--bad_ckpt", type=str)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
