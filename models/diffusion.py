import torch
import lightning as pl
from models.diffusion_unet import UNet
from models.util import instantiate_from_config, s2c, c2s, visualize_voxel
from models.autoencoder import AutoencoderKL
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional
from einops import rearrange
import os
from eval import jensen_shannon_divergence, voxel_to_pointcloud, pairwise_CD, lgan_mmd_cov


class Diffusion(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 num_classes,
                 vae_config,
                 var_scheduler,
                 learning_rate=0,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.network = UNet(**unet_config, num_classes=num_classes)
        self.vae = AutoencoderKL(**vae_config)
        self.vae.eval()
        self.var_scheduler = var_scheduler
        
        self.in_ch = unet_config.in_ch
        self.resolution = unet_config.resolution
        self.learning_rate = learning_rate
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    @property
    def device(self):
        return next(self.network.parameters()).device
    

    def training_step(self, batch, batch_idx):
        x0, label = batch
        x0 = x0.to(self.device)
        label = label.to(self.device)

        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        xt, eps = self.var_scheduler.add_noise(x0, timestep)
        pred_noise = self.network(xt, timestep, class_label=label)

        loss = F.mse_loss(pred_noise, eps)
        self.log("train/loss", loss)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            log_dict = self.log_images()
            self.logger.log_image(**log_dict)

        x0, label = batch
        x0 = x0.to(self.device)
        label = label.to(self.device)

        B = x0.shape[0]
        timestep = torch.linspace(0, 999, steps=B, dtype=torch.int64).to(self.device)
        xt, eps = self.var_scheduler.add_noise(x0, timestep)
        pred_noise = self.network(xt, timestep, class_label=label)

        loss = F.mse_loss(pred_noise, eps)
        self.log("val/loss", loss)
        return {'loss': loss}
    

    # @torch.no_grad()
    # def on_validation_epoch_end(self, *args, **kwrags):
    #     eval_res = self.evaluation()
    #     self.log_dict(eval_res)
    

    # @torch.no_grad()
    # def evaluation(self):
    #     category = "airplane"
    #     cnt = 0
    #     samples = []
    #     while cnt < 100:
    #         bs = min(100 - cnt, 32)
    #         z0 = self.sample(bs)
    #         x0 = self.vae.decode(z0)
    #         x0 = torch.where(c2s(x0) > 0.5, 1, 0).squeeze().cpu()
    #         samples.append(x0)
    #         cnt += bs

    #     X_gen = torch.concat(samples, dim=0)
    #     shapenet_dir = "./data/hdf5_data"  # TODO: CHANGE THE PATH IF NEEDED.
    #     val_set_path = os.path.join(shapenet_dir, f"{category}_voxels_val.npy")
    #     assert os.path.exists(val_set_path), f"{val_set_path} not exist."

    #     val_set = torch.from_numpy(np.load(val_set_path))
    #     X_ref = val_set.float()[:100]

    #     print("[*] Computing JSD...")
    #     jsd_score = jensen_shannon_divergence(X_gen, X_ref)
    #     print(f"JSD: {jsd_score}")
        
    #     res_dic = {'val/JSD': jsd_score}
    #     return res_dic


    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.network.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return opt


    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
        self_guidance: Optional[float] = 0.0,
    ):  
        x_T = torch.randn([batch_size, self.in_ch, self.resolution, self.resolution, self.resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            class_label = class_label.to(self.device)
            null_label = torch.zeros_like(class_label)
            class_label = torch.cat([class_label, null_label])

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                noise_pred = self.network(
                    torch.cat([x_t]*2),
                    timestep=t.to(self.device),
                    class_label=class_label
                )
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = (1 + guidance_scale) * noise_pred_cond - guidance_scale * noise_pred_uncond
            else:
                noise_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label,
                )

            if self_guidance > 0.0:
                assert self.bad_network is not None
                bad_noise_pred = self.bad_network(x_t, timestep=t.to(self.device), class_label=class_label)
                noise_pred = (1 + self_guidance) * noise_pred - self_guidance * bad_noise_pred

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]


    @torch.no_grad()
    def log_images(self):
        class_label = torch.tensor([1,2,3]).to(self.device)
        z0 = self.sample(3, class_label=class_label)
        x0 = self.vae.decode(z0)
        x0 = rearrange(x0, 'b c h w d -> b h w d c')

        # Density reduction to make visualization faster
        top1_values, top1_indices = torch.topk(x0, 1, dim=-1)
        sparse_x0 = torch.zeros_like(x0).scatter_(-1, top1_indices, top1_values)
        sparse_x0 = rearrange(sparse_x0, 'b h w d c -> b c h w d')

        x0 = torch.where(c2s(sparse_x0) > 0.5, 1, 0).squeeze().cpu().numpy()
        imgs = []
        for i in range(3):
            img = visualize_voxel(x0[i])
            imgs.append(img)
        
        log_dict = {
            'key': 'samples',
            'images': imgs,
        }

        if self.network.use_cfg:
            log_dict['caption'] = ['chair', 'airplane', 'table']
        
        return log_dict

