import torch
import lightning as pl
from models.util import instantiate_from_config
from models.distribution import DiagonalGaussianDistribution2D
import torch.nn.functional as F
import numpy as np
from load_data import voxelize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
import wandb

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

num_group = 8

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1,)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1,)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_group, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, tdim=0, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_group, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1,),
        )

        if tdim > 0:
            self.temb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(tdim, out_ch),
            )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_group, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1,),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb=None):
        h = self.block1(x)
        if temb is not None:
            h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Encoder(nn.Module):
    def __init__(self, *, in_ch, ch, ch_mult=(1,2,4,8,16), num_res_blocks,
                 attn_resolutions, dropout=0.0, resolution, z_channels, double_z=True, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_ch

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                attn = curr_res in attn_resolutions
                block.append(ResBlock(block_in, block_out, dropout, attn=attn))
                block_in = block_out
            module_dict = nn.ModuleDict({
                "block": block,
            })
            if i_level != self.num_resolutions-1:
                module_dict["downsample"] = DownSample(block_out)
                curr_res //= 2
            self.down.append(module_dict)
        
        # # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResBlock(block_in, block_in, dropout, attn=True)
        # self.mid.block_2 = ResBlock(block_in, block_in, dropout)

        # end
        self.norm_out = nn.GroupNorm(num_group, block_in)
        self.conv_out = nn.Conv2d(block_in, 
                                  z_channels * 2 if double_z else z_channels, # mean and logvar
                                  3, stride=1, padding=1)

                       

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)
        
        # # middle
        # h = self.mid.block_1(h)
        # h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
    

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8,16), num_res_blocks,
                 attn_resolutions, dropout=0.0,
                 resolution, z_channels, sigmoid_out=False, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.sigmoid_out = sigmoid_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        # # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResBlock(in_ch=block_in,
        #                                out_ch=block_in,
        #                                dropout=dropout,
        #                                attn=True
        #                             )
        # self.mid.block_2 = ResBlock(in_ch=block_in,
        #                                out_ch=block_in,
        #                                dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions-1)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                attn = curr_res in attn_resolutions
                block.append(ResBlock(in_ch=block_in,
                                         out_ch=block_out,
                                         dropout=dropout,
                                         attn=attn))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.upsample = UpSample(block_in)
            curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_group, block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self, z):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # # middle
        # h = self.mid.block_1(h)
        # h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions-1)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        if self.sigmoid_out:
            h = torch.sigmoid(h)
        return h



class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 learning_rate,
                 kl_weight=1.0e-06,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.kl_weight = kl_weight
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

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution2D(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def loss(self, inputs, reconstructions, posteriors,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous(), reduction="none")
        rec_loss = torch.sum(rec_loss, dim=[1, 2, 3]).mean()
        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss)
        loss = rec_loss + self.kl_weight * kl_loss
        return loss, {f"{split}/rec_loss": rec_loss, f"{split}/kl_loss": self.kl_weight  * kl_loss}

    def training_step(self, batch, batch_idx,):
        inputs, _ = batch
        reconstructions, posterior = self(inputs)

        # if optimizer_idx == 0:
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

        # if optimizer_idx == 1:
        #     # train the discriminator
        #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
        #                                         last_layer=self.get_last_layer(), split="train")

        #     self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     return discloss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            log_images = self.log_images(batch)
            self.logger.experiment.log(log_images)
        inputs, _ = batch
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        # return [opt_ae, opt_disc], []
        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x, _ = batch
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec

        log["inputs"] = x

        for k, v in log.items():
            log[k] = wandb.Image(v[0])

        return log