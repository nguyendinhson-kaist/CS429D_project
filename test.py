import torch
from models.module import Encoder, Decoder
from models.autoencoder import AutoencoderKL
from models.diffusion_unet import UNet
from data.dataset import ShapeNetLatentModule
from omegaconf import OmegaConf

from einops import rearrange

# vae_config = OmegaConf.load('./configs/vae/config_nodisc_kl1e-6.yaml')
# ds_module = ShapeNetLatentModule(
#         "./data",
#         batch_size=vae_config.data.batch_size,
#         num_workers=vae_config.data.num_workers,
#     )

# train_dl = ds_module.train_dataloader()
# val_dl = ds_module.val_dataloader()

# data = torch.randn(8, 1, 128, 128, 128).cuda()

# def s2c(data, h=2, w=2, d=2):
#     return rearrange(data, 'b c (H h) (W w) (D d) -> b (c h w d) H W D', h=h, w=w, d=d)

# def c2s(data, h=2, w=2, d=2):
#     return rearrange(data, 'b (c h w d) H W D -> b c (H h) (W w) (D d)', h=h, w=w, d=d)

# data = s2c(data)
# encoder = Encoder(in_ch=8,
#                   ch=128,
#                   num_res_blocks=1,
#                   attn_resolutions=[],
#                   ch_mult=(1,2,2,4),
#                   dropout=0.1,
#                   resolution=64,
#                   z_channels=4,
#                   double_z=True
#                   ).cuda()

# decoder = Decoder(z_channels=4,
#                   ch=128,
#                   out_ch=8,
#                   num_res_blocks=1,
#                   ch_mult=(1,2,2,4),
#                   attn_resolutions=[],
#                   resolution=64,
#                   dropout=0.1,
#                   sigmoid_out=True
#                   ).cuda()

# latent = encoder(data)
# reconstructed = decoder(latent[:, :4])
# import pdb; pdb.set_trace()
bs = 64
data = torch.randn(bs, 8, 8, 8, 8).cuda()

unet = UNet(T=1000, 
            resolution=8, 
            in_ch=8,
            ch=128, 
            ch_mult=[1,2,2,4], 
            attn=[1], 
            num_res_blocks=4, 
            dropout=0.1, 
            use_cfg=False, 
            cfg_dropout=0.1, 
            num_classes=None).cuda()

timestep = torch.zeros(bs).cuda()
output = unet(data, timestep)
import pdb; pdb.set_trace()

