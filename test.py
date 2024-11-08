import torch
from models.module import Encoder, Decoder
from models.autoencoder import AutoencoderKL

from einops import rearrange

data = torch.randn(8, 1, 128, 128, 128).cuda()

def s2c(data, h=2, w=2, d=2):
    return rearrange(data, 'b c (H h) (W w) (D d) -> b (c h w d) H W D', h=h, w=w, d=d)

def c2s(data, h=2, w=2, d=2):
    return rearrange(data, 'b (c h w d) H W D -> b c (H h) (W w) (D d)', h=h, w=w, d=d)

data = s2c(data)
encoder = Encoder(in_ch=8,
                  ch=128,
                  num_res_blocks=1,
                  attn_resolutions=[],
                  ch_mult=(1,2,2,4,4),
                  dropout=0.1,
                  resolution=64,
                  z_channels=4,
                  double_z=True
                  ).cuda()

decoder = Decoder(z_channels=4,
                  ch=128,
                  out_ch=8,
                  num_res_blocks=1,
                  ch_mult=(1,2,2,4,4),
                  attn_resolutions=[],
                  resolution=64,
                  dropout=0.1,
                  sigmoid_out=True
                  ).cuda()

latent = encoder(data)
reconstructed = decoder(latent[:, :4])
import pdb; pdb.set_trace()