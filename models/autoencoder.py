import torch
import lightning as pl
from models.module import Encoder, Decoder
from models.util import instantiate_from_config, s2c, c2s
from models.distribution import DiagonalGaussianDistribution
import torch.nn.functional as F
import numpy as np
from load_data import voxelize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
import wandb
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 learning_rate,
                 kl_weight=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
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
        posterior = DiagonalGaussianDistribution(moments)
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
        weight = torch.where(inputs == 1.0, 1.0, 1e-1)
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous(), reduction="none")
        # rec_loss = torch.sum(rec_loss * weight, dim=[1,2,3,4]).mean()
        rec_loss = torch.mean(rec_loss * weight)
        kl_loss = posteriors.kl()
        kl_loss = kl_loss.mean()
        loss = rec_loss + self.kl_weight * kl_loss
        return loss, {f"{split}/rec_loss": rec_loss, f"{split}/kl_loss": self.kl_weight  * kl_loss}

    def get_input(self, batch):
        x, _ = batch
        x = x.unsqueeze(1).to(self.device)
        return s2c(x)

    def training_step(self, batch, batch_idx,):
        if batch_idx == 0:
            log_images = self.log_images(batch)
            self.logger.experiment.log(log_images)
        inputs = self.get_input(batch)
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
        # if batch_idx == 0:
        #     log_images = self.log_images(batch)
        #     self.logger.experiment.log(log_images)
        inputs = self.get_input(batch)
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
                                  lr=lr, betas=(0.5, 0.9),
                                  weight_decay=1e-5)
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        # return [opt_ae, opt_disc], []
        scheduler = LinearWarmupCosineAnnealingLR(opt_ae, warmup_epochs=2, max_epochs=50)
        return [opt_ae], [{"scheduler": scheduler, "interval": "epoch"}]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = c2s(self.decode(torch.randn_like(posterior.sample())))
            log["reconstructions"] = c2s(xrec)

            log["samples"] = torch.where(log["samples"] > 0.5, 1, 0)
            log["reconstructions"] = torch.where(log["reconstructions"] > 0.5, 1, 0)

        log["inputs"] = c2s(x)

        for k, v in log.items():
            img = visualize_voxel(v[0].squeeze().cpu().numpy())
            log[k] = wandb.Image(np.array(img))

        return log

def visualize_voxel(voxel_grid):
    """
    Visualizes a 3D binary voxel grid using matplotlib.

    Parameters:
    voxel_grid (numpy.ndarray): A 3D binary voxel grid where 1 indicates occupancy and 0 indicates empty.
    """

    # Get the coordinates of occupied voxels
    occupied_voxels = np.argwhere(voxel_grid == 1)

    # Create a 3D plot
    fig = plt.figure()
    plt.tight_layout()

    ax = fig.add_subplot(111, projection='3d')

    # Plot occupied voxels as scatter points
    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 2], occupied_voxels[:, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set the limits for the axes
    ax.set_xlim([0, voxel_grid.shape[0]])
    ax.set_ylim([0, voxel_grid.shape[1]])
    ax.set_zlim([0, voxel_grid.shape[2]])
    
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move the buffer cursor to the beginning
    plt.close()
    # Convert the buffer into a Pillow Image
    img = Image.open(buf)
    return img