import torch
import lightning as pl
from models.module import Encoder, Decoder
from models.util import instantiate_from_config, s2c, c2s, visualize_voxel
from models.distribution import DiagonalGaussianDistribution
import torch.nn.functional as F
import numpy as np
from load_data import voxelize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
import wandb
import plotly.graph_objects as go
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.discriminator import NLayerDiscriminator, weights_init
from tqdm import tqdm

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 disc_config,
                 embed_dim,
                 learning_rate=0,
                 kl_weight=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.automatic_optimization = False
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
        
        self.discriminator = NLayerDiscriminator(input_nc=disc_config.disc_in_channels,
                                                 n_layers=disc_config.disc_num_layers,
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_config.disc_start
        self.disc_loss = hinge_d_loss if disc_config.disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_config.disc_factor
        self.discriminator_weight = disc_config.disc_weight
        self.disc_conditional = disc_config.disc_conditional

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
    
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    

    def loss(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        weight = torch.where(inputs == 1.0, 1.0, 1e-1)
        rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous(), reduction="none")
        rec_loss = torch.mean(rec_loss * weight)
        kl_loss = posteriors.kl()
        # loss = rec_loss + self.kl_weight * kl_loss
        # return loss, {f"{split}/rec_loss": rec_loss, f"{split}/kl_loss": self.kl_weight  * kl_loss}

         # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = rec_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        

    def get_input(self, batch):
        x, _ = batch
        x = x.unsqueeze(1).to(self.device)
        return s2c(x)

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0 and self.global_step % 50 == 0:
        #     log_images = self.log_images(batch)
        #     self.logger.experiment.log(log_images)
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        g_opt, d_opt = self.optimizers()

        # train encoder+decoder
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, global_step=self.global_step, 
                                        optimizer_idx=0, last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        g_opt.zero_grad()
        self.manual_backward(aeloss)
        g_opt.step()
        # return aeloss


        # train the discriminator
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, global_step=self.global_step, 
                                            optimizer_idx=1, last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()
        # return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            log_images = self.log_images(batch)
            self.logger.experiment.log(log_images)
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx=0, 
                                        global_step=self.global_step, last_layer=self.get_last_layer(), split="val")

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
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
        # scheduler = LinearWarmupCosineAnnealingLR(opt_ae, warmup_epochs=2, max_epochs=50)
        # return [opt_ae], [{"scheduler": scheduler, "interval": "step"}]

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

        plotly_samples = get_voxel_plotly(log["samples"][0].squeeze().cpu().numpy())
        plotly_reconstructions = get_voxel_plotly(log["reconstructions"][0].squeeze().cpu().numpy())

        for k, v in log.items():
            img = visualize_voxel(v[0].squeeze().cpu().numpy())
            log[k] = wandb.Image(np.array(img))
        
        log["plotly/samples"] = wandb.Plotly(plotly_samples)
        log["plotly/reconstructions"] = wandb.Plotly(plotly_reconstructions)

        return log
    
    
    @torch.no_grad()
    def inference(self, test_dl, val_dl):
        self.eval()
        save_data = []
        for batch in tqdm(val_dl):
            x = self.get_input(batch)
            xrec, _ = self(x, sample_posterior=False)
            xrec = torch.where(c2s(xrec) > 0.5, 1, 0).squeeze(1)
            save_data.append(xrec.cpu().numpy())
        for batch in tqdm(test_dl):
            x = self.get_input(batch)
            xrec, _ = self(x, sample_posterior=False)
            xrec = torch.where(c2s(xrec) > 0.5, 1, 0).squeeze(1)
            save_data.append(xrec.cpu().numpy())
        return np.concatenate(save_data, axis=0)


def get_voxel_plotly(voxel_grid, lim=128):
    """
    Visualizes a 3D binary voxel grid using Plotly.

    Parameters:
    voxel_grid (numpy.ndarray): A 3D binary voxel grid where 1 indicates occupancy and 0 indicates empty.
    """

    # change to numpy if needed
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.cpu().numpy()

    # Get the coordinates of occupied voxels
    occupied_voxels = np.argwhere(voxel_grid == 1)

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=occupied_voxels[:, 0],
        y=occupied_voxels[:, 2],
        z=occupied_voxels[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
        )
    )])

    # Set labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            xaxis=dict(range=[0, lim]),
            yaxis=dict(range=[0, lim]),
            zaxis=dict(range=[0, lim])
        )
    )

    return fig


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
