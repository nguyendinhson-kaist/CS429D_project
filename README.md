# Install Requirement

```
conda create -n project python=3.10
conda activate project
pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1
pip install -r requirements.txt
```

# Checkpoint Preparation
- In case there is no checkpoint zipped together with this code, please download the checkpoint from this [link](https://drive.google.com/drive/folders/1lZjMsKsdQCYqEGIgsCGPmQ3AWiEOUZr4).

- Unzip the file and the checkpoint is ready for use.


# Data Preparation for LDM
Run the following commands in order:
```
python load_data.py
```
```
python data_processing.py
```

```
python data/prepare_latent.py\
    --gpu 0 \
    --config configs/vae/config_nodisc_kl1e-6_64.yaml \
    --ckpt ckpt/train_vae_11-22-212308_cond_64_nos2c_w1e-1/epoch=238-step=504768.ckpt
```

# Training 
**VAE Training**
```
CUDA_VISIBLE_DEVICES=0 \
python train_vae.py \
    --config configs/vae/config_nodisc_kl1e-6_64.yaml \
    --exp_name cond_64_nos2c_w1e-1
```

**LDM Training**
```
CUDA_VISIBLE_DEVICES=0 \
python train_ldm.py \
    --config configs/diffusion/ldm_cond_64_cfg0.0.yaml \
    --exp_name cond_64_cfg0.0 
```

# Inference
**VAE Inference** (Not important)
Example command to get reconstruciton data for val and test sets:
```
CUDA_VISIBLE_DEVICES=0 \
python inference_vae.py \
    --config configs/vae/config_nodisc_kl1e-6_64.yaml \
    --exp_name cond_64_airplane \
    --ckpt ckpt/train_vae_11-22-212308_cond_64_nos2c_w1e-1/epoch=238-step=504768.ckpt \
    --target_categories airplane
```
(Values for `target_categories`: "airplane", "table", "chair")

Get metric measurement for reconstruction
```
python eval.py airplane output/vae_reconstruction/cond_64_airplane/rec_data.npy
```

**LDM Inference - IMPORTANT**
Command line to `sample 1000 voxels` for each category
```
CUDA_VISIBLE_DEVICES=0 \
python inference_ldm.py \
    --config ./configs/diffusion/ldm_cond_64_cfg0.0.yaml \
    --ckpt ckpt/ldm_11-27-205036_cond_64_JSD_cfg0.0/epoch=399-step=26400.ckpt \
    --output_dir ldm/cond_64_JSD_cfg0.0_ep399 \
    --target_category <category>
```
`<category>` is in 1 value in [ "table", "chair", "airplane" ]

Command lines to get `quantitative measurement`:
```
python eval.py chair output/ldm/cond_64_JSD_cfg0.0_ep399/chair/samples.npy
```
```
python eval.py table output/ldm/cond_64_JSD_cfg0.0_ep399/table/samples.npy
```
```
python eval.py airplane output/ldm/cond_64_JSD_cfg0.0_ep399/airplane/samples.npy
```
