CUDA_VISIBLE_DEVICES=0 \
python inference_vae.py \
--config configs/vae/config_nodisc_kl1e-6.yaml \
--exp_name cond_64_airplane_w3e-1 \
--ckpt logs/train_vae_11-22-131429_cond_64_w3e-1/epoch=189-step=50160.ckpt \
--target_categories airplane