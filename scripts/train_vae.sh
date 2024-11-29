CUDA_VISIBLE_DEVICES=0 \
python train_vae.py \
    --config configs/vae/config_nodisc_kl1e-6_64.yaml \
    --exp_name cond_64_nos2c_w1e-1