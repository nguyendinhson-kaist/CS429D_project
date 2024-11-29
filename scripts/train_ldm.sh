CUDA_VISIBLE_DEVICES=0 \
python train_ldm.py \
    --config configs/diffusion/ldm_cond_64_cfg0.0.yaml \
    --exp_name cond_64_JSD_cfg0.0 
