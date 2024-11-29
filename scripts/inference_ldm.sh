CUDA_VISIBLE_DEVICES=0 \
python inference_ldm.py \
    --config ./configs/diffusion/ldm_cond_64_cfg0.0.yaml \
    --ckpt ckpt/ldm_11-27-205036_cond_64_JSD_cfg0.0/epoch=399-step=26400.ckpt \
    --output_dir ldm/cond_64_JSD_cfg0.0_ep399 \
    --target_category airplane