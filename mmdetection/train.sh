# Relabel 0626
export CUDA_VISIBLE_DEVICES=0,1
# bash ./tools/my_train.sh \
#     configs/dino/dino-5scale_swin-l_8xb2-12e_tb_relabel_0626.py \
#     2 \
#     --amp

# bash ./tools/my_train.sh \
#     configs/dino/dino-4scale_r50_8xb2-12e_tb_relabel_0626.py \
#     2 \
#     --amp

# bash ./tools/my_train.sh \
#     configs/dino/dino-4scale_r50_8xb2-12e_tb_relabel_0626_resize_1024.py \
#     2 \
#     --amp

# bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_8xb32-300e_tb_relabel_0626.py \
#     2 \
#     --amp

# bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626.py \
#     2 \
#     --amp \
#     --cfg-options load_from=work_dirs/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626/epoch_10.pth work_dir=work_dirs/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626/with_mosaic

# bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0626.py \
#     2 \
#     --amp

# bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune.py \
#     2 \
#     --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_semi_meanteacher.py \
#     2
#     # --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_semi_mixpl.py \
#     2 \
#     --resume

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_semi_meanteacher_sup_baseline.py \
#     2 \
#     --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0819_finetune.py \
#     2 \
#     --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_v5_semi_meanteacher_sup_baseline.py \
#     2 \
#     --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_v5_semi_meanteacher.py \
#     2
#     # --amp

# TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_v5_semi_mixpl.py \
#     2
#     # --resume

TORCH_DISTRIBUTED_DEBUG=INFO bash ./tools/my_train.sh \
    configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_v6_finetune.py \
    2
    # --resume



