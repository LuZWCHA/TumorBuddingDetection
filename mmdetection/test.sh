# Relabel 0626
export CUDA_VISIBLE_DEVICES=0,1
# bash ./tools/my_train.sh \
#     configs/dino/dino-5scale_swin-l_8xb2-12e_tb_relabel_0626.py \
#     2 \
#     --amp

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# bash ./tools/my_dist_test.sh \
#     configs/dino/dino-4scale_r50_8xb2-12e_tb_relabel_0626.py \
#     work_dirs/dino-4scale_r50_8xb2-12e_tb_relabel_0626/20240627_094425/epoch_9_copy.pth \
#     2 \
#     --out "work_dirs/dino-4scale_r50_8xb2-12e_tb_relabel_0626/20240627_094425/test_result.pkl"

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
#     --amp


# bash ./tools/my_dist_test.sh \
#     configs/rtmdet/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626_trainset.py \
#     work_dirs/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626/epoch_30.pth \
#     2 \
#     --out "work_dirs/rtmdet_l_swin_b_4xb32-100e_tb_relabel_0626/test_result_e30.pkl"

# bash ./tools/my_dist_test.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0626.py \
#     work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0626/epoch_80.pth \
#     2 \
#     --out "work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0626/test_result_e80.pkl" \
#     --tta

# bash ./tools/my_dist_test.sh \
#     configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune.py \
#     /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune/epoch_50.pth \
#     2 \
#     --out "work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune/test_result_e50_TTA.pkl" \
#     --tta

# bash ./tools/my_dist_test.sh \
#     /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune_trainset.py \
#     /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune/epoch_50.pth \
#     1 \
#     --out "work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune_trainset/test_result_e50.pkl"


bash ./tools/my_dist_test.sh \
    /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16_200e_tb_v5_semi_mixpl.py \
    /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_8000i_tb_v5_semi_mixpl_p_20_finetune/iter_6400.pth \
    2 \
    --out "work_dirs/rtmdet_l_swin_b_p6_4xb16_8000i_tb_v5_semi_mixpl_p_20_finetune/test_result_i6400.pkl"