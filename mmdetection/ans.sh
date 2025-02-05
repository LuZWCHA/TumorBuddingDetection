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
#     2

CONFIG="configs/dino/dino-4scale_r50_8xb2-12e_tb_relabel_0626.py"
PRED="work_dirs/dino-4scale_r50_8xb2-12e_tb_relabel_0626/20240627_094425/test_result.pkl"
SAVE_PATH="work_dirs/tb_detection_relabel"

CONFIG="configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune.py"
PRED="work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune/test_result_e10.pkl"
SAVE_PATH="work_dirs/tb_detection_relabel_0711"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
export PYTHONPATH=$PYTHONPATH
# export QT_QPA_PLATFORM=xcb

python ./tools/analysis_tools/analyze_results.py \
    $CONFIG \
    $PRED \
    --show-score-thr 0.4 \
    $SAVE_PATH

python ./tools/analysis_tools/eval_metric.py \
    $CONFIG \
    $PRED > ${SAVE_PATH}/"metrics.log"

python ./tools/analysis_tools/my_confusion_matrix.py \
    $CONFIG \
    $PRED \
    $SAVE_PATH \
    --score-thr 0.4 \
    --tp-iou-thr 0.3 \
    --nms-iou-thr 0.7

python ./tools/analysis_tools/my_confusion_matrix.py \
    $CONFIG \
    $PRED \
    $SAVE_PATH \
    --score-thr 0.4 \
    --tp-iou-thr 0.3 \
    --nms-iou-thr 0.7 \
    --norm


