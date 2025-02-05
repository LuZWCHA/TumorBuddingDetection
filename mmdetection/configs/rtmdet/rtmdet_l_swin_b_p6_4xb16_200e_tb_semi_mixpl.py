_base_ = [
    './rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune_no_cp.py',
    './mixpl_tb_detection.py'
]

# custom_imports = dict(
#     imports=['mixpl.mixpl'], allow_failed_imports=False)
# max_epochs = 200  # 训练的最大 epoch
# stage2_num_epochs = 20
max_iter = 8000
source_ratio = [2, 4]
unsup_weight = 0.5
train_batch_size_per_gpu=sum(source_ratio)
train_num_workers=10
stage2_num_epochs=20

data_root = _base_.semi_data_root
unlabeled_data_root = data_root

labeled_precent = 20 # n/100

detector = _base_.model
image_size = img_size = _base_.img_size
dataset_type = _base_.dataset_type
metainfo = _base_.metainfo

# load_from="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_4000i_tb_semi_meanteacher_p_20/epoch_1.pth"
load_from="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_200e_tb_semi_meanteacher_sup_baseline_p_20/epoch_40.pth"
# load_from="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_8000i_tb_semi_meanteacher_p_20_finetune/iter_800.pth"


model = dict(
    _delete_=True,
    type='MixPL',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        least_num=1,
        cache_size=8,
        mixup=True,
        mosaic=True,
        mosaic_shape=[(1024, 1024)],
        mosaic_weight=0.5,
        erase=True,
        erase_patches=(1, 10),
        erase_ratio=(0, 0.05),
        erase_thr=0.7,
        cls_pseudo_thr=0.49,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=unsup_weight,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = f'semi_anns/train.1@{labeled_precent}.json'
unlabeled_dataset.ann_file = f'semi_anns/train.1@{labeled_precent}-unlabeled.json'
labeled_dataset.data_prefix = dict(img='train/')
unlabeled_dataset.data_prefix = dict(img='train/')

# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     sampler=dict(batch_size=train_batch_size_per_gpu, source_ratio=source_ratio),
#     dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

train_dataloader = dict(
    _delete_=True,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        source_ratio=source_ratio,
        batch_size=train_batch_size_per_gpu),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(batch_size=train_batch_size_per_gpu,
                      num_workers=train_num_workers)

test_dataloader = val_dataloader

# training schedule for 90k
train_cfg = dict(
    _delete_=True, 
    max_iters=max_iter, 
    type='IterBasedTrainLoop', 
    val_interval=400)

val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=_base_.base_lr * 0.1, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=400, max_keep_ckpts=5))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', skip_buffer=False)]
# resume=True

work_dir = f'./work_dirs/rtmdet_l_swin_b_p6_4xb16_{max_iter}i_tb_semi_mixpl_p_{labeled_precent}_finetune'
# work_dir = f'./work_dirs/rtmdet_l_swin_b_p6_4xb16_{max_epochs}e_tb_semi_meanteacher_p_{labeled_precent}'