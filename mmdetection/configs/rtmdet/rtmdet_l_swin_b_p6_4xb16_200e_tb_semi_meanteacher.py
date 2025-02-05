_base_ = './rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0711_finetune_no_cp.py'

# max_epochs = 200  # 训练的最大 epoch
# stage2_num_epochs = 20
max_iter = 8000
source_ratio = [2, 4]
unsup_weight = 0.5
train_batch_size_per_gpu=sum(source_ratio)
train_num_workers=10
stage2_num_epochs=20

data_root = _base_.data_root
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
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor
    ),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=0.2,
        cls_pseudo_thr=0.49,
        min_pseudo_bbox_wh=(1e-2, 1e-2),
    ),
    semi_test_cfg=dict(predict_on='teacher')
    # semi_test_cfg=dict(predict_on='student')
)

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.1),
    # dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=image_size, pad_val=114.0),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        # update_pad_shape=False,
        skip_img_without_anno=False),
    
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='AdvancedBlur', blur_limit=3, p=0.2), 
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    
    # dict(
    #     type='CachedMixUp',
    #     img_scale=image_size,
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=20,
    #     pad_val=(114, 114, 114)),
    # dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='AdvancedBlur', blur_limit=3, p=0.2), 
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    # dict(type='PackDetInputs')
]

sup_pipeline = train_pipeline + [
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

sup_pipeline_stage2 = train_pipeline_stage2 + [
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# from '../_base_/datasets/semi_coco_detection.py'
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    # [dict(type='ShearX')],
    # [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(
         type='RandomResize',
         scale=img_size,
        #  ratio_range=(0.6, 1.66),
        ratio_range=(1.0,1.0),
         keep_ratio=True),
    dict(type='RandomCrop',
         crop_size=img_size,
         recompute_bbox=True,
         allow_negative_crop=True),
    dict(type='Pad',
         size=img_size,
         pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(
         type='RandomResize',
         scale=img_size,
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop',
         crop_size=img_size,
         recompute_bbox=True,
         allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.1)),
    dict(type='Pad',
         size=img_size,
         pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args = _base_.backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]


labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=f'semi_anns/train.1@{labeled_precent}.json',
    metainfo=metainfo,
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=_base_.backend_args
)

unlabeled_dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=unlabeled_data_root,
    ann_file=f'semi_anns/train.1@{labeled_precent}-unlabeled.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=_base_.backend_args
)

train_dataloader = dict(
    # _delete_=True,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(
        # type='GroupMultiSourceSampler',
        type='MultiSourceSampler',
        # type='MultiSourceSimpleSampler',
        batch_size=train_batch_size_per_gpu,
        source_ratio=source_ratio),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset', 
        datasets=[labeled_dataset, unlabeled_dataset])
)


val_dataloader = dict(batch_size=train_batch_size_per_gpu,
                      num_workers=train_num_workers)

test_dataloader = val_dataloader


train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=400,
    # dynamic_intervals=[(max_iter - 2000, 1000)]
)
val_cfg = dict(type='TeacherStudentValLoop')

custom_hooks = [
    dict(type='MeanTeacherHook', skip_buffer=False),
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=_base_.base_lr * 0.1, weight_decay=0.01),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# save_epoch_intervals = 10
save_iteration_intervals = 400
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    # interval=save_epoch_intervals,
                    by_epoch=False,
                    interval=save_iteration_intervals,
                    max_keep_ckpts=20),
    logger=dict(type='LoggerHook', interval=20)
)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                #   dict(type='TensorboardVisBackend')
])

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.1,
#         update_buffers=True,
#         priority=49),
#     # dict(
#     #     type='PipelineSwitchHook',
#     #     switch_epoch=max_epochs - stage2_num_epochs,
#     #     switch_pipeline=sup_pipeline_stage2)
# ]

work_dir = f'./work_dirs/rtmdet_l_swin_b_p6_4xb16_{max_iter}i_tb_semi_meanteacher_p_{labeled_precent}_finetune'
# work_dir = f'./work_dirs/rtmdet_l_swin_b_p6_4xb16_{max_epochs}e_tb_semi_meanteacher_p_{labeled_precent}'