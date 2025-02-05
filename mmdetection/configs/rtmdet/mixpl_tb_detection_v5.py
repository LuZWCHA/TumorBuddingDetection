# dataset settings
semi_dataset_type = 'TBDataset'
semi_data_root = '/nasdata2/dataset/research_data/coco_format/v5/'

semi_metainfo = {
    'classes': ('TB', 'PDC'),
    'palette': [
        (220, 20, 60), (120, 231, 100)
    ]
}
semi_image_size = (1024, 1024)

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

scale = [(640, 640), (1280, 1280)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

semi_albu_train_transforms = [
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

semi_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=semi_image_size, pad_val=114.0),
    dict(
        type='RandomResize',
        scale=semi_image_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=semi_image_size),
    dict(
        type='Albu',
        transforms=semi_albu_train_transforms,
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
    dict(type='Pad', size=semi_image_size, pad_val=dict(img=(114, 114, 114))),
    
    # dict(
    #     type='CachedMixUp',
    #     img_scale=image_size,
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=20,
    #     pad_val=(114, 114, 114)),
    # dict(type='PackDetInputs')
]

# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = semi_train_pipeline + [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomResize', scale=scale, keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(
         type='RandomResize',
         scale=semi_image_size,
        #  ratio_range=(0.6, 1.66),
        ratio_range=(1.0,1.0),
         keep_ratio=True),
    dict(type='RandomCrop',
         crop_size=semi_image_size,
         recompute_bbox=True,
         allow_negative_crop=True),
    dict(type='Pad',
         size=semi_image_size,
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
         scale=semi_image_size,
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop',
         crop_size=semi_image_size,
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
         size=semi_image_size,
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
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmptyAnnotations'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=image_size, keep_ratio=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# batch_size = 8
# num_workers = 8
# There are two common semi-supervised learning settings on the coco dataset：
# (1) Divide the train2017 into labeled and unlabeled datasets
# by a fixed percentage, such as 1%, 2%, 5% and 10%.
# The format of labeled_ann_file and unlabeled_ann_file are
# instances_train2017.{fold}@{percent}.json, and
# instances_train2017.{fold}@{percent}-unlabeled.json
# `fold` is used for cross-validation, and `percent` represents
# the proportion of labeled data in the train2017.
# (2) Choose the train2017 as the labeled dataset
# and unlabeled2017 as the unlabeled dataset.
# The labeled_ann_file and unlabeled_ann_file are
# instances_train2017.json and image_info_unlabeled2017.json
# We use this configuration by default.
labeled_dataset = dict(
    type=semi_dataset_type,
    data_root=semi_data_root,
    metainfo=semi_metainfo,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline)

unlabeled_dataset = dict(
    type=semi_dataset_type,
    data_root=semi_data_root,
    metainfo=semi_metainfo,
    ann_file='annotations/instances_unlabeled2017.json',
    data_prefix=dict(img='unlabeled2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline)


# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='val.json',
#         metainfo=metainfo,
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=test_pipeline))

# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val2017.json',
#     metric='bbox',
#     classwise=True,
#     format_only=False,
#     outfile_prefix='./work_dirs/tb_detection_relabel/balanced_sampled_test_0711',
#     )
# test_evaluator = val_evaluator