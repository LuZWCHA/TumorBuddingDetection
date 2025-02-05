_base_ = './rtmdet_l_swin_b_4xb32-100e_tb_relabel_0711.py'

# load_from="work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_relabel_0626/epoch_60.pth"
img_size = (1280, 1280)

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2, 1],
        num_heads=[4, 8, 16, 32, 64],
        strides=(4, 2, 2, 2, 2),
        out_indices=(1, 2, 3, 4)),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    bbox_head=dict(
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32, 64])))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='CachedMosaic', img_scale=(1280, 1280), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(2560, 2560),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_size),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    # dict(
    #     type='CachedMixUp',
    #     img_scale=img_size,
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=20,
    #     pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=img_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_size),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# train_dataloader = dict(
#     batch_size=24, num_workers=16, dataset=dict(pipeline=train_pipeline))


# image_size = img_size = _base_.img_size
dataset_type = _base_.dataset_type
metainfo = _base_.metainfo
data_root = _base_.data_root
labeled_percent = 20

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=f'semi_anns/train.1@{labeled_percent}.json',
    metainfo=metainfo,
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=_base_.backend_args
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=labeled_dataset)

val_dataloader = dict(num_workers=10, dataset=dict(pipeline=test_pipeline))

val_evaluator = dict(proposal_nums=(100, 1, 10))
# test_evaluator = val_evaluator
# test_dataloader = val_dataloader

max_epochs = 100
stage2_num_epochs = 20
# base_lr = 0.004

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

interval = 5
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=20  # only keep latest 3 checkpoints
    ),
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        interval=interval * 60,
        draw=True
    )    
)
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

img_scales = [(1280, 1280), (640, 640), (1920, 1920)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='Resize', scale=s, keep_ratio=True)
            #     for s in img_scales
            # ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [
                dict(
                    type='Pad',
                    size=img_size,
                    pad_val=dict(img=(114, 114, 114))),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

work_dir = f"./work_dirs/rtmdet_l_swin_b_p6_4xb16_200e_tb_semi_meanteacher_sup_baseline_p_{labeled_percent}"