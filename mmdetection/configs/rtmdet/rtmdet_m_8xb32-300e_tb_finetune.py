_base_ = './rtmdet_l_8xb32-300e_tb.py'

load_from = "/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_m_8xb32-300e_tb/epoch_180.pth"


model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192))


train_dataloader = dict(
    batch_size=128,
    num_workers=24,
    batch_sampler=None,
    filter_cfg=dict(filter_empty_gt=True, filter_ratio=0.5, min_size=32),
    pin_memory=True)

base_lr = 0.001
max_iters = 2_000
interval = 100
stage2_num_iters=0

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, 
        interval=interval,
        max_keep_ckpts=5  # only keep latest 3 checkpoints
    ),
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        interval=interval,
        draw=True
    )    
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=[(max_iters - stage2_num_iters, 1)])

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]