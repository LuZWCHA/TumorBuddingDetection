_base_ = './dino-4scale_r50_8xb2-12e_tb_relabel_0626.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='MyDetLocalVisualizer',
    vis_backends=vis_backends,
    
)
# /nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/mmdet/visualization/my_local_visualizer.py

default_hooks = dict(
    visualization=dict(
        score_thr=0.15,
        draw=True,
        test_out_dir="test_results",
        interval=2, type='mmdet.DetVisualizationHook')
)


# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='test.json',
#         metainfo=metainfo,
#         data_prefix=dict(img='test/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))

# # test_dataloader = val_dataloader
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'test.json',
#     metric='bbox',
#     classwise=True,
#     format_only=True,
#     outfile_prefix='./work_dirs/tb_detection/test',
#     backend_args=backend_args)