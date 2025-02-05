_base_ = './rtmdet_l_8xb32-300e_tb.py'
load_from = "/nasdata2/private/zwlu/detection/TumorBuddingDetection/pretrained_models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192))
max_epochs = 300
base_lr = 0.005
