_base_ = [
    './uvtr_multi_base.py',
]

seed = 17
deterministic=True

model = dict(
    pretrained_img='ckpts/uvtr/pretrain/uvtr_c_r101_h5.pth', # please download the pretrained model from the our git
    img_backbone=dict(depth=101))