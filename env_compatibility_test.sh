#!/bin/bash


# original bevfusion config
cd bevfusion
torchpack dist-run -np 1 python tools/test.py bevfusion_original_cfg.yaml pretrained/bevfusion-det.pth --eval bbox
cd ..

cd DeepInteraction
python tools/test.py projects/configs/nuscenes/Fusion_0075_refactor.py pretrained/Fusion_0075_refactor.pth --eval=bbox
cd ..

cd UVTR
python3 extra_tools/test.py projects_uvtr/configs/uvtr/multi_modality/uvtr_m_v0075_r101_h5.py pretrained/uvtr_m_v0075_r101_h5.pth --eval=bbox
cd ..

cd bevfusion2
./tools/dist_test.sh configs/bevfusion/bevf_tf_4x8_10e_nusc_aug.py pretrained/bevfusion_tf.pth 1 --eval bbox
cd ..

cd transfusion
./tools/dist_test.sh transfusion_fix.py epoch_6.pth 1 --eval bbox
cd ..

cd bevformer
./tools/dist_test.sh ./projects_bevcam/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth
cd ..
