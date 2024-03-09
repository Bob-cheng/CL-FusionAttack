
# Fusion is Not Enough: Single Modal Attack on Fusion Models for 3D Object Detection

This is the PyTorch implementation of the paper "Fusion is Not Enough: Single Modal Attack on Fusion Models for 3D Object Detection" published in ICLR'2024.

Citation: 

```
@inproceedings{cheng2024fusion,
  title={Fusion is Not Enough: Single Modal Attack on Fusion Models for 3D Object Detection},
  author={Cheng, Zhiyuan and Choi, Hongjun and Feng, Shiwei and Liang, James Chenhao and Tao, Guanhong and Liu, Dongfang and Zuzak, Michael and Zhang, Xiangyu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## Table of Contents
1. [Code preparation](#code-preparation)
2. [Prepare Nuscenes dataset](#prepare-nuscenes-dataset)
3. [Environment preparation](#environment-preparation)
4. [Prepare dataset annotation file and pretrained model](#prepare-dataset-annotation-file-and-pretrained-model)
5. [Config log](#config-log)
6. [Sensitivity heatmap generation](#sensitivity-heatmap-generation)
7. [Patch generation](#patch-generation)

## Code preparation
Clone this repository to folder `~/FusionAttack`

```
git clone <repo_url> FusionAttack
```

## Prepare Nuscenes dataset

Download nuScenes V1.0 full dataset data, mini data and CAN bus expansion data [HERE](https://www.nuscenes.org/download), unzip those files to `/path/to/nuscenes/`. 
```
/path/to/nuscenes/
├── can_bus/
├── maps/
├── mini/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-mini/
├── samples/
├── sweeps/
├── v1.0-test/
├── v1.0-trainval/
```

## Environment preparation 

Create a new conda environment of Python 3.8 called fusionattack:
```
conda create -n fusionattack python=3.8
conda activate fusionattack
```

Install required packages:
```
conda install openmpi=4.0.4 mpi4py=3.0.3 Pillow=8.4.0
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install trimesh==2.35.39 scikit-image==0.19.3 setuptools==59.5.0 tensorboardx timm spconv tqdm torchpack
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmdet==2.20.0 mmsegmentation==0.14.1
pip install nuscenes-devkit
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install gdown
```

Install the customized codebase (mmdet3d):
```
cd ~/FusionAttack/bevfusion
python setup.py develop
```

## Prepare dataset annotation file and pretrained model
### FusionAttack
Link dataset path to subfolder `data/`:
```
cd ~/FusionAttack
# pwd: ~/FusionAttack/
mkdir data 
ln -s /path/to/nuscenes ./data/nuscenes
```
### BEVFusion-MIT
Link dataset path to subfolder `data/`:
```
cd bevfusion
# pwd: ~/FusionAttack/bevfusion/
mkdir data 
ln -s /path/to/nuscenes ./data/nuscenes
```
Prepare the dataset annotations:
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
python tools/create_data.py nuscenes --root-path ./data/nuscenes/mini --out-dir ./data/nuscenes/mini --extra-tag nuscenes --version v1.0-mini
```
Prepare the pretrained model:
```
mkdir pretrained 
wget -P ./pretrained/ https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-det.pth

```

### DeepInteraction
Link dataset path to subfolder `data/`:
```
cd ../DeepInteraction
mkdir data
ln -s /path/to/nuscenes ./data/nuscenes
```
Prepare the pretrained model:
```
mkdir pretrained 
gdown -O ./pretrained/ 1M5eUlXZ8HJ--J53y0FoAHn1QpZGowsdc
```
### BEVFusion-PKU
Link dataset path to subfolder `data/`:
```

cd ../bevfusion2
mkdir data
ln -s /path/to/nuscenes ./data/nuscenes
```
Prepare the pretrained model:
```
mkdir pretrained
gdown -O ./pretrained/ 1tAJA3_5jkE3IAuS_7l8fRNKYSfSEgU5u
```
### UVTR
Link dataset path to subfolder `data/`:
```
cd ../UVTR
mkdir data
ln -s /path/to/nuscenes ./data/nuscenes
```
Generate the unified data info and sampling database for nuScenes dataset:
```
python extra_tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes_unified
python extra_tools/create_data.py nuscenes --root-path ./data/nuscenes/mini --out-dir ./data/nuscenes/mini --extra-tag nuscenes_unified --version v1.0-mini
```
Prepare the pretrained model:
```
mkdir pretrained 
gdown -O ./pretrained/ 1dlxXIS4Cuv6ePxuxMRIaxpG_b1Pk8sqO
```
### TransFusion
Link dataset path to subfolder `data/`:
```
cd ../transfusion
mkdir data
ln -s /path/to/nuscenes ./data/nuscenes
```
Prepare the pretrained model:
```
gdown 1PElbFAAkja8huLTrJRfhBq29mQdVlX7Q
```
### BEVFormer
Link dataset path and `can_bus/` folder to subfolder `data/`:
```
cd ../BEVFormer
mkdir data
ln -s /path/to/nuscenes ./data/nuscenes
ln -s /path/to/nuscenes/can_bus ./data/can_bus
```
Genetate custom annotation files which are different from mmdet3d's:
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
python tools/create_data.py nuscenes --root-path ./data/nuscenes/mini --out-dir ./data/nuscenes/mini --extra-tag nuscenes --version v1.0-mini --canbus ./data
```
Prepare the pretrained model:
```
mkdir ckpts
wget -P ./ckpts/ https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
```

## Config log

Write the desired log dir to `my_config.py`:
```
log_dir = '/path/to/log/dir'
```

Launch Tensorboard to visulize the log:
```
tensorboard --logdir /path/to/log/dir --samples_per_plugin images=200
```

## Sensitivity heatmap generation
Model name mapping:
- BEVFusion-MIT --> `bevfusion`
- BEVFusion-PKU --> `bevfusion2`
- DeepInteraction --> `deepint`
- TransFusion --> `transfusion`
- BEVFormer --> `bevformer`
- UVTR --> `uvtr`

Example command to generate sensitivity heatmap for UVTRT (`uvtr`):
```
cd ~/FusionAttack

CUDA_VISIBLE_DEVICES=2 \
python my_main.py uvtr train \
    --patch_cfg     0 \
    --n_iters        5000 \
    --batch_size    1 \
    --patch_type    whole \
    --mask_step     4 \
    --obj_type      None \
    --test_name     <Name for this test>
```

## Patch generation
### Scene-oriented patch
Example command to generate scene-oriented patch for BEVFusion-PKU (`bevfusion2`):
```
CUDA_VISIBLE_DEVICES=2 \
python my_main.py bevfusion2 train \
    --patch_cfg     3 \
    --n_iters       1500 \
    --batch_size    5 \
    --patch_type    rec \
    --obj_type      None \
    --lr            0.01 \
    --score_tres    0.3 \
    --test_name     <Name for this test>
```

### Object-oriented patch
Example command to generate object-oriented patch for DeepInteraction (`deepint`):
```
CUDA_VISIBLE_DEVICES=2 \
python my_main.py deepint train \
    --patch_cfg     4 \
    --n_iters       1000 \
    --batch_size    5 \
    --patch_type    rec \
    --obj_type      Targeted \
    --lr            0.01\
    --score_tres    0.3\
    --patch_fid     100
    --test_name     <Name for this test>
```

You can see tensorboard for the visualized attack results.