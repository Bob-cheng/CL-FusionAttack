import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchvision.transforms import Resize, ToPILImage, ToTensor
import torchvision.transforms.functional as vis_F
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import PIL.Image as pil
import copy
import cv2


import mmcv
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar

from scipy.special import softmax
from patch_converter import Patch_converter
from my_config import My_config
import math

def sample_train_val(scenes, object_ids, val_ratio=0.2):
    N = len(scenes)
    scenes = np.array(scenes)
    val_n = int(math.ceil(N * val_ratio))
    rs = np.random.RandomState(17)
    rand_idxs = list(np.arange(N))
    rs.shuffle(rand_idxs)
    # val set
    val_scenes = scenes[rand_idxs[:val_n]]
    val_sorted_idx = np.argsort(val_scenes)
    val_list = val_scenes[val_sorted_idx].tolist()
    
    # train set
    train_scenes = scenes[rand_idxs[val_n:]]
    train_sorted_idx = np.argsort(train_scenes)
    train_list = train_scenes[train_sorted_idx].tolist()
    
    # objects
    if object_ids is None:
        train_objs, val_objs = None, None
    else:
        assert len(scenes) == len(object_ids)
        # object_ids = [ids.append(-1) for ids in object_ids]
        object_ids = np.array(object_ids)
        val_objs = object_ids[rand_idxs[:val_n]]
        val_objs = val_objs[val_sorted_idx].tolist()
        train_objs = object_ids[rand_idxs[val_n:]]
        train_objs = train_objs[train_sorted_idx].tolist()

    return train_list, train_objs, val_list, val_objs

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def scale2PILImage(img, img_norm_cfg):
    img_tensor = img.clone().detach()
    for i in range(img_tensor.shape[0]):
        img_tensor[i] *= img_norm_cfg['std'][i]
        img_tensor[i] += img_norm_cfg['mean'][i]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return ToPILImage()(img_tensor)

def vis_patch(
    patch: torch.Tensor, 
    mask: torch.Tensor, 
    ori_img: np.ndarray, # channel order: BGR
    target='original'):
    patch_converter = Patch_converter()
    patch, mask = patch_converter.convert_patch(patch, mask, target)
    # convert patch channel from RGB to BGR and shape from (3,H,W) to (H,W,3)
    patch, mask = patch[[2, 1, 0]].permute((1, 2, 0)).numpy(), mask.permute((1, 2, 0)).numpy()
    patch *= 255
    patch = patch.astype(np.uint8)
    # mask = mask.astype(np.uint8)
    adv_image = ori_img * (1-mask) + patch * mask
    return adv_image.astype(np.uint8)

def numpy2tensor(x):
    x_tensor = torch.from_numpy(x).unsqueeze(0).cuda()
    return x_tensor

def select_objs(scores, obj_gt_indices, object_ids):
    assert obj_gt_indices is not None
    selected_idx = []
    for i in range(len(obj_gt_indices)):
        if obj_gt_indices[i] in object_ids:
            selected_idx.append(i)
    if len(selected_idx) == 0:
        print("You have succeed! No matching bboxes found.")
    else:
        scores = scores[selected_idx]
    return scores


def save_pic(tensor, i, log_dir=''):
    """
        tensor: image of size (C, H, W)
    """
    unloader = ToPILImage() # tensor to PIL image
    image = tensor.clone().cpu()
    # image = image.squeeze(0)
    image = unloader(image)
    if log_dir != '':
        file_path = os.path.join(log_dir, "{}.png".format(i))
    else:
        file_path = "{}.png".format(i)
    image.save(file_path, "PNG")

def get_input_data(model_name, dataloader, timestamp: int, data_iter=None):
    """
    timestamp == -1 means next input data
    """
    data_iter = iter(enumerate(dataloader)) if data_iter is None else data_iter
    def next_data(data_iter, dataloader):
        try:
            sample_idx, input_data = next(data_iter)
        except StopIteration:
            if timestamp == -1:
                input_data, data_iter, sample_idx = None, None, -1 
            else:
                data_iter = iter(enumerate(dataloader))
                sample_idx, input_data = next(data_iter)
        # print("loaded sample index: ", sample_idx)
        return input_data, data_iter, sample_idx
    input_data, data_iter, sample_idx = next_data(data_iter, dataloader)
    if model_name == 'bevfusion':
        # while timestamp != -1 and input_data['metas'].data[0][0]['timestamp'] != timestamp:
        while timestamp != -1 and input_data['metas'].data[0][0]['pts_filename'][-24:-8] != timestamp:
            input_data, data_iter, sample_idx = next_data(data_iter, dataloader)
    elif model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'bevformer' \
        or model_name == 'transfusion' or model_name == 'autoalign':
        while timestamp != -1 and int(input_data['img_metas'][0].data[0][0]['pts_filename'][-24:-8]) != timestamp:
            input_data, data_iter, sample_idx = next_data(data_iter, dataloader)
    elif model_name == 'uvtr':
        while timestamp != -1 and int(input_data['img_metas'].data[0][0]['pts_filename'][-24:-8]) != timestamp:
            input_data, data_iter, sample_idx = next_data(data_iter, dataloader)
    else:
        raise RuntimeError("No metas info available.")
    return input_data, data_iter, sample_idx

def norm_img(patch: torch.Tensor, img_norm_cfg):
    if img_norm_cfg['mean'][0] > 1:
        patch.data.clamp_(0, 255)
    else:
        patch.data.clamp_(0, 1)
    patch_norm = patch.clone()
    if len(patch.shape) == 4:
        for i in range(patch.shape[1]):
            patch_norm[:, i, ...] -= img_norm_cfg['mean'][i]
            patch_norm[:, i, ...] /= img_norm_cfg['std'][i]
    else:
        for i in range(patch.shape[0]):
            patch_norm[i] -= img_norm_cfg['mean'][i]
            patch_norm[i] /= img_norm_cfg['std'][i]
    return patch_norm

def rev_norm(img_norm: torch.Tensor, img_norm_cfg):
    img = img_norm.clone()
    for i in range(img.shape[0]):
        img[i] *= img_norm_cfg['std'][i]
        img[i] += img_norm_cfg['mean'][i]
    if img_norm_cfg['mean'][0] > 1:
        img.data.clamp_(0, 255)
    else:
        img.data.clamp_(0, 1)
    return img

def create_pseudo_area(patch_area):
    # pseudo area is in the bevfusion2 area_ref
    H_phy, W_phy, alpha, dy, dx = patch_area
    H_img, W_img = 448, 800
    assert H_phy <= H_img // My_config.proj_scale and W_phy <= W_img // My_config.proj_scale
    scale = My_config.proj_scale 
    if np.sum(patch_area[2:5]) == -3:
        pseudo_area = ((H_img - H_phy * scale) // 2, (W_img - W_phy * scale) // 2, H_phy * scale, W_phy * scale)
    else:
        pseudo_area = (H_img - H_phy * scale - 1, (W_img - W_phy * scale) // 2, H_phy * scale, W_phy * scale)
    pseudo_area = tuple([int(v) for v in pseudo_area])
    return pseudo_area

def get_phy_patch(model_relate, patch, mask,  pseudo_area, patch_area, deterministic, rs=None):
    H_phy, W_phy, alpha, dy, dx = patch_area
    metas = get_meta_from_inputdata(model_relate['model_name'], model_relate['input_data'])
    k = model_relate['k']
    transform = metas["lidar2image"][k] if "lidar2image" in metas.keys() else metas["lidar2img"][k]
    if np.sum(patch_area[2:5]) == -3: # patch on the target vehicle
        assert isinstance(model_relate['object_ids'], list), 'the target object ids should be list'
        target_idx = model_relate['object_ids'][0]
        model_name = model_relate['model_name']
        input_data = model_relate['input_data']
        if model_name == "bevfusion" or model_name == 'uvtr':
            bboxes = input_data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        elif model_name == "deepint" or model_name == 'bevfusion2' or model_name == 'bevformer'\
                or model_name == 'transfusion' or model_name == 'autoalign':
            bboxes = input_data["gt_bboxes_3d"][0].data[0][0].tensor.numpy()
        if model_name == 'bevfusion':
            bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes_corners = LiDARInstance3DBoxes(bboxes, box_dim=9).corners.numpy()
        obj_corners = bboxes_corners[target_idx]
        patch_trans, mask_trans = model_relate['pc_3d'].compose_targeted_patch(patch, mask, pseudo_area, 
                                    obj_corners, transform, W_phy, H_phy,
                                    deterministic=deterministic, rs=rs)
    else: # patch on the ground 
        patch_trans, mask_trans = model_relate['pc_3d'].compose_3d_patch(patch, 
                    mask, pseudo_area, transform, W_phy, H_phy, dx=dx, dy=dy, 
                    alpha=alpha, deterministic=deterministic, rs=rs)
    return patch_trans, mask_trans

class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()
        self.ky = np.array([
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]]
        ])
        self.kx = np.array([
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]]
        ])
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(torch.from_numpy(self.kx).float().unsqueeze(0),
                                          requires_grad=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(self.ky).float().unsqueeze(0),
                                          requires_grad=False)

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        height, width = input.size()[2:4]
        gx = self.conv_x(input)
        gy = self.conv_y(input)

        # gy = gy.squeeze(0).squeeze(0)
        # cv2.imwrite('gy.png', (gy*255.0).to('cpu').numpy().astype('uint8'))
        # exit()

        self.loss = torch.sum(gx**2 + gy**2)/2.0
        return self.loss

def get_meta_from_inputdata(model_name, input_data):
    if model_name == 'bevfusion':
        metas = input_data["metas"].data[0][0]
    elif model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'bevformer'\
          or model_name == 'transfusion'or model_name == 'autoalign':
        metas = input_data['img_metas'][0].data[0][0]
    elif model_name == 'uvtr':
        metas = input_data['img_metas'].data[0][0]
    else:
        raise NotImplementedError()
    return metas


def visulize_atk(model_relate, input_data, patch, mask, mode, box_notes=True, score_notes=True):
    """
    model_relate: dict include 'model', 'model_name', 'cfg', 'img_norm_cfg'
    patch and mask: only used for visualize purpose
    input_data: the input_data used to feed the model for prediction
    mode: 'pred' or 'gt'
    """
    target_model = model_relate['model']
    model_name = model_relate['model_name']
    cfg = model_relate['cfg']
    img_norm_cfg = model_relate['img_norm_cfg']
    k = model_relate['k']

    if mode == 'pred':
        target_model.eval()
        with torch.inference_mode():
            outputs = target_model(return_loss=False, rescale=True, **input_data)
    if model_name == 'bevfusion':
        object_classes = cfg.object_classes
    else:
        object_classes = cfg.class_names
    metas = get_meta_from_inputdata(model_name, input_data)
    if mode == 'pred':
        if model_name == 'deepint' or model_name == 'uvtr' or model_name == 'bevformer'\
            or model_name == 'bevfusion2' or model_name == 'transfusion' or model_name == 'autoalign':
            result_dict = outputs[0]['pts_bbox']
        elif model_name == 'bevfusion':
            result_dict = outputs[0]
        bboxes = result_dict["boxes_3d"].tensor.numpy()
        scores = result_dict["scores_3d"].numpy()
        labels = result_dict["labels_3d"].numpy()
        box_idxs = result_dict["obj_gt_indices"].int().numpy() \
                    if "obj_gt_indices" in result_dict.keys() and result_dict["obj_gt_indices"] is not None else None
        indices = scores >= My_config.score_thres

        indices_valid = np.array([box_id != -1 for box_id in box_idxs])
        bboxes = bboxes[indices & indices_valid]
        scores = scores[indices & indices_valid]
        labels = labels[indices & indices_valid]
        box_idxs = box_idxs[indices & indices_valid] if box_idxs is not None else None
    else:
        if model_name == "bevfusion" or model_name == 'uvtr':
            bboxes = input_data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = input_data["gt_labels_3d"].data[0][0].numpy()
        elif model_name == "deepint" or model_name == 'bevfusion2' or model_name == 'bevformer'\
              or model_name == 'transfusion' or model_name == 'autoalign':
            bboxes = input_data["gt_bboxes_3d"][0].data[0][0].tensor.numpy()
            labels = input_data["gt_labels_3d"][0].data[0][0].numpy()
        box_idxs = np.arange(len(labels))
        scores = np.ones_like(labels)
    if model_name == 'bevfusion':
        bboxes[..., 2] -= bboxes[..., 5] / 2
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

    if "img" in input_data:
        # k = 1 if model_name == 'bevfusion2' or model_name == 'transfusion' else 0
        image_path = metas["filename"][k]
        if model_name == 'bevfusion' or model_name == 'bevfusion2' or model_name == 'bevformer'\
            or model_name == 'transfusion' or model_name == 'autoalign':
            image = mmcv.imread(image_path) # BGR
            if mode =='pred' and patch is not None:
                image = vis_patch(patch, mask, image, target='original')
        elif model_name == 'deepint':
            image = rev_norm(input_data['img'][0].data[0][0, k, ...],img_norm_cfg)
            image = image[[2, 1, 0]].permute((1,2,0)).detach().numpy() * 255
            image = image.astype(np.uint8)
            if mode =='pred' and patch is not None:
                image = vis_patch(patch, mask, image, target=model_name)
        elif model_name == 'uvtr':
            image = rev_norm(input_data['img'].data[0][0, k, ...],img_norm_cfg)
            image = image.permute((1,2,0)).detach().numpy()
            image = image.astype(np.uint8)
            if mode =='pred' and patch is not None:
                image = vis_patch(patch, mask, image, target=model_name)
        
        annotated_img = visualize_camera(
            None,
            image,
            bboxes=bboxes,
            labels=labels,
            transform=metas["lidar2image"][k] if "lidar2image" in metas.keys() else metas["lidar2img"][k],
            classes=object_classes,
            thickness=2,
            box_idxs=box_idxs if box_notes else None,
            scores=scores if score_notes else None
        )
        annotated_img = torch.from_numpy(annotated_img).permute((2, 0, 1))[[2, 1, 0]]
    if "points" in input_data:
        if model_name == 'bevfusion' or model_name == 'uvtr':
            lidar = input_data["points"].data[0][0].numpy()
        elif model_name == 'deepint'  or model_name == 'bevfusion2'\
              or model_name == 'transfusion' or model_name == 'autoalign':
            lidar = input_data["points"][0].data[0][0].numpy()
        annotated_lidar_img = visualize_lidar(
            None,
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=object_classes,
        )
        annotated_lidar_img = torch.from_numpy(annotated_lidar_img).permute((2, 0, 1))[:3]
        annotated_lidar_img = vis_F.resize(annotated_lidar_img[:3], (1080, 1080))[[2, 1, 0]]
    else:
        annotated_lidar_img = None
    return annotated_img, annotated_lidar_img

def fromTensor2Heatmap(gray_tensor, max_val=1):
    assert gray_tensor.shape[0] == 1
    gray_map = gray_tensor.clone().detach().cpu().numpy().transpose(1,2,0)
    gray_map = 255 * (np.clip(gray_map, 0, max_val) / max_val)
    gray_map = np.uint8(gray_map)
    gray_map = cv2.cvtColor(gray_map, cv2.COLOR_BGR2RGB)
    heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = ToTensor()(pil.fromarray(heatmap))
    return heatmap

def extract_patch(patch_img: torch.Tensor, paint_mask:torch.Tensor):
    _, H, W = patch_img.shape
    paint_mask_2D = paint_mask.squeeze()
    last_row = False
    last_col = False
    h_range = []
    w_range = []
    for i in range(H):
        has_one = False
        for j in range(W):
            if abs(paint_mask_2D[i, j] - 1)  < 0.5:
                has_one = True
                if not last_row:
                    h_range.append(i)
                    last_row = True
                break
        if not has_one and last_row:
            h_range.append(i)
            last_row = False
    if len(h_range) == 1:
        h_range.append(H)
    
    for j in range(W):
        has_one = False
        for i in range(H):
            if abs(paint_mask_2D[i, j] - 1)  < 0.5:
                has_one = True
                if not last_col:
                    w_range.append(j)
                    last_col = True
                break
        if not has_one and last_col:
            w_range.append(j)
            last_col = False
    if len(w_range) == 1:
        w_range.append(W)
    
    return patch_img[:, h_range[0] : h_range[1], w_range[0] : w_range[1]]

