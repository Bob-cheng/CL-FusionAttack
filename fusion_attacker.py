
import os
import torch
import torch.optim as optim
import copy
import math

import numpy as np
from model_loader import Model_loader
from transfer_enabler import Transfer_enabler
from patch_converter import Patch_converter
from PIL import Image
from patch_converter_3d import Patch_converter_3d
from my_utils import get_phy_patch, get_input_data, select_objs, extract_patch,\
                    norm_img, visulize_atk, save_pic, fromTensor2Heatmap, \
                    get_meta_from_inputdata, create_pseudo_area, TVLoss
from my_config import My_config
from torchvision.transforms import Resize, InterpolationMode, ToTensor
from mwUpdater import MaskWeightUpdater
from typing import Union
from torch.optim.lr_scheduler import StepLR



class FusionAttacker(object):
    def __init__(self, model_names) -> None:
        self.model_loader = Model_loader()
        self.model_names = model_names
        self.model_relates = [] # list of dict, [{model, dataloader, cfg}, ]
        for name in self.model_names:
            model, dataloader, cfg = self.model_loader.load_model(name)
            self.model_relates.append(dict(
                model_name=name,
                model=model,
                dataloader=dataloader,
                cfg=cfg,
                data_iter=iter(enumerate(dataloader))
            ))
        self.default_atk_cfg={
            'n_iters': 10000,
            'loss_type': 'log_score_loss',
            'lr': 1e-3,
            'batch_size': 10,
            'replace': None
        }
        self.patch_converter = Patch_converter()
        self.init_img_shape = self.patch_converter.bevfusion_shape
        self.bev2_img_shape = self.patch_converter.deepint_shape # also bevfusion2 shape
        self.best_mean_score = 1
        self.best_loss = 1e5
        self.mask_wt = 1
        self.adv_wt = 1
        self.scene_idx = 0

    def _get_adv_loss(self, model_name, model, input_data, loss_type='score_loss', object_ids:list=None):
        if loss_type == 'all_loss':
            model.train()
            outputs = model(return_loss=False, rescale=True, **input_data)
            adv_loss = - outputs['loss/object/loss_heatmap'] \
                        - outputs['loss/object/layer_-1_loss_bbox'] \
                        - outputs['loss/object/layer_-1_loss_cls']
            log_info = "Heatmap_loss: {:.6f}, class_loss: {:.6f}, bbox_loss: {:.6f}".format(
                outputs['loss/object/loss_heatmap'].item(),
                outputs['loss/object/layer_-1_loss_cls'].item(),
                outputs['loss/object/layer_-1_loss_bbox'].item()
            )
            mean_score = 0
        elif loss_type == 'score_loss' or loss_type == 'score_loss_FP':
            model.eval()
            outputs = model(return_loss=False, rescale=True, **input_data)
            if model_name == 'bevfusion':
                results_dict = outputs[0]
            elif model_name == 'deepint' or model_name == 'uvtr' or model_name == 'bevformer'\
                or model_name == 'bevfusion2' or model_name == 'transfusion' or model_name == 'autoalign':
                results_dict = outputs[0]['pts_bbox']
            scores = results_dict["scores_3d"]
            if object_ids is not None:
                scores = select_objs(scores, results_dict["obj_gt_indices"], object_ids)
            adv_loss = torch.nn.MSELoss()(scores.float(), torch.zeros_like(scores).float().to(scores.device))
            mean_score = torch.mean(scores)
            log_info = f"Score_loss: {adv_loss:.6f}, Mean score: {mean_score:.6f}"
            if loss_type == 'score_loss_FP':
                adv_loss *= -1
        elif loss_type == 'log_score_loss':
            model.eval()
            outputs = model(return_loss=False, rescale=True, **input_data)
            if model_name == 'bevfusion':
                results_dict = outputs[0]
            elif model_name == 'deepint' or model_name == 'uvtr' or model_name == 'bevformer'\
                or model_name == 'bevfusion2' or model_name == 'transfusion' or model_name == 'autoalign':
                results_dict = outputs[0]['pts_bbox']
            scores = results_dict["scores_3d"]
            # scores = scores[scores >= 0.3]
            if object_ids is not None:
                scores = select_objs(scores, results_dict["obj_gt_indices"], object_ids)
            adv_loss = torch.log(torch.mean(scores.float()))
            mean_score = torch.mean(scores)
            log_info = f"Log_score_loss: {adv_loss:.12f}, Mean score: {mean_score:.6f}"
        elif loss_type == 'dense_heatmap_loss':
            model.eval()
            outputs = model(return_loss=False, rescale=True, **input_data)
            if model_name == 'bevfusion':
                results_dict = outputs[0]
            elif model_name == 'deepint' or model_name == 'uvtr' or model_name == 'bevformer' \
                or model_name == 'bevfusion2' or model_name == 'transfusion' or model_name == 'autoalign':
                results_dict = outputs[0]['pts_bbox']
            assert 'dense_heatmap' in results_dict.keys()
            dense_heatmap = results_dict['dense_heatmap'] # shape: 10, 180, 180
            scores = results_dict["scores_3d"]
            mean_score = torch.mean(scores)
            adv_loss = torch.mean(dense_heatmap.sigmoid().square())
            log_info = f"Log_score_loss: {adv_loss:.12f} (dense_heatmap_loss), Mean score: {mean_score:.6f}"
        if self.best_mean_score >  mean_score:
            self.best_mean_score = mean_score
        return adv_loss, mean_score, log_info

    def attack_single(self, model_relate, patch, mask, object_ids, loss_type):
        patch_cvt, mask_cvt = self.patch_converter.convert_patch(patch, mask, model_relate['model_name'])
        adv_input_data = self.apply_patch(model_relate, patch_cvt, mask_cvt)
        # save_pic(adv_image, 0)
        adv_loss, adv_mean_score, adv_info = self._get_adv_loss(
                                                model_relate['model_name'],
                                                model_relate['model'], 
                                                adv_input_data, 
                                                loss_type, 
                                                object_ids)
        return adv_loss, adv_mean_score, adv_info
            
    
    def apply_patch(self, model_relate, patch, mask):
        model_name = model_relate['model_name']
        input_data = model_relate['input_data']
        img_norm_cfg = model_relate['img_norm_cfg']
        k = model_relate['k']
        if model_name == 'uvtr' or model_name == 'autoalign' or model_name == 'bevformer':
            patch_norm = norm_img(patch * 255, img_norm_cfg)
        else:
            patch_norm = norm_img(patch, img_norm_cfg)
        if len(patch.shape) == 3:
            ori_image = model_relate['ori_image']
            new_image = ori_image.clone().detach()
            adv_image = new_image * (1-mask) + patch_norm * mask
            if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
                or model_name == 'autoalign' or model_name == 'bevformer':
                input_data['img'][0].data[0].detach_()
                input_data['img'][0].data[0][0, k, ...] = adv_image
            elif model_name == 'bevfusion' or model_name == 'uvtr':
                input_data['img'].data[0].detach_()
                input_data['img'].data[0][0, k, ...] = adv_image
        elif len(patch.shape) == 4:
            all_ori_image = model_relate['all_ori_image']
            new_image = all_ori_image.clone().detach()
            adv_image = new_image * (1-mask) + patch_norm * mask
            if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
                or model_name == 'autoalign' or model_name == 'bevformer':
                input_data['img'][0].data[0].detach_()
                input_data['img'][0].data[0][0, ...] = adv_image
            elif model_name == 'bevfusion' or model_name == 'uvtr':
                input_data['img'].data[0].detach_()
                input_data['img'].data[0][0, ...] = adv_image

        return input_data
    
    def log(self, i, loss_type, total_adv_loss, img_step=100):
        if i % 1 == 0:
            print(  f"Iteration: {i}, "+
                    f"adv_loss: {total_adv_loss:.6f}, "+
                    f"best_loss: {self.best_loss:.6f}, "+
                    f"best_mean_score: {self.best_mean_score:.6f}"
                )
            My_config.tb_logger.add_scalar(f'Fusion_attack/{loss_type}', total_adv_loss, i)
            My_config.tb_logger.add_scalar(f'Fusion_attack/best_loss', self.best_loss, i)
            My_config.tb_logger.add_scalar(f'Fusion_attack/mean_score', self.best_mean_score, i)
        if i % img_step == 0:
            for model_relate in self.model_relates:
                if i == 0:
                    anno_img, anno_lidar = visulize_atk(model_relate, model_relate['input_data'], None, None, 'gt')
                    My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/gt_img", anno_img, i)
                    if anno_lidar is not None:
                        My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/gt_lidar", anno_lidar, i)
                
                if 'pc_3d' in model_relate.keys():
                    patch, mask = get_phy_patch(model_relate, self.best_patch, self.mask, self.pseudo_area, self.patch_area, True)
                else:
                    patch, mask = self.patch_converter.convert_patch(self.best_patch, 
                                                                    self.mask, 
                                                                    model_relate['model_name'])
                adv_input_data = self.apply_patch(model_relate, patch, mask)
                if len(patch.shape) == 3:
                    anno_img, anno_lidar = visulize_atk(model_relate, adv_input_data, patch, mask, 'pred')
                    My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/patch_img", extract_patch(self.best_patch, self.mask), i)
                else:
                    k = model_relate['k']
                    anno_img, anno_lidar = visulize_atk(model_relate, adv_input_data, patch[k], mask[k], 'pred')

                My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/adv_img", anno_img, i)
                if anno_lidar is not None:
                    My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/adv_lidar", anno_lidar, i)
        if (i+1) % 100 == 0:
            # save patch and mask
            torch.save(self.latest_patch, os.path.join(My_config.log_dir, "bevfusion_patch_latest.pt"))
            torch.save(self.best_patch, os.path.join(My_config.log_dir, "bevfusion_patch.pt"))
            torch.save(self.mask, os.path.join(My_config.log_dir, "bevfusion_mask.pt"))
    
    def from_init_to_mask(self, patch_type ,values: torch.Tensor, mask_size):
        if patch_type == 'whole':
            gamma = 1
            mask = torch.tanh(values * gamma) * 0.5 + 0.5
            mask = Resize(mask_size, InterpolationMode.NEAREST)(mask)
            mask = torch.clamp(mask, 0, 1)
        elif patch_type == 'dynamic':
            l, r, t, b = values
            H, W = mask_size
            x = torch.arange(0, H)
            y = torch.arange(0, W)
            grid_x, grid_y = torch.meshgrid(x, y)
            grid_x.requires_grad = False
            grid_y.requires_grad = False
            mask = 0.25 * (-torch.tanh(grid_x-t) * torch.tanh(grid_x-b) + 1) * (-torch.tanh(grid_y-l) * torch.tanh(grid_y-r) + 1)
            mask = mask.clamp(0, 1).unsqueeze(0)
        return mask

    def get_mask_patch(self, patch_type, patch_area, mask_step=2, physical=False):
        t, l, h, w = patch_area
        C, H, W = self.init_img_shape
        if patch_type == 'rec':
            if physical:
                C, H, W = self.bev2_img_shape
            mask = torch.zeros([1, H, W])
            mask[:, t:t+h, l:l+w] = 1
            patch = torch.rand((C, H, W)).requires_grad_(True)
            return patch, mask
        elif patch_type == 'whole':
            # all mask
            # mask_init = torch.zeros([6, 1, H // mask_step, W // mask_step]).requires_grad_(True)
            # patch = torch.rand([6, C, H, W]).requires_grad_(True)

            # front-camera mask
            mask_init = torch.zeros([1, H // mask_step, W // mask_step]).requires_grad_(True)
            patch = torch.rand([C, H, W]).requires_grad_(True)

            return patch, mask_init
        elif patch_type == 'dynamic':
            mask_init = torch.tensor([0, W, 0, H]).float().requires_grad_(True)
            patch = torch.rand(self.init_img_shape).requires_grad_(True)
            return patch, mask_init
            
    def next_scene(self, model_relate, attack_timestamp: Union[int, list], object_ids, img_rep_dict=None):
        if type(attack_timestamp) is list:
            if self.scene_idx == 0:
                model_relate['data_iter'] = iter(enumerate(model_relate['dataloader']))
            current_ts = attack_timestamp[self.scene_idx]
            curr_objects_ids = object_ids[self.scene_idx] if object_ids is not None else None
            self.scene_idx += 1
            self.scene_idx %= len(attack_timestamp)
        else:
            current_ts = attack_timestamp
            curr_objects_ids = object_ids
        input_data, data_iter, _ = get_input_data(
                                    model_relate['model_name'], 
                                    model_relate['dataloader'], 
                                    current_ts,
                                    model_relate['data_iter'])
        k = 1 if model_relate['model_name'] == 'bevfusion2' or model_relate['model_name'] == 'transfusion' else 0

        new_img_tensor = None
        if img_rep_dict is not None and current_ts in img_rep_dict.keys():
            new_img_filename = os.path.join(My_config.physical_photo_dir, img_rep_dict[current_ts])
            metas = get_meta_from_inputdata(model_relate['model_name'], input_data)
            metas["filename"][k] = new_img_filename
            new_img_tensor = ToTensor()(Image.open(new_img_filename))
            assert new_img_tensor.shape == self.patch_converter.original_shape, "new image shape error"

        if model_relate['model_name'] == 'bevfusion': # [0, 1] image
            img_norm_cfg = input_data['metas'].data[0][0]['img_norm_cfg']
            if max(img_norm_cfg['mean']) > 1:
                img_norm_cfg['mean'] =  np.array(img_norm_cfg['mean']) / 255
                img_norm_cfg['std'] = np.array(img_norm_cfg['std']) /  255
            if new_img_tensor is not None:
                input_data['img'].data[0][0, k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor, img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    model_relate['model_name']
                )[0]
            ori_image = input_data['img'].data[0][0, k, ...].clone().detach()
            all_ori_image = input_data['img'].data[0][0, ...].clone().detach()
        elif model_relate['model_name'] == 'deepint' or model_relate['model_name'] == 'bevfusion2' \
            or model_relate['model_name'] == 'transfusion': # [0, 1] image
            img_norm_cfg = copy.deepcopy(input_data['img_metas'][0].data[0][0]['img_norm_cfg'])
            if max(img_norm_cfg['mean']) > 1:
                img_norm_cfg['mean'] /= 255
                img_norm_cfg['std'] /= 255
            if new_img_tensor is not None:
                input_data['img'][0].data[0][0, k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor, img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    model_relate['model_name']
                )[0]
            ori_image = input_data['img'][0].data[0][0, k, ...].clone().detach()
            all_ori_image = input_data['img'][0].data[0][0, ...].clone().detach()
        elif model_relate['model_name'] == 'autoalign' or model_relate['model_name'] == 'bevformer': # [0, 255] image
            img_norm_cfg = input_data['img_metas'][0].data[0][0]['img_norm_cfg']
            if new_img_tensor is not None:
                input_data['img'][0].data[0][0, k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor*255, img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    model_relate['model_name']
                )[0]
            ori_image = input_data['img'][0].data[0][0, k, ...].clone().detach()
            all_ori_image = input_data['img'][0].data[0][0, ...].clone().detach()
        elif model_relate['model_name'] == 'uvtr': # [0, 255] image
            img_norm_cfg = input_data['img_metas'].data[0][0]['img_norm_cfg']
            if new_img_tensor is not None:
                input_data['img'].data[0][0, k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor*255, img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    model_relate['model_name']
                )[0]
            ori_image = input_data['img'].data[0][0, k, ...].clone().detach()
            all_ori_image = input_data['img'].data[0][0, ...].clone().detach()

        model_relate['input_data'] = input_data
        model_relate['img_norm_cfg'] = img_norm_cfg
        model_relate['ori_image'] = ori_image
        model_relate['all_ori_image'] = all_ori_image
        model_relate['data_iter'] = data_iter
        model_relate['k'] = k
        model_relate['object_ids'] = curr_objects_ids
        return input_data

    def attack(self, attack_timestamp: Union[int,list], patch_area, object_ids=None, atk_cfg_set=None):
        atk_cfg = self.default_atk_cfg
        if atk_cfg_set is not None:
            atk_cfg.update(atk_cfg_set)
        patch_type = atk_cfg['patch_type']
        mask_step = atk_cfg['mask_step']
        adp_thres = atk_cfg['adp_thres'] if atk_cfg['adp_thres'] > 0 else None
        self.mask_wt = atk_cfg['mask_weight']
        self.patch_area = patch_area

        ## Transfer_enabler and Patch_converter_3d both controls the transformation of patch.
        # Transfer_enabler controls 1. color change and 2. naive patch's area transformation.
        # Patch_converter_3d controls physical patch's location transformation.
        self.transfer_enabler =  Transfer_enabler(rot_deg=0, 
            scale_range=None, #(0.9, 1.1),
            trans_range=None, #(0.01, 0.01),
            brightness = 0.5, #0.3,
            contrast=0.1, #0.1,
            saturation=0.3 #0.1,
        )
        for model_relate in self.model_relates:
            for params in model_relate['model'].parameters():
                params.requires_grad_(False)
            input_data = self.next_scene(model_relate, attack_timestamp, object_ids, img_rep_dict=atk_cfg['replace'])
            if attack_timestamp == -1:
                print("Model {} has {} training scenes.".format(
                    model_relate['model_name'],
                    len(model_relate['dataloader'])
                ))
            elif type(attack_timestamp) is list:
                print("Model {} has {} training scenes.".format(
                    model_relate['model_name'],
                    len(attack_timestamp)
                ))
            ## for physical patch
            if len(patch_area) == 5:
                model_relate['pc_3d'] = Patch_converter_3d(
                                            model_relate['model_name'],
                                            resize_range=np.arange(0.7, 1.05 + 0.01, 0.01),
                                            # angle_range=np.arange(-4, 4+2, 2),
                                            # dist_range_x=np.arange(-0.2, 0.2+0.1, 0.1),
                                            # dist_range_y=np.arange(7, 7.5 + 0.1, 0.1),
                                        )
            ## log benign prediction:
            anno_img, anno_lidar = visulize_atk(model_relate, input_data, None, None, 'pred')
            My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/adv_img", anno_img, -1)
            if anno_lidar is not None:
                My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/adv_lidar", anno_lidar, -1)

        C, H, W = self.init_img_shape
        if len(patch_area) == 4:
            patch, mask_init = self.get_mask_patch(patch_type, patch_area, mask_step)
        else: # physical patch
            assert patch_type == 'rec'
            pseudo_area = create_pseudo_area(patch_area)
            patch, mask_init = self.get_mask_patch(patch_type, pseudo_area, mask_step, physical=True)
            self.pseudo_area = pseudo_area
            
        if patch_type == 'whole':
            mask = self.from_init_to_mask(patch_type, mask_init, (H, W))
            optimizer = optim.Adam([patch, mask_init], atk_cfg['lr'], betas=(0.5, 0.9))
        elif patch_type == 'dynamic':
            mask = self.from_init_to_mask(patch_type, mask_init, (H, W))
            optimizer = optim.Adam([patch], atk_cfg['lr'], betas=(0.5, 0.9))
            mask_optimizer = optim.Adam([mask_init], atk_cfg['mask_lr'], betas=(0.5, 0.9))
            mw_updater = MaskWeightUpdater(self.mask_wt, 0.01387, atk_cfg['n_iters'])
        elif patch_type == 'rec':
            mask = mask_init
            optimizer = optim.Adam([patch], atk_cfg['lr'], betas=(0.5, 0.9))
            scheduler = StepLR(optimizer, 100, 0.97)
        
        self.mask = mask
        self.latest_patch = patch.clone().detach()
        self.best_loss = 1e5
        self.best_mean_score = 1
        

        model_num = len(self.model_relates)

        self.tv_loss_module = TVLoss()

        for i in range(atk_cfg['n_iters']):
            # scale the patch to 0-255 for some models are performed at apply_patch()
            patch.data.clamp_(0, 1)
            total_adv_loss = 0
            total_mask_loss = 0
            total_all_loss = 0
            total_patch_grad = 0
            total_mask_grad = 0
            total_pass = 0
            batch_size = atk_cfg['batch_size']
            for atk_idx in range(model_num): # for each model
                model_relate = self.model_relates[atk_idx]
                for _ in range(batch_size): # for a batch of transformations
                    if patch_type == 'rec':
                        if atk_cfg['trans']:
                            patch_trans, mask_trans = self.transfer_enabler.random_trans(patch, mask, patch_area, model_relate)
                        else:
                            if len(patch_area) == 5: # physical patch
                                patch_trans, mask_trans = get_phy_patch(model_relate, patch, mask, 
                                                    pseudo_area, patch_area, deterministic=True)
                            else:
                                patch_trans, mask_trans = patch, mask 
                    elif patch_type == 'whole' or patch_type == 'dynamic':
                        assert batch_size == 1
                        mask = self.from_init_to_mask(patch_type, mask_init, (H, W))
                        patch_trans, mask_trans = patch, mask
                    adv_loss, adv_mean_score, adv_info = self.attack_single(
                                model_relate, 
                                patch_trans, 
                                mask_trans, 
                                model_relate['object_ids'], 
                                atk_cfg['loss_type'])
                    if atk_cfg['tv_loss']:
                        tv_loss = 0.0001 * self.tv_loss_module(patch)
                        print("tv_loss: ", tv_loss)
                        adv_loss += tv_loss

                    if patch_type == 'rec':
                        all_loss = adv_loss
                    elif patch_type == 'whole':
                        if adp_thres is not None:
                            # adaptive control
                            self.adv_wt = 0 if adv_loss < math.log(adp_thres) else 1
                        adv_loss *= self.adv_wt
                        mask_loss = torch.mean(mask) * self.mask_wt
                        all_loss = adv_loss + mask_loss               
                    elif patch_type == 'dynamic':
                        mask_ratio = mw_updater.get_mask_ratio((H, W), mask_init)
                        mask_wt = mw_updater.step(mask_ratio)
                        mask_loss = (mask_ratio) * mask_wt
                        all_loss = adv_loss + mask_loss  
                        mask_optimizer.zero_grad()
                    optimizer.zero_grad()     
                    all_loss.backward()
                    total_pass += 1
                    total_patch_grad += patch.grad
                    total_adv_loss += adv_loss.item()
                    total_all_loss += all_loss.item()
                    if patch_type == 'whole' or patch_type == 'dynamic':
                        total_mask_grad += mask_init.grad
                        total_mask_loss += mask_loss

                    if attack_timestamp == -1 or type(attack_timestamp) is list: # change scene
                        self.next_scene(model_relate, attack_timestamp, object_ids, img_rep_dict=atk_cfg['replace'])
                    # end for batch
                # end for models
            total_adv_loss /= total_pass
            total_all_loss /= total_pass
            total_patch_grad /= total_pass
            patch.grad = total_patch_grad
            if patch_type == 'whole' or patch_type == 'dynamic':
                total_mask_loss /= total_pass
                total_mask_grad /= total_pass
                mask_init.grad = total_mask_grad
            
            if total_all_loss < self.best_loss or patch_type == 'dynamic':
                self.best_loss = total_all_loss
                self.best_patch = patch.clone().detach()
                self.mask = mask.clone().detach()
            self.latest_patch = patch.clone().detach()
            
            optimizer.step()
            
            if patch_type == 'dynamic':
                mask_optimizer.step()
            elif patch_type == 'rec':
                scheduler.step()

            self.log(i,atk_cfg['loss_type'], total_adv_loss, img_step=50)
            if patch_type == 'whole' or patch_type == 'dynamic':
                # mask log
                My_config.tb_logger.add_scalar(f'Fusion_attack/mask_loss', total_mask_loss, i)
                if patch_type == 'dynamic':
                    My_config.tb_logger.add_scalar(f'Fusion_attack/mask_ratio', mask_ratio, i)
                
                print(f"Iteration: {i}, mask_loss: {total_mask_loss:.6f}")
                if i % 50 == 0:
                    if len(mask.shape) == 4:
                        k = model_relate['k']
                        mask_heatmap = fromTensor2Heatmap(mask[k], max_val=0.7)
                    else:
                        mask_heatmap = fromTensor2Heatmap(mask, max_val=0.7)
                    My_config.tb_logger.add_image(f"Fusion_attack/{model_relate['model_name']}/mask", mask_heatmap, i)
                    torch.save(mask.clone().detach(), os.path.join(My_config.log_dir, f"sensitivity_mask_{i}.pt"))