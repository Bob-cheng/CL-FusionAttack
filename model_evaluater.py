import os
import torch
import copy
import numpy as np
import PIL.Image as pil

from patch_converter import Patch_converter
from my_utils import select_objs, norm_img, visulize_atk, get_input_data, extract_patch,\
                        get_phy_patch, create_pseudo_area,rev_norm,get_meta_from_inputdata
from my_config import My_config
from transfer_enabler import Transfer_enabler
from nusc_eval_metric import NuscEvalMetric
from patch_converter_3d import Patch_converter_3d
from torchvision.transforms import ToTensor, ToPILImage

class Model_evaluater(object):
    def __init__(self, model_name, target_model, model_dataloader, cfg) -> None:
        """
        target_model: model to be evaluated
        model_dataloader: dataloader of the model to be evaluated
        model_name: one of ['bevfusion', 'deepint', 'uvtr', 'bevfusion2', 'transfusion', 'autoalign', 'bevformer']
        """
        self.target_model = target_model
        self.model_dataloader = model_dataloader
        self.model_name = model_name # target model name
        self.patch_converter = Patch_converter()
        self.cfg = cfg
        self.img_norm_cfg = None
        self.k = 1 if self.model_name == 'bevfusion2' or self.model_name == 'transfusion' else 0


    def _get_adv_loss(self, model_name, model, input_data, loss_type='log_score_loss', object_ids:list=None):
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
        return adv_loss, mean_score, log_info, results_dict

    def take_input_image(self, model_name, input_data):
        if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
            or model_name == 'autoalign' or model_name == 'bevformer':
            input_image = input_data['img'][0].data[0][0, self.k, ...]
        elif model_name == 'bevfusion' or model_name == 'uvtr':
            input_image = input_data['img'].data[0][0, self.k, ...]
        return input_image

    def put_input_image(self, model_name, input_data, input_image):
        if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
            or model_name == 'autoalign' or model_name == 'bevformer':
            input_data['img'][0].data[0].detach_()
            input_data['img'][0].data[0][0, self.k, ...] = input_image
        elif model_name == 'bevfusion' or model_name == 'uvtr':
            input_data['img'].data[0].detach_()
            input_data['img'].data[0][0, self.k, ...] = input_image
        return input_data
    
    def apply_patch(self, patch, mask, input_data, args):
        model_name = self.model_name
        img_norm_cfg = self.img_norm_cfg
        k = self.k
        if model_name == 'uvtr' or model_name == 'autoalign' or model_name == 'bevformer':
            patch_norm = norm_img(patch * 255, img_norm_cfg)
        else:
            patch_norm = norm_img(patch, img_norm_cfg)

        ori_image =self.ori_image
        new_image = ori_image.clone().detach()
        adv_image = new_image * (1-mask) + patch_norm * mask
        input_data = self.put_input_image(model_name, input_data, adv_image)
        return input_data

    def eval_patch(self, patch, mask, attack_timestamps, patch_area, args, object_ids=None):
        # log the patch under eval:
        My_config.tb_logger.add_image(f'trans_eval/{self.model_name}/patch_img', extract_patch(patch, mask), 0)
        
        is_trans = args['trans']
        batch_size = args['batch_size'] # trans batch size
        trans_batch_size = 1 if is_trans is False else batch_size
        ben_results = [] # List[Tuple[index, token, outputs, obj_idxes]]
        adv_results = [] # List[Tuple[index, token, outputs, obj_idxes]]
        self.total_adv_mean_score= 0
        self.total_ben_mean_score= 0
        data_iter = iter(enumerate(self.model_dataloader))
        if type(attack_timestamps) is list:
            for i, current_ts in enumerate(attack_timestamps):
                input_data, data_iter, dataset_idx = get_input_data(self.model_name, self.model_dataloader, current_ts, data_iter)
                input_data['current_ts'] = current_ts
                curr_object_ids = object_ids[i] if object_ids is not None else None
                ben_detection_rst, adv_detection_rst = self.eval_patch_on_scene(patch, mask, input_data, patch_area, dataset_idx, args,curr_object_ids, trans_batch_size, i)
                print(ben_detection_rst[0][:2])
                ben_results.extend(ben_detection_rst)
                adv_results.extend(adv_detection_rst)
        elif type(attack_timestamps) is int:
            input_data, _, dataset_idx = get_input_data(self.model_name, self.model_dataloader, attack_timestamps, data_iter)
            input_data['current_ts'] = attack_timestamps
            ben_detection_rst, adv_detection_rst = self.eval_patch_on_scene(patch, mask, input_data, patch_area, dataset_idx, args, object_ids, trans_batch_size, 0)
            ben_results.extend(ben_detection_rst)
            adv_results.extend(adv_detection_rst)
            while attack_timestamps == -1:
                input_data, _, dataset_idx = get_input_data(self.model_name, self.model_dataloader, attack_timestamps, data_iter)
                input_data['current_ts'] = attack_timestamps
                if input_data is None: break
                ben_detection_rst, adv_detection_rst = self.eval_patch_on_scene(patch, mask, input_data, patch_area, dataset_idx, args, object_ids, trans_batch_size, 0)
                ben_results.extend(ben_detection_rst)
                adv_results.extend(adv_detection_rst)
        nusc_eval = NuscEvalMetric(self.model_dataloader)
        print("Benign Results: ")
        metrics = nusc_eval.eval_results(ben_results)
        print("Adversarial Results: ")
        metrics = nusc_eval.eval_results(adv_results)
        N_scenes = len(attack_timestamps) if type(attack_timestamps) is list else 1
        mean_ben_score = self.total_ben_mean_score / N_scenes
        mean_adv_score = self.total_adv_mean_score / N_scenes
        print(f"Total mean benign score: {mean_ben_score}, total mean adv score: {mean_adv_score}")
        My_config.tb_logger.add_text(f'trans_eval/{self.model_name}/metrics_summary', str(metrics), 0)
        
    def preprocess_input_data(self, input_data, args):
        img_rep_dict = args['replace']
        current_ts = input_data['current_ts']
        new_img_tensor = None
        if img_rep_dict is not None and current_ts in img_rep_dict.keys():
            new_img_filename = os.path.join(My_config.physical_photo_dir, img_rep_dict[current_ts])
            metas = get_meta_from_inputdata(self.model_name, input_data)
            metas["filename"][self.k] = new_img_filename
            new_img_tensor = ToTensor()(pil.open(new_img_filename))[:3, :, :]
            assert new_img_tensor.shape == self.patch_converter.original_shape, f"new image shape error: {new_img_tensor.shape}"
        
        # set image norm cfg
        if self.model_name == 'deepint' or self.model_name == 'bevfusion2'\
            or self.model_name == 'transfusion': # [0, 1] image
            self.img_norm_cfg = copy.deepcopy(input_data['img_metas'][0].data[0][0]['img_norm_cfg'])
            sample_token = input_data['img_metas'][0].data[0][0]['sample_idx']
            if max(self.img_norm_cfg['mean']) > 1:
                self.img_norm_cfg['mean'] /= 255
                self.img_norm_cfg['std'] /= 255
            if new_img_tensor is not None:
                input_data['img'][0].data[0][0, self.k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor, self.img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    self.model_name
                )[0]
            self.ori_image = input_data['img'][0].data[0][0, self.k, ...].clone().detach()
        elif self.model_name == 'bevfusion': # [0, 1] image
            self.img_norm_cfg = input_data['metas'].data[0][0]['img_norm_cfg']
            sample_token = input_data['metas'].data[0][0]['token']
            if new_img_tensor is not None:
                input_data['img'].data[0][0, self.k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor, self.img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    self.model_name
                )[0]
            self.ori_image = input_data['img'].data[0][0, self.k, ...].clone().detach()
        elif self.model_name == 'uvtr': # [0, 255] image
            self.img_norm_cfg = input_data['img_metas'].data[0][0]['img_norm_cfg']
            sample_token = input_data['img_metas'].data[0][0]['sample_idx']
            if new_img_tensor is not None:
                input_data['img'].data[0][0, self.k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor*255, self.img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    self.model_name
                )[0]
            self.ori_image = input_data['img'].data[0][0, self.k, ...].clone().detach()
        elif self.model_name == 'autoalign' or self.model_name == 'bevformer': # [0, 255] image
            self.img_norm_cfg = input_data['img_metas'][0].data[0][0]['img_norm_cfg']
            sample_token = input_data['img_metas'][0].data[0][0]['sample_idx']
            if new_img_tensor is not None:
                input_data['img'][0].data[0][0, self.k, ...] = self.patch_converter.convert_patch(
                    norm_img(new_img_tensor*255, self.img_norm_cfg), 
                    torch.ones(1, *new_img_tensor.shape[1:]),
                    self.model_name
                )[0]
            self.ori_image = input_data['img'][0].data[0][0, self.k, ...].clone().detach()
        return input_data, sample_token
    

    def eval_patch_on_scene(self, patch, mask, input_data, patch_area, dataset_idx, args, 
                            object_ids=None, trans_batch_size=1, scene_idx=0):
        
        input_data, sample_token = self.preprocess_input_data(input_data, args)
        ben_detection_rst = [] # List[Tuple[index, token, outputs, obj_idxes]]

        model_relate = {
            'model_name': self.model_name,
            'model': self.target_model,
            'cfg': self.cfg,
            'img_norm_cfg': self.img_norm_cfg,
            'k': self.k,
            'input_data': input_data,
            'object_ids': object_ids
        }

        self.patch_area = patch_area
        if len(patch_area) == 5:
            self.pseudo_area = create_pseudo_area(patch_area)
            model_relate['pc_3d'] = Patch_converter_3d(self.model_name)
        else:
            t, l, h, w = patch_area
            mask_center = [t + h//2, l + w//2]

        # eval benign
        with torch.no_grad():
            ben_loss, ben_score, ben_info, outputs = self._get_adv_loss(self.model_name, self.target_model, input_data, 'log_score_loss', object_ids)
        
        anno_img, anno_lidar = visulize_atk(model_relate, input_data, None, None, 'pred', score_notes=False, box_notes=False)
        
        ben_detection_rst.append((dataset_idx, sample_token, outputs, object_ids))

        My_config.tb_logger.add_image(f'trans_eval/{self.model_name}/ben_img', anno_img, scene_idx)
        ToPILImage()(anno_img.clone().cpu()).save(os.path.join(My_config.log_dir, f'ben_imgs/{scene_idx:0>3d}.png'))
        if anno_lidar is not None:
            My_config.tb_logger.add_image(f'trans_eval/{self.model_name}/ben_lidar', anno_lidar, scene_idx)
            ToPILImage()(anno_lidar.clone().cpu()).save(os.path.join(My_config.log_dir, f'ben_lidar/{scene_idx:0>3d}.png'))

        My_config.tb_logger.add_text(f'trans_eval/{self.model_name}/Benign', str(ben_info), scene_idx)
        
        # eval adv
        transfer_enabler = Transfer_enabler()
        total_adv_loss=0
        total_adv_score = 0
        adv_detection_rst = []
        for i in range(trans_batch_size):
            if trans_batch_size == 1:
                if len(patch_area) == 5: # physical
                    patch_trans, mask_trans = get_phy_patch(model_relate, patch, mask, self.pseudo_area, self.patch_area, True)
                else:
                    patch_trans, mask_trans = patch, mask
            else:
                patch_trans, mask_trans = transfer_enabler.random_trans(patch, mask, patch_area)
            patch_conv, mask_conv = self.patch_converter.convert_patch(patch_trans, mask_trans, target=self.model_name)
            adv_input_data = self.apply_patch(patch_conv, mask_conv, input_data, args)
            with torch.no_grad():
                adv_loss, adv_score, adv_info, outputs = self._get_adv_loss(self.model_name, self.target_model, adv_input_data, 'log_score_loss', object_ids)
            print(adv_info)
            anno_img, anno_lidar = visulize_atk(model_relate, adv_input_data, patch_conv, mask_conv, 'pred', box_notes=False, score_notes=False)
            My_config.tb_logger.add_image(f'trans_eval/{self.model_name}/adv_img', anno_img, scene_idx * trans_batch_size + i)
            ToPILImage()(anno_img.clone().cpu()).save(os.path.join(My_config.log_dir, f'adv_imgs/{scene_idx * trans_batch_size + i:0>3d}.png'))
            if anno_lidar is not None:
                My_config.tb_logger.add_image(f'trans_eval/{self.model_name}/adv_lidar', anno_lidar,  scene_idx * trans_batch_size + i)
                ToPILImage()(anno_lidar.clone().cpu()).save(os.path.join(My_config.log_dir, f'adv_lidar/{scene_idx * trans_batch_size + i:0>3d}.png'))
            My_config.tb_logger.add_text(f'trans_eval/{self.model_name}/adv', str(adv_info),  scene_idx * trans_batch_size + i)
            total_adv_loss += adv_loss
            total_adv_score += adv_score
            adv_detection_rst.append((dataset_idx, sample_token, outputs, object_ids))
        avg_adv_loss = total_adv_loss / trans_batch_size
        avg_adv_score = total_adv_score / trans_batch_size
            
        summary = f"average adv loss diff: {avg_adv_loss - ben_loss}, average score diff: {avg_adv_score - ben_score}."
        print(summary)
        My_config.tb_logger.add_text(f'trans_eval/{self.model_name}/Summary', str(summary), scene_idx)
        self.total_ben_mean_score += ben_score
        self.total_adv_mean_score += avg_adv_score
        return ben_detection_rst, adv_detection_rst
        
