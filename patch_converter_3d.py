import numpy as np
from math import cos, sin, radians
import copy
from patch_converter import Patch_converter
import torch
from torchvision.transforms.functional import perspective, InterpolationMode, resize as img_resize

class Patch_converter_3d(object):
    def __init__(self, model_name,
                    angle_range=None, # np.arange(-10, 10 + 1, 1)
                    dist_range_y=None, # np.arange(5, 10 + 1, 1)
                    dist_range_x=None, # np.arange(-1, 1 + 0.5, 0.5)
                    resize_range=None # np.arange(0.5, 1 + 0.1, 0.1)
                    ) -> None:
        self.lidar_height = 1.84
        self.angle_range = angle_range
        self.dist_range_x = dist_range_x
        self.dist_range_y = dist_range_y
        self.resize_range = resize_range
        self.model_name = model_name
        self.pc = Patch_converter()
    
    def compose_targeted_patch(self,
                                ori_patch:torch.Tensor, 
                                ori_mask: torch.Tensor, 
                                ori_patch_area, # Tuple: (top_loc, left_loc, H, W) relative to bevfusion2 size
                                obj_corners: np.ndarray, # shape: (8, 3), order see: LiDARInstance3DBoxes
                                transform,
                                W,
                                H,
                                deterministic=True, 
                                rs:np.random.RandomState=None
                                ):
        """torch.Tensor: Coordinates of corners of the target object's bounding box
        in shape (8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1) ---> p1
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0) ---> p2
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0) ---> p3
                (x0, y0, z0)
        """
        if not deterministic:
            if self.resize_range is not None:
                scale = rs.choice(self.resize_range)
                ori_H, ori_W =  ori_patch.shape[-2:]
                scale_H, scale_W = int(ori_H * scale), int(ori_W * scale)
                ori_patch = img_resize(ori_patch, (scale_H, scale_W))
                ori_patch = img_resize(ori_patch, (ori_H, ori_W))

        back_coords = obj_corners[[2,3,6,7], :]
        center_coord = np.sum(back_coords,axis=0) / 4
        x_vec = obj_corners[7] - obj_corners[3]
        x_vec = x_vec / np.linalg.norm(x_vec) * W/2
        z_vec = np.array([0,0,1]) * H/2
        p1 = center_coord + x_vec + z_vec
        p2 = center_coord + x_vec - z_vec
        p3 = center_coord - x_vec - z_vec
        p4 = center_coord - x_vec + z_vec
        coords = np.hstack([np.vstack([p1, p2, p3, p4]), np.ones((4, 1))])

        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
    
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        coords = coords[:, :2]

        # transform of these models are to original image size, so convert to target model input size
        # deepint, uvtr, bevformer's transformations are to model input size 
        if self.model_name == 'bevfusion' or self.model_name == 'bevfusion2' \
            or self.model_name == 'transfusion' or self.model_name == 'autoalign': 
            coords = self.pc.convert_coord_from_ori(target = self.model_name, coords = coords)

        coords = coords.astype(np.int32)
        coords = coords[[3, 0, 1, 2]] # in the order of tl, tr, br, bl
        coords_uv = [[coord[0], coord[1]] for coord in coords]

        # from bevfusion scale to target scale
        new_patch, new_mask = self.pc.convert_patch(ori_patch, ori_mask, target = self.model_name)
        new_patch_area = self.pc.convert_patch_area(ori_patch_area, target = self.model_name, source='bevfusion2')
        t, l, h, w = new_patch_area
        tl = [l, t]
        tr = [l+w, t]
        br = [l+w, t+h]
        bl = [l, t+h]
        patch_bind = torch.cat([new_mask, new_patch], dim=0).clone()
        patch_bind = perspective(patch_bind, [tl, tr, br, bl], coords_uv, interpolation=InterpolationMode.BILINEAR, fill=0)
        # patch_bind = perspective(patch_bind, [tl, tr, br, bl], coords_uv, interpolation=InterpolationMode.NEAREST, fill=0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        mask_out[mask_out.isnan()] = 0
        patch_out[patch_out.isnan()] = 0
        return patch_out, mask_out


    
    def compose_3d_patch(self, 
                    ori_patch:torch.Tensor, 
                    ori_mask: torch.Tensor, 
                    ori_patch_area, # Tuple: (top_loc, left_loc, H, W) relative to bevfusion2 size
                    transform, 
                    W, 
                    H, 
                    deterministic=True, 
                    dx=0,
                    dy=6.7,
                    alpha=0,
                    rs:np.random.RandomState=None):
        """
            ori_patch: the original patch with optimizable parameters with size relative to bevfusion
            ori_mask: the mask used on originial patch
            ori_patch_area: the patch area of original patch # Tuple: (top_loc, left_loc, H, W) relative to bevfusion size
        """
        if not deterministic:
            assert rs is not None
            if self.dist_range_x is not None:
                dx = rs.choice(self.dist_range_x)
            if self.dist_range_y is not None:
                dy = rs.choice(self.dist_range_y)
            if self.angle_range is not None:
                alpha = rs.choice(self.angle_range)
            if self.resize_range is not None:
                scale = rs.choice(self.resize_range)
                ori_H, ori_W =  ori_patch.shape[-2:]
                scale_H, scale_W = int(ori_H * scale), int(ori_W * scale)
                ori_patch = img_resize(ori_patch, (scale_H, scale_W))
                ori_patch = img_resize(ori_patch, (ori_H, ori_W))
            # print("dx: {}, dy: {}, alpha: {}".format(dx, dy, alpha))

        x_vec = np.array((cos(radians(alpha)), sin(radians(alpha))))
        y_vec = np.array((-sin(radians(alpha)), cos(radians(alpha))))
        m = np.array((dx, dy))
        p1 = np.append(m + W/2 * x_vec + H/2 * y_vec, - self.lidar_height)
        p2 = np.append(m - W/2 * x_vec + H/2 * y_vec, - self.lidar_height)
        p3 = np.append(m - W/2 * x_vec - H/2 * y_vec, - self.lidar_height)
        p4 = np.append(m + W/2 * x_vec - H/2 * y_vec, - self.lidar_height)
        coords = np.hstack([np.vstack([p1, p2, p3, p4]), np.ones((4, 1))])

        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
    
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        coords = coords[:, :2]

        # transform of these models are to original image size, so convert to target model input size
        # deepint, uvtr, bevformer's transformations are to model input size 
        if self.model_name == 'bevfusion' or self.model_name == 'bevfusion2' \
            or self.model_name == 'transfusion' or self.model_name == 'autoalign': 
            coords = self.pc.convert_coord_from_ori(target = self.model_name, coords = coords)

        coords = coords.astype(np.int32)
        coords = coords[[1, 0, 3, 2]] # in the order of tl, tr, br, bl
        coords_uv = [[coord[0], coord[1]] for coord in coords]

        # from bevfusion scale to target scale
        new_patch, new_mask = self.pc.convert_patch(ori_patch, ori_mask, target = self.model_name)
        new_patch_area = self.pc.convert_patch_area(ori_patch_area, target = self.model_name, source='bevfusion2')

        t, l, h, w = new_patch_area
        tl = [l, t]
        tr = [l+w, t]
        br = [l+w, t+h]
        bl = [l, t+h]
        patch_bind = torch.cat([new_mask, new_patch], dim=0).clone()
        # patch_bind = perspective(patch_bind, [tl, tr, br, bl], coords_uv, interpolation=InterpolationMode.BILINEAR, fill=0)
        patch_bind = perspective(patch_bind, [tl, tr, br, bl], coords_uv, interpolation=InterpolationMode.NEAREST, fill=0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        mask_out[mask_out.isnan()] = 0
        patch_out[patch_out.isnan()] = 0
        return patch_out, mask_out

if __name__ == "__main__":
    
    from model_loader import Model_loader
    from my_utils import get_input_data, save_pic
    model_loader = Model_loader()
    model_name = 'bevfusion2'
    model, model_dataloader, target_cfg = model_loader.load_model(model_name)

    ## test scene-oriented attack
    # input_data, data_iter, dataset_idx = get_input_data(model_name, model_dataloader, -1)
    # metas = input_data['img_metas'][0].data[0][0]
    # k = 1 if model_name == 'bevfusion2' or model_name == 'transfusion' else 0
    # transform = metas["lidar2image"][k] if "lidar2image" in metas.keys() else metas["lidar2img"][k]
    # pc_3d = Patch_converter_3d(model_name)
    # W = 2
    # H = 2
    # patch_area = (256 - H * 100 - 1, (704 - W * 100) // 2, H * 100, W * 100)
    # t, l, h, w = patch_area
    # patch = torch.rand((3, 256, 704))
    # mask = torch.zeros((1, 256, 704))
    # mask[:, t:t+h, l:l+w] = 1
    
    # patch, mask = pc_3d.compose_3d_patch(patch, mask, patch_area, transform, W, H, dx=0, dy=6.7, alpha=0)
    # ori_image = input_data['img'][0].data[0][0, k, ...].clone().detach()
    # patched_image = ori_image * (1-mask) + patch * mask
    # save_pic(patched_image, 0)

    ## test object-oriented attack
    from mmdet3d.core import LiDARInstance3DBoxes
    input_data, data_iter, dataset_idx = get_input_data(model_name, model_dataloader, 1542800856950302)
    target_idx=17
    metas = input_data['img_metas'][0].data[0][0]
    k = 1 if model_name == 'bevfusion2' or model_name == 'transfusion' else 0
    transform = metas["lidar2image"][k] if "lidar2image" in metas.keys() else metas["lidar2img"][k]
    pc_3d = Patch_converter_3d(model_name)
    W = 1
    H = 1
    patch_area = (448 - H * 100 - 1, (800 - W * 100) // 2, H * 100, W * 100)
    t, l, h, w = patch_area
    patch = torch.rand((3, 448, 800))
    mask = torch.zeros((1, 448, 800))
    mask[:, t:t+h, l:l+w] = 1
    if model_name == "bevfusion" or model_name == 'uvtr':
        bboxes = input_data["gt_bboxes_3d"].data[0][0].tensor.numpy()
    elif model_name == "deepint" or model_name == 'bevfusion2' or model_name == 'bevformer'\
            or model_name == 'transfusion' or model_name == 'autoalign':
        bboxes = input_data["gt_bboxes_3d"][0].data[0][0].tensor.numpy()
    if model_name == 'bevfusion':
        bboxes[..., 2] -= bboxes[..., 5] / 2
    bboxes_corners = LiDARInstance3DBoxes(bboxes, box_dim=9).corners.numpy()
    obj_corners = bboxes_corners[target_idx]
    patch, mask = pc_3d.compose_targeted_patch(patch, mask, patch_area, 
                                obj_corners, transform, W, H)
    ori_image = input_data['img'][0].data[0][0, k, ...].clone().detach()
    patched_image = ori_image * (1-mask) + patch * mask
    save_pic(patched_image, 0)
    