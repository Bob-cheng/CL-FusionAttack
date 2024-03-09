import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vis_F

class Patch_converter(object):
    def __init__(self) -> None:
        """
        original size (H, W): (900, 1600)
        deepint size: (448, 800)
        bevfusion2 size: (448, 800)
        transfusion size: (448, 800)
        bevfusion size: (256, 704), from original: center bottom crop, scale 0.465
            (900, 1600) * 0.465 = (418, 744)
        uvtr size: (928, 1600), from original: bottom padding 28
        bevformer size: (928, 1600), from original: bottom padding 28
        autoalign size: (640, 1152), from original: scale to (640, 1137) then pad right to 1152
        """
        self.bev_scale = 0.465
        self.deepint_scale = 0.5
        self.original_shape = (3, 900, 1600)
        self.deepint_shape = (3, 448, 800) # also shape of bevfusion2 and transfusion
        self.bevfusion_shape = (3, 256, 704)
        self.uvtr_shape = (3, 928, 1600)
        self.autoalign_shape = (3, 640, 1152)
        self.autoalign_scale = 0.711

    def bev2ori(self, patch_area):
        t, l, h, w = patch_area
        ori_h, ori_w = h / self.bev_scale, w / self.bev_scale
        ori_t = (t + (418 - 256)) / self.bev_scale
        ori_l = (l + (744 - 704) // 2) / self.bev_scale
        ori_patch_area = (int(ori_t), int(ori_l), int(ori_h), int(ori_w))
        return ori_patch_area
    
    def autoalign2ori(self, patch_area):
        t, l, h, w = patch_area
        ori_h, ori_w = h / self.autoalign_scale, w / self.autoalign_scale
        ori_t = t / self.autoalign_scale
        ori_l = l / self.autoalign_scale
        ori_patch_area = (int(ori_t), int(ori_l), int(ori_h), int(ori_w))
        return ori_patch_area
    
    def ori2autoalign(self, patch_area):
        t, l, h, w = patch_area
        return (    
            int(t * self.autoalign_scale),
            int(l * self.autoalign_scale),
            int(h * self.autoalign_scale),
            int(w * self.autoalign_scale)
        )

    def autoalign2ori_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.autoalign_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        patch_bind = patch_bind[:, :, :1137]
        patch_bind = vis_F.resize(patch_bind, self.original_shape[1:])
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out
    
    def ori2autoalign_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.original_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        patch_bind = vis_F.resize(patch_bind, (640, 1137))
        right_pad = 15
        patch_bind = F.pad(patch_bind, (0, right_pad), "constant", 0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out
        
    
    def ori2uvtr(self, patch_area):
        return patch_area

    def ori2uvtr_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.original_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        bottom_pad = 928 - 900
        patch_bind = F.pad(patch_bind, (0, 0, 0, bottom_pad), "constant", 0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def uvtr2ori_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.uvtr_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        bottom_crop = 928 - 900
        patch_bind = patch_bind[:, :-bottom_crop, :]
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def uvtr2ori(self, patch_area):
        return patch_area
    
    def bev2ori_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.bevfusion_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        side_pad = (744 - 704) // 2
        top_pad = (418 - 256)
        patch_bind = F.pad(patch_bind, (side_pad, side_pad, top_pad, 0), "constant", 0)
        patch_bind = vis_F.resize(patch_bind, self.original_shape[1:])
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out
    
    def bev2deepint_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.bevfusion_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        side_pad = (744 - 704) // 2
        top_pad = (418 - 256)
        patch_bind = F.pad(patch_bind, (side_pad, side_pad, top_pad, 0), "constant", 0)
        patch_bind = vis_F.resize(patch_bind, self.deepint_shape[1:])
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def bev2uvtr_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.bevfusion_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        side_pad = (744 - 704) // 2
        top_pad = (418 - 256)
        patch_bind = F.pad(patch_bind, (side_pad, side_pad, top_pad, 0), "constant", 0)
        patch_bind = vis_F.resize(patch_bind, self.original_shape[1:])
        bottom_pad = 928 - 900
        patch_bind = F.pad(patch_bind, (0, 0, 0, bottom_pad), "constant", 0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def deepi2uvtr_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        assert patch_img.shape == self.deepint_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        patch_bind = vis_F.resize(patch_bind, self.original_shape[1:])
        bottom_pad = 928 - 900
        patch_bind = F.pad(patch_bind, (0, 0, 0, bottom_pad), "constant", 0)
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def deepi2ori(self, patch_area):
        t, l, h, w = patch_area
        return (    
            int(t / self.deepint_scale),
            int(l / self.deepint_scale),
            int(h / self.deepint_scale),
            int(w / self.deepint_scale)
        )

    def deepi2ori_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.deepint_shape
        patch_out = vis_F.resize(patch_img, self.original_shape[1:])
        mask_out = vis_F.resize(mask, self.original_shape[1:])
        return patch_out, mask_out

    def deepi2ori_pseudo_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.deepint_shape
        t = self.original_shape[1] - self.deepint_shape[1]
        b = 0
        l = (self.original_shape[2] - self.deepint_shape[2]) // 2
        r = l
        patch_out = vis_F.pad(patch_img, [l,t,r,b])
        mask_out = vis_F.pad(mask, [l,t,r,b])
        return patch_out, mask_out
    
    def ori2deepi_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.original_shape
        patch_out = vis_F.resize(patch_img, self.deepint_shape[1:])
        mask_out = vis_F.resize(mask, self.deepint_shape[1:])
        return patch_out, mask_out


    def ori2deepi(self, patch_area):
        t, l, h, w = patch_area
        return (    
            int(t * self.deepint_scale),
            int(l * self.deepint_scale),
            int(h * self.deepint_scale),
            int(w * self.deepint_scale)
        )
    

    def ori2bev(self, patch_area):
        t, l, h, w = patch_area
        t_bev = t * self.bev_scale - (418 - 256)
        l_bev = l * self.bev_scale - (744 - 704) // 2
        h_bev, w_bev = h * self.bev_scale, w * self.bev_scale
        return (
            int(t_bev),
            int(l_bev),
            int(h_bev),
            int(w_bev)
        )

    def ori2bev_patch(self, patch_img: torch.Tensor, mask: torch.Tensor):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        """
        assert patch_img.shape == self.original_shape
        patch_bind = torch.cat([mask, patch_img], dim=0).clone()
        patch_bind = vis_F.resize(patch_bind, (418, 744))
        side_crop = (744 - 704) // 2
        top_crop= (418 - 256)
        patch_bind = patch_bind[:, top_crop:, side_crop:-side_crop]
        mask_out = patch_bind[[0]]
        patch_out = patch_bind[1:4]
        return patch_out, mask_out

    def convert_coord_from_ori(self, target, coords):
        if target == 'bevfusion2' or target == 'transfusion' or target == 'deepint':
            return coords * self.deepint_scale
        elif target == 'bevfusion':
            coord_ret = coords * self.bev_scale
            coord_ret[:, 0] -= (744 - 704) // 2
            coord_ret[:, 1] -= (418 - 256)
            return coord_ret
        elif target == 'uvtr' or target == 'bevformer' or target == 'original':
            return coords
        elif target == 'autoalign':
            coord_ret = coords * self.autoalign_scale
            return coord_ret

    def convert_coord_to_ori(self, source, coords):
        if source == 'bevfusion2' or source == 'transfusion' or source == 'deepint':
            return coords / self.deepint_scale
        elif source == 'bevfusion':
            coord_ret[:, 1] += (418 - 256)
            coord_ret[:, 0] += (744 - 704) // 2
            coord_ret = coords / self.bev_scale
            return coord_ret
        elif source == 'uvtr' or source == 'bevformer' or source == 'original':
            return coords
        elif source == 'autoalign':
            coord_ret = coords / self.autoalign_scale
            return coord_ret


    def convert_patch_area(self, patch_area, target:str, source:str='bevfusion'):
        if source == 'bevfusion2' or source == 'transfusion':
            source = 'deepint'
        elif source == 'bevformer':
            source = 'uvtr'
        if target == 'bevfusion2' or target == 'transfusion':
            target = 'deepint'
        elif target == 'bevformer':
            target = 'uvtr'
        if source == target:
            return patch_area
        elif source == 'bevfusion' and target == 'original':
            return self.bev2ori(patch_area)
        elif source == 'bevfusion' and target == 'deepint':
            return self.ori2deepi(self.bev2ori(patch_area))
        elif source == 'bevfusion' and target == 'uvtr':
            return self.ori2uvtr(self.bev2ori(patch_area))
        elif source == 'deepint' and target == 'original':
            return self.deepi2ori(patch_area)
        elif source == 'deepint' and target == 'bevfusion':
            return self.ori2bev(self.deepi2ori(patch_area))
        elif source == 'deepint' and target == 'uvtr':
            return self.ori2uvtr(self.deepi2ori(patch_area))
        elif source == 'original' and target == 'bevfusion':
            return self.ori2bev(patch_area)
        elif source == 'original' and target == 'deepint':
            return self.ori2deepi(patch_area)
        elif source == 'original' and target == 'uvtr':
            return self.ori2uvtr(patch_area)
        elif source == 'autoalign' and target == 'original':
            return self.autoalign2ori(patch_area)
        elif source ==  'original'and target == 'autoalign':
            return self.ori2autoalign(patch_area)
        else:
            raise NotImplementedError()

    def convert_pseudo_patch(self, patch_img, mask, target:str, source:str=None):
        if source is None:
            if patch_img.shape == self.original_shape:
                source = 'original'
            elif patch_img.shape == self.bevfusion_shape:
                source = 'bevfusion'
            elif patch_img.shape == self.deepint_shape:
                source = 'deepint'
            elif patch_img.shape == self.uvtr_shape:
                source = 'uvtr'
            elif patch_img.shape == self.autoalign_shape:
                source = 'autoalign'
            else:
                raise RuntimeError("Patch image shape error.")
            
        if target == 'bevfusion2' or target == 'transfusion':
            target = 'deepint'
        elif target == 'bevformer':
            target = 'uvtr'

        if source == target:
            return patch_img, mask

        transformations = {
            ('deepint', 'original'): self.deepi2ori_pseudo_patch,
        }
        try:
            transform_func = transformations[(source, target)]
            return transform_func(patch_img, mask)
        except KeyError:
            raise NotImplementedError(f"source: {source}, target: {target}")




    def convert_patch(self, patch_img, mask, target:str, source:str=None):
        if len(mask.shape) == 4:
            patch_cvt = []
            mask_cvt = []
            for i in range(len(mask)):
                patch_cvt_sub, mask_cvt_sub = self.convert_patch(patch_img[i], mask[i], target)
                patch_cvt.append(patch_cvt_sub)
                mask_cvt.append(mask_cvt_sub)
            patch_cvt = torch.stack(patch_cvt)
            mask_cvt = torch.stack(mask_cvt)
            return patch_cvt, mask_cvt
        
        if source is None:
            if patch_img.shape == self.original_shape:
                source = 'original'
            elif patch_img.shape == self.bevfusion_shape:
                source = 'bevfusion'
            elif patch_img.shape == self.deepint_shape:
                source = 'deepint'
            elif patch_img.shape == self.uvtr_shape:
                source = 'uvtr'
            elif patch_img.shape == self.autoalign_shape:
                source = 'autoalign'
            else:
                raise RuntimeError("Patch image shape error.")
                
        if target == 'bevfusion2' or target == 'transfusion':
            target = 'deepint'
        elif target == 'bevformer':
            target = 'uvtr'

        if source == target:
            return patch_img, mask

        transformations = {
            ('bevfusion', 'original'): self.bev2ori_patch,
            ('bevfusion', 'deepint'): self.bev2deepint_patch,
            ('bevfusion', 'uvtr'): self.bev2uvtr_patch,
            ('bevfusion', 'autoalign'): lambda x, y: self.ori2autoalign_patch(*self.bev2ori_patch(x, y)),
            ('deepint', 'original'): self.deepi2ori_patch,
            ('deepint', 'bevfusion'): lambda x, y: self.ori2bev_patch(*self.deepi2ori_patch(x, y)),
            ('deepint', 'uvtr'): self.deepi2uvtr_patch,
            ('uvtr', 'bevfusion'): lambda x, y: self.ori2bev_patch(*self.uvtr2ori_patch(x, y)),
            ('uvtr', 'deepint'): lambda x, y: self.ori2deepi_patch(*self.uvtr2ori_patch(x, y)),
            ('uvtr', 'original'): self.uvtr2ori_patch,
            ('autoalign', 'original'): self.autoalign2ori_patch,
            ('original', 'autoalign'): self.ori2autoalign_patch,
            ('original', 'bevfusion'): self.ori2bev_patch,
            ('original', 'deepint'): self.ori2deepi_patch,
            ('original', 'uvtr'): self.ori2uvtr_patch,
        }
        try:
            transform_func = transformations[(source, target)]
            return transform_func(patch_img, mask)
        except KeyError:
            raise NotImplementedError(f"source: {source}, target: {target}")

        # if source == target:
        #     return patch_img, mask
        # elif source == 'bevfusion' and target == 'original':
        #     return self.bev2ori_patch(patch_img, mask)
        # elif source == 'bevfusion' and target == 'deepint':
        #     return self.bev2deepint_patch(patch_img, mask)
        # elif source == 'bevfusion' and target == 'uvtr':
        #     return self.bev2uvtr_patch(patch_img, mask)
        # elif source == 'bevfusion' and target == 'autoalign':
        #     return self.ori2autoalign_patch(*self.bev2ori_patch(patch_img, mask))
        # elif source == 'deepint' and target == 'original':
        #     return self.deepi2ori_patch(patch_img, mask)
        # elif source == 'deepint' and target == 'bevfusion':
        #     return self.ori2bev_patch(*self.deepi2ori_patch(patch_img, mask))
        # elif source == 'deepint' and target == 'uvtr':
        #     return self.deepi2uvtr_patch(patch_img, mask)
        # elif source == 'uvtr' and target == 'bevfusion':
        #     return self.ori2bev_patch(*self.uvtr2ori_patch(patch_img, mask))
        # elif source == 'uvtr' and target == 'deepint':
        #     return self.ori2deepi_patch(*self.uvtr2ori_patch(patch_img, mask))
        # elif source == 'uvtr' and target == 'original':
        #     return self.uvtr2ori_patch(patch_img, mask)
        # elif source == 'autoalign' and target == 'original':
        #     return self.autoalign2ori_patch(patch_img, mask)
        # elif source == 'original' and target == 'autoalign':
        #     return self.ori2autoalign_patch(patch_img, mask)
        # elif source == 'original' and target == 'bevfusion':
        #     return self.ori2bev_patch(patch_img, mask)
        # else:
        #     raise NotImplementedError(f"source: {source}, target: {target}")