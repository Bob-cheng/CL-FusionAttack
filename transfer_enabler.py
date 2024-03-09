
import torch
import numpy as np
from torch.nn import Sequential
from torchvision.transforms import RandomAffine, RandomPerspective, ColorJitter
from torchvision.transforms.functional import affine, InterpolationMode
from my_utils import get_phy_patch, create_pseudo_area


class Transfer_enabler(object):
    def __init__(
            self, 
            persp_rat=0.1,
            rot_deg=3, 
            scale_range=(0.9, 1.1), 
            trans_range=(0.01, 0.01), # for image size (256, 704) 
            brightness = 0.3,
            contrast=0.1,
            saturation=0.1
            ) -> None:
        self.trans_seq = Sequential(
                            # RandomPerspective(persp_rat, p=0.7),
                            RandomAffine(rot_deg, 
                                trans_range, 
                                scale_range, 
                                interpolation=InterpolationMode.BILINEAR),
                        )
        self.color_aug = ColorJitter(brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation)
        self.rs = np.random.RandomState(7)
        

    def random_trans(self, patch_img, mask, patch_area, model_relate=None):
        """
        patch_img: tensor with size of (3, H, W)
        mask: tensor with size of (1, H, W)
        mask_center: tuple like (x, y)
        """
        if len(patch_area) == 4:
            t, l, h, w = patch_area
            mask_center = [t + h//2, l + w//2]
            patch_bind = torch.cat([mask, patch_img], dim=0)
            _, H, W = patch_bind.shape
            img_center = np.array([H//2, W//2])
            mask_center = np.array(mask_center)
            move_vec = img_center - mask_center
            patch_bind = affine(img=patch_bind, 
                                angle=0, 
                                translate=[move_vec[1], move_vec[0]], 
                                scale=1, 
                                shear=0,
                                fill=0)
            patch_bind = self.trans_seq(patch_bind)
            patch_bind = affine(img=patch_bind, 
                                angle=0, 
                                translate=[-move_vec[1], -move_vec[0]], 
                                scale=1, 
                                shear=0,
                                fill=0)
            mask_out = patch_bind[[0]]
            patch_out = patch_bind[1:4]
            patch_out = self.color_aug(patch_out)
            return patch_out, mask_out
        elif len(patch_area) == 5 and model_relate is not None:
            patch_img = self.color_aug(patch_img)
            pseudo_area = create_pseudo_area(patch_area)
            patch_trans, mask_trans = get_phy_patch(model_relate, patch_img, mask, 
                pseudo_area, patch_area, deterministic=False, rs=self.rs)
            return patch_trans, mask_trans
        else: 
            raise NotImplementedError('patch_area unrecognized.')
        




