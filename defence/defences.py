import numpy as np
import torch
from torchvision.transforms import transforms
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image as pil
import io
from scipy import ndimage


def perturb_input(model_name, input_data, camera_ratio, Lidar_ratio):
    if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
        or model_name == 'autoalign':
        input_image = input_data['img'][0].data[0] # shape: (1, 6, 3, H, W)
        input_lidar = input_data['points'][0].data[0][0]
    elif model_name == 'bevfusion' or model_name == 'uvtr':
        input_image = input_data['img'].data[0] # shape: ([34688, 5])
        input_lidar = input_data['points'].data[0][0]

    if camera_ratio < 1:
        num_pixels = int(torch.numel(input_image) * (1-camera_ratio))
        indices = torch.randperm(torch.numel(input_image))[0:num_pixels]
        input_image.view(-1)[indices] = 0
    
    if Lidar_ratio < 1:
        # print(input_lidar.shape)
        z_axis = input_lidar[:, 2]
        meaningful_indices = torch.where(z_axis > -1.8)[0]
        num_points = int(len(meaningful_indices) * (1-Lidar_ratio))
        indices = torch.randperm(len(meaningful_indices))[0:num_points]
        input_lidar[meaningful_indices[indices], 2] = -1.8
        input_lidar[meaningful_indices[indices], 1] = torch.rand(num_points) * 140 - 70
        input_lidar[meaningful_indices[indices], 0] = torch.rand(num_points) * 140 - 70
        input_lidar[meaningful_indices[indices], 3] = 0
    
    if model_name == 'deepint' or model_name == 'bevfusion2' or model_name == 'transfusion'\
        or model_name == 'autoalign':
        input_data['img'][0].data[0] = input_image
        input_data['points'][0].data[0][0] = input_lidar
    elif model_name == 'bevfusion' or model_name == 'uvtr':
        input_data['img'].data[0] = input_image
        input_data['points'].data[0][0] = input_lidar

    return input_data

def apply_image_defence(img: torch.Tensor, defence_name, param):
    """
    img: the input image with a size of (3, H, W) in the range of [0, 1]
    """
    if defence_name == 'bitdepth':
        img_out = bitdepth_defense(img, depth=param)
    elif defence_name == 'jpeg':
        img_out = jpeg_defense(img, quality=param)
    elif defence_name == 'blur':
        img_out = blurring_defense(img, ksize=param)
    elif defence_name == 'gaussian':
        img_out = gaussian_noise(img, std=param)
    elif defence_name == 'black':
        img_out = img * 0
    else:
        raise NotImplementedError('defence method not implemented')
    return img_out


def bitdepth_defense(img_rgb: torch.Tensor, depth=5):
    depth=int(depth)
    max_value = np.rint(2 ** depth - 1)
    img_rgb = torch.round(img_rgb * max_value)  # int 0-max_val
    img_rgb = img_rgb / max_value  # float 0-1
    return img_rgb

def jpeg_defense(img: torch.Tensor, quality=20):
    device=img.device
    tmp = io.BytesIO()
    transforms.ToPILImage()(img.cpu()).save(tmp, format='jpeg', quality=int(quality))
    # transforms.ToPILImage()(img.squeeze(0).cpu()).save(tmp, format='png')
    _img = transforms.ToTensor()(pil.open(tmp)).to(device)
    return _img

def blurring_defense(img: torch.Tensor, ksize=5):
    ksize = int(ksize)
    device=img.device
    img_np = img.cpu().numpy() * 255
    img_np = img_np.astype(np.uint8)
    rgb = ndimage.filters.median_filter(img_np, size=(1,ksize, ksize), mode='reflect')
    # rgb = img_np
    rgb = rgb.astype(np.uint8)
    rgb_tensor = torch.from_numpy(rgb).float() / 255
    rgb_tensor = rgb_tensor.to(device)
    return rgb_tensor

def gaussian_noise(img: torch.Tensor, std=0.1):
    device = img.device
    noise = torch.normal(torch.zeros_like(img), torch.ones_like(img) * std).to(device)
    img += noise
    img.clamp_(0, 1)
    return img