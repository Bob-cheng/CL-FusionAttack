import os
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from mmdet3d.core.visualizer.image_vis import project_pts_on_img, map_pointcloud_to_image
from mmdet3d.core.utils.gaussian import generate_guassian_depth_target
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_v6(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        img = np.stack(
            [mmcv.imfrombytes(self.file_client.get(name), flag=self.color_type) \
                for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_v4(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged', 
                    project_pts_to_img_depth=False, 
                    cam_depth_range=[4.0, 45.0, 1.0],
                    constant_std=0.5):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std=constant_std

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        if self.img_scale is None:
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        else:
            img = np.stack(
                [self.pad(mmcv.imread(name, self.color_type)) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        if self.project_pts_to_img_depth:
            # results['img_fields'].append('img_depth')
            results['img_depth'] = []
            for i in range(len(results['img'])):
                # project_pts_on_img(results['points'].tensor.numpy(), results['img'][i], results['lidar2img'][i])
                depth = map_pointcloud_to_image(
                    results['points'].tensor.numpy(), 
                    results['img'][i], 
                    results['caminfo'][i]['sensor2lidar_rotation'], 
                    results['caminfo'][i]['sensor2lidar_translation'], 
                    results['caminfo'][i]['cam_intrinsic'], show=False)
                guassian_depth, min_depth, std_var = generate_guassian_depth_target(torch.from_numpy(depth).unsqueeze(0), stride=8, cam_depth_range=self.cam_depth_range, constant_std=self.constant_std)
                depth = torch.cat([min_depth[0].unsqueeze(-1), guassian_depth[0]], dim=-1)
                results['img_depth'].append(depth)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_v2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # filename = results['image_paths']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps_v2(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_masks_bev"] = labels
        return data

@PIPELINES.register_module()
class LoadPointsFromFile_v2(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class LoadPointsFromFile_v4(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)
            attribute_dims = dict(height=3)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += 'shift_height={}, '.format(self.shift_height)
        repr_str += 'file_client_args={}), '.format(self.file_client_args)
        repr_str += 'load_dim={}, '.format(self.load_dim)
        repr_str += 'use_dim={})'.format(self.use_dim)
        return repr_str

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps_v4(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 point_cloud_angle_range=None):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        if point_cloud_angle_range is not None:
            self.filter_by_angle = True
            self.point_cloud_angle_range = point_cloud_angle_range
            print(point_cloud_angle_range)
        else:
            self.filter_by_angle = False
            # self.point_cloud_angle_range = point_cloud_angle_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        # print(points_numpy[points_numpy[:,1]>0])
        pts_phi = (np.arctan(points_numpy[:, 0] / points_numpy[:, 1]) + (points_numpy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2) 
        
        pts_phi[pts_phi>np.pi] -= np.pi * 2
        pts_phi = pts_phi/np.pi*180
        
        assert np.all(-180 <= pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(pts_phi>=self.point_cloud_angle_range[0], pts_phi<=self.point_cloud_angle_range[1])
        # points_numpy = points_numpy[filt]
        # print(points_numpy[points_numpy[:,1]>0])

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 7.5))
        # ax = Axes3D(fig)
        # ax.view_init(30, 150)
        # ax.scatter(xs=[-54, 54], ys=[-54, 54], zs=[-10, 10], c='white')
        # ax.scatter(xs=points_numpy[:, 0], ys=points_numpy[:, 1], zs=points_numpy[:, 2], c='blue', s=2)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.gca().set_box_aspect((54, 54, 10))
        # plt.savefig('/data2/linzhiwei/project/TransFusion_SST/tmp/before.png')

        # fig = plt.figure(figsize=(10, 7.5))
        # points_numpy = points_numpy[filt]
        # ax = Axes3D(fig)
        # ax.view_init(30, 150)
        # ax.scatter(xs=[-54, 54], ys=[-54, 54], zs=[-10, 10], c='white')
        # ax.scatter(xs=points_numpy[:, 0], ys=points_numpy[:, 1], zs=points_numpy[:, 2], c='blue', s=2)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.gca().set_box_aspect((54, 54, 10))
        # plt.savefig('/data2/linzhiwei/project/TransFusion_SST/tmp/after.png')

        # print(a)
        return points[filt]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        if self.filter_by_angle:
            points = self.filter_point_by_angle(points)

        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'



@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results

@PIPELINES.register_module()
class LoadAnnotations3D_v4(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
