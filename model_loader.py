import os
import sys

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model, build_detector
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

class Model_loader(object):
    def __init__(self) -> None:
        ## original bevfusion
        self.bevfusion_config = 'bevfusion/bevfusion_original_cfg.yaml'
        self.bevfusion_ckpoint = 'bevfusion/pretrained/bevfusion-det.pth'

        ## DeepInteraction
        self.deepint_config = 'DeepInteraction/projects/configs/nuscenes/Fusion_0075_refactor.py'
        self.deepint_ckpoint = 'DeepInteraction/pretrained/Fusion_0075_refactor.pth'

        ## UVTR Fusion
        self.uvtr_config = 'UVTR/projects_uvtr/configs/uvtr/multi_modality/uvtr_m_v0075_r101_h5.py'
        self.uvtr_ckpoint = 'UVTR/pretrained/uvtr_m_v0075_r101_h5.pth'

        ## original bevfusion2
        self.bevfusion2_ckpoint = 'bevfusion2/pretrained/bevfusion_tf.pth'
        self.bevfusion2_config = 'bevfusion2/configs/bevfusion/bevf_tf_4x8_10e_nusc_aug.py'

        self.transfusion_config = 'transfusion/transfusion_fix.py'
        self.transfusion_ckpoint = 'transfusion/epoch_6.pth'

        # bevformer original
        self.bevformer_config = 'bevformer/projects_bevcam/configs/bevformer/bevformer_base.py'
        self.bevformer_ckpoint = 'bevformer/ckpts/bevformer_r101_dcn_24ep.pth'
        
    
    def load_bevfusion_model(self):
        from torchpack.utils.config import configs
        from mmdet3d.utils import recursive_eval
        configs.load(self.bevfusion_config, recursive=True)
        cfg = Config(recursive_eval(configs), filename=self.bevfusion_config)
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        cfg.model.pretrained = None
        samples_per_gpu = 1
        # Single test dataset
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # init distributed env first, since logger depends on the dist info.
        distributed = False

        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.bevfusion_ckpoint, map_location="cpu")
        # old versions did not save class info in checkpoints, this walkaround is 
        # for backward compatibility
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        # not distributed
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader)
        return model, data_loader, cfg
    
    def load_uvtr_model(self):
        sys.path.append('./UVTR')
        cfg = Config.fromfile(self.uvtr_config)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])

        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        
        distributed = False
        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        # cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.uvtr_ckpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
        return model, data_loader, cfg

    def load_deepint_model(self):
        sys.path.append('./DeepInteraction')
        cfg = Config.fromfile(self.deepint_config)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        
        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)

        distributed = False

        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)
        
        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        # cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.deepint_ckpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # not distrubuted
        model = MMDataParallel(model, device_ids=[0])
        return model, data_loader, cfg
    
    def load_transfusion_model(self):
        sys.path.append('./transfusion')
        cfg = Config.fromfile(self.transfusion_config)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            # cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        
        distributed = False

        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        # cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.transfusion_ckpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # not distrubuted
        model = MMDataParallel(model, device_ids=[0])
        return model, data_loader, cfg
    
    def load_bevfusion2_model(self):
        sys.path.append('./bevfusion2')
        cfg = Config.fromfile(self.bevfusion2_config)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            # cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        
        distributed = False

        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        # cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.bevfusion2_ckpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # not distrubuted
        model = MMDataParallel(model, device_ids=[0])
        return model, data_loader, cfg

    def load_bevformer_model(self):
        sys.path.append('./bevformer')
        # from projects_bevcam.mmdet3d_plugin.datasets.builder import build_dataloader as build_dataloader_bevformer
        cfg = Config.fromfile(self.bevformer_config)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        
        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        else:
            raise NotImplementedError('list test config not implemented.')

        distributed = False

        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)
        
        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
            # nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )

        # build the model and load checkpoint
        # cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.bevformer_ckpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # not distrubuted
        model = MMDataParallel(model, device_ids=[0])
        return model, data_loader, cfg
    

    def load_model(self, model_name):
        if model_name == 'bevfusion':
            return self.load_bevfusion_model()
        elif model_name == 'deepint':
            return self.load_deepint_model()
        elif model_name == 'uvtr':
            return self.load_uvtr_model()
        elif model_name == 'bevfusion2':
            return self.load_bevfusion2_model()
        elif model_name == 'transfusion':
            return self.load_transfusion_model()
        elif model_name == 'autoalign':
            return self.load_autoalign_model()
        elif model_name == 'bevformer':
            return self.load_bevformer_model()

if __name__ == '__main__':
    model_loader = Model_loader()
    # print("load transfusion model")
    # model, data_loader, cfg = model_loader.load_model('transfusion')
    # print("load uvtr model")
    # model, data_loader, cfg = model_loader.load_model('uvtr')
    # print("load deepint model")
    # model, data_loader, cfg = model_loader.load_model('deepint')
    # print("load bevfusion model")
    # model, data_loader, cfg = model_loader.load_model('bevfusion')
    # print("load bevfusion2 model")
    # model, data_loader, cfg = model_loader.load_model('bevfusion2')
    print("load bevformer model")
    model, data_loader, cfg = model_loader.load_model('bevformer')
