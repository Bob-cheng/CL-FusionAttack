from mmcv.utils import Registry
import warnings

# from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS
from mmcv.cnn import MODELS as MMCV_MODELS
from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS)
from mmseg.models.builder import SEGMENTORS

FUSIONMODELS = Registry("fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")

MODELS = Registry('models', parent=MMCV_MODELS)
FUSION_LAYERS = MODELS
VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_model(cfg, train_cfg=None, test_cfg=None):
    if cfg.type in ['BEVFusion', 'BEVFusion_v1test', 'BEVFusion_v1test2']:
        return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    elif cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)

def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)
