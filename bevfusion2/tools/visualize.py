import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)

    cfg = Config.fromfile(args.config)

    # torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        # dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES
        distributed = True
        # distributed = False
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
        model.eval()

    for data in tqdm(dataflow):
        metas = data["img_metas"][0].data[0][0]
        name = "{}-{}".format(metas['pts_filename'][-24:-8], metas["sample_idx"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(return_loss=False, rescale=True, **data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"][0].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"][0].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            scores = np.ones_like(labels)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]['pts_bbox']:
            bboxes = outputs[0]['pts_bbox']["boxes_3d"].tensor.numpy()
            scores = outputs[0]['pts_bbox']["scores_3d"].numpy()
            labels = outputs[0]['pts_bbox']["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"][0].data[0].numpy()
            masks = masks.astype(bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        box_idxs = np.arange(len(labels))

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2img"][k],
                    classes=cfg.class_names,
                    box_idxs=None, #box_idxs,
                    scores=None, #scores
                )

        if "points" in data:
            lidar = data["points"][0].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.class_names,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()

