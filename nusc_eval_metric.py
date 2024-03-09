from nuscenes import NuScenes
from my_detection_eval import DetectionEval
import tempfile
from os import path as osp
import mmcv
import numpy as np
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset, output_to_nusc_box, lidar_nusc_box_to_global

class NuscEvalMetric(object):
    def __init__(self, dataset_loader) -> None:
        self.dataset = dataset_loader.dataset
        if hasattr(self.dataset, 'data_root'):
            data_root = self.dataset.data_root
        elif hasattr(self.dataset, 'dataset_root'):
            data_root = self.dataset.dataset_root
        self.nusc = NuScenes(version=self.dataset.version, dataroot=data_root, verbose=False)

    def eval_results(self, results, verbose=False):
        """
        results: # List[Tuple[index, token, outputs, obj_idxs]]
        """
        sample_idxs = [rst[0] for rst in results]
        sample_tokens = [rst[1] for rst in results]
        detect_outputs = [rst[2] for rst in results]
        object_idxs = [rst[3] for rst in results]
        exclude_object_idxs_dict = {}
        if object_idxs[0] is not None:
            for i, token in enumerate(sample_tokens):
                exclude_object_idxs_dict[token] = object_idxs[i]
        else:
            exclude_object_idxs_dict = None

        # detect_outputs = [ out[0]['pts_bbox'] for out in detect_outputs]

        metrics = {}
        ## TODO: write result path
        if "boxes_3d" in detect_outputs[0]:
            result_files, tmp_dir = self.format_results(detect_outputs,sample_idxs, None)

            if isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files, sample_tokens, exclude_object_idxs_dict=exclude_object_idxs_dict))

            if tmp_dir is not None:
                tmp_dir.cleanup()
        return metrics

    def _evaluate_single(self, result_path, sample_tokens, exclude_object_idxs_dict=None):
        output_dir = osp.join(*osp.split(result_path)[:-1])
        my_eval = DetectionEval(
            self.nusc,
            self.dataset.eval_detection_configs,
            result_path=result_path,
            sample_tokens=sample_tokens,
            exclude_objs=exclude_object_idxs_dict,
            output_dir=output_dir,
            verbose=True
        )
        my_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        for name in self.dataset.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_ap_dist_{}".format(name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_{}".format(name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["object/{}".format(NuScenesDataset.ErrNameMapping[k])] = val

        detail["object/nds"] = metrics["nd_score"]
        detail["object/map"] = metrics["mean_ap"]
        return detail
    
    def format_results(self, results, sample_idxs, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        # assert len(results) == len(
        #     self
        # ), "The length of results is not equal to the dataset len: {} != {}".format(
        #     len(results), len(self)
        # )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, sample_idxs, jsonfile_prefix)
        return result_files, tmp_dir
    
    def _format_bbox(self, results, sample_idxs, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.dataset.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.dataset.data_infos[sample_idxs[sample_id]]["token"]
            boxes = lidar_nusc_box_to_global(
                self.dataset.data_infos[sample_idxs[sample_id]],
                boxes,
                mapped_class_names,
                self.dataset.eval_detection_configs,
                self.dataset.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.dataset.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
