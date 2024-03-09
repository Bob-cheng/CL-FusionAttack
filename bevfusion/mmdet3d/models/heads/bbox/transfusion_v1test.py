import copy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

__all__ = ["TransFusionHead_v1test"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class TransFusionHead_v1test(nn.Module):
    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=1,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_iou=dict(
            type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"
        ),
        loss_bbox=dict(type="L1Loss", reduction="mean"),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead_v1test, self).__init__()

        self.fp16_enabled = False

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.num_decoder_layers = 2 #num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )


        if self.initialize_by_heatmap:
            # a convolution for fused feat
            self.fused_conv = build_conv_layer(
                dict(type="Conv2d"),
                80, #in_channels,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

            # self.heatmap_feat_conv =  build_conv_layer(
            #     dict(type="Conv2d"),
            #     128,
            #     128,
            #     kernel_size=10,
            #     padding=0,
            #     bias=bias,
            #     stride=10
            # )
            
            layers = []
            layers.append(
                ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_cfg=dict(type="Conv2d"),
                    norm_cfg=dict(type="BN2d"),
                )
            )
            layers.append(
                build_conv_layer(
                    dict(type="Conv2d"),
                    hidden_channel,
                    num_classes,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                )
            )
            self.heatmap_head = nn.Sequential(*layers)
            # self.heatmap_head_img = copy.deepcopy(self.heatmap_head) # for debug purpose
            # for param in self.heatmap_head.parameters():
            #     param.requires_grad_(False)
            
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos)


        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel,
                    num_heads,
                    ffn_channel,
                    dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )
        
        # heatmap fusion decoder
        # self.heatmap_feat_decoder = TransformerDecoderLayer(
        #             hidden_channel,
        #             nhead=8,
        #             dim_feedforward=128,
        #             dropout=dropout,
        #             activation=activation,
        #             self_posembed=PositionEmbeddingLearned(2, hidden_channel),
        #             cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
        #             cross_only=True
        #         )

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                )
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid], indexing='ij'
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, pure_lidar_inputs, pure_cam_inputs, img_inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        # lidar_feat = self.shared_conv(inputs)
        # lidar_feat = self.fused_conv(inputs)
        pure_cam_feat = self.fused_conv(pure_cam_inputs)

        #################################
        # image to BEV
        #################################
        # lidar_feat_flatten = lidar_feat.view(
        #     batch_size, lidar_feat.shape[1], -1
        # )  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(inputs.device)

        #################################
        # image guided query initialization
        #################################
        if self.initialize_by_heatmap:
            ## generate heatmap with fused lidar and camera feature (overall-sensitivity)
            # dense_heatmap = self.heatmap_head(lidar_feat) # lidar_feat: 1, 128, 180, 180

            # ################# for debug purpose ################
            # pure_lidar_feat = self.lidar_conv(pure_lidar_inputs) # pure_lidar_feat: 1, 128, 180, 180
            pure_lidar_feat = self.shared_conv(pure_lidar_inputs) # pure_lidar_feat: 1, 128, 180, 180
            pure_lidar_feat_flatten = pure_lidar_feat.view(
                batch_size, pure_lidar_feat.shape[1], -1
            )  # [BS, C, H*W]

            ## use transformer decoder to generate heatmap using both lidar (Query) and camera data(K,V)
            # pure_cam_feat_downsample = self.heatmap_feat_conv(pure_cam_feat)
            # pure_cam_feat_downsample_flatten = pure_cam_feat_downsample.view(batch_size, pure_lidar_feat.shape[1], -1)
            # downsampled_pos = self.create_2D_grid(*pure_cam_feat_downsample.shape[-2:]).to(bev_pos.device)
            # heatmap_feat_flatten = self.heatmap_feat_decoder(pure_lidar_feat_flatten, pure_cam_feat_downsample_flatten, bev_pos, downsampled_pos)
            heatmap_feat_flatten = pure_cam_feat.view(
                batch_size, pure_cam_feat.shape[1], -1
            )  # [BS, C, H*W]
            # dense_heatmap = self.heatmap_head_img(heatmap_feat_flatten.view(pure_lidar_feat.shape))
            # dense_heatmap_lidar = self.heatmap_head(pure_lidar_feat)
            dense_heatmap = self.heatmap_head(pure_lidar_feat)
            ## generate heatmap with only lidar feature, (no image gradients)
            # dense_heatmap = self.heatmap_head(pure_lidar_feat)
            # ####################################################

            dense_heatmap_img = None
            # heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_lidar.detach().sigmoid()) / 2
            heatmap = dense_heatmap.detach().sigmoid()
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(
                heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
            )
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg["dataset"] == "nuScenes":
                local_max[
                    :,
                    8,
                ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[
                    :,
                    9,
                ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
                local_max[
                    :,
                    1,
                ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[
                    :,
                    2,
                ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
                ..., : self.num_proposals
            ]
            # top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='trunc')
            top_proposals_index = top_proposals % heatmap.shape[-1]
            
            # query_feat = lidar_feat_flatten.gather(
            #     index=top_proposals_index[:, None, :].expand(
            #         -1, lidar_feat_flatten.shape[1], -1
            #     ),
            #     dim=-1,
            # )

            ########### for debug purpose #########
            query_feat = pure_lidar_feat_flatten.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, pure_lidar_feat_flatten.shape[1], -1
                ),
                dim=-1,
            )
            #######################################

            self.query_labels = top_proposals_class

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
                0, 2, 1
            )
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(
                index=top_proposals_index[:, None, :]
                .permute(0, 2, 1)
                .expand(-1, -1, bev_pos.shape[-1]),
                dim=1,
            )

        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(inputs.device)  # [BS, num_proposals, 2]
            query_pos = self.query_pos
        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        Q_N = query_feat.shape[2]
        KV_N = heatmap_feat_flatten.shape[2]
        total_mask = query_feat.new_zeros(batch_size * self.num_heads, Q_N, KV_N)
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            ##### for debug purpose #####
            # query_feat = self.decoder[i](
            #     query_feat, lidar_feat_flatten, query_pos, bev_pos
            # )

            query_feat = self.decoder[i](
                query_feat, heatmap_feat_flatten, query_pos, bev_pos, attn_mask=total_mask.detach()
            )
            #############################
            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

            # for  next level masked training
            if i != self.num_decoder_layers - 1:
                res_layer["query_heatmap_score"] = heatmap.gather(
                    index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
                    dim=-1,
                )  # [bs, num_classes, num_proposals]
                # print(res_layer["query_heatmap_score"].shape, res_layer['heatmap'].shape)
                bbox_res = self.get_bboxes([[res_layer]], metas)
                total_mask = []
                for bbox_res_single in bbox_res:
                    bboxes_3d, scores, labels_3d, _ = tuple(bbox_res_single)
                    indices = scores > 0.01
                    if indices.sum() > 0:
                        bboxes_3d   = bboxes_3d[indices]
                        scores      = scores[indices]
                        labels_3d   = labels_3d[indices]
                    bboxes_3d = bboxes_3d.tensor
                    voxel_size = torch.tensor(self.train_cfg["voxel_size"])
                    pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
                    mask = bboxes_3d.new_zeros(1, 180, 180)
                    for idx in range(len(bboxes_3d)):
                        width = bboxes_3d[idx][3]
                        length = bboxes_3d[idx][4]
                        width = width / voxel_size[0] / self.train_cfg["out_size_factor"] * 5
                        length = length / voxel_size[1] / self.train_cfg["out_size_factor"] * 5
                        if width > 0 and length > 0:
                            radius = gaussian_radius((length, width), min_overlap=self.train_cfg["gaussian_overlap"])
                            radius = max(self.train_cfg["min_radius"], int(radius))
                            x, y = bboxes_3d[idx][0], bboxes_3d[idx][1]
                            coor_x = ((x - pc_range[0])/ voxel_size[0] / self.train_cfg["out_size_factor"])
                            coor_y = ( (y - pc_range[1])/ voxel_size[1] / self.train_cfg["out_size_factor"])

                            center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=mask.device)
                            center_int = center.to(torch.int32)

                            # original
                            # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                            # NOTE: fix
                            draw_heatmap_gaussian( mask[0], center_int[[1, 0]], radius)
                    total_mask.append(mask)
                total_mask = torch.stack(total_mask) # shape, (4, 1, 180, 180), values: [0,1]
                total_mask[total_mask.isnan()] = 0
                #convert to -inf and 0
                non_obj_idxes = total_mask == 0
                obj_idxes = ~non_obj_idxes
                total_mask[non_obj_idxes] = -1e9
                total_mask[obj_idxes] = 0
                # convert to BS * self.num_heads, Q_N, KV_N
                total_mask = total_mask.view(batch_size, 1, KV_N).expand(batch_size,  Q_N, KV_N)
                total_mask = total_mask.unsqueeze(1).expand(batch_size, self.num_heads,  Q_N, KV_N).reshape(batch_size * self.num_heads, Q_N, KV_N)

                # print(total_mask.shape, pure_cam_feat.shape)
                # pure_cam_feat_mask = pure_cam_feat * total_mask.detach()
                # # pure_cam_feat_mask = pure_cam_feat + total_mask.detach()
                # heatmap_feat_flatten = pure_cam_feat_mask.view( batch_size, pure_cam_feat.shape[1], -1)  # [BS, C, H*W]


        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.initialize_by_heatmap:
            ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
                index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
                dim=-1,
            )  # [bs, num_classes, num_proposals]
            ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, lidar_feats, cam_feats, metas):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        if isinstance(lidar_feats, torch.Tensor):
            lidar_feats = [lidar_feats]
        if isinstance(cam_feats, torch.Tensor):
            cam_feats = [cam_feats]
        res = multi_apply(self.forward_single, feats, lidar_feats, cam_feats, [None], [metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx : batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        obj_gt_indices = torch.cat(res_tuple[-1], dim=0)
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap, obj_gt_indices
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, obj_gt_indices
        # heatmap = torch.cat(res_tuple[7], dim=0)
        # return (
        #     labels,
        #     label_weights,
        #     bbox_targets,
        #     bbox_weights,
        #     ious,
        #     num_pos,
        #     matched_ious,
        #     heatmap,
        #     obj_gt_indices
        # )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict["center"].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1), :
            ]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1),
            ]

            if self.train_cfg.assigner.type == "HungarianAssigner3D":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == "HeuristicAssigner":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        obj_gt_indices = -1 * bboxes_tensor.new_ones(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            # bbox_weights[pos_inds, :] = 1.0
            bbox_weights[pos_inds, :] = torch.ones_like(bbox_weights[pos_inds, :]).float()

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
                obj_gt_indices[pos_inds] = sampling_result.pos_assigned_gt_inds
            if self.train_cfg.pos_weight <= 0:
                # label_weights[pos_inds] = 1.0
                label_weights[pos_inds] = torch.ones_like(label_weights[pos_inds])
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            # label_weights[neg_inds] = 1.0
            label_weights[neg_inds] = torch.ones_like(label_weights[neg_inds])

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
            ).to(device)
            grid_size = torch.tensor(self.train_cfg["grid_size"])
            pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
            voxel_size = torch.tensor(self.train_cfg["voxel_size"])
            feature_map_size = (
                # grid_size[:2] // self.train_cfg["out_size_factor"]
                torch.div(grid_size[:2], self.train_cfg["out_size_factor"], rounding_mode='trunc')
            )  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(
                self.num_classes, feature_map_size[1], feature_map_size[0]
            )
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius = max(self.train_cfg["min_radius"], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (
                        (x - pc_range[0])
                        / voxel_size[0]
                        / self.train_cfg["out_size_factor"]
                    )
                    coor_y = (
                        (y - pc_range[1])
                        / voxel_size[1]
                        / self.train_cfg["out_size_factor"]
                    )

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    # original
                    # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                    # NOTE: fix
                    draw_heatmap_gaussian(
                        heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                    )

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return (
                labels[None],
                label_weights[None],
                bbox_targets[None],
                bbox_weights[None],
                ious[None],
                int(pos_inds.shape[0]),
                float(mean_iou),
                heatmap[None],
                obj_gt_indices
            )
        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return (
                labels[None], 
                label_weights[None], 
                bbox_targets[None], 
                bbox_weights[None], 
                ious[None], 
                int(pos_inds.shape[0]), 
                float(mean_iou), 
                obj_gt_indices
            )


    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # (
        #     labels,
        #     label_weights,
        #     bbox_targets,
        #     bbox_weights,
        #     ious,
        #     num_pos,
        #     matched_ious,
        #     heatmap,
        #     obj_gt_indices
        # ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if self.initialize_by_heatmap:
            (labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap, obj_gt_indices) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            (labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, obj_gt_indices) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict["dense_heatmap"]),
                heatmap,
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
            )
            loss_dict["loss_heatmap"] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1"
            else:
                prefix = f"layer_{idx_layer}"

            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_score = preds_dict["heatmap"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict["center"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_height = preds_dict["height"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_rot = preds_dict["rot"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_dim = preds_dict["dim"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size]
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                    ...,
                    idx_layer
                    * self.num_proposals : (idx_layer + 1)
                    * self.num_proposals,
                ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get("code_weights", None)
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            )
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_loss_bbox = self.loss_bbox(
                preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1)
            )

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f"matched_ious"] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, metas, 
                img=None, 
                rescale=False, 
                for_roi=False,
                gt_bboxes_3d=None,
                gt_labels_3d=None
                ):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        if gt_bboxes_3d is not None and gt_labels_3d is not None:
        # if False: # for training purpose
            obj_gt_indices = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])[-1]
        else:
            obj_gt_indices = None
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals :].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            if self.initialize_by_heatmap:
                one_hot = F.one_hot(
                    self.query_labels, num_classes=self.num_classes
                ).permute(0, 2, 1)
                batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot
            else:
                batch_score = batch_score

            batch_center = preds_dict[0]["center"][..., -self.num_proposals :]
            batch_height = preds_dict[0]["height"][..., -self.num_proposals :]
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals :]
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals :]
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals :]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True if obj_gt_indices is None else False,
            )

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]
                if obj_gt_indices is not None:
                    assert len(labels) == len(obj_gt_indices)
                ## adopt circle nms for different categories
                if self.test_cfg["nms_type"] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task["radius"] > 0:
                            if self.test_cfg["nms_type"] == "circle":
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task["radius"],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]["box_type_3d"](
                                        boxes3d[task_mask][:, :7], 7
                                    ).bev
                                )
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task["radius"],
                                    pre_maxsize=self.test_cfg["pre_maxsize"],
                                    post_max_size=self.test_cfg["post_maxsize"],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices
                            ]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                        obj_gt_indices=obj_gt_indices[keep_mask] if obj_gt_indices is not None else None
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels, obj_gt_indices=obj_gt_indices)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        if len(rets[0]) == 1:
            res = [
                [
                    metas[0]["box_type_3d"](
                        rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                    ),
                    rets[0][0]["scores"],
                    rets[0][0]["labels"].int(),
                    rets[0][0]["obj_gt_indices"]
                ]
            ]
        else:
            res = []
            for i in range(batch_size):
                res.append([
                    metas[0]["box_type_3d"](
                        rets[0][i]["bboxes"], box_dim=rets[0][i]["bboxes"].shape[-1]
                    ),
                    rets[0][i]["scores"],
                    rets[0][i]["labels"].int(),
                    rets[0][i]["obj_gt_indices"]
                ])
        return res
