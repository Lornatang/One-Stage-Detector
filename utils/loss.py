# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn

from .utils import GIOU_xywh_torch
from .utils import iou_xywh_torch


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction=reduction)  # must be nn.BCEWithLogitsLoss()

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        return loss


class YoloV3Loss(nn.Module):
    def __init__(self, cfg=None):
        super(YoloV3Loss, self).__init__()
        self.iou_threshold = cfg.TRAIN.IOU_THRESHOLD
        self.strides = cfg.MODEL.STRIDES

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.strides

        loss_s, loss_s_giou, loss_s_conf, loss_s_cls = self.cal_loss_per_layer(p[0], p_d[0],
                                                                               label_sbbox,
                                                                               sbboxes,
                                                                               strides[0])
        loss_m, loss_m_giou, loss_m_conf, loss_m_cls = self.cal_loss_per_layer(p[1], p_d[1],
                                                                               label_mbbox,
                                                                               mbboxes,
                                                                               strides[1])
        loss_l, loss_l_giou, loss_l_conf, loss_l_cls = self.cal_loss_per_layer(p[2], p_d[2],
                                                                               label_lbbox,
                                                                               lbboxes,
                                                                               strides[2])

        loss = loss_l + loss_m + loss_s
        loss_giou = loss_s_giou + loss_m_giou + loss_l_giou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_giou, loss_conf, loss_cls

    def cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.
        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.
        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.
        :param stride: The scale of the feature map relative to the original image
        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # loss giou
        giou = GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou) * label_mix

        # loss confidence
        iou = iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.iou_threshold).float()

        loss_conf = (label_obj_mask * FOCAL(pred=p_conf, true=label_obj_mask) +
                     label_noobj_mask * FOCAL(pred=p_conf, true=label_obj_mask)) * label_mix

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        return loss, loss_giou, loss_conf, loss_cls
