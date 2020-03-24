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

from configs import yolov3_voc
from model.backbones import Darknet53
from model.layer import YOLO
from model.module.fpn import FPN


class VOC(nn.Module):
    def __init__(self):
        super(VOC, self).__init__()

        self.na = torch.Tensor(yolov3_voc.MODEL["ANCHORS"])  # number of anchors
        self.strides = torch.Tensor(yolov3_voc.MODEL["STRIDES"])
        self.nc = yolov3_voc.DATA["NUM_CLASSES"]  # number of classes
        self.oc = yolov3_voc.MODEL["ANCHORS_PER_SCLAE"] * (
                self.nc + 5)  # output channels

        self.backbone = Darknet53()
        self.fpn = FPN(in_channels=[1024, 512, 256],
                       out_channels=[self.oc,
                                     self.oc,
                                     self.oc])

        # small anchors
        self.small = YOLO(anchors=self.na[0],
                          num_classes=self.nc,
                          stride=self.strides[0])
        # medium anchors
        self.medium = YOLO(anchors=self.na[1],
                           num_classes=self.nc,
                           stride=self.strides[1])
        # large anchors
        self.large = YOLO(anchors=self.na[2],
                          num_classes=self.nc,
                          stride=self.strides[2])

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = []

        output_small, output_medium, output_large = self.backbone(x)
        pred_small, pred_medium, pred_large = self.fpn(output_small,
                                                       output_medium,
                                                       output_large)

        out.append(self.small(pred_small))
        out.append(self.medium(pred_medium))
        out.append(self.medium(pred_large))

        if self.training:
            pred, raw = list(zip(*out))
            return pred, raw  # small, medium, large
        else:
            pred, raw = list(zip(*out))
            return pred, torch.cat(raw, 0)
