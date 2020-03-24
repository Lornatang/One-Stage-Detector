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
from model.backbones import Tiny
from model.layer import YOLO


class TinyVOC(nn.Module):
    def __init__(self):
        super(TinyVOC, self).__init__()

        self.na = torch.Tensor(yolov3_voc.MODEL["ANCHORS"])  # number of anchors
        self.strides = torch.Tensor(yolov3_voc.MODEL["STRIDES"])
        self.nc = yolov3_voc.DATA["NUM_CLASSES"]  # number of classes
        self.oc = yolov3_voc.MODEL["ANCHORS_PER_SCLAE"] * (
                self.nc + 5)  # output channels

        self.backbone = Tiny()

        # small anchors
        self.small = YOLO(anchors=self.na[0],
                          num_classes=self.nc,
                          stride=self.strides[0])
        # medium anchors
        self.medium = YOLO(anchors=self.na[1],
                           num_classes=self.nc,
                           stride=self.strides[1])

    def forward(self, x):
        out = []

        pred_small, pred_medium = self.backnone(x)

        out.append(self.small(pred_small))
        out.append(self.medium(pred_medium))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # small, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
