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

from model.backbones import Darknet53
from model.backbones import YOLOLayer
from model.backbones import FPN


class YOLOv3(nn.Module):
    def __init__(self, cfg=None):
        super(YOLOv3, self).__init__()

        assert cfg is not None, "Error: Network structure profile cannot be empty!"

        # number of anchors
        self.na = torch.FloatTensor([[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],
                                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
                                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]])
        self.strides = torch.FloatTensor(cfg.MODEL.STRIDES)
        self.nc = len(cfg.CLASSES)  # number of classes
        self.oc = cfg.MODEL.PRIORS_ANCHOR * (self.nc + 5)  # output channels

        self.backbone = Darknet53()
        self.fpn = FPN(in_channels=[1024, 512, 256],
                       out_channels=[self.oc,
                                     self.oc,
                                     self.oc])

        # small anchors
        self.small = YOLOLayer(anchors=self.na[0],
                               num_classes=self.nc,
                               stride=self.strides[0])
        # medium anchors
        self.medium = YOLOLayer(anchors=self.na[1],
                                num_classes=self.nc,
                                stride=self.strides[1])
        # large anchors
        self.large = YOLOLayer(anchors=self.na[2],
                               num_classes=self.nc,
                               stride=self.strides[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        out = []

        output_small, output_medium, output_large = self.backbone(x)
        pred_small, pred_medium, pred_large = self.fpn(output_large,
                                                       output_medium,
                                                       output_small)

        out.append(self.small(pred_small))
        out.append(self.medium(pred_medium))
        out.append(self.large(pred_large))

        if self.training:
            pred, raw = list(zip(*out))
            return pred, raw  # small, medium, large
        else:
            pred, raw = list(zip(*out))
            return pred, torch.cat(raw, 0)
