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

import numpy as np
import torch
import torch.nn as nn

import yolo.configs.voc
from yolo.model import BasicBlock
from yolo.model import FPN
from yolo.model import Head
from yolo.model.backbones import Darknet53


class VOC(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, init_weights=True):
        super(VOC, self).__init__()

        self.anchors = torch.FloatTensor(yolo.configs.voc.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(yolo.configs.voc.MODEL["STRIDES"])
        self.num_classes = yolo.configs.voc.DATA["NUM"]
        self.out_filters = yolo.configs.voc.MODEL["ANCHORS_PER_SCLAE"] * (
                self.num_classes + 5)

        self.backbone = Darknet53()
        self.fpn = FPN(inplanes=[1024, 512, 256],
                       planes=[self.out_filters,
                               self.out_filters,
                               self.out_filters])

        # small
        self.s_head = Head(nC=self.num_classes, anchors=self.anchors[0],
                           stride=self.strides[0])
        # medium
        self.m_head = Head(nC=self.num_classes, anchors=self.anchors[1],
                           stride=self.strides[1])
        # large
        self.l_head = Head(nC=self.num_classes, anchors=self.anchors[2],
                           stride=self.strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.backbone(x)
        x_s, x_m, x_l = self.fpn(x_l, x_m, x_s)

        out.append(self.s_head(x_s))
        out.append(self.m_head(x_m))
        out.append(self.l_head(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=52):
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, BasicBlock):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
