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
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding,
                 batch_norm=False, activation=None):
        super(BasicBlock, self).__init__()

        layers = [nn.Conv2d(inplanes, planes, kernel_size,
                            stride, padding, bias=not batch_norm)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(planes))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)

        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes, meplanes, planes):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            BasicBlock(inplanes, meplanes, 1, 1, 0,
                       batch_norm=True, activation="leakyrelu"),
            BasicBlock(meplanes, planes, 3, 1, 1,
                       batch_norm=True, activation="leakyrelu")
        )

    def forward(self, x):
        shortcut = x
        out = self.main(x)
        out += shortcut

        return out


class Head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)

        p_de = self.__decode(p.clone())

        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3,
                                                           1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1,
                              5 + self.__nC) if not self.training else pred_bbox


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class FPN(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """

    def __init__(self, inplanes, planes):
        super(FPN, self).__init__()

        fi_0, fi_1, fi_2 = inplanes
        fo_0, fo_1, fo_2 = planes

        # large
        self.__conv_set_0 = nn.Sequential(
            BasicBlock(fi_0, 512, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(512, 1024, 3, 1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(1024, 512, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(512, 1024, 3, 1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(1024, 512, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
        )

        self.__conv0_0 = BasicBlock(512, 1024, 3, 1, 1, batch_norm=True,
                                    activation="leakyrelu")
        self.__conv0_1 = BasicBlock(1024, fo_0, 1, 1, 0)

        self.__conv0 = BasicBlock(512, 256, 1, 1, 0, batch_norm=True,
                                  activation="leakyrelu")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium
        self.__conv_set_1 = nn.Sequential(
            BasicBlock(fi_1 + 256, 256, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(256, 512, 3, 1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(512, 256, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(256, 512, 3, 1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(512, 256, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),

        )
        self.__conv1_0 = BasicBlock(256, 512, 3, 1, 1, batch_norm=True,
                                    activation="leakyrelu")
        self.__conv1_1 = BasicBlock(512, fo_1, 1, 1, 0)

        self.__conv1 = BasicBlock(256, 128, 1, 1, 0, batch_norm=True,
                                  activation="leakyrelu")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            BasicBlock(fi_2 + 128, 128, 1, 1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(fi_2 + 128, 128, 1,
                       1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(128, 256, 3,
                       1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(256, 128, 1,
                       1, 0, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(128, 256, 3,
                       1, 1, batch_norm=True,
                       activation="leakyrelu"),
            BasicBlock(256, 128, 1,
                       1, 0, batch_norm=True,
                       activation="leakyrelu"),
        )
        self.__conv2_0 = BasicBlock(128, 256,
                                    3, 1,
                                    1, batch_norm=True, activation="leakyrelu")
        self.__conv2_1 = BasicBlock(256, fo_2,
                                    1,
                                    1, 0)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large
