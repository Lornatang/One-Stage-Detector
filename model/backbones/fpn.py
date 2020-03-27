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
import torch.nn as nn

from model.module.conv import ConvBNLeakyReLU
from model.module.route import Route
from model.module.upsampling import Upsample


class FPN(nn.Module):
    """ Adapt to more size pictures, FPN for YOLOv3.
    Source paper see http://arxiv.org/pdf/1612.03144.
    """

    def __init__(self, in_channels, out_channels):
        """

        Args:
            in_channels (list): Number of channels in the input image.
            out_channels (list): Number of channels produced by the convolution.
        """
        super(FPN, self).__init__()

        in_channel_0, in_channel_1, in_channel_2 = in_channels
        out_channel_0, out_channel_1, out_channel_2 = out_channels

        # large
        self.convset0 = nn.Sequential(
            ConvBNLeakyReLU(in_channel_0, 512, 1, 1, 0),
            ConvBNLeakyReLU(512, 1024, 3, 1, 1),
            ConvBNLeakyReLU(1024, 512, 1, 1, 0),
            ConvBNLeakyReLU(512, 1024, 3, 1, 1),
            ConvBNLeakyReLU(1024, 512, 1, 1, 0),
        )

        self.conv0_0 = ConvBNLeakyReLU(512, 1024, 3, 1, 1)
        self.conv0_1 = nn.Conv2d(1024, out_channel_0, 1, 1, 0)

        self.conv0 = ConvBNLeakyReLU(512, 256, 1, 1, 0)
        self.upsample0 = Upsample(scale_factor=2)
        self.route0 = Route()

        # medium
        self.convset1 = nn.Sequential(
            ConvBNLeakyReLU(in_channel_1 + 256, 256, 1, 1, 0),
            ConvBNLeakyReLU(256, 512, 3, 1, 1),
            ConvBNLeakyReLU(512, 256, 1, 1, 0),
            ConvBNLeakyReLU(256, 512, 3, 1, 1),
            ConvBNLeakyReLU(512, 256, 1, 1, 0)

        )
        self.conv1_0 = ConvBNLeakyReLU(256, 512, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(512, out_channel_1, 1, 1, 0)

        self.conv1 = ConvBNLeakyReLU(256, 128, 1, 1, 0)
        self.upsample1 = Upsample(scale_factor=2)
        self.route1 = Route()

        # small
        self.convset2 = nn.Sequential(
            ConvBNLeakyReLU(in_channel_2 + 128, 128, 1, 1, 0),
            ConvBNLeakyReLU(128, 256, 3, 1, 1),
            ConvBNLeakyReLU(256, 128, 1, 1, 0),
            ConvBNLeakyReLU(128, 256, 3, 1, 1),
            ConvBNLeakyReLU(256, 128, 1, 1, 0),
        )
        self.conv2_0 = ConvBNLeakyReLU(128, 256, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(256, out_channel_2, 1, 1, 0)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.convset0(x0)
        out0 = self.conv0_0(r0)
        out0 = self.conv0_1(out0)

        # medium
        r1 = self.conv0(r0)
        r1 = self.upsample0(r1)
        x1 = self.route0(x1, r1)
        r1 = self.convset1(x1)
        out1 = self.conv1_0(r1)
        out1 = self.conv1_1(out1)

        # small
        r2 = self.conv1(r1)
        r2 = self.upsample1(r2)
        x2 = self.route1(x2, r2)
        r2 = self.convset2(x2)
        out2 = self.conv2_0(r2)
        out2 = self.conv2_1(out2)

        return out2, out1, out0
