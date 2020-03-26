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

from model.module.activition import Mish
from model.module.activition import Swish


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 batch_norm=None, activation=None):
        super(BasicConv2d, self).__init__()

        self.batch_norm = batch_norm
        self.activation = activation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=not batch_norm)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "relu6":
            self.activation = nn.ReLU6(inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "mish":
            self.activation = Mish()

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        if self.activation:
            out = self.activation(out)

        return out


class DeepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeepConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.main(x)

        return out


def fuse_conv_and_bn(conv, bn):
    # source from https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
