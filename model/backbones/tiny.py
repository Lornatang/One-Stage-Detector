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

from ..module import BasicConv2d


class Tiny(torch.nn.Module):

    def __init__(self):
        super(Tiny, self).__init__()
        self.conv1 = BasicConv2d(3, 16, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BasicConv2d(16, 32, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicConv2d(32, 64, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = BasicConv2d(64, 128, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = BasicConv2d(128, 256, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = BasicConv2d(256, 512, 3, 1, 1, batch_norm=True, activation="leakyrelu")
        self.maxpool6 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.conv7 = BasicConv2d(512, 1024, 3, 1, 1, batch_norm=True, activation="leakyrelu")

    def forward(self, x):
        x = self.conv1(x)  # 416 * 416 * 16
        x = self.maxpool1(x)  # 208 * 208 * 16

        x = self.conv2(x)  # 208 * 208 * 32
        x = self.maxpool2(x)  # 104 * 104 * 32

        x = self.conv3(x)  # 104 * 104 * 64
        x = self.maxpool3(x)  # 52 * 52 * 64

        small_output = self.conv4(x)  # 52 * 52 * 128
        x = self.maxpool4(small_output)  # 26 * 26 * 128

        medium_output = self.conv5(x)  # 26 * 26 * 256
        x = self.maxpool5(medium_output)  # 13 * 13 * 256

        x = self.conv6(x)  # 13 * 13 * 512
        x = self.maxpool6(x)  # 13 * 13 * 512
        large_output = self.conv7(x)  # 13 * 13 * 1024

        return small_output, medium_output, large_output
