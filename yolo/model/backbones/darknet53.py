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

from yolo.model import BasicBlock
from yolo.model import ResidualBlock


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv = BasicBlock(3, 32, 3, 1, 1, batch_norm=True,
                               activation='leakyrelu')

        self.conv_5_0 = BasicBlock(32, 64, 3, 2, 1, batch_norm=True,
                                   activation='leakyrelu')
        self.rb_5_0 = ResidualBlock(inplanes=64, meplanes=32, planes=64)

        self.conv_5_1 = BasicBlock(64, 128, 3, 2, 1, batch_norm=True,
                                   activation='leakyrelu')
        self.rb_5_1_0 = ResidualBlock(inplanes=128, meplanes=64, planes=128)
        self.rb_5_1_1 = ResidualBlock(inplanes=128, meplanes=64, planes=128)

        self.conv_5_2 = BasicBlock(128, 256, 3, 2, 1, batch_norm=True,
                                   activation='leakyrelu')
        self.rb_5_2_0 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_1 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_2 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_3 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_4 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_5 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_6 = ResidualBlock(inplanes=256, meplanes=128, planes=256)
        self.rb_5_2_7 = ResidualBlock(inplanes=256, meplanes=128, planes=256)

        self.conv_5_3 = BasicBlock(256, 512, 3, 2, 1, batch_norm=True,
                                   activation='leakyrelu')
        self.rb_5_3_0 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_1 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_2 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_3 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_4 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_5 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_6 = ResidualBlock(inplanes=512, meplanes=256, planes=512)
        self.rb_5_3_7 = ResidualBlock(inplanes=512, meplanes=256, planes=512)

        self.conv_5_4 = BasicBlock(512, 1024, 3, 2, 1, batch_norm=True,
                                   activation='leakyrelu')
        self.rb_5_4_0 = ResidualBlock(inplanes=1024, meplanes=512, planes=1024)
        self.rb_5_4_1 = ResidualBlock(inplanes=1024, meplanes=512, planes=1024)
        self.rb_5_4_2 = ResidualBlock(inplanes=1024, meplanes=512, planes=1024)
        self.rb_5_4_3 = ResidualBlock(inplanes=1024, meplanes=512, planes=1024)

    def forward(self, x):
        x = self.conv(x)

        x0_0 = self.conv_5_0(x)
        x0_1 = self.rb_5_0(x0_0)

        x1_0 = self.conv_5_1(x0_1)
        x1_1 = self.rb_5_1_0(x1_0)
        x1_2 = self.rb_5_1_1(x1_1)

        x2_0 = self.conv_5_2(x1_2)
        x2_1 = self.rb_5_2_0(x2_0)
        x2_2 = self.rb_5_2_1(x2_1)
        x2_3 = self.rb_5_2_2(x2_2)
        x2_4 = self.rb_5_2_3(x2_3)
        x2_5 = self.rb_5_2_4(x2_4)
        x2_6 = self.rb_5_2_5(x2_5)
        x2_7 = self.rb_5_2_6(x2_6)
        x2_8 = self.rb_5_2_7(x2_7)  # small

        x3_0 = self.conv_5_3(x2_8)
        x3_1 = self.rb_5_3_0(x3_0)
        x3_2 = self.rb_5_3_1(x3_1)
        x3_3 = self.rb_5_3_2(x3_2)
        x3_4 = self.rb_5_3_3(x3_3)
        x3_5 = self.rb_5_3_4(x3_4)
        x3_6 = self.rb_5_3_5(x3_5)
        x3_7 = self.rb_5_3_6(x3_6)
        x3_8 = self.rb_5_3_7(x3_7)  # medium

        x4_0 = self.conv_5_4(x3_8)
        x4_1 = self.rb_5_4_0(x4_0)
        x4_2 = self.rb_5_4_1(x4_1)
        x4_3 = self.rb_5_4_2(x4_2)
        x4_4 = self.rb_5_4_3(x4_3)  # large

        return x2_8, x3_8, x4_4
