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
from pathlib import Path

import numpy as np
import torch

from model.module.conv import BasicConv2d


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in "weights"

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == "darknet53.conv.74":
        cutoff = 75
    elif file == "yolov3-tiny.conv.15":
        cutoff = 15

    # Read weights file
    with open(weights, "rb") as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # (int32) version info: major, minor, revision
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    count = 0
    ptr = 0
    for m in self.modules():
        if isinstance(m, BasicConv2d):
            # print(m._BasicConv2dconv)
            # print(m.named_parameters())
            # only initing backbone conv's weights
            if count == cutoff:
                break
            count += 1

            conv_layer = m.conv
            if m.batch_norm:
                # Load BN bias, weights, running mean and running variance
                bn_layer = m.batch_norm
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b

            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

# TODO:
# def save_weights(self, path="model.weights", cutoff=-1):
#     # Converts a PyTorch model to Darket format (*.pt to *.weights)
#     # Note: Does not work if model.fuse() is applied
#     with open(path, "wb") as f:
#
#         # Iterate through layers
#         for i, (mdef, module) in enumerate(
#                 zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
#             if mdef["type"] == "convolutional":
#                 conv_layer = module[0]
#                 # If batch norm, load bn first
#                 if mdef["batch_normalize"]:
#                     bn_layer = module[1]
#                     bn_layer.bias.data.cpu().numpy().tofile(f)
#                     bn_layer.weight.data.cpu().numpy().tofile(f)
#                     bn_layer.running_mean.data.cpu().numpy().tofile(f)
#                     bn_layer.running_var.data.cpu().numpy().tofile(f)
#                 # Load conv bias
#                 else:
#                     conv_layer.bias.data.cpu().numpy().tofile(f)
#                 # Load conv weights
#                 conv_layer.weight.data.cpu().numpy().tofile(f)
