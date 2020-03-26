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
import math
from copy import deepcopy

import torch
import torch.nn as nn


class CosineDecayLR(object):
    def __init__(self, optimizer, max_batches, lr, warmup):
        """ Cosine decay scheduler about all batches training.

        Args:
            optimizer (torch.optim): Stochastic gradient descent (optionally with momentum).
            max_batches (int): The maximum number of steps in the training process.
            lr (float): Learning rate.
            warmup (int): In the training begin, the lr is smoothly increase from 0 to lr_init,
                which means "warmup", this means warmup steps, if 0 that means don't use lr warmup.

        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = CosineDecayLR(optimizer, 10000, 0.1, 0.0001, 400)
            >>> for epoch in range(50):
            >>>     for iters in range(200):
            >>>        scheduler.step(200 * epoch + iters)
        """
        super(CosineDecayLR, self).__init__()
        self.optimizer = optimizer
        self.max_bacthes = max_batches
        self.lr = lr
        self.lr_end = self.lr * 0.01
        self.warmup = warmup

    def step(self, iters):
        if self.warmup and iters < self.warmup:
            lr = self.lr / self.warmup * iters
        else:
            max_bacthes = self.max_bacthes - self.warmup
            iters = iters - self.warmup
            lr = self.lr_end + 0.5 * (self.lr - self.lr_end) * (
                    1 + math.cos(iters / max_bacthes * math.pi))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9998, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))
