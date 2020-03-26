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
import argparse
import glob
import multiprocessing as mp
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from easydet.config import get_cfg
from model.network.yolov3_tiny import YOLOv3Tiny
from test import evaluate
from utils import CosineDecayLR
from utils import VocDataset
from utils import init_seeds
from utils import select_device
from utils.loss import YoloV3Loss
from utils.process_darknet_weights import load_darknet_weights

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Set bacth_size for builtin models
    cfg.TRAIN.MINI_BATCH_SIZE = args.batch_size
    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.MINI_BATCH_SIZE * 4
    # Set iou_threshold for builtin models
    cfg.TRAIN.IOU_THRESHOLD = args.iou_threshold
    # Set workers for builtin models
    cfg.TRAIN.GPUS = torch.cuda.device_count()
    cfg.TRAIN.WORKERS = cfg.TRAIN.GPUS * 4
    # Set weights for builtin models
    cfg.TRAIN.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Easydet training for built-in models.")
    parser.add_argument(
        "--config-file",
        default="./configs/YOLOV3.yaml",
        metavar="FILE",
        help="path to config file. (default: ./configs/YOLOV3.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size, (default: 16) this is the total "
             "batch size of all GPUs on the current node when"
             "using Data Parallel or Distributed Data Parallel"
             "Effective batch size is batch_size * accumulate."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown. (default: 0.5).",
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='path to weights file. (default: ``).'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume training flag.'
    )
    parser.add_argument(
        "--device",
        default="0",
        help="device id (default: ``0``)"
    )
    return parser


def train(cfg):
    # Initialize
    init_seeds()
    image_size_min = 6.6  # 320 / 32 / 1.5
    image_size_max = 28.5  # 320 / 32 / 28.5
    if cfg.TRAIN.MULTI_SCALE:
        image_size_min = round(cfg.TRAIN.IMAGE_SIZE / 32 / 1.5)
        image_size_max = round(cfg.TRAIN.IMAGE_SIZE / 32 * 1.5)
        image_size = image_size_max * 32  # initiate with maximum multi_scale size
        print(f"Using multi-scale {image_size_min * 32} - {image_size}")

    # Remove previous results
    for files in glob.glob("results.txt"):
        os.remove(files)

    # Initialize model
    model = YOLOv3Tiny(cfg).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.TRAIN.LR,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.DECAY,
                          nesterov=True)

    # Define the loss function calculation formula of the model
    compute_loss = YoloV3Loss(cfg)

    epoch = 0
    start_epoch = 0
    best_maps = 0.0
    context = None

    # Dataset
    # Apply augmentation hyperparameters
    train_dataset = VocDataset(anno_file_type=cfg.TRAIN.DATASET, image_size=cfg.TRAIN.IMAGE_SIZE,
                               cfg=cfg)
    # Dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.MINI_BATCH_SIZE,
                                  num_workers=cfg.TRAIN.WORKERS,
                                  shuffle=cfg.TRAIN.SHUFFLE,
                                  pin_memory=cfg.TRAIN.PIN_MENORY)

    if cfg.TRAIN.WEIGHTS.endswith(".pth"):
        state = torch.load(cfg.TRAIN.WEIGHTS, map_location=device)
        # load model
        try:
            state["state_dict"] = {k: v for k, v in state["state_dict"].items()
                                   if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(state["state_dict"], strict=False)
        except KeyError as e:
            error_msg = f"{cfg.TRAIN.WEIGHTS} is not compatible with {cfg.CONFIG_FILE}. "
            error_msg += f"Specify --weights `` or specify a --config-file "
            error_msg += f"compatible with {cfg.TRAIN.WEIGHTS}. "
            raise KeyError(error_msg) from e

        # load optimizer
        if state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
            best_maps = state["best_maps"]

        # load results
        if state.get("training_results") is not None:
            with open("results.txt", "w") as file:
                file.write(state["training_results"])  # write results.txt

        start_epoch = state["batches"] + 1 // len(train_dataloader)
        del state

    elif len(cfg.TRAIN.WEIGHTS) > 0:
        # possible weights are "*.weights", "yolov3-tiny.conv.15",  "darknet53.conv.74" etc.
        load_darknet_weights(model, cfg.TRAIN.WEIGHTS)
    else:
        print("Pre training model weight not loaded.")

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        # skip print amp info
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # source https://arxiv.org/pdf/1812.01187.pdf
    scheduler = CosineDecayLR(optimizer,
                              max_batches=cfg.TRAIN.MAX_BATCHES,
                              lr=cfg.TRAIN.LR,
                              warmup=cfg.TRAIN.WARMUP_BATCHES)

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend="nccl",  # "distributed backend"
                                # distributed training init method
                                init_method="tcp://127.0.0.1:9999",
                                # number of nodes for distributed training
                                world_size=1,
                                # distributed training node rank
                                rank=0)
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.backbone = model.module.backbone

    # Model EMA
    # TODO: ema = ModelEMA(model, decay=0.9998)

    # Start training
    batches_num = len(train_dataloader)  # number of batches
    # 'loss_GIOU', 'loss_Confidence', 'loss_Classification' 'loss'
    results = (0, 0, 0, 0)
    epochs = cfg.TRAIN.MAX_BATCHES // len(train_dataloader)
    print(f"Using {cfg.TRAIN.WORKERS} dataloader workers.")
    print(f"Starting training {cfg.TRAIN.MAX_BATCHES} batches for {epochs} epochs...")

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()

        # init batches
        batches = 0
        mean_losses = torch.zeros(4)
        print("\n")
        print(("%10s" * 7) % ("Batch", "memory", "GIoU", "conf", "cls", "total", " image_size"))
        progress_bar = tqdm(enumerate(train_dataloader), total=batches_num)
        for index, (images, small_label_bbox, medium_label_bbox, large_label_bbox,
                    small_bbox, medium_bbox, large_bbox) in progress_bar:

            # number integrated batches (since train start)
            batches = index + len(train_dataloader) * epoch

            scheduler.step(batches)

            images = images.to(device)

            small_label_bbox = small_label_bbox.to(device)
            medium_label_bbox = medium_label_bbox.to(device)
            large_label_bbox = large_label_bbox.to(device)

            small_bbox = small_bbox.to(device)
            medium_bbox = medium_bbox.to(device)
            large_bbox = large_bbox.to(device)

            # Hyper parameter Burn-in
            if batches <= cfg.TRAIN.WARMUP_BATCHES:
                for m in model.named_modules():
                    if m[0].endswith('BatchNorm2d'):
                        m[1].track_running_stats = batches == cfg.TRAIN.WARMUP_BATCHES

            # Run model
            pred, raw = model(images)

            # Compute loss
            loss, loss_giou, loss_conf, loss_cls = compute_loss(pred,
                                                                raw,
                                                                small_label_bbox,
                                                                medium_label_bbox,
                                                                large_label_bbox,
                                                                small_bbox,
                                                                medium_bbox,
                                                                large_bbox)

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize accumulated gradient
            if batches % cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.MINI_BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                # TODO: ema.update(model)

            # Print batch results
            # update mean losses
            loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
            mean_losses = (mean_losses * index + loss_items) / (index + 1)
            memory = f"{torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0:.2f}G"
            context = ("%10s" * 2 + "%10.3g" * 5) % (
                "%g/%g" % (batches + 1, cfg.TRAIN.MAX_BATCHES), memory, *mean_losses,
                train_dataset.image_size)
            progress_bar.set_description(context)

            # Multi-Scale training
            if cfg.TRAIN.MULTI_SCALE:
                #  adjust img_size (67% - 150%) every 10 batch size
                if batches % cfg.TRAIN.RESIZE_INTERVAL == 0:
                    train_dataset.image_size = random.randrange(image_size_min,
                                                                image_size_max + 1) * 32

            # Write Tensorboard results
            if tb_writer:
                # 'loss_GIOU', 'loss_Confidence', 'loss_Classification' 'loss'
                titles = ["GIoU", "Confidence", "Classification", "Train loss"]
                for xi, title in zip(list(mean_losses) + list(results), titles):
                    tb_writer.add_scalar(title, xi, index)

        # Process epoch results
        # TODO: ema.update_attr(model)
        final_epoch = epoch + 1 == epochs

        # Calculate mAP
        # skip first epoch
        maps = 0.
        if epoch > 0:
            maps = evaluate(cfg, args)

        # Write epoch results
        with open("results.txt", "a") as f:
            # 'loss_GIOU', 'loss_Confidence', 'loss_Classification' 'loss', 'maps'
            f.write(context + "%10.3g" * 1 % maps)
            f.write("\n")

        # Update best mAP
        if maps > best_maps:
            best_maps = maps

        # Save training results
        with open("results.txt", 'r') as f:
            # Create checkpoint
            state = {'batches': batches,
                     'best_maps': maps,
                     'training_results': f.read(),
                     'state_dict': model.state_dict(),
                     'optimizer': None
                     if final_epoch else optimizer.state_dict()}

        # Save last checkpoint
        torch.save(state, "weights/checkpoint.pth")

        # Save best checkpoint
        if best_maps == maps:
            state = {'batches': -1,
                     'best_maps': None,
                     'training_results': None,
                     'state_dict': model.state_dict(),
                     'optimizer': None}
            torch.save(state, "weights/model_best.pth")

        # Delete checkpoint
        del state

    print(f"{epoch - start_epoch} epochs completed "
          f"in {(time.time() - start_time) / 3600:.3f} hours.\n")
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    args.weights = "weights/checkpoint.pth" if args.resume else args.weights

    print(args)

    device = select_device(args.device, apex=mixed_precision)

    if device.type == "cpu":
        mixed_precision = False

    try:
        os.makedirs("weights")
    except OSError:
        pass

    try:
        # Start Tensorboard with "tensorboard --logdir=runs"
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter()
    except:
        pass

    train(cfg)
