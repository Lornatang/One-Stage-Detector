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
import multiprocessing as mp
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

from easydet.config import get_cfg
from evaluate import Evaluator
from model.network import YOLOv3
from model.network.yolov3_tiny import YOLOv3Tiny
from utils import load_darknet_weights
from utils import select_device
from utils.visualize import visualize_boxes


def setup_cfg(args):
    # Load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Set task for builtin models
    cfg.TEST.TASK = args.task
    # Set confidence_threshold for builtin models
    cfg.TEST.CONFIDENCE_THRESHOLD = args.confidence_threshold
    # Set multi_scale for builtin models
    cfg.TEST.MULTI_SCALE = args.multi_scale
    # Set flip for builtin models
    cfg.TEST.FLIP = args.flip
    # Set workers for builtin models
    cfg.TEST.GPUS = torch.cuda.device_count()
    cfg.TEST.WORKERS = cfg.TEST.GPUS * 4
    # Set weights for builtin models
    if args.weights:
        cfg.TEST.WEIGHTS = args.weights
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
        "--task",
        type=str,
        default="eval",
        help="`eval`, `visual`, `study`, `benchmark`")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.001,
        help="Minimum score for instance predictions to be shown. (default: 0.5).",
    )
    parser.add_argument(
        '--multi-scale',
        action='store_true',
        help='Adjust (67% - 150%) image size for test.'
    )
    parser.add_argument(
        '--flip',
        action='store_true',
        help='Randomly flip the test image.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='./weights/yolov3_voc_50200.pth',
        help='path to weights file. (default: ``).'
    )
    parser.add_argument(
        "--device",
        default="",
        help="device id (default: ``)"
    )
    return parser


def evaluate(cfg, args):
    device = select_device(args.device)
    # Initialize/load model
    if cfg.MODEL.META_ARCHITECTURE:

        # Initialize model
        model = YOLOv3(cfg).to(device)

        # Load weights
        if cfg.TEST.WEIGHTS.endswith(".pth"):
            state = torch.load(cfg.TEST.WEIGHTS, map_location=device)
            model.load_state_dict(state["state_dict"])
        else:
            load_darknet_weights(model, cfg.TEST.WEIGHTS)

        if device.type != "cpu" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        warnings.warn("WARNNING: Backbone network cannot be empty! "
                      f"Default load Darknet53 meta architecture for `{cfg.CONFIG_FILE}`!")
        model = YOLOv3(cfg).to(device)

    if cfg.TEST.TASK == "visual":
        images = os.listdir(os.path.join(os.getcwd(), "data", "test"))
        for filename in images:
            path = os.path.join(os.path.join(os.getcwd(), "data", "test", filename))

            images = cv2.imread(path)
            assert images is not None

            bboxes_prd = Evaluator(model, cfg=cfg).get_bbox(images)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image=images, boxes=boxes, labels=class_inds, probs=scores,
                                class_labels=cfg.CLASSES)
                path = os.path.join(f"./outputs/{filename}")

                cv2.imwrite(path, images)

    elif cfg.TEST.TASK == "eval":
        maps = 0.
        with torch.no_grad():
            aps = Evaluator(model, visiual=True, cfg=cfg).calculate_aps(cfg.TEST.MULTI_SCALE,
                                                                        cfg.TEST.FLIP)

            for i in aps:
                print(f"{i:25s} --> mAP : {aps[i]:.4f}")
                maps += aps[i]
            maps = maps / len(cfg.CLASSES)
            print(f'mAP:{maps:.6f}')

        return maps


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    print(args)

    evaluate(cfg, args)
