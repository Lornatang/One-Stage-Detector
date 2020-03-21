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
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import yolo.configs.voc
from . import LabelSmooth
from . import Mixup
from . import RandomAffine
from . import RandomCrop
from . import RandomHorizontalFilp
from . import Resize
from . import iou_xywh_numpy


class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        self.img_size = img_size  # For Multi-training
        self.classes = yolo.configs.voc.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(
            self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(
            bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __load_annotations(self, anno_type):

        assert anno_type in ['train',
                             'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(yolo.configs.voc.PROJECT_PATH, 'data',
                                 anno_type + "_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)

        return annotations

    def __parse_annotation(self, annotation):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array(
            [list(map(float, box.split(','))) for box in anno[1:]])

        img, bboxes = RandomHorizontalFilp()(np.copy(img),
                                             np.copy(bboxes))
        img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = Resize((self.img_size, self.img_size), True)(
            np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.
        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.
        """

        anchors = np.array(yolo.configs.voc.MODEL["ANCHORS"])
        strides = np.array(yolo.configs.voc.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = yolo.configs.voc.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]),
                           anchors_per_scale, 6 + self.num_classes))
                 for i in range(3)]
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in
                       range(3)]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:,
                                                                np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(
                    bbox_xywh_scaled[i, 0:2]).astype(
                    np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = iou_xywh_numpy(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


def parse_voc_annotation(data_path, file_type, anno_path,
                         use_difficult_bbox=False):
    """
    解析 pascal voc数据集的annotation, 表示的形式为[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: 数据集的路径 , 如 D:\doc\data\VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: 文件的类型， 'trainval''train''val'
    :param anno_path: 标签存储路径
    :param use_difficult_bbox: 是否适用difficult==1的bbox
    :return: 数据集大小
    """
    classes = yolo.configs.voc.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main',
                                 file_type + '.txt')
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'JPEGImages',
                                      image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations',
                                      image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (
                        int(difficult) == 1):  # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find('bndbox')
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join(
                    [xmin, ymin, xmax, ymax, str(class_id)])
            annotation += '\n'
            # print(annotation)
            f.write(annotation)
    return len(image_ids)
