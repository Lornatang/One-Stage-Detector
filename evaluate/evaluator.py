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
import shutil

import cv2
import torch
from tqdm import tqdm

from utils.visualize import *
from utils import *
from .voc_eval import voc_eval


def get_image_from_tensor(image, test_shape):
    image = Resize((test_shape, test_shape), correct_box=False)(image, None).transpose(2, 0, 1)
    return torch.from_numpy(image[np.newaxis, ...]).float()


class Evaluator(object):
    def __init__(self, model, visiual=True, cfg=None):
        self.classes = cfg.CLASSES
        self.pred_result_path = os.path.join(cfg.DATA_ROOT, 'results')
        self.test_dataset_path = os.path.join(cfg.DATA_ROOT, 'VOCtest_06-Nov-2007', 'VOCdevkit',
                                              'VOC2007')
        self.confidence_threshold = cfg.TEST.CONFIDENCE_THRESHOLD
        self.nms_threshold = cfg.TEST.NMS_THRESHOLD
        self.val_shape = cfg.TEST.IMAGE_SIZE

        self.visiual = visiual
        self.visual_image = 0

        self.model = model
        self.device = next(model.parameters()).device

    def calculate_aps(self, multi_test=False, flip_test=False):
        image_index = os.path.join(self.test_dataset_path, 'ImageSets', 'Main', 'test.txt')
        with open(image_index, 'r') as f:
            lines = f.readlines()
            image_indexs = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for index in tqdm(image_indexs):
            image_path = os.path.join(self.test_dataset_path, 'JPEGImages', index + '.jpg')
            image = cv2.imread(image_path)
            bboxes_pred = self.get_bbox(image, multi_test, flip_test)

            if bboxes_pred.shape[0] != 0 and self.visiual and self.visual_image < 100:
                boxes = bboxes_pred[..., :4]
                class_indexs = bboxes_pred[..., 5].astype(np.int32)
                scores = bboxes_pred[..., 4]

                visualize_boxes(image=image, boxes=boxes, labels=class_indexs,
                                probs=scores, class_labels=self.classes)
                path = os.path.join("../", "data/results/{}.jpg".format(self.visual_image))
                cv2.imwrite(path, image)

                self.visual_image += 1

            for bbox in bboxes_pred:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([index, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'det_' + class_name + '.txt'),
                          'a') as f:
                    f.write(s)

        return self.calculate_all_ap()

    def get_bbox(self, image, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(
                    self.predict(image, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.predict(image[:, ::-1], test_input_size,
                                               valid_scale)
                    bboxes_flip[:, [0, 2]] = image.shape[1] - bboxes_flip[:,
                                                              [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.predict(image, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.confidence_threshold, self.nms_threshold)

        return bboxes

    def predict(self, image, test_shape, test_scale):
        raw_image = np.copy(image)
        raw_height, raw_width, _ = raw_image.shape

        image = get_image_from_tensor(image, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(image)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.convert_pred(pred_bbox, test_shape, (raw_height, raw_width), test_scale)

        return bboxes

    def convert_pred(self, pred_bbox, test_input_size, raw_image_shape, test_scale):
        """ The prediction frame is filtered to remove the frame with unreasonable scale

        Notes:
            Step 1: No matter what kind of data enhancement we use in training,
                the transformation here will not change.
            Assuming that we use the data enhancement method r for the input test picture,
                the conversion method of bbox here is the reverse process of method R.
            Step 2: Cut out the part beyond the original image in the predicted bbox.
            Step 3: Set coor of invalid bbox to 0.
            Step 4: Remove bbox not in valid range.
            Step 5: Remove bbox whose score is lower than the score `iou_threshold`.
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # Step 1
        raw_height, raw_width = raw_image_shape
        resize_ratio = min(1.0 * test_input_size / raw_width, 1.0 * test_input_size / raw_height)
        dw = (test_input_size - resize_ratio * raw_width) / 2
        dh = (test_input_size - resize_ratio * raw_height) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # Step 2
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:],
                                               [raw_width - 1, raw_height - 1])],
                                   axis=-1)
        # Step 3
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                     (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # Step 4
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((test_scale[0] < bboxes_scale),
                                    (bboxes_scale < test_scale[1]))

        # Step 5
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.confidence_threshold

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes

    def calculate_all_ap(self, iou_threshold=0.5, use_07_metric=False):
        """ Calculate the AP value for each category

        Args:
            iou_threshold (float): Measure the degree of overlap between the two regions.
                (default:``0.5``)
            use_07_metric (bool): Whether to use VOC07's 11 point AP computation
                (default:``False``)

        Returns:
            A dict.
        """
        filename = os.path.join(self.pred_result_path, 'det_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annotation_path = os.path.join(self.test_dataset_path, 'Annotations', '{:s}.xml')
        images_set = os.path.join(self.test_dataset_path, 'ImageSets', 'Main', 'test.txt')
        aps = {}
        for i, object_classes in enumerate(self.classes):
            recall, precision, ap = voc_eval(filename, annotation_path, images_set, object_classes,
                                             cachedir, iou_threshold, use_07_metric)
            aps[object_classes] = ap

        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return aps
