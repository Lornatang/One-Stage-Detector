import argparse
import os
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '..')

from utils import init_seeds
from configs import yolov3_voc
from evaluate import Evaluator
from model.network.yolov3_voc import VOC
from utils import CosineDecayLR
from utils import VocDataset
from utils import select_device
from utils.loss import YoloV3Loss

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


class Trainer(object):
    def __init__(self, weight_path, resume):
        # Initialize
        init_seeds()

        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = yolov3_voc.TRAIN["EPOCHS"]
        self.weight = weight_path
        self.multi_scale_train = yolov3_voc.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = VocDataset(anno_file_type="train",
                                        img_size=yolov3_voc.TRAIN[
                                            "TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=yolov3_voc.TRAIN[
                                               "BATCH_SIZE"],
                                           num_workers=yolov3_voc.TRAIN[
                                               "NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = VOC().to(device)
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(),
                                   lr=yolov3_voc.TRAIN["LR_INIT"],
                                   momentum=yolov3_voc.TRAIN["MOMENTUM"],
                                   weight_decay=yolov3_voc.TRAIN[
                                       "WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(strides=yolov3_voc.MODEL["STRIDES"],
                                    iou_threshold_loss=yolov3_voc.TRAIN[
                                        "IOU_THRESHOLD_LOSS"])
        if args.weight != "":
            self.__load_model_weights(weight_path, resume)

        self.scheduler = CosineDecayLR(self.optimizer,
                                       T_max=self.epochs * len(
                                           self.train_dataloader),
                                       lr_init=yolov3_voc.TRAIN[
                                           "LR_INIT"],
                                       lr_min=yolov3_voc.TRAIN[
                                           "LR_END"],
                                       warmup=yolov3_voc.TRAIN[
                                                  "WARMUP_EPOCHS"] * len(
                                           self.train_dataloader))

    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=device)
            self.yolov3.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.yolov3.load_darknet_weights(weight_path)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0],
                                   "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0],
                                   "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0],
                                           'backup_epoch%g.pt' % epoch))
        del chkpt

    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (
                    imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes,
                    mbboxes,
                    lbboxes) in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader) * epoch + i)

                imgs = imgs.to(device)
                label_sbbox = label_sbbox.to(device)
                label_mbbox = label_mbbox.to(device)
                label_lbbox = label_lbbox.to(device)
                sbboxes = sbboxes.to(device)
                mbboxes = mbboxes.to(device)
                lbboxes = lbboxes.to(device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d,
                                                                      label_sbbox,
                                                                      label_mbbox,
                                                                      label_lbbox,
                                                                      sbboxes,
                                                                      mbboxes,
                                                                      lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor(
                    [loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:
                    s = (
                            'Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                            'lr: %g') % (epoch, self.epochs - 1, i,
                                         len(self.train_dataloader) - 1,
                                         mloss[0], mloss[1], mloss[2], mloss[3],
                                         self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(
                        range(10, 20)) * 32
                    print("multi_scale_img_size : {}".format(
                        self.train_dataset.img_size))

            mAP = 0
            if epoch >= 20:
                print('*' * 20 + "Validate" + '*' * 20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g' % (mAP))

            self.__save_model_weights(epoch, mAP)
            print('best mAP : %g' % self.best_mAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str,
                        default='',
                        help='weight file path')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training flag')
    parser.add_argument("--device", default="",
                        help="device id (i.e. 0 or 0,1 or cpu)")
    args = parser.parse_args()

    device = select_device(args.device, apex=mixed_precision)

    Trainer(weight_path=args.weight, resume=args.resume).train()
