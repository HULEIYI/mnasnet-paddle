# """
# =================================================
# @Project -> File    ：AIStudio -> TestCodeforImageNet.py 
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/22 17:02 
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

import os
import time

import paddle
from paddle.vision import datasets
from paddle.io import DataLoader
import paddle.vision.transforms as T

import MnasNetAllPaddle

ROOT = './ImageNet/'
BATCH_SIZE = 16
IMG_SIZE = 224


# 定义计数类；
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getImagenet(root, train=True, transform=None):
    if train:
        root = os.path.join(root, 'ILSVRC2012_img_train')
    else:
        root = os.path.join(root, 'ILSVRC2012_img_val')
    return datasets.DatasetFolder(root=root,
                                  transform=transform)


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()  # TODO:十分重要！！！

    end = time.time()
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target.cast('int64'))

            output = output.cast('float32')
            loss = loss.cast('float32')

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target)
            target = target.reshape([-1, 1])
            prec1 = paddle.metric.accuracy(output, target.cast('int64'), k=1)
            prec5 = paddle.metric.accuracy(output, target.cast('int64'), k=5)
            losses.update(loss.item(), input.shape[0])
            top1.update(prec1.item(), input.shape[0])
            top5.update(prec5.item(), input.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                        loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}\t'' * Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    device = paddle.device.get_device()

    # 获取测试数据集；
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize, ])
    ImageNet_val = getImagenet(ROOT, False, transform)
    val_loader = DataLoader(ImageNet_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0,
                            use_shared_memory=False)
    # TODO:drop_last?

    print(ImageNet_val[0])

    # 设置损失函数；
    criterion = paddle.nn.CrossEntropyLoss()

    # 定义模型；
    # paddlepaddle;
    model_paddle = MnasNetAllPaddle.mnasnetb1_0(pretrained=True)
    model_paddle.to(device=device)

    # 测试集；
    validate(val_loader, model_paddle, criterion)
