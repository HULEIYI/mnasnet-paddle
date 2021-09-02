# """
# =================================================
# @Project -> File    ：AIStudio -> TrainCodeforImageNet.py
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/20 18:16
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

"""
    尝试以脚本格式；
"""

# import pkgs;
import paddle
import paddle.nn as nn
from paddle.vision import datasets, transforms
from paddle.io import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import random

import os
import argparse  # TODO:以脚本格式获取参量；
from pathlib import Path

import time

# HyperParams;
# path;
IMAGENET_PATH = "./ImageNet/"
CKP_PATH = "./Log/ckp.pdparam"
MODEL_PATH = "./Log/mnasneta1_0_ImageNet.pdparam"  # 实际上和CKP_PATH相同；

# random parameters;
SEED = 42

# devices;
DEVICES = paddle.device.get_device()
NUM_CARDS = 0.5

# preprocessing parameters;
BATCH_SIZE = int(128 * NUM_CARDS)  # paper里是1024在8张TPU上；
WOKERS = 4
RESIZE_IMG = 256
IMG_SIZE = 224  # TODO:是先resize至256后crop至224；
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# optimizer parameters;
OPT_DECAY = 0.9
MOMENTUM = 0.9
OPT_EPSILON = 0.001

# training parameters;
# TODO:这些设置均在单卡上；
BASE_LR = 0.008 * NUM_CARDS  # 0.256/32;
DECAY_GAMMA = 0.97
DECAY_EPOCHS = 2.4
LABEL_SMOOTH = 0.1
TRAINING_STEPS = int(3503192 / NUM_CARDS)

NUM_CLASSES = 1000

# screen parameters;
PRINT_FREQ = int(78 / NUM_CARDS)


# set the random seed;
def setRandomSeed(seed: int = 42):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# a fundamental class for statistics;
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


# hyperparam class；
# class HyperParams():
#     pass
#
#
# _argparse = argparse.ArgumentParser(prog='TrainMnasNetA', description='the trainning code for MnasNetA on ImageNet')

# model class;
import MnasNetAllPaddle


# read ImageNet;


def getImageNet(root: str, train: bool = True, transform: transforms.Compose = None, bs: int = BATCH_SIZE,
                workers: int = WOKERS):
    if train:
        path = os.path.join(root, 'ILSVRC2012_img_val')
        train_set = datasets.DatasetFolder(root=path, transform=transform)
        return DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False, num_workers=workers, )
    else:
        path = os.path.join(root, 'ILSVRC2012_img_val')
        val_set = datasets.DatasetFolder(root=path, transform=transform)
        return DataLoader(val_set, batch_size=bs, shuffle=False, drop_last=False, num_workers=workers, )


# training func;
def trainModel(model, opt, lr_schedule, criterion, cur_steps, train_loader, train_log):
    """
        one epoch;
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set train model;
    model.train()  # TODO:很重要！！！

    # record the time;
    time_point = time.time()

    # start training for one epoch;
    for i, (inp, oup) in enumerate(train_loader):
        # TODO:如果中途训练中断，从断点处开始训练；
        if i < cur_steps % len(train_loader):
            continue

        # update current steps;
        cur_steps += 1  # TODO：

        # measure data loading time
        data_time.update(time.time() - time_point)

        # set the tensor to gpu;
        inp = inp.cuda()
        oup = oup.cuda()
        oup_smooth = paddle.nn.functional.one_hot(oup, NUM_CLASSES)
        oup_smooth = paddle.nn.functional.label_smooth(oup_smooth)  # TODO:标签软化；

        # compute output;
        model_oup = model(inp)
        loss = criterion(model_oup, oup_smooth)

        # compute gradient and do opt step;
        opt.clear_grad()
        loss.backward()
        opt.step()

        model_oup = model_oup.cast('float32')
        loss = loss.cast('float32')

        # measure accuracy and record loss;
        oup = oup.reshape([-1, 1])
        prec1 = paddle.metric.accuracy(model_oup, oup.cast('int64'), k=1)
        prec5 = paddle.metric.accuracy(model_oup, oup.cast('int64'), k=5)
        losses.update(loss.item(), inp.shape[0])
        top1.update(prec1.item(), inp.shape[0])
        top5.update(prec5.item(), inp.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - time_point)
        time_point = time.time()

        if cur_steps < 5 * len(train_loader):  # 有5个epoch的warmup；
            lr_schedule.step()
        elif cur_steps % int(2.4 * len(train_loader)) == 0:  # 每2.4epoch更新一次lr；
            lr_schedule.step()

        if cur_steps % PRINT_FREQ == 0:
            print("current lr:\t", opt.get_lr(), "\n")
            print('Steps: [{steps}/{all_steps}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                steps=cur_steps, all_steps=TRAINING_STEPS, batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    # return;
    return cur_steps


# eval func;
def evalModel(model, criterion, val_loader, val_log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()  # TODO:十分重要！！！

    time_point = time.time()
    with paddle.no_grad():
        for i, (inp, oup) in enumerate(val_loader):
            # measure data loading time;
            data_time.update(time.time() - time_point)

            inp = inp.cuda()
            oup = oup.cuda()

            # compute output;
            model_oup = model(inp)
            loss = criterion(model_oup, oup.cast('int64'))

            model_oup = model_oup.cast('float32')
            loss = loss.cast('float32')

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target)
            oup = oup.reshape([-1, 1])
            prec1 = paddle.metric.accuracy(model_oup, oup.cast('int64'), k=1)
            prec5 = paddle.metric.accuracy(model_oup, oup.cast('int64'), k=5)
            losses.update(loss.item(), inp.shape[0])
            top1.update(prec1.item(), inp.shape[0])
            top5.update(prec5.item(), inp.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - time_point)
            time_point = time.time()

            if i % PRINT_FREQ == 0:
                print('Steps: [{steps}/{all_steps}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    steps=i, all_steps=len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}\t'' * Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


# save func;
def saveCkp(model, opt, cur_steps, best_prec1, best_prec5, ):
    ckp_dict = {}
    ckp_dict['cur_lr'] = opt.get_lr()
    ckp_dict['opt'] = opt.state_dict()
    ckp_dict['cur_steps'] = cur_steps  # 存储目前的steps/iterations;
    ckp_dict['ckp_model'] = model.state_dict()
    ckp_dict['best_prec1'] = best_prec1
    ckp_dict['best_prec5'] = best_prec5

    paddle.save(ckp_dict, CKP_PATH)


if __name__ == '__main__':
    # fix seed;
    setRandomSeed(SEED)

    # define global params;
    cur_steps = 0

    best_prec1 = 0
    best_prec5 = 0

    train_log = {"loss": [], "accuracy": []}  # record the log;
    val_log = {"loss": [], "accuracy": []}

    # get the dataset;
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    transform_train = transforms.Compose([transforms.Resize(RESIZE_IMG),
                                          transforms.RandomCrop(IMG_SIZE),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize, ])
    transform_val = transforms.Compose([transforms.Resize(RESIZE_IMG),
                                        transforms.CenterCrop(IMG_SIZE),
                                        transforms.ToTensor(),
                                        normalize, ])
    # TODO:训练和验证的增广不一致；
    train_loader = getImageNet(IMAGENET_PATH, True, transform_train, BATCH_SIZE, WOKERS)
    val_loader = getImageNet(IMAGENET_PATH, False, transform_val, BATCH_SIZE, WOKERS)

    # set the model;
    # MnasNetA;
    model = MnasNetAllPaddle.mnasneta1_0(num_classes=1000)
    model.to(device=DEVICES)  # TODO:记得将模型放置在gpu上；

    # set the training environment;
    # TODO:last_epoch要好好设置；
    base_lr = BASE_LR
    ckp_loaded = None

    ckp_path = Path(CKP_PATH)
    if ckp_path.is_file():
        print("=> loading checkpoint from '{}'".format(CKP_PATH))

        ckp_loaded = paddle.load(CKP_PATH)
        base_lr = ckp_loaded['cur_lr']
        cur_steps = ckp_loaded['cur_steps']
        best_prec1 = ckp_loaded['best_prec1']
        best_prec5 = ckp_loaded['best_prec5']

        print("=> loaded checkpoint at step {}".format(ckp_loaded['cur_steps']))
    else:
        print("=> no checkpoint found at '{}'".format(CKP_PATH))

    if cur_steps < 5 * len(train_loader):
        base_lr = BASE_LR

    # TODO:warm up !!!
    lr_schedule = paddle.optimizer.lr.ExponentialDecay(learning_rate=base_lr, gamma=DECAY_GAMMA, last_epoch=-1,
                                                       verbose=False)  # TODO:在使用时计算2.4epochs更新一次；
    warmup_schedule = paddle.optimizer.lr.LinearWarmup(lr_schedule, 5 * len(train_loader), 0, BASE_LR, last_epoch=-1,
                                                       verbose=False)
    opt = paddle.optimizer.RMSProp(learning_rate=warmup_schedule, rho=OPT_DECAY, momentum=MOMENTUM, epsilon=OPT_EPSILON,
                                   parameters=model.parameters())
    criterion_train = nn.CrossEntropyLoss(soft_label=True)  # TODO:标签要进行软化；
    criterion_val = nn.CrossEntropyLoss()

    # TODO:加载模型参数；
    if ckp_loaded is not None:
        model.set_state_dict(ckp_loaded['ckp_model'])

        print(warmup_schedule.state_dict())
        opt.set_state_dict(ckp_loaded['opt'])  # TODO:只用加载opt的权重，LR_Scheduler会自动加载！！！

        print(warmup_schedule.state_dict())
        print(opt.state_dict()['LR_Scheduler'])
        print(ckp_loaded['opt']['LR_Scheduler'])

        print('=> resume the model and optimizer')
    else:
        print('=> got the random initial model and optimizer')

    # start training；
    while cur_steps < TRAINING_STEPS:
        print("TRAIN:\n")
        cur_steps = trainModel(model, opt, warmup_schedule, criterion_train, cur_steps, train_loader, train_log)
        if cur_steps % len(train_loader) == 0:
            print("TEST:\n")
            prec1, prec5 = evalModel(model, criterion_val, val_loader, val_log)
            if prec1 > best_prec1:
                best_prec1 = prec1
                best_prec5 = prec5
                paddle.save(model.state_dict(), MODEL_PATH)  # 存储最优epoch；

            saveCkp(model, opt, cur_steps, best_prec1, best_prec5)  # 每个epoch后都存储一次ckp；

        print("BEST:\n", "Top1:\t", best_prec1, "Top5:\t", best_prec5)

    # save the params and model;（finally save）
    # 存储一次ckp；
    saveCkp(model, opt, cur_steps, best_prec1, best_prec5, )

    # 单独存储一次模型；
    # paddle.save(model.state_dict(), MODEL_PATH)  # 以gpu模式存储的；# TODO:存储不得，可能最后并非最好的；
