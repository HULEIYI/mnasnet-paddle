# """
# =================================================
# @Project -> File    ：AIStudio -> TestLRSchedule.py 
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/24 16:31 
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

import paddle
import MnasNetAllPaddle

import matplotlib.pyplot as plt

import time

NUM_CARDS = 1
BASE_LR = 0.008 * NUM_CARDS
DECAY_GAMMA = 0.97
DECAY_EPOCHS = 2.4
LABEL_SMOOTH = 0.1
TRAINING_STEPS = int(3503192 / NUM_CARDS)

OPT_DECAY = 0.9
MOMENTUM = 0.9
OPT_EPSILON = 0.001

DEVICES = paddle.device.get_device()

base_lr = BASE_LR

# MnasNetA;
model = MnasNetAllPaddle.mnasneta1_0()
model.to(device=DEVICES)  # TODO:记得将模型放置在gpu上；

lr_schedule = paddle.optimizer.lr.ExponentialDecay(learning_rate=base_lr, gamma=DECAY_GAMMA, last_epoch=-1,
                                                   verbose=True)  # TODO:在使用时计算2.4epochs更新一次；
warmup_schedule = paddle.optimizer.lr.LinearWarmup(lr_schedule, 5 * 10010, 0, BASE_LR, last_epoch=-1,
                                                   verbose=True)
opt = paddle.optimizer.RMSProp(learning_rate=warmup_schedule, rho=OPT_DECAY, momentum=MOMENTUM, epsilon=OPT_EPSILON,
                               parameters=model.parameters())

cur_steps = 0

lr = []

OPT_PATH = "./Log/OPT.pdparam"
try:
    opt_loaded = paddle.load(OPT_PATH)
    print(warmup_schedule.state_dict())
    print(opt.state_dict())
    opt.set_state_dict(opt_loaded)
    print(opt_loaded)
    print(opt_loaded.keys())
    print(opt.state_dict().keys())

    print(warmup_schedule.state_dict())
    print(opt.state_dict())
except:
    pass

while cur_steps < TRAINING_STEPS:
    for i in range(10010):
        cur_steps += 1
        if cur_steps < 5 * 10010:
            warmup_schedule.step()
        elif (cur_steps - 5 * 10010) % int(2.4 * 10010) == 0:
            warmup_schedule.step()

        time.sleep(2)

        lr.append(opt.get_lr())

        paddle.save(opt.state_dict(), OPT_PATH)

plt.plot(range(len(lr)), lr)
plt.show()
