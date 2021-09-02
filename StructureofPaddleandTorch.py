# """
# =================================================
# @Project -> File    ：AIStudio -> MnasNetPaddleTe.py
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/12 18:07
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

import MnasNetAllPaddle
import MnasNetTorch
import numpy as np

import paddle
import torch
import paddle.nn as nn

if __name__ == "__main__":
    device = paddle.device.get_device()

    # paddlepaddle;
    model_paddle = MnasNetAllPaddle.mnasneta1_0(pretrained=False)
    model_paddle.to(device=device)
    model_paddle.eval()

    # pytorch;
    # model_torch = ResNetTorch.resnet50(pretrained=True)
    # model_torch.cuda()
    # model_torch.eval()

    for m in model_paddle.sublayers():
        # print(m)
        if isinstance(m, nn.Conv2D):
            print(m._in_channels)
            print(m._out_channels)
            print(m._kernel_size)
            # print(m.weight_attr)
            # m.weight_attr = nn.initializer.KaimingNormal()  # TODO:如何设置fan_out?
            # nn.init.kaiming_normal_(m.weight, mode="fan_out",
            #                         nonlinearity="relu")
            if m.bias is not None:
                pass
                # print(m.bias_attr)
                # m.bias_attr = nn.initializer.Constant(0)
                # nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            print(m)
            # print(m.weight_attr)
            # print(m.bias_attr)

            # m.weight_attr = nn.initializer.Constant(1)
            # nn.init.ones_(m.weight)
            # m.bias_attr = nn.initializer.Constant(0)
            # nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            print(m)
            # print(m.weight_attr)
            # print(m.bias_attr)
            # m.weight_attr = nn.initializer.KaimingUniform()
            # nn.init.kaiming_uniform_(m.weight, mode="fan_out",
            #                          nonlinearity="sigmoid")
            # m.bias_attr = nn.initializer.Constant(0)
            # nn.init.zeros_(m.bias)

    # print(model_paddle)
    # print("resnet50 model paddle\n", model_paddle.state_dict().keys(), "\n")
    # print("resnet50 model torch\n", model_torch.state_dict().keys(), "\n")
