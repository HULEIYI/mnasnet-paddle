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

if __name__ == "__main__":
    device = paddle.device.get_device()

    # paddlepaddle;
    model_paddle = MnasNetAllPaddle.mnasnetb1_0(pretrained=True)
    model_paddle.to(device=device)
    model_paddle.eval()

    # pytorch;
    model_torch = MnasNetTorch.mnasnet1_0(pretrained=True)
    model_torch.cuda()
    model_torch.eval()

    # fake tensor；
    np.random.seed(322)
    inp_tensor = np.random.random(size=(1, 3, 256, 256)).astype('float32')
    inp_paddle = paddle.to_tensor(inp_tensor).cuda()
    inp_torch = torch.tensor(inp_tensor).cuda()
    print(inp_paddle.dtype)
    print(inp_torch.dtype)

    print("mnasnetb1_0 output of paddlepaddle\n", model_paddle(inp_paddle))
    print("mnasnetb1_0 output of pytorch\n", model_torch(inp_torch))
