# """
# =================================================
# @Project -> File    ：AIStudio -> UtilsforPaddlePaddle.py 
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/12 17:39 
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

import torch
import paddle
import numpy as np


def weightTransfromTorch2PaddleFile(weight_path):
    inp_weight = weight_path
    oup_weight = "PaddleWeight.pdparams"

    torch_dict = torch.load(inp_weight)['model']
    paddle_dict = {}

    fc_names = ["classifier.1.weight", "classifier.4.weight", "classifier.6.weight"]
    for k in torch_dict:
        weight = torch_dict[k].cpu().numpy()
        flag = [i in k for i in fc_names]

        if any(flag):
            print("weight {} need to be trans".format(k))
            weight = weight.transpose()
        paddle_dict[k] = weight

    paddle.save(paddle_dict, oup_weight)


def weightTransfromTorch2PaddleDict(torch_dict):
    paddle_dict = {}

    for k in torch_dict:
        k_split = k.split('.')
        weight = torch_dict[k].cpu().detach().numpy()

        if k_split[0] == "fc" or k_split[0] == "classifier":
            print("weight {} need to be trans".format(k))
            weight = weight.transpose()
            paddle_dict[k] = weight
        elif k_split[-1] == "running_mean":
            paddle_dict[k.replace("running_mean", "_mean")] = weight
        elif k_split[-1] == "running_var":
            paddle_dict[k.replace("running_var", "_variance")] = weight
        else:
            paddle_dict[k] = weight

    return paddle_dict
