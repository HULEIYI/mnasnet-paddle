# """
# =================================================
# @Project -> File    ：AIStudio -> CheckCEandSoOn.py 
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/24 17:21 
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
import paddle
import numpy as np

input_data = paddle.uniform([5, 100], dtype="float64")
label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
weight_data = np.random.random([100]).astype("float64")

input = paddle.to_tensor(input_data)
label = paddle.to_tensor(label_data)
weight = paddle.to_tensor(weight_data)

label = paddle.nn.functional.one_hot(label, 100).cast('double')

ce_loss = paddle.nn.CrossEntropyLoss(soft_label=True, weight=weight, reduction='mean')
output = ce_loss(input, label)
print(output)
