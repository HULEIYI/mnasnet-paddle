# """
# =================================================
# @Project -> File    ：AIStudio -> MnasNetPaddle.py
# @IDE                ：PyCharm
# @Author             ：IsHuuAh
# @Date               ：2021/8/19 22:53
# @email              ：18019050827@163.com
# ==================================================
# """
# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
import warnings

import paddle
from paddle import Tensor
import paddle.nn as nn
from typing import Any, Dict, List
import paddle.nn.functional as F

import torch
from UtilsforPaddlePaddle import weightTransfromTorch2PaddleDict

__all__ = ['MNASNet', 'mnasnetb0_5', 'mnasnetb0_75', 'mnasnetb1_0', 'mnasnetb1_3']

# 定义常量；
_MODEL_URLS = {
    "mnasnetb0_5":
        "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnetb0_75": None,
    "mnasnetb1_0":
        "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mnasnetb1_3": None,
    "mnasneta0_5": None,
    "mnasneta0_75": None,
    "mnasneta1_0": None,
    "mnasneta1_3": None,
}

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 0.99  # 论文是0.99；
# _BN_MOMENTUM = 1 - (1 - 0.9997)  # TODO:要确定paddle是否也是！tf和torch应该是一致的；
_BN_WEIGHT_DECAY = 1e-5  # 其实不用设置，默认是1e-5;
_BN_EPSILON = 1e-3

_SE_FACTOR = 4  # 控制SE模块的;


# 使用torch的权重初始化paddle；
def getTorchWeightfromURL(arch, progress):
    state_dict = torch.hub.load_state_dict_from_url(url=_MODEL_URLS[arch], progress=progress)
    return weightTransfromTorch2PaddleDict(state_dict)


class Identity(nn.Layer):
    def __init__(self, in_ch: int):
        super().__init__()

    def forward(self, x):
        return x


class SqueezeExcitation(nn.Layer):  # TODO:check the implement of SE；
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int,
                 squeeze_factor: int = 4):  # TODO:squeeze_factor = 4 or 16? Answer：paper tf源码使用的4；
        super().__init__()
        squeeze_channels = _round_to_multiple_of(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2D(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_channels=squeeze_channels, out_channels=input_channels, kernel_size=1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.sigmoid(scale)  # TODO:是sigmoid还是hardsigmoid；

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class _InvertedResidual(nn.Layer):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int,
            stride: int,
            expansion_factor: int,
            bn_momentum: float = 0.1,
            se: int = 0
    ) -> None:
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        assert se in [0, 1]

        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        if se == 0:
            self.layers = nn.Sequential(
                # Pointwise
                nn.Conv2D(in_channels=in_ch, out_channels=mid_ch, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(num_features=mid_ch, momentum=bn_momentum, epsilon=_BN_EPSILON),
                nn.ReLU(),
                # Depthwise
                nn.Conv2D(in_channels=mid_ch, out_channels=mid_ch, kernel_size=kernel_size, padding=kernel_size // 2,
                          stride=stride, groups=mid_ch, bias_attr=False),
                nn.BatchNorm2D(num_features=mid_ch, momentum=bn_momentum, epsilon=_BN_EPSILON),
                nn.ReLU(),
                # Linear pointwise. Note that there's no activation.
                nn.Conv2D(in_channels=mid_ch, out_channels=out_ch, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(num_features=out_ch, momentum=bn_momentum, epsilon=_BN_EPSILON))
        else:  # mnasnetA;
            self.layers = nn.Sequential(
                # Pointwise
                nn.Conv2D(in_channels=in_ch, out_channels=mid_ch, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(num_features=mid_ch, momentum=bn_momentum, epsilon=_BN_EPSILON),
                nn.ReLU(),
                # Depthwise
                nn.Conv2D(in_channels=mid_ch, out_channels=mid_ch, kernel_size=kernel_size, padding=kernel_size // 2,
                          stride=stride, groups=mid_ch, bias_attr=False),
                nn.BatchNorm2D(num_features=mid_ch, momentum=bn_momentum, epsilon=_BN_EPSILON),
                nn.ReLU(),
                SqueezeExcitation(input_channels=mid_ch, squeeze_factor=_SE_FACTOR),  # TODO:SE;这里太灵活，可优化；
                # Linear pointwise. Note that there's no activation.
                nn.Conv2D(in_channels=mid_ch, out_channels=out_ch, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(num_features=out_ch, momentum=bn_momentum, epsilon=_BN_EPSILON))

    def forward(self, input: Tensor) -> Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch: int, out_ch: int, kernel_size: int, stride: int, exp_factor: int, repeats: int,
           bn_momentum: float, se: int) -> nn.Sequential:
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride,
                              expansion_factor=exp_factor, bn_momentum=bn_momentum, se=se)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=1,
                              expansion_factor=exp_factor, bn_momentum=bn_momentum, se=se))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) -> List[int]:
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


# TODO:由于paddle没有fan_out模式，故有以下计算函数，另，paddle只支持relu损失函数的Kaiming初始方法；
def _calculate_fan_in_out(inp_chnl=None, oup_chnl=None, kernel_size=None):
    if inp_chnl == None and oup_chnl == None and kernel_size == None:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    else:
        if kernel_size == None:  # Linear；
            fan_in = inp_chnl
            fan_out = oup_chnl
        else:
            receptive_field_size = kernel_size[0] * kernel_size[1]
            num_input_fmaps = inp_chnl
            num_output_fmaps = oup_chnl
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


# TODO:MnasNet-B1；
class MNASNet(nn.Layer):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(
            self,
            alpha: float,
            num_classes: int = 1000,
            dropout: float = 0.2,
            ver: str = 'a'
    ) -> None:
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        assert ver in ['a', 'b']
        self.alpha = alpha
        self.num_classes = num_classes

        ver_mul = 0
        if ver == 'a':
            ver_mul = 1  # 控制a和b；

        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2D(in_channels=3, out_channels=depths[0], kernel_size=3, padding=1, stride=2, bias_attr=False),
            nn.BatchNorm2D(num_features=depths[0], momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON),
            nn.ReLU(),
            # Depthwise separable, no skip.
            nn.Conv2D(in_channels=depths[0], out_channels=depths[0], kernel_size=3, padding=1, stride=1,
                      groups=depths[0], bias_attr=False),
            nn.BatchNorm2D(num_features=depths[0], momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON),
            nn.ReLU(),
            nn.Conv2D(in_channels=depths[0], out_channels=depths[1], kernel_size=1, padding=0, stride=1,
                      bias_attr=False),
            nn.BatchNorm2D(num_features=depths[1], momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM, se=0 * ver_mul),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM, se=1 * ver_mul),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM, se=0 * ver_mul),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM, se=1 * ver_mul),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM, se=1 * ver_mul),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM, se=0 * ver_mul),
            # Final mapping to classifier input.
            nn.Conv2D(in_channels=depths[7], out_channels=1280, kernel_size=1, padding=0, stride=1, bias_attr=False),
            nn.BatchNorm2D(num_features=1280, momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON),
            nn.ReLU(),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout),  # TODO:p是否进行1-p；此处，paddle和torch的策略一致，均是在训练时增大输出结果；
                                        nn.Linear(in_features=1280, out_features=num_classes))
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                fan_in, fan_out = _calculate_fan_in_out(m._in_channels, m._out_channels, m._kernel_size)
                m.weight_attr = paddle.ParamAttr(nn.initializer.KaimingNormal(fan_in=fan_out))  # TODO:如何设置fan_out?
                # nn.init.kaiming_normal_(m.weight, mode="fan_out",
                #                         nonlinearity="relu")
                if m.bias is not None:
                    m.bias_attr = paddle.ParamAttr(nn.initializer.Constant(0))
                    # nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight_attr = paddle.ParamAttr(nn.initializer.Constant(1))
                # nn.init.ones_(m.weight)
                m.bias_attr = paddle.ParamAttr(nn.initializer.Constant(0))
                # nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_in, fan_out = _calculate_fan_in_out(m.weight.shape[0], m.weight.shape[1])
                m.weight_attr = paddle.ParamAttr(nn.initializer.KaimingUniform(fan_in=fan_out))
                # nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                #                          nonlinearity="sigmoid")
                m.bias_attr = paddle.ParamAttr(nn.initializer.Constant(0))
                # nn.init.zeros_(m.bias)


def _load_pretrained(model_name: str, model: nn.Layer, progress: bool) -> None:
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError(
            "No checkpoint is available for model type {}".format(model_name))
    # checkpoint_url = _MODEL_URLS[model_name]
    state_dict = getTorchWeightfromURL(arch=model_name, progress=progress)
    model.set_state_dict(state_dict)

    # model.load_state_dict(
    #     load_state_dict_from_url(checkpoint_url, progress=progress))


def mnasnetb0_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, ver='b', **kwargs)
    if pretrained:
        _load_pretrained("mnasnetb0_5", model, progress)
    return model


def mnasnetb0_75(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, ver='b', **kwargs)
    if pretrained:
        _load_pretrained("mnasnetb0_75", model, progress)
    return model


def mnasnetb1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, ver='b', **kwargs)
    if pretrained:
        _load_pretrained("mnasnetb1_0", model, progress)
    return model


def mnasnetb1_3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, ver='b', **kwargs)
    if pretrained:
        _load_pretrained("mnasnetb1_3", model, progress)
    return model


def mnasneta0_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, ver='a', **kwargs)
    if pretrained:
        _load_pretrained("mnasneta0_5", model, progress)
    return model


def mnasneta0_75(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, ver='a', **kwargs)
    if pretrained:
        _load_pretrained("mnasneta0_75", model, progress)
    return model


def mnasneta1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, ver='a', **kwargs)
    if pretrained:
        _load_pretrained("mnasneta1_0", model, progress)
    return model


def mnasneta1_3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, ver='a', **kwargs)
    if pretrained:
        _load_pretrained("mnasneta1_3", model, progress)
    return model
