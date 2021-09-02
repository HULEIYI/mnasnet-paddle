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
import argparse  # TODO:以脚本格式获取参量；

# set coefficients;
_argparse = argparse.ArgumentParser(prog='TrainMnasNetA', description='the trainning code for MnasNetA on ImageNet')

# dataset/model parameters;
_argparse.add_argument('--dir', required=True,
                       help='path to dataset')

_argparse.add_argument('--model', default='MnasNetA', type=str,
                       help='Name of model to train (default: "countception"')
_argparse.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
_argparse.add_argument('--init_ckp', default='', type=str,
                       help='Initialize model from this checkpoint (default: none)')
_argparse.add_argument('--resume', default='', type=str,
                       help='Resume full model and optimizer state from checkpoint (default: none)')
_argparse.add_argument('--no_resume_opt', action='store_true', default=False,
                       help='prevent resume of optimizer state when resuming model')
_argparse.add_argument('-nc', '--num-classes', type=int, default=1000,
                       help='number of label classes (default: 1000)')
_argparse.add_argument('-gp', '--global_pool_type', default=None, type=str,
                       help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
_argparse.add_argument('-ims', '--img_size', type=int, default=None,
                       help='Image patch size (default: None => model default)')
_argparse.add_argument('-cp', '--crop_pct', default=None, type=float,
                       help='Input image center crop percent (for validation only)')
_argparse.add_argument('--mean', type=float, nargs='+', default=None,
                       help='Override mean pixel value of dataset')
_argparse.add_argument('--std', type=float, nargs='+', default=None,
                       help='Override std deviation of of dataset')
_argparse.add_argument('-interpmode', '--interpolation_mode', default='', type=str,
                       help='Image resize interpolation type (overrides model)')  # TODO:插值算法；
_argparse.add_argument('-bs', '--batch_size', type=int, default=32,
                       help='input batch size for training (default: 32)')
_argparse.add_argument('-vbsm', '--validation_batch_size_multiplier', type=int, default=1,
                       help='ratio of validation batch size to training batch size (default: 1)')

# optimizer parameters;
_argparse.add_argument('--opt', default='sgd', type=str,
                       help='Optimizer (default: "sgd"')
_argparse.add_argument('--opt_eps', default=None, type=float,
                       help='Optimizer Epsilon (default: None, use opt default)')
_argparse.add_argument('--opt_betas', default=None, type=float, nargs='+',
                       help='Optimizer Betas (default: None, use opt default)')
_argparse.add_argument('--momentum', type=float, default=0.9,
                       help='Optimizer momentum (default: 0.9)')
_argparse.add_argument('-wdecay', '--weight_decay', type=float, default=0.0001,
                       help='weight decay (default: 0.0001)')
_argparse.add_argument('-cg', '--clip_grad', type=float, default=None,
                       help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
_argparse.add_argument('-lrsch', '--lr_schedule', default='step', type=str,
                       help='LR scheduler (default: "step"')
_argparse.add_argument('--lr', type=float, default=0.01, metavar='LR',
                       help='learning rate (default: 0.01)')
_argparse.add_argument('--lr_noise', type=float, nargs='+', default=None,
                       help='learning rate noise on/off epoch percentages')
_argparse.add_argument('--lr_noise_pct', type=float, default=0.67,
                       help='learning rate noise limit percent (default: 0.67)')
_argparse.add_argument('--lr_noise_std', type=float, default=1.0,
                       help='learning rate noise std-dev (default: 1.0)')
_argparse.add_argument('--lr_cycle_mul', type=float, default=1.0,
                       help='learning rate cycle len multiplier (default: 1.0)')
_argparse.add_argument('--lr_cycle_limit', type=int, default=1,
                       help='learning rate cycle limit')
_argparse.add_argument('--warmup_lr', type=float, default=0.0001,
                       help='warmup learning rate (default: 0.0001)')
_argparse.add_argument('--min_lr', type=float, default=1e-5,
                       help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
_argparse.add_argument('--epochs', type=int, default=200,
                       help='number of epochs to train (default: 2)')
_argparse.add_argument('--start_epoch', default=None, type=int,
                       help='manual epoch number (useful on restarts)')
_argparse.add_argument('--decay_epochs', type=float, default=30,
                       help='epoch interval to decay LR')
_argparse.add_argument('--warmup_epochs', type=int, default=3,
                       help='epochs to warmup LR, if scheduler supports')
_argparse.add_argument('--cooldown_epochs', type=int, default=10,
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
_argparse.add_argument('--patience_epochs', type=int, default=10,
                       help='patience epochs for Plateau LR scheduler (default: 10')
_argparse.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                       help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
_argparse.add_argument('--no_aug', action='store_true', default=False,
                       help='Disable all training augmentation, override other train aug args')
_argparse.add_argument('-rescale', '--resize_scale', type=float, nargs='+', default=[0.08, 1.0],
                       help='Random resize scale (default: 0.08 1.0)')
_argparse.add_argument('-reratio', '--resize_ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.],
                       help='Random resize aspect ratio (default: 0.75 1.33)')
_argparse.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability')
_argparse.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability')
_argparse.add_argument('-coljit', '--color_jitter', type=float, default=0.4,
                       help='Color jitter factor (default: 0.4)')
_argparse.add_argument('--autoaug', type=str, default=None,
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
_argparse.add_argument('--aug_splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
_argparse.add_argument('--jsd', action='store_true', default=False,
                       help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
_argparse.add_argument('--reprob', type=float, default=0.,
                       help='Random erase prob (default: 0.)')
_argparse.add_argument('--remode', type=str, default='const',
                       help='Random erase mode (default: "const")')
_argparse.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
_argparse.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
_argparse.add_argument('--mixup', type=float, default=0.0,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.)')
_argparse.add_argument('--cutmix', type=float, default=0.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
_argparse.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
_argparse.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
_argparse.add_argument('--mixup_switch_prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
_argparse.add_argument('--mixup_mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
_argparse.add_argument('--mixup_off_epoch', default=0, type=int,
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
_argparse.add_argument('--smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
_argparse.add_argument('--train_interpolation', type=str, default='random',
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
_argparse.add_argument('--drop', type=float, default=0.0,
                       help='Dropout rate (default: 0.)')
_argparse.add_argument('--drop_connect', type=float, default=None,
                       help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
_argparse.add_argument('--drop_path', type=float, default=None,
                       help='Drop path rate (default: None)')
_argparse.add_argument('--drop_block', type=float, default=None,
                       help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
_argparse.add_argument('--bn_tf', action='store_true', default=False,
                       help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
_argparse.add_argument('--bn_momentum', type=float, default=None,
                       help='BatchNorm momentum override (if not None)')
_argparse.add_argument('--bn_eps', type=float, default=None,
                       help='BatchNorm epsilon override (if not None)')
_argparse.add_argument('--sync_bn', action='store_true',
                       help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
_argparse.add_argument('--dist_bn', type=str, default='',
                       help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
_argparse.add_argument('--split_bn', action='store_true',
                       help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
_argparse.add_argument('--model_ema', action='store_true', default=False,
                       help='Enable tracking moving average of model weights')
_argparse.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
_argparse.add_argument('--model_ema_decay', type=float, default=0.9998,
                       help='decay factor for model weights moving average (default: 0.9998)')

# Misc
# 硬件设置和环境配置；
_argparse.add_argument('--rand_seed', type=int, default=322,
                       help='random seed (default: 322)')  # TODO:常用random seed；
_argparse.add_argument('--log_interval', type=int, default=50,
                       help='how many batches to wait before logging training status')
_argparse.add_argument('--recovery_interval', type=int, default=0,
                       help='how many batches to wait before writing recovery checkpoint')
_argparse.add_argument('--workers', type=int, default=4,
                       help='how many training processes to use (default: 1)')  # TODO:常用number of workers；
_argparse.add_argument('--save_images', action='store_true', default=False,
                       help='save images of input bathes every log interval for debugging')
_argparse.add_argument('--amp', action='store_true', default=False,
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
_argparse.add_argument('--apex_amp', action='store_true', default=False,
                       help='Use NVIDIA Apex AMP mixed precision')
_argparse.add_argument('--native_amp', action='store_true', default=False,
                       help='Use Native Torch AMP mixed precision')
_argparse.add_argument('--channels_last', action='store_true', default=False,
                       help='Use channels_last memory layout')
_argparse.add_argument('--pin_mem', action='store_true', default=False,
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')  # TODO:常用pin memory；
_argparse.add_argument('--no_prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher')
_argparse.add_argument('--output', default='', type=str,
                       help='path to output folder (default: none, current dir)')
_argparse.add_argument('--eval_metric', default='top1', type=str,
                       help='Best metric (default: "top1"') # TODO:常用evaluate metric；
_argparse.add_argument('--ttaug', type=int, default=0,
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
_argparse.add_argument("--local_rank", default=0, type=int)
_argparse.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                       help='use the multi-epochs-loader to save time at the beginning of every epoch')
_argparse.add_argument('--torchscript', dest='torchscript', action='store_true',
                       help='convert model torchscript for inference')

# set constants;

# read ImageNet;
