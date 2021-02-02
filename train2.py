#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
import csv
import os
from shutil import copyfile
import sys

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import MLP
import util
from torch.optim.lr_scheduler import OneCycleLR
import math
import torch.nn.functional as F
import datetime

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data', default='caltech101', type=str, metavar='MODEL',
                    help='Dataset to use')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

# Bob's arguments
parser.add_argument('--tl', action='store_true', default=False,
                    help='When true, only trains last two layers of network')
parser.add_argument('--actfun', default='swish', type=str, metavar='ACTFUN',
                    help='Controls which activation function is used in the network')
parser.add_argument('--p', type=int, default=1, metavar='p',
                    help='Number of pre-activation permutations')
parser.add_argument('--k', type=int, default=2, metavar='k',
                    help='Higher order activation group size')
parser.add_argument('--g', type=int, default=1, metavar='g',
                    help='Inter layer group size')
parser.add_argument('--check-path', default='', type=str, metavar='PATH',
                    help='Path for recording checkpoints')
parser.add_argument('--control-amp', default='', type=str, metavar='PATH',
                    help='Allows user to specify whether or not we want to use amp')
parser.add_argument('--extra-channel-mult', default=1.0, type=float, metavar='PATH',
                    help='Allows us to specify additional channel multiplier for higher order activations')
parser.add_argument('--weight-init', type=str,
                    help='Weight init method')
parser.add_argument('--load-path', default='', type=str, metavar='PATH',
                    help='Path for loading initial checkpoints')
parser.add_argument('--partial_ho_actfun', default='', type=str,
                    help='Tells network when to apply higher order activations only to specific blocks')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _logger.info(
        '====================\n\n'
        'Actfun: {}\n'
        'LR: {}\n'
        'Epochs: {}\n'
        'p: {}\n'
        'k: {}\n'
        'g: {}\n'
        'Extra channel multiplier: {}\n'
        'Weight Init: {}\n'
        '\n===================='.format(args.actfun, args.lr, args.epochs, args.p, args.k, args.g,
                                        args.extra_channel_mult, args.weight_init))

    # ================================================================================= Loading models
    pre_model = create_model(
        args.model,
        pretrained=True,
        actfun='swish',
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        p=args.p,
        k=args.k,
        g=args.g,
        extra_channel_mult=args.extra_channel_mult,
        weight_init_name=args.weight_init,
        partial_ho_actfun=args.partial_ho_actfun
    )
    pre_model_layers = list(pre_model.children())
    pre_model = torch.nn.Sequential(*pre_model_layers[:-1])
    pre_model.to(device)

    model = MLP.MLP(actfun=args.actfun,
                    input_dim=1280,
                    output_dim=args.num_classes,
                    k=args.k,
                    p=args.p,
                    g=args.g,
                    num_params=1_000_000,
                    permute_type='shuffle')
    model.to(device)

    # ================================================================================= Loading dataset
    util.seed_all(args.seed)
    if args.tl:
        if args.data == 'caltech101' and not os.path.exists('caltech101'):
            dir_root = r'101_ObjectCategories'
            dir_new = r'caltech101'
            dir_new_train = os.path.join(dir_new, 'train')
            dir_new_val = os.path.join(dir_new, 'val')
            dir_new_test = os.path.join(dir_new, 'test')
            if not os.path.exists(dir_new):
                os.mkdir(dir_new)
                os.mkdir(dir_new_train)
                os.mkdir(dir_new_val)
                os.mkdir(dir_new_test)

            for dir2 in os.listdir(dir_root):
                if dir2 != 'BACKGROUND_Google':
                    curr_path = os.path.join(dir_root, dir2)
                    new_path_train = os.path.join(dir_new_train, dir2)
                    new_path_val = os.path.join(dir_new_val, dir2)
                    new_path_test = os.path.join(dir_new_test, dir2)
                    if not os.path.exists(new_path_train):
                        os.mkdir(new_path_train)
                    if not os.path.exists(new_path_val):
                        os.mkdir(new_path_val)
                    if not os.path.exists(new_path_test):
                        os.mkdir(new_path_test)

                    train_upper = int(0.8 * len(os.listdir(curr_path)))
                    val_upper = int(0.9 * len(os.listdir(curr_path)))
                    curr_files_all = os.listdir(curr_path)
                    curr_files_train = curr_files_all[:train_upper]
                    curr_files_val = curr_files_all[train_upper:val_upper]
                    curr_files_test = curr_files_all[val_upper:]

                    for file in curr_files_train:
                        copyfile(os.path.join(curr_path, file),
                                 os.path.join(new_path_train, file))
                    for file in curr_files_val:
                        copyfile(os.path.join(curr_path, file),
                                 os.path.join(new_path_val, file))
                    for file in curr_files_test:
                        copyfile(os.path.join(curr_path, file),
                                 os.path.join(new_path_test, file))
        time.sleep(5)

    # create the train and eval datasets
    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # ================================================================================= Optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr,
                           epochs=args.epochs,
                           steps_per_epoch=int(math.floor(len(dataset_train) / args.batch_size)),
                           cycle_momentum=False
                           )

    # ================================================================================= Save file / checkpoints
    fieldnames = [
        'dataset', 'seed', 'epoch', 'time', 'actfun', 'model', 'batch_size', 'alpha_primes', 'alphas',
        'num_params', 'var_nparams', 'var_nsamples', 'k', 'p', 'g', 'perm_method', 'gen_gap',
        'epoch_train_loss', 'epoch_train_acc', 'epoch_aug_train_loss', 'epoch_aug_train_acc',
        'epoch_val_loss', 'epoch_val_acc', 'hp_idx', 'curr_lr', 'found_lr', 'epochs'
    ]
    filename = 'out_{}_{}_{}_{}'.format(datetime.date.today(), args.actfun, args.data, args.seed)
    outfile_path = os.path.join(args.save_path, filename) + '.csv'
    checkpoint_path = os.path.join(args.check_path, filename) + '.pth'
    if not os.path.exists(outfile_path):
        with open(outfile_path, mode='w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    epoch = 1
    checkpoint = torch.load(checkpoint_path) if os.path.exists(checkpoint_path) else None
    if checkpoint is not None:
        pre_model.load_state_dict(checkpoint['pre_model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        pre_model.to(device)
        model.to(device)
        print("*** LOADED CHECKPOINT ***"
              "\n{}"
              "\nSeed: {}"
              "\nEpoch: {}"
              "\nActfun: {}"
              "\np: {}"
              "\nk: {}"
              "\ng: {}"
              "\nperm_method: {}".format(checkpoint_path, checkpoint['curr_seed'],
                                         checkpoint['epoch'], checkpoint['actfun'],
                                         checkpoint['p'], checkpoint['k'], checkpoint['g'],
                                         checkpoint['perm_method']))

    best_val_acc = 0

    if args.mix_pre_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # ================================================================================= Training
    while epoch <= args.epochs:

        if args.check_path != '':
            torch.save({'pre_model_state_dict': pre_model.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_seed': args.seed,
                        'epoch': epoch,
                        'actfun': args.actfun,
                        'p': args.p, 'k': args.k, 'g': args.g,
                        'perm_method': 'shuffle'
                        }, checkpoint_path)

        util.seed_all((args.seed * args.num_epochs) + epoch)
        start_time = time.time()
        if args.mix_pre:
            scaler = torch.cuda.amp.GradScaler()

        # ---- Training
        model.train()
        total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
        for batch_idx, (x, targetx) in enumerate(loader_train):
            x, targetx = x.to(device), targetx.to(device)
            optimizer.zero_grad()
            if args.mix_pre:
                with torch.cuda.amp.autocast():
                    output = model(x)
                    train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif args.mix_pre_apex:
                output = model(x)
                train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            else:
                output = model(x)
                train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                train_loss.backward()
                optimizer.step()
            if args.optim == 'onecycle' or args.optim == 'onecycle_sgd':
                scheduler.step()
            _, prediction = torch.max(output.data, 1)
            num_correct += torch.sum(prediction == targetx.data)
            num_total += len(prediction)
        epoch_aug_train_loss = total_train_loss / n
        epoch_aug_train_acc = num_correct * 1.0 / num_total

        alpha_primes = []
        alphas = []
        if model.actfun == 'combinact':
            for i, layer_alpha_primes in enumerate(model.all_alpha_primes):
                curr_alpha_primes = torch.mean(layer_alpha_primes, dim=0)
                curr_alphas = F.softmax(curr_alpha_primes, dim=0).data.tolist()
                curr_alpha_primes = curr_alpha_primes.tolist()
                alpha_primes.append(curr_alpha_primes)
                alphas.append(curr_alphas)

        model.eval()
        with torch.no_grad():
            total_val_loss, n, num_correct, num_total = 0, 0, 0, 0
            for batch_idx, (y, targety) in enumerate(loader_eval):
                y, targety = y.to(device), targety.to(device)
                output = model(y)
                val_loss = criterion(output, targety)
                total_val_loss += val_loss
                n += 1
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
            epoch_val_loss = total_val_loss / n
            epoch_val_acc = num_correct * 1.0 / num_total
        lr_curr = 0
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']
        print(
            "    Epoch {}: LR {:1.5f} ||| aug_train_acc {:1.4f} | val_acc {:1.4f} ||| "
            "aug_train_loss {:1.4f} | val_loss {:1.4f} ||| time = {:1.4f}"
                .format(epoch, lr_curr, epoch_aug_train_acc, epoch_val_acc,
                        epoch_aug_train_loss, epoch_val_loss, (time.time() - start_time)),
            flush=True
        )

        if args.hp_idx is None:
            hp_idx = -1
        else:
            hp_idx = args.hp_idx

        epoch_train_loss = 0
        epoch_train_acc = 0
        if epoch == args.epochs:
            with torch.no_grad():
                total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
                for batch_idx, (x, targetx) in enumerate(loader_train):
                    x, targetx = x.to(device), targetx.to(device)
                    output = model(x)
                    train_loss = criterion(output, targetx)
                    total_train_loss += train_loss
                    n += 1
                    _, prediction = torch.max(output.data, 1)
                    num_correct += torch.sum(prediction == targetx.data)
                    num_total += len(prediction)
                epoch_aug_train_loss = total_train_loss / n
                epoch_aug_train_acc = num_correct * 1.0 / num_total

                total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
                for batch_idx, (x, targetx) in enumerate(loader_eval):
                    x, targetx = x.to(device), targetx.to(device)
                    output = model(x)
                    train_loss = criterion(output, targetx)
                    total_train_loss += train_loss
                    n += 1
                    _, prediction = torch.max(output.data, 1)
                    num_correct += torch.sum(prediction == targetx.data)
                    num_total += len(prediction)
                epoch_train_loss = total_val_loss / n
                epoch_train_acc = num_correct * 1.0 / num_total

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'dataset': args.dataset,
                             'seed': args.seed,
                             'epoch': epoch,
                             'time': (time.time() - start_time),
                             'actfun': model.actfun,
                             'model': args.model,
                             'batch_size': args.batch_size,
                             'alpha_primes': alpha_primes,
                             'alphas': alphas,
                             'num_params': util.get_model_params(model),
                             'var_nparams': args.var_n_params,
                             'var_nsamples': args.var_n_samples,
                             'k': args.k,
                             'p': args.p,
                             'g': args.g,
                             'perm_method': 'shuffle',
                             'gen_gap': float(epoch_val_loss - epoch_train_loss),
                             'epoch_train_loss': float(epoch_train_loss),
                             'epoch_train_acc': float(epoch_train_acc),
                             'epoch_aug_train_loss': float(epoch_aug_train_loss),
                             'epoch_aug_train_acc': float(epoch_aug_train_acc),
                             'epoch_val_loss': float(epoch_val_loss),
                             'epoch_val_acc': float(epoch_val_acc),
                             'hp_idx': hp_idx,
                             'curr_lr': lr_curr,
                             'found_lr': args.lr,
                             'epochs': args.epochs
                             })

        epoch += 1


if __name__ == '__main__':
    main()
