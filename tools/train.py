# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from mmcv.utils import TORCH_VERSION, digit_version
import sys
sys.path.insert(0, '/home/eason/workspace_perception/UniLiDAR/')
from projects.unilidar_plugin.occupancy.apis.train import custom_train_model, custom_train_multidb_model
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
os.environ['JOBLIB_TEMP_FOLDER'] = '/data1' 
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# import os
# os.environ["WANDB_API_KEY"] = YOUR_API_KEY
os.environ["WANDB_MODE"] = "offline"
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin') and cfg.plugin:
        assert cfg.plugin_dir is not None
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]

        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
        
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
        
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name='mmdet')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    model.init_weights()
    for param in model.parameters():
        param.requires_grad = True
    # for n, p in model.named_parameters():
    #     if p.grad is None:
    #         print(f'{n} has no grad')
    if cfg.get('fine_tune', False):
        if cfg.get('unilidar', False):
            if cfg.get('sample_from_voxel', False):
                # 首先，冻结所有参数
                for param in model.parameters():
                    param.requires_grad = False

                # 定义需要微调的参数名称列表
                fine_tuning_params = [
                    'pts_bbox_head.fine_mlp_1.0.weight', 'pts_bbox_head.fine_mlp_1.0.bias',
                    'pts_bbox_head.fine_mlp_1.1.weight', 'pts_bbox_head.fine_mlp_1.1.bias',
                    'pts_bbox_head.fine_mlp_1.3.weight', 'pts_bbox_head.fine_mlp_1.3.bias',
                    'pts_bbox_head.fine_mlp_2.0.weight', 'pts_bbox_head.fine_mlp_2.0.bias',
                    'pts_bbox_head.fine_mlp_2.1.weight', 'pts_bbox_head.fine_mlp_2.1.bias',
                    'pts_bbox_head.fine_mlp_2.3.weight', 'pts_bbox_head.fine_mlp_2.3.bias'
                ]

                # 然后，根据参数的名称，解冻特定的参数
                for name, param in model.named_parameters():
                    if name in fine_tuning_params:
                        param.requires_grad = True
        else:
            if cfg.get('sample_from_voxel', False):
                # 首先，冻结模型中的所有参数
                for param in model.parameters():
                    param.requires_grad = False

                # 定义要解冻（进行微调）的参数名称列表
                params_to_unfreeze = [
                    'pts_bbox_head.fine_mlp.0.weight',
                    'pts_bbox_head.fine_mlp.0.bias',
                    'pts_bbox_head.fine_mlp.1.weight',
                    'pts_bbox_head.fine_mlp.1.bias',
                    'pts_bbox_head.fine_mlp.3.weight',
                    'pts_bbox_head.fine_mlp.3.bias'
                ]

                # 遍历模型的所有命名参数
                for name, param in model.named_parameters():
                    if name in params_to_unfreeze:
                        # 如果参数名称在列表中，则解冻该参数（允许其接受梯度）
                        param.requires_grad = True

    
    if cfg.get('unilidar', False):
        datasets_nu = [build_dataset(cfg.data_nu.train)]
        datasets_sk = [build_dataset(cfg.data_sk.train)]
        
        model.CLASSES_nu = datasets_nu[0].CLASSES
        model.CLASSES_sk = datasets_sk[0].CLASSES
        
        custom_train_multidb_model(
            model,
            datasets_nu,
            datasets_sk,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
    
    else:
        datasets = [build_dataset(cfg.data.train)]
        
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
    
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
