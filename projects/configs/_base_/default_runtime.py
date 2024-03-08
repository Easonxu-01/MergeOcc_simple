'''
Author: EASON XU
Date: 2023-12-07 01:49:10
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2023-12-21 07:54:59
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/_base_/default_runtime.py
'''
checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project= None,
                name='eason',
                job_type='training',
                notes= None
            )
        )
        
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
