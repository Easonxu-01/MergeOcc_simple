'''
Author: EASON XU
Date: 2023-12-07 01:49:10
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-01-29 12:36:37
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/occupancy/apis/__init__.py
'''
from .train import custom_train_model, custom_train_multidb_model
from .mmdet_train import custom_train_detector, custom_train_multidb_detector
# from .test import custom_multi_gpu_test