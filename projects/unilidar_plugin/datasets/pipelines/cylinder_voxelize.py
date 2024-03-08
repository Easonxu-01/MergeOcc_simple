'''
Author: EASON XU
Date: 2023-10-03 01:57:21
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2023-10-10 07:40:13
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/cylinder_voxelize.py
'''
import numba as nb
import numpy as np
import time
import copy
import math
import trimesh
import mmcv
from functools import partial
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16, force_fp32
from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F

@force_fp32()
@PIPELINES.register_module()
class cylinder_voxelize(object):
    def __init__(self, rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
        fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], 
        grid_size=[512,360,40], pc_range=None):
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug

        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.pc_range = np.array(pc_range)
        self.trans_std = [0.1, 0.1, 0.1]
    
    def __call__(self, results):
        # lidar branch
        if 'points' in results.keys():
            all_ori = results['points']
            all= results['points'].tensor.cpu().numpy()
            points = results['points'].tensor.cpu().numpy()[:, :3]
            xyz = points[:, :3] 

            # random data augmentation by rotation
            if self.rotate_aug:
                rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                xyz[:, :2] = np.dot(xyz[:, :2], j)

            # random data augmentation by flip x , y or x+y
            if self.flip_aug:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    xyz[:, 0] = -xyz[:, 0]
                elif flip_type == 2:
                    xyz[:, 1] = -xyz[:, 1]
                elif flip_type == 3:
                    xyz[:, :2] = -xyz[:, :2]
            # random points augmentation by scale x & y
            if  self.scale_aug:
                noise_scale = np.random.uniform(0.95, 1.05)
                xyz[:, 0] = noise_scale * xyz[:, 0]
                xyz[:, 1] = noise_scale * xyz[:, 1]
            
            # random points augmentation by translate xyz
            if self.transform_aug:
                noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                            np.random.normal(0, self.trans_std[1], 1),
                                            np.random.normal(0, self.trans_std[2], 1)]).T

                xyz[:, 0:3] += noise_translate
                
            # convert coordinate into polar coordinates
            xyz_pol = cart2polar(xyz)

            # assert self.fixed_volume_space
            # max_bound = np.asarray(self.max_volume_space)
            # min_bound = np.asarray(self.min_volume_space)
            # # get grid index
            # crop_range = max_bound - min_bound
            # intervals = crop_range / (self.grid_size)
            
            # if (intervals == 0).any(): 
            #     print("Zero interval!")
            # xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
            # grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
            all[:,:3] = xyz_pol
            all_ori.tensor = torch.from_numpy(all)
            results.update({'points':all_ori})
            # results['points'].tensor[:, :3] = torch.from_numpy(all)
            # results['gt_occ'] = processed_label                

        return results
        
@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(data):
    points = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int32)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[2] for d in data]).astype(np.float)
    point_label = np.stack([d[3] for d in data]).astype(np.int32)
    return torch.from_numpy(points), \
        torch.from_numpy(label2stack), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label)


def seg_custom_collate_fn(data):
    voxel_label = np.stack([d[0] for d in data]).astype(np.int32)
    grid_ind_stack = np.stack([d[1] for d in data]).astype(np.float32)
    point_label = np.stack([d[2] for d in data]).astype(np.int32)
    point_feat = np.stack([d[3] for d in data]).astype(np.float32)
    grid_ind_vox_stack = np.stack([d[4] for d in data]).astype(np.float32)
    
    return torch.from_numpy(point_feat), \
        torch.from_numpy(voxel_label), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label), \
        torch.from_numpy(grid_ind_vox_stack)


def occ_custom_collate_fn(data):
    voxel_position_coarse = np.stack([d[0] for d in data]).astype(np.float32)
    voxel_label = np.stack([d[1] for d in data]).astype(np.int32)
    grid_ind_stack = np.stack([d[2] for d in data]).astype(np.float32)
    point_feat = np.stack([d[3] for d in data]).astype(np.float32)
    
    return torch.from_numpy(voxel_position_coarse), \
        torch.from_numpy(point_feat), \
        torch.from_numpy(voxel_label), \
        torch.from_numpy(grid_ind_stack)
        
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
