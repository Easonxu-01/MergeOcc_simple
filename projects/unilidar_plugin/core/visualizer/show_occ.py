'''
Author: EASON XU
Date: 2023-12-07 01:49:10
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-02-29 14:08:49
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/core/visualizer/show_occ.py
'''

import torch.nn.functional as F
import torch
import numpy as np
from os import path as osp
import os
from mmcv.parallel import DataContainer
def save_occ(pred_c, pred_f, img_metas, path, visible_mask=None, gt_occ=None, dataset_flag=None, free_id=0, thres_low=0.4, thres_high=0.99):

    """
    visualization saving for paper:
    1. gt
    2. pred_f pred_c
    3. gt visible
    4. pred_f visible
    """
    if pred_f is not None:
        if not isinstance(gt_occ, list) and not isinstance(gt_occ, torch.TensorType):
            gt_occ = gt_occ.data[0]
        if isinstance(gt_occ, list):        
            if all(isinstance(x, torch.Tensor) for x in gt_occ):
                for i in range(len(gt_occ)):
                    gt_occ[i] = gt_occ[i].reshape(256, 256, 32)
                gt_occ = torch.stack(gt_occ, dim=0)
        if len(gt_occ.shape) != 4:
            gt_occ = gt_occ.reshape(-1, 1, 256, 256, 32)
        pred_c = F.softmax(pred_c, dim=1)
        pred_f = F.softmax(pred_f, dim=1)
        B = pred_c.shape[0]
        
        for b in range(B):
            pred_c = pred_c[b].cpu().numpy()  # C W H D
            pred_f = pred_f[b].cpu().numpy() # C W H D
            if isinstance(gt_occ, DataContainer):
                gt_occ = gt_occ.data[0][b].cpu().numpy()
            else:
                gt_occ = gt_occ[b].cpu().numpy()  # W H D
            gt_occ[gt_occ==0] = 0
            _, W, H, D = pred_c.shape
            coordinates_3D_c = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
            _, W, H, D = pred_f.shape
            coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
            pred_c = np.argmax(pred_c, axis=0) # (W, H, D)
            pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
            occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
            occ_pred_c_mask = (pred_c.reshape(-1))!=free_id
            occ_gt_mask = (gt_occ.reshape(-1))!=free_id
            pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
            pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
            gt_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
            if len(visible_mask) == B:
                if dataset_flag==1:
                    visible_mask = visible_mask[b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    scene_token = img_metas.data[0][b]['scene_token']
                    lidar_token = img_metas.data[0][b]['lidar_token']
                    save_path = osp.join(path, 'nuscene', scene_token, lidar_token)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
                elif dataset_flag==2:
                    visible_mask = visible_mask[b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    path_ori = img_metas.data[0][b]['voxel_semantic_mask_path']
                    extracted_part = path_ori.split('/', 1)[1].rsplit('.', 1)[0]
                    save_path = osp.join(path, extracted_part)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
            else:
                if dataset_flag==1:
                    visible_mask = visible_mask[2*b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    scene_token = img_metas.data[0][2*b]['scene_token']
                    lidar_token = img_metas.data[0][2*b]['lidar_token']
                    save_path = osp.join(path, 'nuscene', scene_token, lidar_token)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
                elif dataset_flag==2:
                    visible_mask = visible_mask[2*b+1].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    path_ori = img_metas.data[0][2*b+1]['voxel_semantic_mask_path']
                    extracted_part = path_ori.split('/', 1)[1].rsplit('.', 1)[0]
                    save_path = osp.join(path, extracted_part)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
    else:
        if not isinstance(gt_occ, list) and not isinstance(gt_occ, torch.Tensor):
            gt_occ = gt_occ.data[0]
        if isinstance(gt_occ, list):        
            if all(isinstance(x, torch.Tensor) for x in gt_occ):
                for i in range(len(gt_occ)):
                    gt_occ[i] = gt_occ[i].reshape(256, 256, 32)
                gt_occ = torch.stack(gt_occ, dim=0)
        if len(gt_occ.shape) != 4:
            gt_occ = gt_occ.reshape(-1, 1, 256, 256, 32)
        pred_c = F.softmax(pred_c, dim=1)
        B = pred_c.shape[0]
        
        for b in range(B):
            pred_c = pred_c[b].cpu().numpy()  # C W H D
            if isinstance(gt_occ, DataContainer):
                gt_occ = gt_occ.data[0][b].cpu().numpy()
            else:
                gt_occ = gt_occ[b].cpu().numpy()  # W H D
            gt_occ[gt_occ==0] = 0
            _, W, H, D = pred_c.shape
            coordinates_3D_c = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
            W, H, D = gt_occ.shape
            coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
            pred_c = np.argmax(pred_c, axis=0) # (W, H, D)
            occ_pred_c_mask = (pred_c.reshape(-1))!=free_id
            occ_gt_mask = (gt_occ.reshape(-1))!=free_id
            pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
            gt_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls

            if len(visible_mask) == B:
                if dataset_flag==1:
                    visible_mask = visible_mask[b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    scene_token = img_metas.data[0][b]['scene_token']
                    lidar_token = img_metas.data[0][b]['lidar_token']
                    save_path = osp.join(path, 'nuscene', scene_token, lidar_token)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    # np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    # np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
                elif dataset_flag==2:
                    visible_mask = visible_mask[b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    path_ori = img_metas.data[0][b]['voxel_semantic_mask_path']
                    extracted_part = path_ori.split('/', 1)[1].rsplit('.', 1)[0]
                    save_path = osp.join(path, extracted_part)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    # np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    # np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
            else:
                if dataset_flag==1:
                    visible_mask = visible_mask[2*b].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    scene_token = img_metas.data[0][2*b]['scene_token']
                    lidar_token = img_metas.data[0][2*b]['lidar_token']
                    save_path = osp.join(path, 'nuscene', scene_token, lidar_token)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    # np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    # np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)
                elif dataset_flag==2:
                    visible_mask = visible_mask[2*b+1].cpu().numpy().reshape(-1) > 0  # WHD
                    gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
                    path_ori = img_metas.data[0][2*b+1]['voxel_semantic_mask_path']
                    extracted_part = path_ori.split('/', 1)[1].rsplit('.', 1)[0]
                    save_path = osp.join(path, extracted_part)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
                    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
                    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
                    save_gt_path = osp.join(save_path, 'gt.npy')
                    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
                    # np.save(save_pred_f_path, pred_f_save)
                    np.save(save_pred_c_path, pred_c_save)
                    # np.save(save_pred_f_v_path, pred_f_visible_save)
                    np.save(save_gt_path, gt_save)
                    np.save(save_gt_v_path, gt_visible_save)