'''
Author: EASON XU
Date: 2023-10-16 14:43:08
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-02-25 11:52:23
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/occupancy/losses/occ_loss.py
'''
import numpy as np
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss
from mmdet3d.models import losses
from projects.unilidar_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss    
from projects.unilidar_plugin.occupancy.dense_heads.lovasz_softmax import lovasz_softmax
from projects.unilidar_plugin.utils.nusc_param import nusc_class_frequencies, nusc_class_names
from projects.unilidar_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names
from mmdet3d.models.builder import LOSSES

@LOSSES.register_module()
class OccLoss(nn.Module):
    def __init__(self, 
                balance_cls_weight=True,
                loss_weight_cfg=None,
                cascade_ratio=1,
                sample_from_voxel=False,
                sample_from_img=False,
                dual=False,
                dataset_flag = 1,
                num_cls=17,
                empty_idx=0,):
        super(OccLoss, self).__init__()
        self.dual = dual
        if not self.dual:
            if balance_cls_weight:
                    if dataset_flag == 1:
                        self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
                    elif dataset_flag == 2:
                        self.class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))
                    else:
                        self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 
            else:
                self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 

            self.class_names = nusc_class_names if dataset_flag == 1 else kitti_class_names 
        else:
            if balance_cls_weight:
                    self.class_weights_1 = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
                    self.class_weights_2 = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))
            else:
                self.class_weights_1 = torch.ones(num_cls[0])/num_cls[0]  # FIXME hardcode 
                self.class_weights_2 = torch.ones(num_cls[1])/num_cls[1]  # FIXME hardcode 

            self.class_names_1 = nusc_class_names 
            self.class_names_2 = kitti_class_names 
            
        self.empty_idx = empty_idx
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
            "loss_voxel_ce_weight": 1.0,
            "loss_voxel_sem_scal_weight": 1.0,
            "loss_voxel_geo_scal_weight": 1.0,
            "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        self.cascade_ratio = cascade_ratio 
        self.sample_from_voxel = sample_from_voxel
        self.sample_from_img = sample_from_img
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
           

    def loss_voxel(self, output_voxels, target_voxels, tag, flag=None):
        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        if isinstance(target_voxels, list):        
            if all(isinstance(x, torch.Tensor) for x in target_voxels):
                for i in range(len(target_voxels)):
                    target_voxels[i] = target_voxels[i].reshape(256, 256, 32)
                target_voxels = torch.stack(target_voxels, dim=0)
        if len(target_voxels.shape) != 4:
            target_voxels = target_voxels.reshape(-1, 256, 256, 32)
        if B != target_voxels.shape[0]:
            target_voxels = target_voxels[0:B, ...]
        ratio_h = target_voxels.shape[1] // H
        ratio_w = target_voxels.shape[2] // W
        ratio_d = target_voxels.shape[3] // D
        if ratio_h != 1:
            target_voxels = target_voxels.reshape(B, H, ratio_h, W, ratio_w, D, ratio_d).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio_h*ratio_w*ratio_d)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        # assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}
        
        if not self.dual:
            class_weights = self.class_weights
        else:
            if flag==1:
                class_weights = self.class_weights_1
            elif flag==2:
                class_weights = self.class_weights_2
        
        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict


    def loss_point(self, fine_coord, fine_output, target_voxels, tag):

        if isinstance(target_voxels, list):        
            if all(isinstance(x, torch.Tensor) for x in target_voxels):
                for i in range(len(target_voxels)):
                    target_voxels[i] = target_voxels[i].reshape(256, 256, 32)
                target_voxels = torch.stack(target_voxels, dim=0)
        if len(target_voxels.shape) != 4:
            target_voxels = target_voxels.reshape(-1, 256, 256, 32)
        selected_gt = target_voxels[:, fine_coord[0,:], fine_coord[1,:], fine_coord[2,:]].long()[0]
        assert torch.isnan(selected_gt).sum().item() == 0, torch.isnan(selected_gt).sum().item()
        assert torch.isnan(fine_output).sum().item() == 0, torch.isnan(fine_output).sum().item()

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(fine_output, selected_gt, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(fine_output, dim=1), selected_gt, ignore=255)


        return loss_dict

    
    def forward(self,
            output=None,
            target_voxels=None, target_points=None, dataset_flag=None,
            img_metas=None,visible_mask=None, **kwargs):
        
        if not self.dual or (isinstance(dataset_flag, list) and len(dataset_flag) == 1):
            output_voxels = output['output_voxels']
            output_coords_fine = output['output_coords_fine']
            output_voxels_fine = output['output_voxels_fine']
            flag = dataset_flag[0]
            loss_dict = {}
            for index, output_voxel in enumerate(output_voxels):
                res = self.loss_voxel(output_voxels= output_voxel, target_voxels=target_voxels,  tag='c_{}'.format(index), flag=flag)
                loss_dict.update(res)
            if self.cascade_ratio != 1:
                loss_batch_dict = {}
                if self.sample_from_voxel or self.sample_from_img:
                    for index, (fine_coord, fine_output) in enumerate(zip(output_coords_fine, output_voxels_fine)):
                        this_batch_loss = self.loss_point(fine_coord, fine_output, target_voxels, tag='fine')
                        for k, v in this_batch_loss.items():
                            if k not in loss_batch_dict:
                                loss_batch_dict[k] = v
                            else:
                                loss_batch_dict[k] = loss_batch_dict[k] + v
                    for k, v in loss_batch_dict.items():
                        loss_dict[k] = v / len(output_coords_fine)
        else:
            
            output_voxels_1 = output.get('output_voxels_1', None)
            output_voxels_2 = output.get('output_voxels_2', None)
            output_coords_fine_1 = output.get('output_coords_fine_1', None)
            output_coords_fine_2 = output.get('output_coords_fine_2', None)
            output_voxels_fine_1 = output.get('output_voxels_fine_1', None)
            output_voxels_fine_2 = output.get('output_voxels_fine_2', None)
            if isinstance(output_voxels_1, list):
                output_voxels_1 = torch.cat(output_voxels_1, dim=0)
            if isinstance(output_voxels_2, list):
                output_voxels_2 = torch.cat(output_voxels_2, dim=0)
            # if isinstance(output_coords_fine_1, list):
            #     output_coords_fine_1 = torch.cat(output_coords_fine_1, dim=0)
            # if isinstance(output_coords_fine_2, list):
            #     output_coords_fine_2 = torch.cat(output_coords_fine_2, dim=0)
            # if isinstance(output_voxels_fine_1, list):
            #     output_voxels_fine_1 = torch.cat(output_voxels_fine_1, dim=1)
            # if isinstance(output_voxels_fine_2, list):
            #     output_voxels_fine_2 = torch.cat(output_voxels_fine_2, dim=1)
            
            idx1, idx2 = 0, 0
            # 根据dataset_flag将GT分配给对应的列表
            voxels_1 = [None] * sum(flag.item() == 1 for flag in dataset_flag)
            voxels_2 = [None] * sum(flag.item() == 2 for flag in dataset_flag)
            for idx, flag in enumerate(dataset_flag):
                    if flag.item() == 1:
                        voxels_1[idx1] = (target_voxels[idx:idx+1][0])
                        idx1 += 1
                    else:
                        voxels_2[idx2] = (target_voxels[idx:idx+1][0])
                        idx2 += 1
            if voxels_1 != []:
                target_voxels_1 = torch.stack(voxels_1, dim=0) 
            if voxels_2 != []:
                target_voxels_2 = torch.stack(voxels_2, dim=0) 
            
            loss_dict = {}
            res = {}

            # 索引映射，用于追踪output_voxels_1和output_voxels_2在原始output_voxels中的位置
            index_1, index_2 = 0, 0  # 分别追踪output_voxels_1和output_voxels_2的索引
            for idx, flag in enumerate(dataset_flag):
                if flag.item() == 1:
                    # 计算属于dataset 1的voxel的损失，并使用原始顺序的索引作为tag
                    res = self.loss_voxel(output_voxels=output_voxels_1[index_1].unsqueeze(0), target_voxels=target_voxels_1[index_1].unsqueeze(0), tag=f'c_{idx}', flag=1)
                    index_1 += 1
                else:
                    # 计算属于dataset 2的voxel的损失，并使用原始顺序的索引作为tag
                    res = self.loss_voxel(output_voxels=output_voxels_2[index_2].unsqueeze(0), target_voxels=target_voxels_2[index_2].unsqueeze(0), tag=f'c_{idx}', flag=2)
                    index_2 += 1
                loss_dict.update(res)
            
            if self.cascade_ratio != 1:
                loss_batch_dict = {}
                if self.sample_from_voxel or self.sample_from_img:
                    for index, (fine_coord_1, fine_output_1) in enumerate(zip(output_coords_fine_1, output_voxels_fine_1)):
                        this_batch_loss_1 = self.loss_point(fine_coord_1, fine_output_1, target_voxels_1, tag='fine')
                        for k, v in this_batch_loss_1.items():
                            if k not in loss_batch_dict:
                                loss_batch_dict[k] = v
                            else:
                                loss_batch_dict[k] = loss_batch_dict[k] + v
                    for index, (fine_coord_2, fine_output_2) in enumerate(zip(output_coords_fine_2, output_voxels_fine_2)):
                        this_batch_loss_2 = self.loss_point(fine_coord_2, fine_output_2, target_voxels_2, tag='fine')
                        for k, v in this_batch_loss_2.items():
                            if k not in loss_batch_dict:
                                loss_batch_dict[k] = v
                            else:
                                loss_batch_dict[k] = loss_batch_dict[k] + v
                                
                    for k, v in loss_batch_dict.items():
                        loss_dict[k] = v / (len(output_coords_fine_1) + len(output_coords_fine_2))
            
        return loss_dict