import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from .lovasz_softmax import lovasz_softmax
from projects.unilidar_plugin.utils import coarse_to_fine_coordinates, project_points_on_img
from projects.unilidar_plugin.utils.nusc_param import nusc_class_frequencies, nusc_class_names
from projects.unilidar_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss

@HEADS.register_module()
class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        num_img_level=1,
        soft_weights=False,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        fine_topk=20000,
        dual = False,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        final_occ_size=[256, 256, 20],
        empty_idx=0,
        visible_loss=False,
        cascade_ratio=1,
        sample_from_voxel=False,
        sample_from_img=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super(OccHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.fine_topk = fine_topk
        self.dual = dual
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()
        self.final_occ_size = final_occ_size
        self.visible_loss = visible_loss
        self.cascade_ratio = cascade_ratio
        self.sample_from_voxel = sample_from_voxel
        self.sample_from_img = sample_from_img

        if not self.dual:
            if self.cascade_ratio != 1: 
                if self.sample_from_voxel or self.sample_from_img:
                    fine_mlp_input_dim = 0 if not self.sample_from_voxel else 128
                    if sample_from_img:
                        self.img_mlp_0 = nn.Sequential(
                            nn.Conv2d(512, 128, 1, 1, 0),
                            nn.GroupNorm(16, 128),
                            nn.ReLU(inplace=True)
                        )
                        self.img_mlp = nn.Sequential(
                            nn.Linear(128, 64),
                            nn.GroupNorm(16, 64),
                            nn.ReLU(inplace=True),
                        )
                        fine_mlp_input_dim += 64

                    self.fine_mlp = nn.Sequential(
                        nn.Linear(fine_mlp_input_dim, 64),
                        nn.GroupNorm(16, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, out_channel)
                )
            
            # voxel-level prediction
            self.occ_convs = nn.ModuleList()
            for i in range(self.num_level):
                mid_channel = self.in_channels[i] // 2
                occ_conv = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                            out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                    build_norm_layer(norm_cfg, mid_channel)[1],
                    nn.ReLU(inplace=True))
                self.occ_convs.append(occ_conv)


            self.occ_pred_conv = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, mid_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                            out_channels=out_channel, kernel_size=1, stride=1, padding=0))

            self.soft_weights = soft_weights
            self.num_img_level = num_img_level
            self.num_point_sampling_feat = self.num_level
            if self.soft_weights:
                soft_in_channel = mid_channel
                self.voxel_soft_weights = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                            out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                            out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
        else:
            if self.cascade_ratio != 1: 
                if self.sample_from_voxel or self.sample_from_img:
                    fine_mlp_input_dim = 0 if not self.sample_from_voxel else 128
                    if sample_from_img:
                        self.img_mlp_0 = nn.Sequential(
                            nn.Conv2d(512, 128, 1, 1, 0),
                            nn.GroupNorm(16, 128),
                            nn.ReLU(inplace=True)
                        )
                        self.img_mlp = nn.Sequential(
                            nn.Linear(128, 64),
                            nn.GroupNorm(16, 64),
                            nn.ReLU(inplace=True),
                        )
                        fine_mlp_input_dim += 64

                    self.fine_mlp_1 = nn.Sequential(
                        nn.Linear(fine_mlp_input_dim, 64),
                        nn.GroupNorm(16, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, out_channel[0])
                )
                    self.fine_mlp_2 = nn.Sequential(
                        nn.Linear(fine_mlp_input_dim, 64),
                        nn.GroupNorm(16, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, out_channel[1])
                )
            
            # voxel-level prediction
            self.occ_convs_1 = nn.ModuleList()
            for i in range(self.num_level):
                mid_channel = self.in_channels[i] // 2
                occ_conv_1 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                            out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                    build_norm_layer(norm_cfg, mid_channel)[1],
                    nn.ReLU(inplace=True))
                self.occ_convs_1.append(occ_conv_1)


            self.occ_pred_conv_1 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, mid_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                            out_channels=out_channel[0], kernel_size=1, stride=1, padding=0))
            
            self.occ_convs_2 = nn.ModuleList()
            for i in range(self.num_level):
                mid_channel = self.in_channels[i] // 2
                occ_conv_2 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                            out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                    build_norm_layer(norm_cfg, mid_channel)[1],
                    nn.ReLU(inplace=True))
                self.occ_convs_2.append(occ_conv_2)


            self.occ_pred_conv_2 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, mid_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                            out_channels=out_channel[1], kernel_size=1, stride=1, padding=0))

            self.soft_weights = soft_weights
            self.num_img_level = num_img_level
            self.num_point_sampling_feat = self.num_level
            if self.soft_weights:
                soft_in_channel = mid_channel
                self.voxel_soft_weights_1 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                            out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                            out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
                self.voxel_soft_weights_2 = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                            out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                            out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
        # # loss functions
        # if balance_cls_weight:
        #     self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        # else:
        #     self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

        # self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        
    def forward_coarse_voxel(self, voxel_feats, dataset_flag):
        output_occs = []
        output = {}
        if not self.dual:
            for feats, occ_conv in zip(voxel_feats, self.occ_convs):
                output_occs.append(occ_conv(feats))

            if self.soft_weights:
                voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
                voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
            else:
                voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

            out_voxel_feats = 0
            _, _, H, W, D= output_occs[0].shape
            for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
                feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                out_voxel_feats += feats * weights.unsqueeze(1)
            output['out_voxel_feats'] = [out_voxel_feats]

            out_voxel = self.occ_pred_conv(out_voxel_feats)
            output['occ'] = [out_voxel]
            output['dataset_flag'] = dataset_flag
            
        elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
            # if self.training == False:
            flag = dataset_flag[0]
            if flag.item() == 1:
                for feats, occ_conv in zip(voxel_feats, self.occ_convs_1):
                    output_occs.append(occ_conv(feats))

                if self.soft_weights:
                    voxel_soft_weights = self.voxel_soft_weights_1(output_occs[0])
                    voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
                else:
                    voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

                out_voxel_feats = 0
                _, _, H, W, D= output_occs[0].shape
                for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
                    feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                    out_voxel_feats += feats * weights.unsqueeze(1)
                output['out_voxel_feats_1'] = [out_voxel_feats]
                out_voxel = self.occ_pred_conv_1(out_voxel_feats)
                output['occ_1'] = [out_voxel]
            elif flag.item() == 2:
                for feats, occ_conv in zip(voxel_feats, self.occ_convs_2):
                    output_occs.append(occ_conv(feats))

                if self.soft_weights:
                    voxel_soft_weights = self.voxel_soft_weights_2(output_occs[0])
                    voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
                else:
                    voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

                out_voxel_feats = 0
                _, _, H, W, D= output_occs[0].shape
                for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
                    feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                    out_voxel_feats += feats * weights.unsqueeze(1)
                output['out_voxel_feats_2'] = [out_voxel_feats]
                out_voxel = self.occ_pred_conv_2(out_voxel_feats)
                output['occ_2'] = [out_voxel]
        else:
            # 按照dataset_flag区分每个尺度的特征，准备两个列表来收集
            feats_for_head1 = {scale_idx: [] for scale_idx in range(len(voxel_feats))}  # 用于self.head1
            feats_for_head2 = {scale_idx: [] for scale_idx in range(len(voxel_feats))}  # 用于self.head2

            # 根据dataset_flag将特征分配给对应的列表
            for idx, flag in enumerate(dataset_flag):
                for scale_idx in range(len(voxel_feats)):
                    if flag.item() == 1:
                        feats_for_head1[scale_idx].append(voxel_feats[scale_idx][idx:idx+1])
                    else:
                        feats_for_head2[scale_idx].append(voxel_feats[scale_idx][idx:idx+1])

            # 处理特征
            input_for_head1 = []
            input_for_head2 = []
            outputs_head1 = []
            outputs_head2 = []

            for scale_idx in range(len(voxel_feats)):
                if feats_for_head1[scale_idx] is not None:  # 检查是否有特征送入head1
                    # 将同一尺度的特征合并为一个batch进行处理
                    batch_for_head1 = torch.cat(feats_for_head1[scale_idx], dim=0)
                    input_for_head1.append(batch_for_head1)
                else:
                    input_for_head1.append(None)  # 如果没有特征，用None占位

                if feats_for_head2[scale_idx] is not None:  # 检查是否有特征送入head2
                    batch_for_head2 = torch.cat(feats_for_head2[scale_idx], dim=0)
                    input_for_head2.append(batch_for_head2)
                else:
                    input_for_head2.append(None)
            
            #head1
            for feats, occ_conv in zip(input_for_head1, self.occ_convs_1):
                outputs_head1.append(occ_conv(feats))

            if self.soft_weights:
                voxel_soft_weights_1 = self.voxel_soft_weights_1(outputs_head1[0])
                voxel_soft_weights_1 = torch.softmax(voxel_soft_weights_1, dim=1)
            else:
                voxel_soft_weights_1 = torch.ones([outputs_head1[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(outputs_head1[0].device) / self.num_point_sampling_feat

            out_voxel_feats_1 = 0
            _, _, H, W, D= outputs_head1[0].shape
            for feats, weights in zip(outputs_head1, torch.unbind(voxel_soft_weights_1, dim=1)):
                feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                out_voxel_feats_1 += feats * weights.unsqueeze(1)
            output['out_voxel_feats_1'] = [out_voxel_feats_1]
            out_voxel_1 = self.occ_pred_conv_1(out_voxel_feats_1)
            output['occ_1'] = [out_voxel_1]
            #head2
            for feats, occ_conv in zip(input_for_head2, self.occ_convs_2):
                outputs_head2.append(occ_conv(feats))

            if self.soft_weights:
                voxel_soft_weights_2 = self.voxel_soft_weights_2(outputs_head2[0])
                voxel_soft_weights_2 = torch.softmax(voxel_soft_weights_2, dim=1)
            else:
                voxel_soft_weights_2 = torch.ones([outputs_head2[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(outputs_head2[0].device) / self.num_point_sampling_feat

            out_voxel_feats_2 = 0
            _, _, H, W, D= outputs_head2[0].shape
            for feats, weights in zip(outputs_head2, torch.unbind(voxel_soft_weights_2, dim=1)):
                feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                out_voxel_feats_2 += feats * weights.unsqueeze(1)
            output['out_voxel_feats_2'] = [out_voxel_feats_2]
            out_voxel_2 = self.occ_pred_conv_2(out_voxel_feats_2)
            output['occ_2'] = [out_voxel_2]
            
            # 重组特征
            final_outputs = [[None] for _ in range(len(dataset_flag))]  # 初始化最终输出的结构
            final_occ = [[None] for _ in range(len(dataset_flag))]  # 初始化最终输出的结构
            # 指针，用于跟踪output1和output2中的当前位置
            idx1, idx2 = 0, 0

            # 根据dataset_flag的值决定如何从output1和output2中取出相应的输出
            for flag in dataset_flag:
                if flag.item() == 1:
                    final_outputs[idx1 + idx2] = out_voxel_feats_1[idx1]
                    final_occ[idx1 + idx2] = out_voxel_1[idx1]
                    idx1 += 1
                else:
                    final_outputs[idx1 + idx2] = out_voxel_feats_2[idx2]
                    final_occ[idx1 + idx2] = out_voxel_2[idx2]
                    idx2 += 1

                    
            output['out_voxel_feats'] = final_outputs
            output['occ'] = final_occ
                
        return output
            
    def forward(self, voxel_feats, dataset_flag, img_feats=None, pts_feats=None, transform=None, **kwargs):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        # forward voxel 
        output = self.forward_coarse_voxel(voxel_feats, dataset_flag)
        if not self.dual:
            out_voxel_feats = output['out_voxel_feats'][0]
            coarse_occ = output['occ'][0]

            if self.cascade_ratio != 1:
                if self.sample_from_img or self.sample_from_voxel:
                    coarse_occ_mask = coarse_occ.argmax(1) != self.empty_idx
                    assert coarse_occ_mask.sum() > 0, 'no foreground in coarse voxel'
                    B, W, H, D = coarse_occ_mask.shape
                    coarse_coord_x, coarse_coord_y, coarse_coord_z = torch.meshgrid(torch.arange(W).to(coarse_occ.device),
                                torch.arange(H).to(coarse_occ.device), torch.arange(D).to(coarse_occ.device), indexing='ij')
                    
                    output['fine_output'] = []
                    output['fine_coord'] = []

                    if self.sample_from_img and img_feats is not None:
                        img_feats_ = img_feats[0]
                        B_i,N_i,C_i, W_i, H_i = img_feats_.shape
                        img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                        img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]

                    for b in range(B):
                        append_feats = []
                        this_coarse_coord = torch.stack([coarse_coord_x[coarse_occ_mask[b]],
                                                        coarse_coord_y[coarse_occ_mask[b]],
                                                        coarse_coord_z[coarse_occ_mask[b]]], dim=0)  # 3, N
                        if self.training:
                            this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio, topk=self.fine_topk)  # 3, 8N/64N
                        else:
                            this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio)  # 3, 8N/64N

                        output['fine_coord'].append(this_fine_coord)

                        if self.sample_from_voxel:
                            this_fine_coord = this_fine_coord.float()
                            this_fine_coord[0, :] = (this_fine_coord[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                            this_fine_coord[1, :] = (this_fine_coord[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                            this_fine_coord[2, :] = (this_fine_coord[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                            this_fine_coord = this_fine_coord[None,None,None].permute(0,4,1,2,3).float()
                            # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                            new_feat = F.grid_sample(out_voxel_feats[b:b+1].permute(0,1,4,3,2), this_fine_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
                            append_feats.append(new_feat[0,:,:,0,0].permute(1,0))
                            assert torch.isnan(new_feat).sum().item() == 0
                            
                        if not self.dual:
                            output['fine_output'].append(self.fine_mlp(torch.concat(append_feats, dim=1)))
                        else:
                            flag = dataset_flag[0]
                            if flag.item() == 1:
                                output['fine_output'].append(self.fine_mlp_1(torch.concat(append_feats, dim=1)))
                                
                            elif flag.item() == 2:
                                output['fine_output'].append(self.fine_mlp_2(torch.concat(append_feats, dim=1)))

            res = {
                'output_voxels': output['occ'],
                'output_voxels_fine': output.get('fine_output', None),
                'output_coords_fine': output.get('fine_coord', None),
            }

        else:
            if (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                # if self.training == False:
                flag = dataset_flag[0]
                if flag.item() == 1:
                    if output['out_voxel_feats_1'] is not None:
                        out_voxel_feats_1 = output['out_voxel_feats_1'][0]
                        coarse_occ_1 = output['occ_1'][0]
                    else:
                        out_voxel_feats_1 = output['out_voxel_feats_1']
                        coarse_occ_1 = output['occ_1']
                    res = {
                            'output_voxels_1': output['occ_1'],
                        }
                if flag.item() == 2:
                    if output['out_voxel_feats_2'] is not None:
                        out_voxel_feats_2 = output['out_voxel_feats_2'][0]
                        coarse_occ_2 = output['occ_2'][0]
                    else:
                        out_voxel_feats_2 = output['out_voxel_feats_2']
                        coarse_occ_2 = output['occ_2']
                    res = {
                            'output_voxels_2': output['occ_2'],
                        }
                    # out_voxel_feats = output['out_voxel_feats']
                    # coarse_occ = output['occ']
                    
                if self.cascade_ratio != 1:
                    if self.sample_from_img or self.sample_from_voxel:
                        if flag.item() == 1:
                            coarse_occ_mask_1 = coarse_occ_1.argmax(1) != self.empty_idx
                            # assert coarse_occ_mask_1.sum() > 0, 'no foreground in coarse voxel 1'
                            B1, W, H, D = coarse_occ_mask_1.shape
                            coarse_coord_x_1, coarse_coord_y_1, coarse_coord_z_1 = torch.meshgrid(torch.arange(W).to(coarse_occ_1.device),
                                        torch.arange(H).to(coarse_occ_1.device), torch.arange(D).to(coarse_occ_1.device), indexing='ij')
                            output['fine_output_1'] = []
                            output['fine_coord_1'] = []

                            if self.sample_from_img and img_feats is not None:
                                img_feats_ = img_feats[0]
                                B_i,N_i,C_i, W_i, H_i = img_feats_.shape
                                img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                                img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]

                            for b in range(B1):
                                append_feats_1 = []
                                this_coarse_coord_1 = torch.stack([coarse_coord_x_1[coarse_occ_mask_1[b]],
                                                                coarse_coord_y_1[coarse_occ_mask_1[b]],
                                                                coarse_coord_z_1[coarse_occ_mask_1[b]]], dim=0)
                                this_fine_coord_1 = coarse_to_fine_coordinates(this_coarse_coord_1, self.cascade_ratio)  # 3, 8N/64N
                                output['fine_coord_1'].append(this_fine_coord_1)
                                
                                if self.sample_from_voxel:
                                    this_fine_coord_1 = this_fine_coord_1.float()
                                    this_fine_coord_1[0, :] = (this_fine_coord_1[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                                    this_fine_coord_1[1, :] = (this_fine_coord_1[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                                    this_fine_coord_1[2, :] = (this_fine_coord_1[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                                    this_fine_coord_1 = this_fine_coord_1[None,None,None].permute(0,4,1,2,3).float()
                                    
                                    # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                                    new_feat_1 = F.grid_sample(out_voxel_feats_1[b:b+1].permute(0,1,4,3,2), this_fine_coord_1, mode='bilinear', padding_mode='zeros', align_corners=False)
                                    append_feats_1.append(new_feat_1[0,:,:,0,0].permute(1,0))
                                    assert torch.isnan(new_feat_1).sum().item() == 0
                            
                            
                                output['fine_output_1'].append(self.fine_mlp_1(torch.concat(append_feats_1, dim=1)))
                                    
                            res = {
                                    'output_voxels_1': output['occ_1'],
                                    'output_voxels_fine_1': output.get('fine_output_1', None),
                                    'output_coords_fine_1': output.get('fine_coord_1', None),
                                }
                                    
                        elif flag.item() == 2:
                            coarse_occ_mask_2 = coarse_occ_2.argmax(1) != self.empty_idx
                            # assert coarse_occ_mask_2.sum() > 0, 'no foreground in coarse voxel 2'
                            B2, W, H, D = coarse_occ_mask_2.shape
                            coarse_coord_x_2, coarse_coord_y_2, coarse_coord_z_2 = torch.meshgrid(torch.arange(W).to(coarse_occ_2.device),
                                        torch.arange(H).to(coarse_occ_2.device), torch.arange(D).to(coarse_occ_2.device), indexing='ij')
                            output['fine_output_2'] = []
                            output['fine_coord_2'] = []

                            if self.sample_from_img and img_feats is not None:
                                img_feats_ = img_feats[0]
                                B_i,N_i,C_i, W_i, H_i = img_feats_.shape
                                img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                                img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]

                            for b in range(B2):
                                append_feats_2 = []
                                this_coarse_coord_2 = torch.stack([coarse_coord_x_2[coarse_occ_mask_2[b]],
                                                                coarse_coord_y_2[coarse_occ_mask_2[b]],
                                                                coarse_coord_z_2[coarse_occ_mask_2[b]]], dim=0)
                                this_fine_coord_2 = coarse_to_fine_coordinates(this_coarse_coord_2, self.cascade_ratio)  # 3, 8N/64N
                                output['fine_coord_2'].append(this_fine_coord_2)
                                
                                if self.sample_from_voxel:
                                    this_fine_coord_2 = this_fine_coord_2.float()
                                    this_fine_coord_2[0, :] = (this_fine_coord_2[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                                    this_fine_coord_2[1, :] = (this_fine_coord_2[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                                    this_fine_coord_2[2, :] = (this_fine_coord_2[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                                    this_fine_coord_2 = this_fine_coord_2[None,None,None].permute(0,4,1,2,3).float()
                                    
                                    # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                                    new_feat_2 = F.grid_sample(out_voxel_feats_2[b:b+1].permute(0,1,4,3,2), this_fine_coord_2, mode='bilinear', padding_mode='zeros', align_corners=False)
                                    append_feats_2.append(new_feat_2[0,:,:,0,0].permute(1,0))
                                    assert torch.isnan(new_feat_2).sum().item() == 0
                                    
                                output['fine_output_2'].append(self.fine_mlp_2(torch.concat(append_feats_2, dim=1)))
                                
                            res = {
                                    'output_voxels_2': output['occ_2'],
                                    'output_voxels_fine_2': output.get('fine_output_2', None),
                                    'output_coords_fine_2': output.get('fine_coord_2', None),
                                }
                            
            else:
                if output['out_voxel_feats_1'] is not None:
                    out_voxel_feats_1 = output['out_voxel_feats_1'][0]
                    coarse_occ_1 = output['occ_1'][0]
                else:
                    out_voxel_feats_1 = output['out_voxel_feats_1']
                    coarse_occ_1 = output['occ_1']
                if output['out_voxel_feats_2'] is not None:
                    out_voxel_feats_2 = output['out_voxel_feats_2'][0]
                    coarse_occ_2 = output['occ_2'][0]
                else:
                    out_voxel_feats_2 = output['out_voxel_feats_2']
                    coarse_occ_2 = output['occ_2']
                out_voxel_feats = output['out_voxel_feats']
                coarse_occ = output['occ']
                
                if self.cascade_ratio != 1:
                    if self.sample_from_img or self.sample_from_voxel:
                        coarse_occ_mask_1 = coarse_occ_1.argmax(1) != self.empty_idx
                        coarse_occ_mask_2 = coarse_occ_2.argmax(1) != self.empty_idx
                        # assert coarse_occ_mask_1.sum() > 0, 'no foreground in coarse voxel 1'
                        # assert coarse_occ_mask_2.sum() > 0, 'no foreground in coarse voxel 2'
                        
                        B1, W, H, D = coarse_occ_mask_1.shape
                        B2, W, H, D = coarse_occ_mask_2.shape
                        coarse_coord_x_1, coarse_coord_y_1, coarse_coord_z_1 = torch.meshgrid(torch.arange(W).to(coarse_occ_1.device),
                                    torch.arange(H).to(coarse_occ_1.device), torch.arange(D).to(coarse_occ_1.device), indexing='ij')
                        coarse_coord_x_2, coarse_coord_y_2, coarse_coord_z_2 = torch.meshgrid(torch.arange(W).to(coarse_occ_2.device),
                                    torch.arange(H).to(coarse_occ_2.device), torch.arange(D).to(coarse_occ_2.device), indexing='ij')
                        
                        output['fine_output_1'] = []
                        output['fine_output_2'] = []
                        output['fine_coord_1'] = []
                        output['fine_coord_2'] = []

                        if self.sample_from_img and img_feats is not None:
                            img_feats_ = img_feats[0]
                            B_i,N_i,C_i, W_i, H_i = img_feats_.shape
                            img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                            img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]

                        for b in range(B1):
                            append_feats_1 = []
                            this_coarse_coord_1 = torch.stack([coarse_coord_x_1[coarse_occ_mask_1[b]],
                                                            coarse_coord_y_1[coarse_occ_mask_1[b]],
                                                            coarse_coord_z_1[coarse_occ_mask_1[b]]], dim=0)  # 3, N
                            if self.training:
                                this_fine_coord_1 = coarse_to_fine_coordinates(this_coarse_coord_1, self.cascade_ratio, topk=self.fine_topk)  # 3, 8N/64N
                            else:
                                this_fine_coord_1 = coarse_to_fine_coordinates(this_coarse_coord_1, self.cascade_ratio)  # 3, 8N/64N
                                
                            output['fine_coord_1'].append(this_fine_coord_1)

                            if self.sample_from_voxel:
                                this_fine_coord_1 = this_fine_coord_1.float()
                                this_fine_coord_1[0, :] = (this_fine_coord_1[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                                this_fine_coord_1[1, :] = (this_fine_coord_1[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                                this_fine_coord_1[2, :] = (this_fine_coord_1[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                                this_fine_coord_1 = this_fine_coord_1[None,None,None].permute(0,4,1,2,3).float()
                                
                                # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                                new_feat_1 = F.grid_sample(out_voxel_feats_1[b:b+1].permute(0,1,4,3,2), this_fine_coord_1, mode='bilinear', padding_mode='zeros', align_corners=False)
                                append_feats_1.append(new_feat_1[0,:,:,0,0].permute(1,0))
                                assert torch.isnan(new_feat_1).sum().item() == 0
                                
                                
                            output['fine_output_1'].append(self.fine_mlp_1(torch.concat(append_feats_1, dim=1)))
                            
                        for b in range(B2):
                            append_feats_2 = []
                            this_coarse_coord_2 = torch.stack([coarse_coord_x_2[coarse_occ_mask_2[b]],
                                                            coarse_coord_y_2[coarse_occ_mask_2[b]],
                                                            coarse_coord_z_2[coarse_occ_mask_2[b]]], dim=0)  # 3, N
                            if self.training:
                                this_fine_coord_2 = coarse_to_fine_coordinates(this_coarse_coord_2, self.cascade_ratio, topk=self.fine_topk)  # 3, 8N/64N
                            else:
                                this_fine_coord_2 = coarse_to_fine_coordinates(this_coarse_coord_2, self.cascade_ratio)  # 3, 8N/64N

                            output['fine_coord_2'].append(this_fine_coord_2)

                            if self.sample_from_voxel:
                                this_fine_coord_2 = this_fine_coord_2.float()
                                this_fine_coord_2[0, :] = (this_fine_coord_2[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                                this_fine_coord_2[1, :] = (this_fine_coord_2[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                                this_fine_coord_2[2, :] = (this_fine_coord_2[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                                this_fine_coord_2 = this_fine_coord_2[None,None,None].permute(0,4,1,2,3).float()
                                # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                                new_feat_2 = F.grid_sample(out_voxel_feats_2[b:b+1].permute(0,1,4,3,2), this_fine_coord_2, mode='bilinear', padding_mode='zeros', align_corners=False)
                                append_feats_2.append(new_feat_2[0,:,:,0,0].permute(1,0))
                                assert torch.isnan(new_feat_2).sum().item() == 0
                                
                            output['fine_output_2'].append(self.fine_mlp_2(torch.concat(append_feats_2, dim=1)))

            res = {
                'output_voxels_1': output.get('occ_1', None),
                'output_voxels_2': output.get('occ_2', None),
                'output_voxels': output.get('occ', None),
                'output_voxels_fine': output.get('fine_output', None),
                'output_voxels_fine_1': output.get('fine_output_1', None),
                'output_voxels_fine_2': output.get('fine_output_2', None),
                'output_coords_fine': output.get('fine_coord', None),
                'output_coords_fine_1': output.get('fine_coord_1', None),
                'output_coords_fine_2': output.get('fine_coord_2', None),
            }
            
        return res


        
