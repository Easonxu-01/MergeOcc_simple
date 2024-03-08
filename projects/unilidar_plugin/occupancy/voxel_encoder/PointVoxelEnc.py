import math
from functools import partial
from mmcv.runner import BaseModule

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch.nn.functional as F
import torchsparse.nn.functional as TSF
import torch_scatter
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
import spconv.pytorch as spconv
# from spconv.pytorch import functional as Fsp
from projects.unilidar_plugin.utils.pvvp import initial_voxelize, voxelize, point_to_voxel, voxel_to_point
from mmdet3d.models.builder import MIDDLE_ENCODERS

import copy


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,bias=False)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, bias=False)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, bias=False)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, bias=False)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, bias=False)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, bias=False)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)
        resA.F = resA.F + shortcut.F

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spnn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, bias=False)
            else:
                self.pool = spnn.Conv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1), bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)

        resA.F = resA.F + shortcut.F

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA

    
class PointBranch(nn.Module):
    def __init__(self, pt_in_dim, pt_out_dim):
        super(PointBranch, self).__init__()
        self.pt_in_dim = pt_in_dim
        self.pt_out_dim = pt_out_dim
        self.range = self.pt_out_dim-self.pt_in_dim
        self.point_transforms = nn.Sequential(
                nn.Linear(self.pt_in_dim, int(self.pt_in_dim+self.range//2)),
                nn.BatchNorm1d(int(self.pt_in_dim+self.range//2)),
                nn.LeakyReLU(True),
                nn.Linear(int(self.pt_in_dim+self.range//2), self.pt_out_dim),
                nn.BatchNorm1d(self.pt_out_dim),
                nn.LeakyReLU(True),
                nn.MaxPool1d(kernel_size=2, stride=2))
        self.up = nn.Linear(int(self.pt_out_dim//2), self.pt_out_dim)
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pt_fea):
        point_features = self.point_transforms(pt_fea) + pt_fea
        return self.up(point_features)


@MIDDLE_ENCODERS.register_module()
class PointVoxelEnc(nn.Module):
    def __init__(self, input_channel, init_size, spatial_shape, **kwargs):
        super().__init__()
        self.input_channel = input_channel  
        self.init_size = init_size
        self.spatial_shape = spatial_shape
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(self.input_channel),

            nn.Linear(self.input_channel, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
        )
        self.fea_compression = nn.Sequential(
            nn.Linear(256, self.init_size),
            nn.LeakyReLU()
        )
        self.downCntx = ResContextBlock(self.init_size, self.init_size, indice_key="pre")
        self.resBlock2 = ResBlock(self.init_size, 2 * self.init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * self.init_size, 4 * self.init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * self.init_size, 8 * self.init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * self.init_size, 16 * self.init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")
        self.PBD1 = PointBranch(self.init_size, 2 * self.init_size)
        self.PBD2 = PointBranch(2 * self.init_size, 4 * self.init_size)
        self.PBD3 = PointBranch(4 * self.init_size, 8 * self.init_size)
        self.PBD4 = PointBranch(8 * self.init_size, 16 * self.init_size)

    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = torch.flip(coors,dims=[1]).float() #bzyx->xyzb
        # coors = torch.cat([coors[:, 1:], coors[:, :1]], dim=1) #batch_idx->last
        #channel:4-32
        init_features = self.fea_compression(self.PPmodel(voxel_features))
        #channel:32
        V = SparseTensor(init_features, coors.float())
        #channel:16
        P0 = PointTensor(V.F, V.C.float())
        #channel:16
        V0 = initial_voxelize(P0, 0.5, 0.5)
        #channel:32 
        V0 = self.downCntx(V0)
        #channel:64
        down1c, down1b = self.resBlock2(V0)
        
        P1 = self.PBD1(P0.F)
        down1c_pts = voxel_to_point(down1c, P0, nearest=False)
        down1c_pts.F = down1c_pts.F + P1
        P1_fusion = down1c_pts
        V1 = point_to_voxel(down1c, P1_fusion)
        
        
        down2c, down2b = self.resBlock3(V1)
        P2 = self.PBD2(P1_fusion.F)
        down2c_pts = voxel_to_point(down2c, P1_fusion, nearest=False)
        down2c_pts.F = down2c_pts.F + P2
        P2_fusion = down2c_pts
        V2 = point_to_voxel(down2c, P2_fusion)
        
        down3c, down3b = self.resBlock4(V2)
        P3 = self.PBD3(P2_fusion.F)
        down3c_pts = voxel_to_point(down3c, P2_fusion, nearest=False)
        down3c_pts.F = down3c_pts.F + P3
        P3_fusion = down3c_pts
        V3 = point_to_voxel(down3c, P3_fusion)
        
        down4c, down4b = self.resBlock5(V3)
        P4 = self.PBD4(P3_fusion.F)
        down4c_pts = voxel_to_point(down4c, P3_fusion, nearest=False)
        down4c_pts.F = down4c_pts.F + P4
        P4_fusion = down4c_pts
        V4 = point_to_voxel(down4c, P4_fusion)
        
        V4.C[:, 0] = torch.clamp(V4.C[:, 0], max=torch.max(coors[:,0]))
        V4.C[:, 1] = torch.clamp(V4.C[:, 1], max=torch.max(coors[:,1]))
        V4.C[:, 2] = torch.clamp(V4.C[:, 2], max=torch.max(coors[:,2]))
        P4_fusion.C[:, 0] = torch.clamp(P4_fusion.C[:, 0], max=torch.max(coors[:,0]))
        P4_fusion.C[:, 1] = torch.clamp(P4_fusion.C[:, 1], max=torch.max(coors[:,1]))
        P4_fusion.C[:, 2] = torch.clamp(P4_fusion.C[:, 2], max=torch.max(coors[:,2]))
        
        batch_idx_V = V4.C[:,3].clone()
        V4.C[:,3] = (V4.C[:,2]/4).floor() 
        V4.C[:,2] = (V4.C[:,1]/16).floor() 
        V4.C[:,1] = (V4.C[:,0]/16).floor()  
        V4.C[:,0] = batch_idx_V
        V4.C = V4.C.to(torch.int32) 
        batch_idx_P = P4_fusion.C[:,3].clone()
        P4_fusion.C[:,3] = (P4_fusion.C[:,2]/4).floor() 
        P4_fusion.C[:,2] = (P4_fusion.C[:,1]/16).floor() 
        P4_fusion.C[:,1] = (P4_fusion.C[:,0]/16).floor()  
        P4_fusion.C[:,0] = batch_idx_P
        P4_fusion.C = P4_fusion.C.to(torch.int32) 
        
        batch_size_v = V4.C[-1, 0] + 1 
        assert batch_size_v == batch_size, "batch_size mismatch"
        pts_feats = spconv.SparseConvTensor(P4_fusion.F, P4_fusion.C, [self.spatial_shape[0]//16, self.spatial_shape[1]//16, self.spatial_shape[2]//4], batch_size)
        pts_feats = pts_feats.dense()
        pts_voxel_feats = spconv.SparseConvTensor(V4.F, V4.C, [self.spatial_shape[0]//16, self.spatial_shape[1]//16, self.spatial_shape[2]//4], batch_size)
        pts_voxel_feats = pts_voxel_feats.dense()
        
        
        # return V4.F, P4_fusion.F, V4.C, P4_fusion.C
        return pts_voxel_feats, pts_feats
        
        
        

