'''
Author: EASON XU
Date: 2023-10-12 01:46:14
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2023-10-12 06:48:09
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/utils/pvvp.py
'''

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


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = TSF.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = TSF.sphashquery(pc_hash, sparse_hash)
    counts = TSF.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = TSF.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = TSF.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor

# z: PointTensor(the C is float())
# return: SparseTensor
def voxelize(z, init_res=None, after_res=None, voxel_type='max'):
    pc_hash = torchsparse.nn.functional.sphash(z.C.int())
    sparse_hash, inds, idx_query = torch_unique(pc_hash)
    counts = torchsparse.nn.functional.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = z.C[inds].int()
    if voxel_type == 'max':
        inserted_feat = torch_scatter.scatter_max(z.F, idx_query, dim=0)[0]
    elif voxel_type == 'mean':
        inserted_feat = torch_scatter.scatter_mean(z.F, idx_query, dim=0)
    else:
        raise NotImplementedError("Wrong voxel_type")
    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts

    return new_tensor

def torch_unique(x):
    # x should be 1 dim and no grad
    unique, inverse = torch.unique(x, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, inds, inverse

# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = TSF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = TSF.sphash(x.C)
        idx_query = TSF.sphashquery(pc_hash, sparse_hash)
        counts = TSF.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = TSF.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = TSF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = TSF.sphash(x.C.to(z.F.device))
        idx_query = TSF.sphashquery(old_hash, pc_hash)
        weights = TSF.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = TSF.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = TSF.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor