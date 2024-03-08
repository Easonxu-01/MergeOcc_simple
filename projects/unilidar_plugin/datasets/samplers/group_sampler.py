
# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random
from IPython import embed


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

@SAMPLER.register_module()
class BalancedDistributedGroupSampler(Sampler):
    """
    在分布式训练中，确保每个batch都包含来自两个不同数据集的数据。
    如果一个数据集的长度小于另一个，将会重复采样较短的数据集以满足batch大小的需求。
    """

    def __init__(self, dataset, samples_per_gpu=2, num_replicas=None, rank=None, seed=0):
        self.rank, self.num_replicas = get_dist_info()
        if num_replicas is not None:
            self.num_replicas = num_replicas
        if rank is not None:
            self.rank = rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu  # 每个GPU的样本数
        self.seed = seed
        self.epoch = 0

        assert hasattr(self.dataset, 'dataflag'), "Dataset must have a 'dataflag' attribute"
        self.flag = self.dataset.dataflag
        self.dataset1_indices = np.where(self.flag == 0)[0]
        self.dataset2_indices = np.where(self.flag == 1)[0]

        self.total_samples = len(self.dataset1_indices) + len(self.dataset2_indices)
        self.total_size = self.total_samples  # 总尺寸现在正确反映整个数据集的大小

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.rank)

        # 确保每个数据集的索引数量能够被samples_per_gpu整除，为此可能需要重复某些索引
        total_samples_per_dataset = max(len(self.dataset1_indices), len(self.dataset2_indices))
        total_samples_needed = total_samples_per_dataset * 2  # 需要的总样本数是两倍的最大数据集长度
        total_samples_per_gpu = self.samples_per_gpu * self.num_replicas  # 每个GPU的样本数乘以副本数

        # 计算总样本数，使其能够被total_samples_per_gpu整除
        if total_samples_needed % total_samples_per_gpu != 0:
            total_samples_needed += total_samples_per_gpu - (total_samples_needed % total_samples_per_gpu)

        # 根据需要重复采样，确保每个数据集的索引数量相等
        dataset1_indices = np.random.choice(self.dataset1_indices, total_samples_needed // 2, replace=True)
        dataset2_indices = np.random.choice(self.dataset2_indices, total_samples_needed // 2, replace=True)

        # 打乱索引
        np.random.shuffle(dataset1_indices)
        np.random.shuffle(dataset2_indices)

        # 交替合并两个数据集的索引
        indices = np.empty(total_samples_needed, dtype=int)
        indices[0::2] = dataset1_indices[:total_samples_needed // 2]
        indices[1::2] = dataset2_indices[:total_samples_needed // 2]

        # 分配给每个进程的索引部分
        indices_per_replica = len(indices) // self.num_replicas
        offset = indices_per_replica * self.rank
        indices = indices[offset:offset + indices_per_replica]

        return iter(indices)

    def __len__(self):
        # 返回的长度是按照当前replica计算的，以反映每个epoch中实际的迭代次数
        return 2 * max(len(self.dataset1_indices),len(self.dataset2_indices)) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
# class BalancedDistributedGroupSampler(Sampler):
#     """
#     在分布式训练中，确保每个batch都包含来自两个不同数据集的数据。
#     如果一个数据集的长度小于另一个，将会重复采样较短的数据集以满足batch大小的需求。
#     """

#     def __init__(self, dataset, samples_per_gpu=2, num_replicas=None, rank=None, seed=0):
#         self.rank, self.num_replicas = get_dist_info()
#         if num_replicas is not None:
#             self.num_replicas = num_replicas
#         if rank is not None:
#             self.rank = rank

#         self.dataset = dataset
#         self.samples_per_gpu = samples_per_gpu  # 每个GPU的样本数
#         self.seed = seed
#         self.epoch = 0

#         assert hasattr(self.dataset, 'dataflag'), "Dataset must have a 'dataflag' attribute"
#         self.flag = self.dataset.dataflag
#         self.dataset1_indices = np.where(self.flag == 0)[0]
#         self.dataset2_indices = np.where(self.flag == 1)[0]

#         # 确保每个数据集的索引数量至少是总数的一半
#         self.num_samples_per_dataset = max(len(self.dataset1_indices), len(self.dataset2_indices))
#         self.total_size = self.num_samples_per_dataset * 2 * self.num_replicas  # 总尺寸是每个数据集样本数两倍，乘以副本数

#     def __iter__(self):
#         # 为每个rank生成独特的种子
#         g = torch.Generator()
#         g.manual_seed(self.epoch + self.seed + self.rank)

#         # 改动：计算每个数据集的采样数量时，确保它能被 samples_per_gpu 整除
#         total_samples_per_dataset = (self.num_samples_per_dataset + self.samples_per_gpu - 1) // self.samples_per_gpu * self.samples_per_gpu

#         # 改动：确保 dataset1_indices 和 dataset2_indices 的长度相等
#         if len(self.dataset1_indices) < total_samples_per_dataset:
#             padded_size = total_samples_per_dataset - len(self.dataset1_indices) % total_samples_per_dataset
#             dataset1_indices = np.pad(self.dataset1_indices, (0, padded_size), mode='wrap')
#         else:
#             dataset1_indices = self.dataset1_indices[:total_samples_per_dataset]

#         if len(self.dataset2_indices) < total_samples_per_dataset:
#             padded_size = total_samples_per_dataset - len(self.dataset2_indices) % total_samples_per_dataset
#             dataset2_indices = np.pad(self.dataset2_indices, (0, padded_size), mode='wrap')
#         else:
#             dataset2_indices = self.dataset2_indices[:total_samples_per_dataset]

#         # 打乱索引
#         np.random.shuffle(dataset1_indices)
#         np.random.shuffle(dataset2_indices)

#         # 交替合并两个数据集的索引
#         indices = np.empty((total_samples_per_dataset * 2,), dtype=int)
#         indices[0::2] = dataset1_indices[:total_samples_per_dataset]
#         indices[1::2] = dataset2_indices[:total_samples_per_dataset]

#         # 分配给每个进程的索引部分
#         indices_per_replica = len(indices) // self.num_replicas
#         offset = indices_per_replica * self.rank
#         indices = indices[offset:offset + indices_per_replica]

#         return iter(indices)

#     def __len__(self):
#         return self.total_size // self.num_replicas

#     def set_epoch(self, epoch):
#         self.epoch = epoch