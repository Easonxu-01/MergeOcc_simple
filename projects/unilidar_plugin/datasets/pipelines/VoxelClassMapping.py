'''
Author: EASON XU
Date: 2024-01-18 14:56:53
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-02-07 02:29:23
Description: 头部注释

FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/VoxelClassMapping.py
'''
import trimesh
import mmcv
import numpy as np
import numba as nb

from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F
import copy


@PIPELINES.register_module()
class VoxelClassMapping(object):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        # 确保必要的键存在
        assert 'pts_semantic_mask' in results
        assert 'voxel_semantic_mask' in results
        
        # pts_semantic_mask = results['pts_semantic_mask'].astype(np.uint16)
        # voxel_semantic_mask = results['voxel_semantic_mask'].astype(np.uint16)
        pts_semantic_mask = results['pts_semantic_mask']
        voxel_semantic_mask = results['voxel_semantic_mask']
        
        if 'seg_label_mapping' not in results:
            if 'labels_map' in results:
                results['seg_label_mapping'] = {}
                results['seg_label_mapping'] = results['labels_map']   
        assert 'seg_label_mapping' in results
        seg_label_mapping = results['seg_label_mapping']

        if isinstance(seg_label_mapping, dict):
            # 获取可能的最大标签值
            max_label_mapping = max(seg_label_mapping.keys())
            max_pts_label = pts_semantic_mask.max()
            max_voxel_label = voxel_semantic_mask.max()
            max_label = max(max_label_mapping, max_pts_label, max_voxel_label)

            # 创建一个足够大的映射数组
            mapping_array = np.zeros(max_label + 1, dtype=int)

            # 填充映射数组
            for original_label, mapped_label in seg_label_mapping.items():
                mapping_array[original_label] = mapped_label
            # in completion we have to distinguish empty and invalid voxels.
            # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
            mapping_array[mapping_array == 0] = 255  # map 0 to 'invalid'
            mapping_array[0] = 0  # only 'empty' stays 'empty'

            # 使用映射数组转换 pts_semantic_mask
            converted_pts_sem_mask = torch.from_numpy(mapping_array[pts_semantic_mask].astype(np.float32))
            converted_voxel_semantic_mask = torch.from_numpy(mapping_array[voxel_semantic_mask].astype(np.float16))
            results['pts_semantic_mask'] = converted_pts_sem_mask
            results['voxel_semantic_mask'] = converted_voxel_semantic_mask



        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    
    
    
# @PIPELINES.register_module()
# class PointSegClassMapping(object):
#     """Map original semantic class to valid category ids.

#     Map valid classes as 0~len(valid_cat_ids)-1 and
#     others as len(valid_cat_ids).

#     Args:
#         valid_cat_ids (tuple[int]): A tuple of valid category.
#         max_cat_id (int): The max possible cat_id in input segmentation mask.
#             Defaults to 40.
#     """

#     def __init__(self, valid_cat_ids, max_cat_id=40):
#         assert max_cat_id >= np.max(valid_cat_ids), \
#             'max_cat_id should be greater than maximum id in valid_cat_ids'

#         self.valid_cat_ids = valid_cat_ids
#         self.max_cat_id = int(max_cat_id)

#         # build cat_id to class index mapping
#         neg_cls = len(valid_cat_ids)
#         self.cat_id2class = np.ones(
#             self.max_cat_id + 1, dtype=np.int) * neg_cls
#         for cls_idx, cat_id in enumerate(valid_cat_ids):
#             self.cat_id2class[cat_id] = cls_idx

#     def __call__(self, results):
#         """Call function to map original semantic class to valid category ids.

#         Args:
#             results (dict): Result dict containing point semantic masks.

#         Returns:
#             dict: The result dict containing the mapped category ids. \
#                 Updated key and value are described below.

#                 - pts_semantic_mask (np.ndarray): Mapped semantic masks.
#         """
#         assert 'pts_semantic_mask' in results
#         pts_semantic_mask = results['pts_semantic_mask']

#         converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

#         results['pts_semantic_mask'] = converted_pts_sem_mask
#         return results

#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
#         repr_str += f'max_cat_id={self.max_cat_id})'
#         return repr_str