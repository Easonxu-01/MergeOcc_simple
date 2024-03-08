'''
Author: EASON XU
Date: 2023-12-21 13:07:43
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-03-02 14:38:28
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/loading_voxels_sk.py
'''
#import open3d as o3d
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
from mmdet3d.core.points import get_points_type

@PIPELINES.register_module()
class LoadPointsFromFile_RPR(object):
    """Load Points From File and restrict the range of point clouds.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 RPR=False,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 dataset_flag = 1,
                 shift_coors=[0, 0, 0],
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.point_cloud_range = np.array(point_cloud_range)
        self.shift_coors = np.array(shift_coors)
        if isinstance(RPR, tuple):
            RPR = RPR[0]
        self.RPR = RPR
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.dataset_flag = [dataset_flag]

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mmcv.check_file_exist(pts_filename)
            points = _read_pointcloud_SemKITTI(pts_filename)
            # pts_bytes = self.file_client.get(pts_filename)
            # points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return torch.from_numpy(points)

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        # if self.dataset_flag[0] == 1:
        #     # 计算符合条件的索引
        #     indices = np.where((points[:, 0] >= self.point_cloud_range[1]) & \
        #                     (points[:, 0] < self.point_cloud_range[4]) & \
        #                     (points[:, 1] >= self.point_cloud_range[0]) & \
        #                     (points[:, 1] < self.point_cloud_range[3]) & \
        #                     (points[:, 2] >= self.point_cloud_range[2]) & \
        #                     (points[:, 2] < self.point_cloud_range[5]))[0]
        # elif self.dataset_flag[0] == 2:
        #     # 计算符合条件的索引
        #     indices = np.where((points[:, 0] >= self.point_cloud_range[0]) & \
        #                     (points[:, 0] < self.point_cloud_range[3]) & \
        #                     (points[:, 1] >= self.point_cloud_range[1]) & \
        #                     (points[:, 1] < self.point_cloud_range[4]) & \
        #                     (points[:, 2] >= self.point_cloud_range[2]) & \
        #                     (points[:, 2] < self.point_cloud_range[5]))[0]
        indices = np.where((points[:, 0] >= self.point_cloud_range[0]) & \
                            (points[:, 0] < self.point_cloud_range[3]) & \
                            (points[:, 1] >= self.point_cloud_range[1]) & \
                            (points[:, 1] < self.point_cloud_range[4]) & \
                            (points[:, 2] >= self.point_cloud_range[2]) & \
                            (points[:, 2] < self.point_cloud_range[5]))[0]


        # 使用索引提取符合条件的点
        if isinstance(self.RPR, tuple):
            self.RPR = self.RPR[0]
        if self.RPR == True:
            points = points[indices]
            

        if self.shift_height:
            # floor_height = np.percentile(points[:, 2], 0.99)
            # height = points[:, 2] - floor_height
            # points = np.concatenate(
            #     [points[:, :3],
            #      np.expand_dims(height, 1), points[:, 3:]], 1)
            # attribute_dims = dict(height=3)
            shifted_points = points[:,:3] - self.shift_coors
            points[:,:3] = shifted_points
            

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))
        
        # if len(points.shape)==2:
        #     points = points.reshape(1, points.shape[0], points.shape[1])
        if points.shape[1]==5:
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        results['RPR_indices'] = indices
        results['dataset_flag'] = self.dataset_flag
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str




@PIPELINES.register_module()
class LoadVoxels(object):

    def __init__(self, to_float32=True, use_semantic=False, cylinder=False, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False, file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.cylinder = cylinder
        self.occ_path = occ_path
        self.cal_visible = cal_visible

        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
        self.file_client_args=file_client_args
        self.file_client = None

    
    def __call__(self, results):
        # rel_path = 'scene_{0}/occupancy/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        # #  [z y x cls] or [z y x vx vy vz cls]
        # pcd = np.load(os.path.join(self.occ_path, rel_path))
        # pcd_label = pcd[..., -1:]
        # pcd_label[pcd_label==0] = 255
        # pcd = pcd[..., [2,1,0]]
        # pcd[:,:3] = cart2polar(pcd[:,:3])if self.cylinder else pcd[:,:3]
        # pcd_np_cor = self.voxel2world(pcd[:,:3] + 0.5)  # x y z
        # untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # # bevdet augmentation
        # pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        # pcd_np_cor = self.world2voxel(pcd_np_cor)

        # # make sure the point is in the grid
        # pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        # transformed_occ = copy.deepcopy(pcd_np_cor)
        # pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # # velocity
        # if self.use_vel:
        #     pcd_vel = pcd[..., [3,4,5]]  # x y z
        #     pcd_vel = (results['bda_mat'] @ torch.from_numpy(pcd_vel).unsqueeze(-1).float()).squeeze(-1).numpy()
        #     pcd_vel = np.concatenate([pcd_np, pcd_vel], axis=-1)  # [x y z cls vx vy vz]
        #     results['gt_vel'] = pcd_vel

        # # 255: noise, 1-16 normal classes, 0 unoccupied
        # pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        # pcd_np = pcd_np.astype(np.int64)
        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        # processed_label = nb_process_label(processed_label, pcd_np)
        # results['gt_occ'] = processed_label
        
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']
        voxel_path = results['voxel_path'] 
        voxel_semantic_mask_path = results['voxel_semantic_mask_path'] 
        voxel_occ_mask_path = results['voxel_occ_mask_path'] 
        voxel_invalidation = results['voxel_invalidation'] 
        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = _read_ptslabel_SemKITTI(pts_semantic_mask_path)
        except ConnectionError:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype='int').copy()
        try:
            mmcv.check_file_exist(voxel_path)
            voxel = _read_occupancy_SemKITTI(voxel_path)
        except ConnectionError:
            mask_bytes = self.file_client.get(voxel_path)
            # add .copy() to fix read-only bug
            voxel = np.frombuffer(
                mask_bytes, dtype='int').copy()
        try:
            mmcv.check_file_exist(voxel_semantic_mask_path)
            voxel_semantic_mask = _read_voxellabel_SemKITTI(voxel_semantic_mask_path)
        except ConnectionError:
            mask_bytes = self.file_client.get(voxel_semantic_mask_path)
            # add .copy() to fix read-only bug
            voxel_semantic_mask = np.frombuffer(
                mask_bytes, dtype='int').copy()
        try:
            mmcv.check_file_exist(voxel_occ_mask_path)
            voxel_occ_mask = _read_occluded_SemKITTI(voxel_occ_mask_path)
        except ConnectionError:
            mask_bytes = self.file_client.get(voxel_occ_mask_path)
            # add .copy() to fix read-only bug
            voxel_occ_mask = np.frombuffer(
                mask_bytes, dtype='int').copy()
        try:
            mmcv.check_file_exist(voxel_invalidation)
            voxel_invalid = _read_invalid_SemKITTI(voxel_invalidation)
        except ConnectionError:
            mask_bytes = self.file_client.get(voxel_invalidation)
            # add .copy() to fix read-only bug
            voxel_invalid = np.frombuffer(
                mask_bytes, dtype='int').copy()
            
        points = results['points']
        if len(points.shape)==2:
            if points.shape[0] != pts_semantic_mask.shape[0]:
                RPR_indices = results['RPR_indices']
                pts_semantic_mask = pts_semantic_mask[RPR_indices]
                results.pop('RPR_indices')
        elif len(points.shape)==3:
            if points.shape[1] != pts_semantic_mask.shape[0]:
                RPR_indices = results['RPR_indices']
                pts_semantic_mask = pts_semantic_mask[RPR_indices]
                results.pop('RPR_indices')
                
        voxel_semantic_mask[
                np.isclose(voxel_invalid, 1)
            ] = 255  # Setting to unknown all voxels marked on invalid mask...
        
        # results['pts_semantic_mask'] = torch.from_numpy(pts_semantic_mask.astype(np.float32))
        results['pts_semantic_mask'] = pts_semantic_mask
        results['voxel'] = torch.from_numpy(voxel)
        # results['voxel_semantic_mask'] = torch.from_numpy(voxel_semantic_mask.astype(np.float32))
        results['voxel_semantic_mask'] = voxel_semantic_mask.reshape(self.grid_size)
        results['voxel_occ_mask'] = torch.from_numpy(voxel_occ_mask)
        results['voxel_invalid'] = torch.from_numpy(voxel_invalid)
        
        results['pts_seg_fields'].append('pts_semantic_mask')
        results['seg_fields'].append('voxel')
        results['seg_fields'].append('voxel_semantic_mask')
        results['seg_fields'].append('voxel_occ_mask')
        results['seg_fields'].append('voxel_invalid')
        
        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # # camera branch
            # if 'img_inputs' in results.keys():
            #     _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
            #     occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
            #                                     rots, trans, intrins, post_rots, post_trans)  # N 6 3
            #     N, n_cam, _ = occ_uvds.shape
            #     img_visible_mask = np.zeros((N, n_cam))
            #     img_h, img_w = results['img_inputs'][0].shape[-2:]
            #     for cam_idx in range(n_cam):
            #         basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
            #                     (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
            #                     (occ_uvds[:, cam_idx, 2] >= 0)

            #         basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
            #         M = basic_valid_occ.shape[0]  # TODO M~=?
            #         basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
            #         basic_valid_occ = basic_valid_occ.cpu().numpy()
            #         basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
            #         depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
            #         nb_valid_mask = np.zeros((M), dtype=np.bool)
            #         nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
            #         img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

            #     img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
            #     img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

            #     img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
            #     img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
            #     img_pcd_np = img_pcd_np.astype(np.int64)
            #     img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
            #     voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
            #     visible_mask = visible_mask | voxel_img
            #     results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            visible_mask = torch.from_numpy(visible_mask)
            results['visible_mask'] = visible_mask

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
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


# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
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

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def pack(array):
    """ convert a boolean array into a bitwise array. """
    array = array.reshape((-1))
    #compressing bit flags.
    # yapf: disable
    compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    # yapf: enable

    return np.array(compressed, dtype=np.uint8)

def _read_SemKITTI(path, dtype, do_unpack):
    bin = np.fromfile(path, dtype=dtype)  # Flattened array
    if do_unpack:
        bin = unpack(bin)
    return bin


def _read_ptslabel_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint32, do_unpack=False)
    return label

def _read_voxellabel_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False)
    return label


def _read_invalid_SemKITTI(path):
    invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return invalid


def _read_occluded_SemKITTI(path):
    occluded = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return occluded


def _read_occupancy_SemKITTI(path):
    occupancy = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
    return occupancy


def _read_pointcloud_SemKITTI(path):
    'Return pointcloud semantic kitti with remissions (x, y, z, intensity)'
    pointcloud = _read_SemKITTI(path, dtype=np.float32, do_unpack=False)
    pointcloud = pointcloud.reshape((-1, 4))
    return pointcloud


def _read_calib_SemKITTI(calib_path):
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
        key, value = line.split(':', 1)
        calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
    return calib_out