import numpy as np
import os 
from mmdet.datasets import DATASETS
from projects.unilidar_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results_sk
from .semantickitti_dataset import SemanticKittiDataset
from mmdet3d.datasets import SemanticKITTIDataset
@DATASETS.register_module()
class SemantickittiVoxelDataset(SemanticKITTIDataset):
    
    METAINFO = {
        'classes': ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign'),
        'palette': [[100, 150, 245], [100, 230, 245], [30, 60, 150],
                    [80, 30, 180], [100, 80, 250], [155, 30, 30],
                    [255, 40, 200], [150, 30, 90], [255, 0, 255],
                    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0],
                    [255, 120, 50], [0, 175, 0], [135, 60, 0], [150, 240, 80],
                    [255, 240, 150], [255, 0, 0]],
        'seg_valid_class_ids':
        tuple(range(19)),
        'seg_all_class_ids':
        tuple(range(19)),
    }

    def __init__(self, occ_size, seg_label_mapping, pc_range, occ_root, **kwargs):
        super().__init__(**kwargs)
        self.load_interval = 1
        for key in list(self.data_infos.keys()):
            if key != 'data_list':
                info= self.data_infos[key]
                self.data_infos.pop(key)
            if key == 'data_list':
                data = self.data_infos[key]
                self.data_infos.pop(key)
        self.data_infos = data
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['sample_id']))
        self.data_infos = self.data_infos[::self.load_interval]
        self.occ_size = occ_size
        self.seg_label_mapping = seg_label_mapping
        self.pc_range = pc_range
        self.occ_root = occ_root
        self._set_group_flag()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):

        info = self.data_infos[index]
        # bs=len(index), 
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['sample_id'],
            lidar_points=info['lidar_points'],
            pts_filename = info['lidar_points']['lidar_path'],
            pts_semantic_mask_path=info['pts_semantic_mask_path'],    
            num_pts_feats=info['lidar_points']['num_pts_feats'],
            voxels=info['voxels'],
            voxel_path=info['voxels']['voxel_path'],
            num_voxel_feats=info['voxels']['num_voxel_feats'],
            voxel_semantic_mask_path=info['voxel_semantic_mask_path'], 
            voxel_occ_mask_path=info['voxel_occ_mask_path'],
            voxel_invalidation=info['voxel_invalidation'],
            seg_label_mapping = self.seg_label_mapping,
            curr=info,
        )
        
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            lidar2cam_dic = {}
            
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam_type] = lidar2cam_rt.T

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                ))
        if self.modality['use_lidar']:
            # FIXME alter lidar path
            input_dict['pts_filename'] = os.path.join(self.data_root, input_dict['pts_filename'])
            input_dict['pts_semantic_mask_path'] = os.path.join(self.data_root, input_dict['pts_semantic_mask_path'])
            input_dict['voxel_path'] = os.path.join(self.data_root, input_dict['voxel_path'])
            input_dict['voxel_semantic_mask_path'] = os.path.join(self.data_root, input_dict['voxel_semantic_mask_path'])      
            input_dict['voxel_occ_mask_path'] = os.path.join(self.data_root, input_dict['voxel_occ_mask_path'])   
            input_dict['voxel_invalidation'] = os.path.join(self.data_root, input_dict['voxel_invalidation'])
        return input_dict

    def evaluate_2(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
            
        ''' evaluate SC '''
        if 'SC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine fine Evaluation')
                logger.info(res_table)
            
        return eval_results

    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric_2'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric_2'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
            
        ''' evaluate SC '''
        if 'SC_metric_fine_2' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine_2'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine_2' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine_2'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine Evaluation')
                logger.info(res_table)
            
        return eval_results
