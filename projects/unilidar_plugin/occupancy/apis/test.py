# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util
import mcubes
#import open3d as o3d
import pdb
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from projects.unilidar_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results
from projects.unilidar_plugin.core import save_occ

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    
    # init predictions
    SC_metric = []
    SC_metric_1 = []
    SC_metric_2 = []
    SSC_metric = []
    SSC_metric_1 = []
    SSC_metric_2 = []
    SC_metric_fine = []
    SC_metric_fine_1 = []
    SC_metric_fine_2 = []
    SSC_metric_fine = []
    SSC_metric_fine_1 = []
    SSC_metric_fine_2 = []
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            dataset_flag = data['dataset_flag'].data[0]
            if len(dataset_flag)==1 or dataset_flag[0]==dataset_flag[1]:
                if show:
                    pred_f = result.get('pred_f')
                    save_occ(result['pred_c'], pred_f, data['img_metas'], out_dir, data['visible_mask'], data['gt_occ'])
                
                if 'SC_metric' in result.keys():
                    SC_metric.append(result['SC_metric'])
                if 'SSC_metric' in result.keys():
                    SSC_metric.append(result['SSC_metric'])
                if 'SSC_metric_fine' in result.keys():
                    SSC_metric_fine.append(result['SSC_metric_fine'])
            else:
                if show:
                    pred_f_1 = result.get('pred_f_1')
                    pred_f_2 = result.get('pred_f_2')
                    save_occ(result['pred_c_1'], pred_f_1, data['img_metas'], out_dir, data['visible_mask'].data[0][::2], data['gt_occ'].data[0][::2],dataset_flag[0])
                    save_occ(result['pred_c_2'], pred_f_2, data['img_metas'], out_dir, data['visible_mask'].data[0][1::2], data['gt_occ'].data[0][1::2],dataset_flag[1])
                    
                if 'SC_metric' in result.keys():
                    SC_metric.append(result['SC_metric'])
                if 'SC_metric_1' in result.keys():
                    SC_metric_1.append(result['SC_metric_1'])
                if 'SC_metric_2' in result.keys():
                    SC_metric_2.append(result['SC_metric_2'])
                if 'SSC_metric_1' in result.keys():
                    SSC_metric_1.append(result['SSC_metric_1'])
                if 'SSC_metric_2' in result.keys():
                    SSC_metric_2.append(result['SSC_metric_2'])
                if 'SSC_metric_fine_1' in result.keys():
                    SSC_metric_fine_1.append(result['SSC_metric_fine_1'])
                if 'SSC_metric_fine_2' in result.keys():
                    SSC_metric_fine_2.append(result['SSC_metric_fine_2'])
        prog_bar.update()


    # collect lists from multi-GPUs
    res = {}
    if 'SC_metric' in result.keys():
        SC_metric = [sum(SC_metric)]
        res['SC_metric'] = SC_metric
        
    if 'SC_metric_1' in result.keys():
        SC_metric_1 = [sum(SC_metric_1)]
        res['SC_metric_1'] = SC_metric_1
        
    if 'SC_metric_2' in result.keys():
        SC_metric_2 = [sum(SC_metric_2)]
        res['SC_metric_2'] = SC_metric_2
        
    if 'SSC_metric' in result.keys():
        if all(tensor.shape == SSC_metric[0].shape for tensor in SSC_metric):
            SSC_metric = [sum(SSC_metric)]
            res['SSC_metric'] = SSC_metric
    
    if 'SSC_metric_1' in result.keys():
        SSC_metric_1 = [sum(SSC_metric_1)]
        res['SSC_metric_1'] = SSC_metric_1
        
    if 'SSC_metric_2' in result.keys():
        SSC_metric_2 = [sum(SSC_metric_2)]
        res['SSC_metric_2'] = SSC_metric_2
    
    if 'SSC_metric_fine' in result.keys():
        if all(tensor.shape == SSC_metric_fine[0].shape for tensor in SSC_metric_fine):
            SSC_metric_fine = [sum(SSC_metric_fine)]
            res['SSC_metric_fine'] = SSC_metric_fine
        
    if 'SSC_metric_fine_1' in result.keys():
        SSC_metric_fine_1 = [sum(SSC_metric_fine_1)]
        res['SSC_metric_fine_1'] = SSC_metric_fine_1

    if 'SSC_metric_fine_2' in result.keys():
        SSC_metric_fine_2 = [sum(SSC_metric_fine_2)]
        res['SSC_metric_fine_2'] = SSC_metric_fine_2

    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    
    # init predictions
    SC_metric = []
    SC_metric_1 = []
    SC_metric_2 = []
    SSC_metric = []
    SSC_metric_1 = []
    SSC_metric_2 = []
    SC_metric_fine = []
    SC_metric_fine_1 = []
    SC_metric_fine_2 = []
    SSC_metric_fine = []
    SSC_metric_fine_1 = []
    SSC_metric_fine_2 = []
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            dataset_flag = data['dataset_flag'].data[0]
            if not model.module.pts_bbox_head.dual:
                if show:
                    pred_f = result.get('pred_f')
                    save_occ(result['pred_c'], pred_f, data['img_metas'], out_dir, data['visible_mask'].data[0], data['gt_occ'].data[0],dataset_flag[0])
                
                if 'SC_metric' in result.keys():
                    SC_metric.append(result['SC_metric'])
                if 'SSC_metric' in result.keys():
                    SSC_metric.append(result['SSC_metric'])
                if 'SC_metric_fine' in result.keys():
                    SC_metric_fine.append(result['SC_metric_fine'])
                if 'SSC_metric_fine' in result.keys():
                    SSC_metric_fine.append(result['SSC_metric_fine'])
            else:
                if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                    if model.module.training == False:
                        flag = dataset_flag[0]
                        if flag.item() == 1:
                            if show:
                                pred_f_1 = result.get('pred_f_1')
                                save_occ(result['pred_c_1'], pred_f_1, data['img_metas'], out_dir, data['visible_mask'].data[0], data['gt_occ'].data[0],dataset_flag[0])

                            if 'SC_metric' in result.keys():
                                SC_metric.append(result['SC_metric'])
                            if 'SC_metric_1' in result.keys():
                                SC_metric_1.append(result['SC_metric_1'])
                            if 'SSC_metric_1' in result.keys():
                                SSC_metric_1.append(result['SSC_metric_1'])
                            if 'SC_metric_fine' in result.keys():
                                SC_metric_fine.append(result['SC_metric_fine'])
                            if 'SC_metric_fine_1' in result.keys():
                                    SC_metric_fine_1.append(result['SC_metric_fine_1'])  
                            if 'SSC_metric_fine_1' in result.keys():
                                SSC_metric_fine_1.append(result['SSC_metric_fine_1'])
                                
                        elif flag.item() == 2:
                            if show:
                                pred_f_2 = result.get('pred_f_2')
                                save_occ(result['pred_c_2'], pred_f_2, data['img_metas'], out_dir, data['visible_mask'].data[0], data['gt_occ'].data[0],dataset_flag[0])

                            if 'SC_metric' in result.keys():
                                SC_metric.append(result['SC_metric'])
                            if 'SC_metric_2' in result.keys():
                                SC_metric_2.append(result['SC_metric_2'])
                            if 'SSC_metric_2' in result.keys():
                                SSC_metric_2.append(result['SSC_metric_2'])
                            if 'SC_metric_fine' in result.keys():
                                SC_metric_fine.append(result['SC_metric_fine'])
                            if 'SC_metric_fine_2' in result.keys():
                                SC_metric_fine_2.append(result['SC_metric_fine_2']) 
                            if 'SSC_metric_fine_2' in result.keys():
                                SSC_metric_fine_2.append(result['SSC_metric_fine_2'])
                            
                else:
                    if show:
                        pred_f_1 = result.get('pred_f_1')
                        pred_f_2 = result.get('pred_f_2')
                        save_occ(result['pred_c_1'], pred_f_1, data['img_metas'], out_dir, data['visible_mask'].data[0][::2], data['gt_occ'].data[0][::2],dataset_flag[0])
                        save_occ(result['pred_c_2'], pred_f_2, data['img_metas'], out_dir, data['visible_mask'].data[0][1::2], data['gt_occ'].data[0][1::2],dataset_flag[1])
                        
                    if 'SC_metric' in result.keys():
                        SC_metric.append(result['SC_metric'])
                    if 'SC_metric_1' in result.keys():
                        SC_metric_1.append(result['SC_metric_1'])
                    if 'SC_metric_2' in result.keys():
                        SC_metric_2.append(result['SC_metric_2'])
                    if 'SSC_metric_1' in result.keys():
                        SSC_metric_1.append(result['SSC_metric_1'])
                    if 'SSC_metric_2' in result.keys():
                        SSC_metric_2.append(result['SSC_metric_2'])
                    if 'SC_metric_fine' in result.keys():
                        SC_metric_fine.append(result['SC_metric_fine'])
                    if 'SC_metric_fine_1' in result.keys():
                            SC_metric_fine_1.append(result['SC_metric_fine_1'])  
                    if 'SC_metric_fine_2' in result.keys():
                        SC_metric_fine_2.append(result['SC_metric_fine_2']) 
                    if 'SSC_metric_fine_1' in result.keys():
                        SSC_metric_fine_1.append(result['SSC_metric_fine_1'])
                    if 'SSC_metric_fine_2' in result.keys():
                        SSC_metric_fine_2.append(result['SSC_metric_fine_2'])
            batch_size = 1
            
        # if i == 5:
        #     break

                
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect lists from multi-GPUs
    res = {}
    if 'SC_metric' in result.keys():
        SC_metric = [sum(SC_metric)]
        SC_metric = collect_results_cpu(SC_metric, len(dataset), tmpdir)
        res['SC_metric'] = SC_metric
        
    if 'SC_metric_1' in result.keys():
        SC_metric_1 = [sum(SC_metric_1)]
        SC_metric_1 = collect_results_cpu(SC_metric_1, len(dataset), tmpdir)
        res['SC_metric_1'] = SC_metric_1
        
    if 'SC_metric_2' in result.keys():
        SC_metric_2 = [sum(SC_metric_2)]
        SC_metric_2 = collect_results_cpu(SC_metric_2, len(dataset), tmpdir)
        res['SC_metric_2'] = SC_metric_2
        
    if 'SSC_metric' in result.keys():
        if all(tensor.shape == SSC_metric[0].shape for tensor in SSC_metric):
            SSC_metric = [sum(SSC_metric)]
            SSC_metric = collect_results_cpu(SSC_metric, len(dataset), tmpdir)
            res['SSC_metric'] = SSC_metric
    
    if 'SSC_metric_1' in result.keys():
        SSC_metric_1 = [sum(SSC_metric_1)]
        SSC_metric_1 = collect_results_cpu(SSC_metric_1, len(dataset), tmpdir)
        res['SSC_metric_1'] = SSC_metric_1
        
    if 'SSC_metric_2' in result.keys():
        SSC_metric_2 = [sum(SSC_metric_2)]
        SSC_metric_2 = collect_results_cpu(SSC_metric_2, len(dataset), tmpdir)
        res['SSC_metric_2'] = SSC_metric_2
        
    if 'SC_metric_fine' in result.keys():
            SC_metric_fine = [sum(SC_metric_fine)]
            SC_metric_fine = collect_results_cpu(SC_metric_fine, len(dataset), tmpdir)
            res['SC_metric_fine'] = SC_metric_fine
        
    if 'SC_metric_fine_1' in result.keys():
        SC_metric_fine_1 = [sum(SC_metric_fine_1)]
        SC_metric_fine_1 = collect_results_cpu(SC_metric_fine_1, len(dataset), tmpdir)
        res['SC_metric_fine_1'] = SC_metric_fine_1
        
    if 'SC_metric_fine_2' in result.keys():
        SC_metric_fine_2 = [sum(SC_metric_fine_2)]
        SC_metric_fine_2 = collect_results_cpu(SC_metric_fine_2, len(dataset), tmpdir)
        res['SC_metric_fine_2'] = SC_metric_fine_2
    
    if 'SSC_metric_fine' in result.keys():
        if all(tensor.shape == SSC_metric_fine[0].shape for tensor in SSC_metric_fine):
            SSC_metric_fine = [sum(SSC_metric_fine)]
            SSC_metric_fine = collect_results_cpu(SSC_metric_fine, len(dataset), tmpdir)
            res['SSC_metric_fine'] = SSC_metric_fine
        
    if 'SSC_metric_fine_1' in result.keys():
        SSC_metric_fine_1 = [sum(SSC_metric_fine_1)]
        SSC_metric_fine_1 = collect_results_cpu(SSC_metric_fine_1, len(dataset), tmpdir)
        res['SSC_metric_fine_1'] = SSC_metric_fine_1

    if 'SSC_metric_fine_2' in result.keys():
        SSC_metric_fine_2 = [sum(SSC_metric_fine_2)]
        SSC_metric_fine_2 = collect_results_cpu(SSC_metric_fine_2, len(dataset), tmpdir)
        res['SSC_metric_fine_2'] = SSC_metric_fine_2

    return res


def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    # collect all parts
    if rank == 0:
    
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        # sort the results
        if type == 'list':
            ordered_results = []
            for res in part_list:  
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
        
        else:
            raise NotImplementedError
        
        # remove tmp dir
        shutil.rmtree(tmpdir)
    
    # 因为我们是分别eval SC和SSC,如果其他rank提前return,开始评测SSC
    # 而rank0的shutil.rmtree可能会删除其他rank正在写入SSC metric的文件
    dist.barrier()

    if rank != 0:
        return None
    
    return ordered_results


