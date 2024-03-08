'''
Author: EASON XU
Date: 2024-01-31 08:32:21
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-02-28 06:34:01
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/merge_dataset.py
'''

import numpy as np
import torch
from torch.utils.data import Dataset
import random
from projects.unilidar_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results_sk, format_SSC_results
class ConcatenatedDataset(Dataset):
    def __init__(self, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        if isinstance(dataset_1[0], list):
            self.len_1 = dataset_1[0].datasets[0].flag.size
        else:
            self.len_1 = dataset_1[0].flag.size
        if isinstance(dataset_2[0], list):
            self.len_2 = dataset_2[0].datasets[0].flag.size
        else:
            self.len_2 = dataset_2[0].flag.size
        if isinstance(dataset_1[0], list) and isinstance(dataset_2[0], list):
            self.flag = np.concatenate([dataset_1[0].datasets[0].flag,dataset_2[0].datasets[0].flag])
        else:
            self.flag = np.concatenate([dataset_1[0].flag,dataset_2[0].flag])
        # 新数据集的长度为两个数据集的长度之和
        self.total_len = self.len_1 + self.len_2
        # 创建一个标志数组，用于区分属于哪个数据集的样本
        # 假设：0代表第一个数据集，1代表第二个数据集
        self.dataflag = np.zeros(self.total_len, dtype=int)
        self.dataflag[self.len_1:] = 1  # 第二个数据集的标志设置为1

    def __len__(self):
        return self.total_len
    
    def __len1__(self):
        return self.len_1
    
    def __len2__(self):
        return self.len_2

    def __getitem__(self, idx):
        if idx < self.len_1:
            # 如果索引在第一个数据集的范围内，返回第一个数据集的数据
            if isinstance(self.dataset_1[0], list):
                return self.dataset_1[0].datasets[0][idx]
            else:
                return self.dataset_1[0][idx]
        else:
            if idx < self.total_len:
                # 否则，返回第二个数据集的数据，注意调整索引
                if isinstance(self.dataset_2[0], list):
                    return self.dataset_2[0].datasets[0][idx-self.len_1]
                else:
                    return self.dataset_2[0][idx-self.len_1]
            else:
                if isinstance(self.dataset_2[0], list):
                    return self.dataset_2[0].datasets[0][random.randint(0, self.len_2 - 1)]
                else:
                    return self.dataset_2[0][random.randint(0, self.len_2 - 1)]
                
    def evaluate_sk(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        # ''' evaluate SC '''
        # evaluation_semantic = sum(results['SC_metric'])
        # ious = cm_to_ious(evaluation_semantic)
        # res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        # for key, val in res_dic.items():
        #     eval_results['SC_{}'.format(key)] = val
        # if logger is not None:
        #     logger.info('SC_total Evaluation')
        #     logger.info(res_table)
        
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
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine_2'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results
    
    def evaluate_nu(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        # ''' evaluate SC '''
        # evaluation_semantic = sum(results['SC_metric'])
        # ious = cm_to_ious(evaluation_semantic)
        # res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        # for key, val in res_dic.items():
        #     eval_results['SC_{}'.format(key)] = val
        # if logger is not None:
        #     logger.info('SC_total Evaluation')
        #     logger.info(res_table)
            
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric_1'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric_1'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine_1'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results
    
    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        # ''' evaluate SC '''
        # evaluation_semantic = sum(results['SC_metric'])
        # ious = cm_to_ious(evaluation_semantic)
        # res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        # for key, val in res_dic.items():
        #     eval_results['SC_{}'.format(key)] = val
        # if logger is not None:
        #     logger.info('SC_total Evaluation')
        #     logger.info(res_table)
        
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
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_sk(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results