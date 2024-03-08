import torch
import collections 
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
# from torchsparse import PointTensor, SparseTensor
import spconv.pytorch as spconv
import numba as nb
import numpy as np
import time
import copy

@DETECTORS.register_module()
class OccNet(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            loss_bbox=None,
            spatial_shape=None,
            **kwargs):
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        # self.cylinder = cylinder
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.spatial_shape = spatial_shape
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.loss_bbox = builder.build_loss(loss_bbox)
        # self.dummy_layer = DummyLayer()

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_enc_feats = self.image_encoder(img[0])
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return x, depth, img_feats

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        for idx,pt in enumerate(pts):
            pt = pt.cuda()
            pts[idx] = pt
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        # pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        # pts_feats = pts_enc_feats['pts_feats']
        # Voxel_features, Pts_features, Voxel_coors, Pts_coors = self.pts_fusion_layer(voxel_features, coors)
        # return Voxel_features, Pts_features, Voxel_coors, Pts_coors
        Voxel, Pts = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return Voxel, Pts
    
    def extract_feat(self, points, img, img_metas, lidar2ego_rotation, lidar2ego_translation):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None
        if img is not None:
            img_voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_voxel_feats, pts_feats  = self.extract_pts_feat(points)
            
            
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(voxel_feature=pts_voxel_feats,pts_feature=pts_feats,lidar2ego_rotation=lidar2ego_rotation, lidar2ego_translation=lidar2ego_translation)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats_enc = self.occ_encoder(voxel_feats)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats_enc, img_feats, pts_feats, depth)
    
    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            dataset_flag=None,
            transform=None,
            img_feats=None,
            pts_feats=None,
            
            visible_mask=None,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            dataset_flag=dataset_flag,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.loss_bbox(
            output=outs,
            target_voxels=gt_occ,
            target_points=points_occ,
            dataset_flag=dataset_flag,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
    
        # losses = self.pts_bbox_head.loss(
        #     output_voxels=outs['output_voxels'],
        #     output_voxels_fine=outs['output_voxels_fine'],
        #     output_coords_fine=outs['output_coords_fine'],
        #     target_voxels=gt_occ,
        #     target_points=points_occ,
        #     img_metas=img_metas,
        #     visible_mask=visible_mask,
        # )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            dataset_flag=None,
            lidar2ego_rotation=None,
            lidar2ego_translation=None,
            **kwargs,
        ):
        # points = self.dummy_layer(points)
        
        # points.requires_grad = True
        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, lidar2ego_rotation=lidar2ego_rotation, lidar2ego_translation=lidar2ego_translation)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[-2], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        transform = img_inputs[1:8] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ,
                        points_occ, img_metas, dataset_flag, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            voxel=None,
            dataset_flag=None,
            lidar2ego_rotation=None,
            lidar2ego_translation=None,
            **kwargs,
        ):
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, voxel=voxel, dataset_flag=dataset_flag, visible_mask=visible_mask, lidar2ego_rotation=lidar2ego_rotation, lidar2ego_translation=lidar2ego_translation, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, voxel=None, dataset_flag=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None,lidar2ego_rotation=None, lidar2ego_translation=None, **kwargs):
        
        if isinstance(gt_occ, list):        
            if all(isinstance(x, torch.Tensor) for x in gt_occ):
                for i in range(len(gt_occ)):
                    gt_occ[i] = gt_occ[i].reshape(256, 256, 32)
                gt_occ = torch.stack(gt_occ, dim=0)
                gt_occ = gt_occ.reshape(-1, 256, 256, 32)
        
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img, img_metas=img_metas, lidar2ego_rotation=lidar2ego_rotation, lidar2ego_translation=lidar2ego_translation)

        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            dataset_flag=dataset_flag,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        if not self.pts_bbox_head.dual:
            pred_c = output['output_voxels'][0]
            SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
            SSC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        pred_c_1 = output['output_voxels_1'][0]
                        SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_1 
                    elif flag.item() == 2:
                        pred_c_2 = output['output_voxels_2'][0]
                        SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_2
            else:
                pred_c_1, pred_c_2 = output['output_voxels_1'][0], output['output_voxels_2'][0]
                SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric = SC_metric_1 + SC_metric_2
            # SSC_metric = SSC_metric_1 + SSC_metric_2
        
        pred_f = None
        SSC_metric_fine = None
        if not self.pts_bbox_head.dual:
            output_voxels_fine = output.get('output_voxels_fine', None)
            if output_voxels_fine is not None:
                if output['output_coords_fine'] is not None:
                    fine_pred = output['output_voxels_fine'][0]  # N ncls
                    fine_coord = output['output_coords_fine'][0]  # 3 N
                    pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                    pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
                else:
                    pred_f = output['output_voxels_fine'][0]
                    
                SC_metric_fine, _ = self.evaluation_semantic(pred_f, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f,  gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        output_voxels_fine_1 = output.get('output_voxels_fine_1', None)
                        if output_voxels_fine_1 is not None:
                            if output['output_coords_fine_1'] is not None:
                                fine_pred_1 = output['output_voxels_fine_1'][0]  # N ncls
                                fine_coord_1 = output['output_coords_fine_1'][0]  # 3 N
                                pred_f_1 = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred_1.shape[1], 1, 1, 1).float()
                                pred_f_1[:, :, fine_coord_1[0], fine_coord_1[1], fine_coord_1[2]] = fine_pred_1.permute(1, 0)[None]
                                SC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                                SSC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                                SC_metric_fine = SC_metric_fine_1
                    elif flag.item() == 2:
                        output_voxels_fine_2 = output.get('output_voxels_fine_2', None)
                        if output_voxels_fine_2 is not None:
                            if output['output_coords_fine_2'] is not None:
                                fine_pred_2 = output['output_voxels_fine_2'][0]
                                fine_coord_2 = output['output_coords_fine_2'][0]
                                pred_f_2 = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred_2.shape[1], 1, 1, 1).float()
                                pred_f_2[:, :, fine_coord_2[0], fine_coord_2[1], fine_coord_2[2]] = fine_pred_2.permute(1, 0)[None]
                                SC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                                SSC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                                SC_metric_fine = SC_metric_fine_2
            else:
                output_voxels_fine_1 = output.get('output_voxels_fine_1', None)
                output_voxels_fine_2 = output.get('output_voxels_fine_2', None)
                if output['output_voxels_fine_1'] is not None:
                    if output_voxels_fine_1 is not None:
                        fine_pred_1 = output['output_voxels_fine_1'][0]  # N ncls
                        fine_coord_1 = output['output_coords_fine_1'][0]  # 3 N
                        pred_f_1 = self.empty_idx * torch.ones_like(gt_occ[::2])[:, None].repeat(1, fine_pred_1.shape[1], 1, 1, 1).float()
                        pred_f_1[:, :, fine_coord_1[0], fine_coord_1[1], fine_coord_1[2]] = fine_pred_1.permute(1, 0)[None]
                if output_voxels_fine_2 is not None:
                    if output['output_coords_fine_2'] is not None:
                        fine_pred_2 = output['output_voxels_fine_2'][0]  # N ncls
                        fine_coord_2 = output['output_coords_fine_2'][0]  # 3 N
                        pred_f_2 = self.empty_idx * torch.ones_like(gt_occ[1::2])[:, None].repeat(1, fine_pred_2.shape[1], 1, 1, 1).float()
                        pred_f_2[:, :, fine_coord_2[0], fine_coord_2[1], fine_coord_2[2]] = fine_pred_2.permute(1, 0)[None]

                    SC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ[::2], dataset_flag[::2], eval_type='SC', visible_mask=visible_mask)
                    SSC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ[::2], dataset_flag[::2], eval_type='SSC', visible_mask=visible_mask)
                    SC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SC', visible_mask=visible_mask)
                    SSC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SSC', visible_mask=visible_mask)
                    SC_metric_fine = SC_metric_fine_1 + SC_metric_fine_2
                    # SSC_metric_fine = SSC_metric_fine_1 + SSC_metric_fine_2
        
        if not self.pts_bbox_head.dual:
            test_output = {
                'SC_metric': SC_metric,
                'SSC_metric': SSC_metric,
                'pred_c': pred_c,
            }

            if SSC_metric_fine is not None:
                test_output['SC_metric_fine'] = SC_metric_fine
                test_output['SSC_metric_fine'] = SSC_metric_fine
                test_output['pred_f'] = pred_f
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_1': SC_metric_1,
                            'SSC_metric_1': SSC_metric_1,
                            'pred_c_1': pred_c_1,
                        }
                        # SSC_metric_fine_1_q=test_output.get('SSC_metric_fine_1',None)
                        if SSC_metric_fine_1 is not None:
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_1'] = SC_metric_fine_1
                            test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                            test_output['pred_f_1'] = pred_f_1
                    if flag.item() == 2:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_2': SC_metric_2,
                            'SSC_metric_2': SSC_metric_2,
                            'pred_c_2': pred_c_2,
                        }
                        # SSC_metric_fine_2=test_output.get('SSC_metric_fine_2',None)
                        if SSC_metric_fine_2 is not None:
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_2'] = SC_metric_fine_2
                            test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                            test_output['pred_f_2'] = pred_f_2
                        
            else:
                test_output = {
                    'SC_metric': SC_metric,
                    'SC_metric_1': SC_metric_1,
                    'SC_metric_2': SC_metric_2,
                    'SSC_metric_1': SSC_metric_1,
                    'SSC_metric_2': SSC_metric_2,
                    'pred_c_1': pred_c_1,
                    'pred_c_2': pred_c_2,
                }

                if SSC_metric_fine_1 is not None:
                    test_output['SC_metric_fine'] = SC_metric_fine
                    test_output['SC_metric_fine_1'] = SC_metric_fine_1
                    test_output['SC_metric_fine_2'] = SC_metric_fine_2
                    test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                    test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                    test_output['pred_f_1'] = pred_f_1
                    test_output['pred_f_2'] = pred_f_2
                

        return test_output


    def evaluation_semantic(self, pred, gt, dataset_flag, eval_type, visible_mask=None):
        if isinstance(gt, list):        
            if all(isinstance(x, torch.Tensor) for x in gt):
                for i in range(len(gt)):
                    gt[i] = gt[i].reshape(256, 256, 32)
                gt = torch.stack(gt, dim=0)
        if len(gt.shape) != 4:
            gt = gt.reshape(-1, 256, 256, 32)
        _, H, W, D = gt.shape
        # if not isinstance(pred, list):
        #     pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        # else:
        #     pred_1 = pred[0]
        #     pred_2 = pred[1]
        #     pred_1 = F.interpolate(pred_1, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred_1 = torch.argmax(pred_1[0], dim=0).cpu().numpy()
        #     pred_2 = F.interpolate(pred_2, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred_2 = torch.argmax(pred_2[0], dim=0).cpu().numpy()
        #     pred = np.concatenate((pred_1,pred_2), axis=0)
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
            
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)
        #TODO:
        # ignore noise
        if isinstance(dataset_flag, list):
            if len(dataset_flag) == 1:
                dataset_flag = dataset_flag[0].item()
            elif all(flag.item() == dataset_flag[0].item() for flag in dataset_flag):
                dataset_flag = dataset_flag[0].item()
            else:
                raise NotImplementedError("Different dataset_flag when eval")
        else:
            assert isinstance(dataset_flag, torch.Tensor)
            dataset_flag = dataset_flag.item()
                
        if dataset_flag == 1:
            noise_mask = gt != 255
            max_label = 17
        elif dataset_flag == 2:
            noise_mask = gt != 255
            max_label = 20

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=max_label)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=max_label)
            return hist, hist_occ
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            dataset_flag=None,
            lidar2ego_rotation=None,
            lidar2ego_translation=None,
            **kwargs,
        ):

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img_inputs, img_metas=img_metas, lidar2ego_rotation=lidar2ego_rotation, lidar2ego_translation=lidar2ego_translation)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            dataset_flag=dataset_flag,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)

# class DummyLayer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dummy = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
#     def forward(self,x):
#         return x + self.dummy - self.dummy #(also tried x+self.dummy)
