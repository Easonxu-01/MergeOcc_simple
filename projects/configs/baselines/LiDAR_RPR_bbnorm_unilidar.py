'''
Author: EASON XU
Date: 2024-01-26 02:42:08
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2024-03-02 15:20:58
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/baselines/LiDAR_128x128x10_unilidar.py
'''
_base_ = [
    '../_base_/default_runtime.py',
    '../datasets/custom_nus-3d.py',
]
dataset_type_sk = 'SemantickittiVoxelDataset'
data_root_sk = 'data/semantickitti/'
dataset_type_nu = 'NuscOCCDataset'
data_root_nu = 'data/nuscenes/'
file_client_args = dict(backend='disk')
#test
gpu_ids = [0,1,2,3,4,5,6,7]
seed=0

class_names_sk = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign'
]
# For nuScenes we usually do 10-class detection
class_names_nu = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
labels_map_sk = {
    0 : 0,     # "unlabeled"
    1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
    10 : 1,     # "car"
    11 : 2,     # "bicycle"
    13 : 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15 : 3,     # "motorcycle"
    16 : 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18 : 4,    # "truck"
    20 : 5,   # "other-vehicle"
    30 : 6,     # "person"
    31 : 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,   # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

metainfo_sk = dict(
    classes=class_names_sk, seg_label_mapping=labels_map_sk, max_label=259, voxel_label_mapping=labels_map_sk)
input_modality_sk = dict(use_lidar=True, use_camera=False)
input_modality_nu = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
backend_args = None

plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path_sk = "./data/semantickitti"
occ_path_nu = "./data/nuScenes-Occupancy"
train_ann_file_sk = "./data/semantickitti/semantickitti_infos_train.pkl"
val_ann_file_sk = "./data/semantickitti/semantickitti_infos_val.pkl"
train_ann_file_nu = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file_nu = "./data/nuscenes/nuscenes_occ_infos_val.pkl"

fine_tune = False
unilidar=True
test_dual = False
cylinder=False
RPR = True
coor_alignment_nu = False
coor_alignment_sk = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# ori_point_cloud_range_sk = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
point_cloud_range = [0.0, -3.1415926, -5.0, 51.2, 3.1415926, 3.0] if cylinder else [0, -25.6, -3.4, 51.2, 25.6, 3.0]
pcr_nu = [-25.6, 0, -3.4, 25.6, 51.2, 3.0]
occ_size_nu = [512, 512, 40]
occ_size_sk = [256, 256, 32]
final_occ_size= occ_size_sk
voxel_channels = [80, 160, 320, 640]
empty_idx_sk = 0  
empty_idx_nu = 0  # noise 0-->255
num_cls_sk = 20  
num_cls_nu = 17  # 0 free, 1-16 obj
visible_mask = True

sample_from_voxel = False
sample_from_img = False

numC_Trans = 80
init_size = 64
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

find_unused_parameters = False

num_beams_sk = 64
num_beams_nu = 32

#TODO
start_vertical_angle=-30.67
end_vertical_angle=10.67    
num_beam_points=1080
#TODO

spatial_shape = [256, 256, 32] if cylinder else [1024, 1024, 128] #model init voxel spatial shape
# voxel_size = [(point_cloud_range[3]-point_cloud_range[0])/spatial_shape[0], \
#                 (point_cloud_range[4]-point_cloud_range[1])/spatial_shape[1], \
#                 (point_cloud_range[5]-point_cloud_range[2])/spatial_shape[2]] #model init voxel size
voxel_size=[0.05,0.05,0.05]
cascade_ratio = final_occ_size[0] * 8 // spatial_shape[0] 
dataset_flag_nu = 1 
dataset_flag_sk = 2



# find_unused_parameters = unilidar
model = dict(
    type='OccNet',
    loss_norm=True,
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        voxel_size= voxel_size,  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=init_size // 4,
        out_channel=init_size //4 * 5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[spatial_shape[0], spatial_shape[1], spatial_shape[2]]
        ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=init_size //4 * 5,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag_nu, eps=1e-3, momentum=0.01, voxel_coord=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag_nu, eps=1e-3, momentum=0.01, voxel_coord=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=final_occ_size,
        fine_topk=15000,
        dual = unilidar,
        empty_idx= empty_idx_sk if empty_idx_nu == empty_idx_sk else False,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=[num_cls_nu, num_cls_sk] if unilidar else False,
        point_cloud_range=point_cloud_range,
    ),
    loss_bbox=dict(
        type='OccLoss',
        balance_cls_weight=False,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        dual=True,
        num_cls=[num_cls_nu, num_cls_sk] if unilidar else False,),
    empty_idx=empty_idx_sk if empty_idx_nu == empty_idx_sk else 0,
    spatial_shape=spatial_shape,
)


train_pipeline_sk = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        shift_height=coor_alignment_sk,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_sk,
        shift_coors=[0, 0, -0.2],),
    dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range if RPR else ori_point_cloud_range,
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'dataset_flag', 'voxel_semantic_mask'])
]
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0.1, 0.1, 0.1],
    # ),

test_pipeline_sk = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        shift_height=coor_alignment_sk,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_sk,
        shift_coors=[0, 0, -0.2],),
    dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range if RPR else ori_point_cloud_range, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'dataset_flag', 'voxel_semantic_mask', 'visible_mask'],meta_keys=['voxel_semantic_mask_path'])
]
# keys=['points', 'voxel',  'dataset_flag', 'pts_semantic_mask', 'voxel_semantic_mask', 'voxel_occ_mask','voxel_invalid']
test_config_sk=dict(
    type=dataset_type_sk,
    occ_root=occ_path_sk,
    data_root=data_root_sk,
    ann_file=val_ann_file_sk,
    pipeline=test_pipeline_sk,
    modality=input_modality_sk,
    classes=class_names_sk,
    occ_size=occ_size_sk,
    seg_label_mapping=labels_map_sk,
    pc_range=point_cloud_range,
    filter_empty_gt=True,
)

train_config_sk=dict(
        type=dataset_type_sk,
        data_root=data_root_sk,
        occ_root=occ_path_sk,
        ann_file=train_ann_file_sk,
        pipeline=train_pipeline_sk,
        modality=input_modality_sk,
        classes=class_names_sk,
        test_mode=False,
        occ_size=occ_size_sk,
        seg_label_mapping=labels_map_sk,
        pc_range=point_cloud_range,
        filter_empty_gt=True,
),

data_sk = dict(
    train=train_config_sk,
    val=test_config_sk,
    test=test_config_sk,
)

bda_aug_conf_nu = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

# train_pipeline_nu = [
#     dict(
#         type='LoadPointsFromFile_RPR',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         shift_height=coor_alignment_nu,
#         RPR=RPR,
#         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
#         dataset_flag=dataset_flag_nu,
#         shift_coors=[0, 0, -0.2],),
#     dict(type='LoadPointsFromMultiSweeps_RPR',
#         sweeps_num=10,
#         RPR=RPR,
#         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
#     dict(
#         type='LoadAnnotationsBEVDepth',
#         bda_aug_conf=bda_aug_conf_nu,
#         classes=class_names_nu,
#         input_modality=input_modality_nu),
#     dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path_nu, grid_size=occ_size_nu, use_vel=False,
#             unoccupied=empty_idx_nu, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=point_cloud_range, cal_visible=visible_mask),
#     dict(type='OccDefaultFormatBundle3D', class_names=class_names_nu),
#     dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag']),
# ]
#     # dict(type='cylinder_voxelize',rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
#     #     fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], grid_size = [512, 360, 32],
#     #     pc_range=point_cloud_range),
# test_pipeline_nu = [
#     dict(
#         type='LoadPointsFromFile_RPR',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         shift_height=coor_alignment_nu,
#         RPR=RPR,
#         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
#         dataset_flag=dataset_flag_nu,
#         shift_coors=[0, 0, -0.2],),
#     dict(type='LoadPointsFromMultiSweeps_RPR',
#         sweeps_num=10,
#         RPR=RPR,
#         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
#     dict(
#         type='LoadAnnotationsBEVDepth',
#         bda_aug_conf=bda_aug_conf_nu,
#         classes=class_names_nu,
#         input_modality=input_modality_nu,
#         is_train=False),
#     dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path_nu, grid_size=occ_size_nu, use_vel=False,
#             unoccupied=empty_idx_nu, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=point_cloud_range, cal_visible=visible_mask),
#     dict(type='OccDefaultFormatBundle3D', class_names=class_names_nu, with_label=False), 
#     dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag', 'visible_mask'],meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
# ]


train_pipeline_nu = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment_nu,
        RPR=RPR,
        point_cloud_range=pcr_nu if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
    dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=RPR,
        point_cloud_range=pcr_nu if RPR else ori_point_cloud_range,),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf_nu,
        classes=class_names_nu,
        input_modality=input_modality_nu),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path_nu, grid_size=occ_size_nu, use_vel=False,
            unoccupied=empty_idx_nu, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=pcr_nu, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names_nu),
    dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag']),
]
    # dict(type='cylinder_voxelize',rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
    #     fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], grid_size = [512, 360, 32],
    #     pc_range=point_cloud_range),
test_pipeline_nu = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment_nu,
        RPR=RPR,
        point_cloud_range=pcr_nu if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
    dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=RPR,
        point_cloud_range=pcr_nu if RPR else ori_point_cloud_range,),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf_nu,
        classes=class_names_nu,
        input_modality=input_modality_nu,
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path_nu, grid_size=occ_size_nu, use_vel=False,
            unoccupied=empty_idx_nu, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=pcr_nu, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names_nu, with_label=False), 
    dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag', 'visible_mask'],meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]
# keys=['gt_occ', 'points', 'dataset_flag',  'lidar2ego_rotation', 'lidar2ego_translation']
            # meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']
    # dict(type='cylinder_voxelize',rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
    #     fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], grid_size = [512, 360, 32],
    #     pc_range=point_cloud_range),

test_config_nu=dict(
    type=dataset_type_nu,
    occ_root=occ_path_nu,
    data_root=data_root_nu,
    ann_file=val_ann_file_nu,
    pipeline=test_pipeline_nu,
    classes=class_names_nu,
    modality=input_modality_nu,
    occ_size=occ_size_nu,
    pc_range=point_cloud_range,
)

train_config_nu=dict(
        type=dataset_type_nu,
        data_root=data_root_nu,
        occ_root=occ_path_nu,
        ann_file=train_ann_file_nu,
        pipeline=train_pipeline_nu,
        classes=class_names_nu,
        modality=input_modality_nu,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size_nu,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR'),

data_nu = dict(
    train=train_config_nu,
    val=test_config_nu,
    test=test_config_nu,
)
data = data_nu

data_merge = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    shuffler_sampler=dict(type='BalancedDistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=15)

evaluation_sk = dict(
    interval=1,
    pipeline=test_pipeline_sk,
    save_best='SSC_mean',
    rule='greater',
)
evaluation_nu = dict(
    interval=1,
    pipeline=test_pipeline_nu,
    save_best='SSC_mean',
    rule='greater',
)
evaluation = evaluation_nu

load_from = 'ckpts/unilidar_merged_bcw_RPRmin_nutr_bbnorm_12812816_nu_36.4_19.0.pth'
work_dir = './work_dirs/unilidar_merged_bcw_RPRmin(nutr)_bbnorm_12812816'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mm_UniLiDAR',
                name='unilidar_merged_bcw_RPRmin(nutr)_bbnorm_12812816',
                job_type='training',
                notes='the-first-test',
            )
        )
    ])

checkpoint_config = dict(interval=2)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

