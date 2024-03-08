import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
import argparse

# # 创建一个解析器
# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, default='/pred_f.npy')
# args = parser.parse_args()

colors_nu = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

colors_sk = np.array([
    [0  , 0  , 0, 255],
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255]]).astype(np.uint8)
# colors_sk[:, :3] = colors_sk[:, :3] // 3 * 2

#mlab.options.offscreen = True
RPR = False
voxel_size = 1.0
shift_nu = [-256, -256, -25]
shifr_sk = [0, -128, -17]
restrict_point_range = [0, -25.6, -3.4, 51.2, 25.6, 3]
nu_ori = [-51.2, -51.2, -5, 51.2, 51.2, 3]
sk_ori = [0, -25.6, -3.4, 51.2, 25.6, 3]
flag = 1
pc_range = nu_ori if flag==1 else sk_ori
visual_path = '/Users/eason/Downloads/uninu/0e7ede02718341558414865d5c604745/fd02aacd87ef4fdfbc0e24c359fb2642/pred_c.npy'
fov_voxels = np.load(visual_path)

fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
uni = np.unique(fov_voxels[:,3])
fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
fov_voxels[fov_voxels[:, 3]==255, 3] = 0
# 计算限定范围在processed_label中的索引范围
pc_range_min = np.array(pc_range[:3])
pc_range_max = np.array(pc_range[3:])
restrict_pc_range_min = np.array(restrict_point_range[:3])
restrict_pc_range_max = np.array(restrict_point_range[3:])

# 计算索引范围
start_index = ((restrict_pc_range_min - pc_range_min) / 0.2).astype(int)[::-1]
end_index = ((restrict_pc_range_max - pc_range_min) / 0.2).astype(int)[::-1]

if fov_voxels[:, 2].max() > 256 and RPR:
    filtered_fov_voxels = fov_voxels[(fov_voxels[:, 0] >= start_index[0]) & (fov_voxels[:, 0] <= end_index[0]) &
                   (fov_voxels[:, 1] >= start_index[1]) & (fov_voxels[:, 1] <= end_index[1]) &
                   (fov_voxels[:, 2] >= start_index[2]) & (fov_voxels[:, 2] <= end_index[2])]
else:
    filtered_fov_voxels = fov_voxels


figure = mlab.figure(size=(512, 512), bgcolor=(1, 1, 1))
# figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
# pdb.set_trace()
plt_plot_fov = mlab.points3d(
    filtered_fov_voxels[:, 0],
    filtered_fov_voxels[:, 1],
    filtered_fov_voxels[:, 2],
    filtered_fov_voxels[:, 3],
    colormap="viridis",
    scale_factor=voxel_size,
    mode="cube",
    opacity=1.0,
    vmin=0,
    vmax=19,
)


plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors_nu if flag==1 else colors_sk
mlab.view(azimuth=0, elevation=90, distance='auto', focalpoint='auto')

#mlab.savefig('temp/mayavi.png')
mlab.show()
