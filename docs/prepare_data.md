# Prepare nuScenes-Occupancy

**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```

**2. Download the generated [train](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl)/[val](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) pickle files and put them in data. Folder structure:**

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl/
│   │   ├── nuscenes_occ_infos_val.pkl/
```

**2. Pre-compute depth map for fast training (depth-aware view transform module, same logic as [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)):**
```
python ./tools/gen_data/gen_depth_gt.py
```
**Folder structure:**
```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   ├── depth_gt/
```

**3. Download and unzip our annotation for nuScenes-Occupancy:**

| Subset | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Size |
| :---: | :---: | :---: | :---: |
| trainval-v0.1 | [link](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing) | [link](https://pan.baidu.com/s/1Wu1EYa7vrh8KS8VPTIny5Q) (code:25ue) | approx. 5G |

```
mv nuScenes-Occupancy-v0.1.7z ./data
cd ./data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
```
**Folder structure:**

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   ├── depth_gt/
│   ├── nuScenes-Occupancy/
```

# Prepare SemanticKITTI

**1. Download the semanticKITTI dataset [HERE](https://http://www.semantic-kitti.org/dataset.html), Folder structure:*

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   ├── depth_gt/
│   ├── nuScenes-Occupancy/
│   ├── semantickitti/
|   │   ├──sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── voxels/ 
            |       ├── 000000.label
            |       ├── 000000.occluded
            |       ├── 000000.invalid
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
          └── ...
```

**2. Download the generated [train](https://drive.google.com/file/d/1AlbseAbUkBrVjEZTbDbYcsA3l6LqqTEO/view?usp=drive_link)/[val](https://drive.google.com/file/d/1gF7rHdZqzcu2mwjzflKF5jcE-wOue0CI/view?usp=drive_link) /[test](https://drive.google.com/file/d/1InCnqx2oIKxIB9Kjb89RPcLood-a__2q/view?usp=drive_link) pickle files and put them in data. Folder structure:**

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   ├── depth_gt/
│   ├── nuScenes-Occupancy/
│   ├── semantickitti/
│   │   ├── semantickitti_infos_train.pkl/
│   │   ├── semantickitti_infos_val.pkl/
│   │   ├── semantickitti_infos_test.pkl/
```

**Folder structure:**

```
mergeocc_simple
├── data/
│   ├── nuscenes/
│   ├── depth_gt/
│   ├── nuScenes-Occupancy/
│   ├── semantickitti/
```

