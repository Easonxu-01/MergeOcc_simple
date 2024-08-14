# MergeOcc: Bridge the domain gap between different LiDARs for robust occupancy prediction

# Abstract

LiDAR-based 3D occupancy prediction evolved rapidly alongside the emergence of large datasets. 
Nevertheless, the potential of existing diverse datasets remains underutilized as they kick in individually. Models trained on a specific dataset often suffer considerable performance degradation when deployed to real-world scenarios or datasets involving disparate LiDARs.
This paper aims to develop a generalized model called MergeOcc, to simultaneously handle different LiDARs by leveraging multiple datasets.
The gaps among LiDAR datasets primarily manifest in geometric disparities and semantic inconsistencies. Thus, MergeOcc incorporates a novel model featuring a geometric realignment module and a semantic label mapping module to enable multiple datasets training (MDT). 
The effectiveness of MergeOcc is validated through experiments on two prominent datasets for autonomous vehicles: OpenOccupancy-nuScenes and SemanticKITTI. The results demonstrate its enhanced robustness and remarkable performance across both types of LiDARs, outperforming several SOTA multi-modality methods. Notably, despite using an identical model architecture and hyper-parameter set, MergeOcc can significantly surpass the baseline due to its exposure to more diverse data.MergeOcc is considered the first cross-dataset 3D occupancy prediction pipeline that effectively bridges the domain gap for seamless deployment across heterogeneous platforms.

# Getting Started

- [Installation](docs/install.md) 

- [Prepare Dataset](docs/prepare_data.md)

- [Training, Evaluation, Visualization](docs/trainval.md)

# **overview  framework**

![framework](./framework.png)

# Primary Results

![IoUDrop_d](./IoUDrop_d.png)



#  Acknowledgement

Many thanks to these excellent projects:
- [OpenOcuupancy](https://github.com/JeffWang987/OpenOccupancy)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [Unidet](https://github.com/xingyizhou/UniDet)
