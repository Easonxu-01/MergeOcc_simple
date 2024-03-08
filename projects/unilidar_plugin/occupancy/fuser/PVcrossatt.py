'''
Author: EASON XU
Date: 2023-10-05 07:21:40
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2023-10-25 01:35:58
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/occupancy/fuser/PVcrossatt.py
'''
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import transforms3d as tfs
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.builder import FUSION_LAYERS



def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices

def generate_grid_3D(height: int, width: int, depth: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    zs = torch.linspace(0, 1, depth)

    indices = torch.stack(torch.meshgrid((xs, ys, zs), indexing='xy'), 0)       # 3 h w d
    indices = F.pad(indices, (0, 0, 0, 0, 0, 0, 0, 1), value=1)                  # 4 h w d
    indices = indices[None]                                                      # 1 4 h w d 

    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


    
class CrossAttention3D(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b c H W D)
        k: (b c h w d)
        v: (b c h w d)
        """
        _, _, H, W, D = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b c H W D -> b (H W D) c').contiguous()
        k = rearrange(k, 'b c h w d -> b (h w d) c').contiguous()
        v = rearrange(v, 'b c h w d -> b (h w d) c').contiguous()

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head).contiguous()
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head).contiguous()
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head).contiguous()

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        # dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W D -> b (H W D) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W D) c -> b c H W D', H=H, W=W, D=D).contiguous()

        return z

@FUSION_LAYERS.register_module()
class PVcrossattention(nn.Module):
    def __init__(
            self,
            num_beams,
            start_vertical_angle,
            end_vertical_angle,
            num_beam_points,
            feat_dim: int,
            dim: int,
            spatial_shape,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            skip: bool = True,
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        
        # 1 4 h w d
        spatial_mesh = generate_grid_3D(self.spatial_shape[0], self.spatial_shape[1], self.spatial_shape[2])
        spatial_mesh[:, :, 0] *= self.spatial_shape[0]
        spatial_mesh[:, :, 1] *= self.spatial_shape[1]
        spatial_mesh[:, :, 2] *= self.spatial_shape[2]
        
        self.register_buffer('spatial_mesh', spatial_mesh, persistent=False)

        self.voxel_feature_linear = nn.Sequential(
            nn.BatchNorm3d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv3d(feat_dim, 2 * dim, 1, bias=False))
        
        self.pts_feature_proj = nn.Sequential(
            nn.BatchNorm3d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv3d(feat_dim, 2 * dim, 1, bias=False))
        
        # self.feature_proj = nn.Sequential( 
        #     nn.BatchNorm2d(feat_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(feat_dim, dim, 1, bias=False))
        
        self.Lidar_ori_embed = nn.Sequential(nn.Linear(num_beams, dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(dim, dim))
        self.lidar_embed = nn.Sequential(nn.Linear(4, dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(dim, dim))

        # 4 num_beams
        self.lidar_ori= torch.cat([torch.linspace(0, num_beams-1, num_beams).view(-1, num_beams),
                                   torch.linspace(start_vertical_angle, end_vertical_angle, num_beams).view(-1, num_beams), 
                                   torch.linspace(0, num_beams-1, num_beams).view(-1, num_beams),
                                   torch.linspace(num_beam_points, (num_beams-1) * num_beam_points, num_beams).view(-1, num_beams),], dim=0)
        self.cross_attend = CrossAttention3D(feat_dim, heads, dim_head, qkv_bias)
        self.skip = skip
        

    def forward(
        self,
        voxel_feature,
        pts_feature,
        lidar2ego_rotation,
        lidar2ego_translation,
    ):
        """
        voxel_feature: (b, c, H, W, D)
        pts_feature: (num_points, channels)
        lidar2ego_rotation: [bs tensors of (1*4)]
        lidar2ego_translation: (bs tensors of (1*3))

        Returns: (b, c, H, W, D)
        """
        x = voxel_feature.contiguous()
        B, _, H, W, D = voxel_feature.shape
        mesh = self.spatial_mesh                                               # 1 1 4 h w d
        _, _, h, w, d= mesh.shape
        assert H==h and W==w and D==d, "Voxel_feature's spatial shape not match"  
        #To Tensor
        if isinstance(lidar2ego_rotation, list) and isinstance(lidar2ego_translation, list):
            lidar2ego_rotation = torch.stack(lidar2ego_rotation, dim=1).view(B, -1, 4).contiguous().detach().cpu().numpy() #bs 1 4
            lidar2ego_translation = torch.stack(lidar2ego_translation, dim=1).view(B, -1, 3).contiguous().detach().cpu() #bs 1 3
        else:
            lidar2ego_rotation = lidar2ego_rotation.view(B, -1, 4).contiguous().detach().cpu().numpy()
            lidar2ego_translation = lidar2ego_translation.view(B, -1, 3).contiguous().detach().cpu()
        # quaternion to rotation matrix 
        q = np.array([lidar2ego_rotation[:, 0, 0], lidar2ego_rotation[:, 0, 1], lidar2ego_rotation[:, 0, 2], lidar2ego_rotation[:, 0, 3]]).reshape(B,-1,4)
        R = torch.zeros((B,3,3))
        for i in range(B):
            R[i,...] = torch.from_numpy(tfs.quaternions.quat2mat(q[i,...].squeeze())).float()
        # translation to #bs 3 1
        R = R.cuda()
        T = lidar2ego_translation.view(B, 3, 1).cuda()
        #concate
        RT = torch.cat([R, T], dim=2) # bs 3 4  
        P = torch.tensor([0, 0, 0, 1], dtype=torch.float32).view(1, 1, 4).cuda()
        P = P.repeat(B, 1, 1) # bs 1 4
        assert RT.shape[0] == P.shape[0] and RT.shape[0] == B, "RT and Batch_size's shape not match"
        # P[:, :3, 3] = RT[:, :, 3]
        L_E = torch.cat([RT, P], dim=1).to(torch.float32).cuda().contiguous() # bs 4 4
            
        L = L_E[..., -1:]                                                      # b 4 1
        L_flat = L.permute(0,2,1).repeat(1, 4, 1).contiguous()                              # b 4 4

        mesh_flat = rearrange(mesh, '... h w d -> ... (h w d)')                # 1 4 (h w d)
        mesh_flat = mesh_flat.repeat(B, 1, 1).to(torch.float32).contiguous().cuda()                # b 4 (h w d)
        
        L_flat = torch.einsum('b n c, b c s -> b n s', L_flat, mesh_flat)      # b 4 (h w d)
        L_flat = L_flat.permute(0,2,1).to(torch.float32).contiguous()                                          # b (h w d) 4
        L_embed = self.lidar_embed(L_flat)                                     # b (h w d) d
        L_embed = rearrange(L_embed, 'b (h w d) c -> b c h w d', h=h, w=w, d=d) # b d h w d
        
        
        Lidar = torch.einsum('b n c, b c s -> b n s', L_E, mesh_flat)          # b 4 (h w d)
        Lidar = Lidar.permute(0,2,1).to(torch.float32).contiguous()                                           # b (h w d) 4
        Lidar_embed = self.lidar_embed(Lidar)                                  # b (h w d) d
        Lidar_embed = rearrange(Lidar_embed, 'b (h w d) c -> b c h w d', h=h, w=w, d=d) # b d h w d 
        
        Lidar_ori = self.lidar_ori[None]                                          # 1 4 num_beams
        Lidar_ori = self.lidar_ori.repeat(B, 1, 1).to(torch.float32).cuda()                            # b 4 num_beams
        Lidar_ori = torch.einsum('b i c, b i s -> b c s', Lidar_ori, mesh_flat)   # b num_beams (h w d)
        Lidar_ori = Lidar_ori.permute(0,2,1).to(torch.float32).contiguous()                                      # b (h w d) num_beams
        Lidar_ori_embed = self.Lidar_ori_embed(Lidar_ori)                           #b (h w d) d
        Lidar_ori_embed = rearrange(Lidar_ori_embed, 'b (h w d) c -> b c h w d', h=h, w=w, d=d) # b d h w d
        
        Lidar_embed = Lidar_embed - L_embed                                             # b d h w d
        Lidar_embed = torch.concat((Lidar_embed, Lidar_ori_embed), dim=1)               # b 2d h w d
        Lidar_embed = Lidar_embed / (Lidar_embed.norm(dim=1, keepdim=True) + 1e-7)      # b 2d h w d 
        Lidar_embed = Lidar_embed.contiguous()

        voxel_feature = self.voxel_feature_linear(voxel_feature)                            # b 2d H W D
        voxel_embed = torch.concat((voxel_feature,Lidar_embed),dim=1)                      # b 4d H W D
        query_pos = voxel_embed / (voxel_embed.norm(dim=1, keepdim=True) + 1e-7)           # b 4d H W D
        
        
        key_proj = self.pts_feature_proj(pts_feature)                                      # b 2d H W D
        val_proj = self.pts_feature_proj(pts_feature)                                      # b 2d H W D
        query = query_pos.contiguous()                                                                  # b 4d d H W
        key = torch.concat((key_proj,Lidar_embed),dim=1).contiguous()                                   # b 4d d h w
        val = torch.concat((val_proj,Lidar_embed),dim=1).contiguous()                                   # b 4d d h w
        
        return self.cross_attend(query, key, val, skip=x if self.skip else None)
