#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def build_rot_y(th):
        return torch.Tensor([
        [np.cos(th), 0, -np.sin(th)],
        [0, 1, 0],
        [np.sin(th), 0, np.cos(th)]])

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.depth = depth
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        def gen_hemisphere_view(self, theta, center):
            cam_focus = torch.Tensor(self.T) - center
            rot_y = build_rot_y(theta)
            t_n = torch.Tensor(rot_y @ cam_focus + center)

            E_ref = torch.Tensor(getWorld2View2(self.R, self.T))
            E_n = torch.Tensor(getWorld2View2(self.R, t_n))
            ref_img = torch.Tensor(self.original_image).cpu()
            ref_depth = torch.Tensor(self.depth)
            K_ref = torch.Tensor(self.cam_intr)

            new_look_at = center - t_n
            new_look_at = new_look_at / torch.linalg.norm(new_look_at)
            new_right = rot_y @ self.R[:,1]
            new_right = new_right / torch.linalg.norm(new_right)
            new_up = torch.cross(new_look_at.float(), new_right.float())

            R_n = torch.stack((new_right, new_up, new_look_at), dim=1)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

