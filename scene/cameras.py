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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2dist
from utils.general_utils import PILtoTorch, NP_resize
from copy import deepcopy
from scipy.interpolate import griddata
from torchvision import transforms

import matplotlib.pyplot as plt


def loadWarpCam(cam, img, warp_depth, mask, R_n, T_n):
    
    img = torch.Tensor(img).cuda().permute(2,0,1)
    c = Camera(colmap_id=None, R=R_n, T=T_n, 
                  FoVx=cam.FoVx, FoVy=cam.FoVy, 
                  image=img, depth=cam.depth, cam_intr=cam.cam_intr, gt_alpha_mask=None,
                  image_name=cam.image_name, uid=None, data_device='cuda')
    #c = deepcopy(cam)
    #c.original_image = warp_image.detach().cuda()
    #c.R = R_n
    #c.T = T_n
    #convert below tensor to boolean
    c.warp_mask = (torch.Tensor(mask) > 0).unsqueeze(0).detach().cuda()
    c.warp_depth = torch.Tensor(warp_depth).detach().cuda()

    return c



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, cam_intr, gt_alpha_mask,
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
        self.cam_intr = cam_intr
        self.warp_depth = None

        self.w_image=None
        self.w_depth=None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
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
        self.warp_mask = None
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    
    def build_rot_y(self, th):
        return np.array([
        [np.cos(th), 0, -np.sin(th)],
        [0, 1, 0],
        [np.sin(th), 0, np.cos(th)]])
    
    def project_with_depth(self, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
        width, height = depth_ref.shape[2], depth_ref.shape[1]
        batchsize = depth_ref.shape[0]

        y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                    torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
        y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
        y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

        xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), torch.stack(
            (x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * (depth_ref.view(batchsize, -1).unsqueeze(1)))

   
        xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                            torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]
        # print(xyz_src.shape)  B*3*20480

        K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480
        depth_src = K_xyz_src[:, 2:3, :]
        xy_src = K_xyz_src[:, :2, :] / K_xyz_src[:, 2:3, :]
        x_src = xy_src[:, 0, :].view([batchsize, height, width])
        y_src = xy_src[:, 1, :].view([batchsize, height, width])
        # print(x_src.shape) #B*128*160

        return x_src, y_src, depth_src
    # (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')


    def forward_warp(self, data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
        x_res, y_res, depth_src = self.project_with_depth(
            depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
        width, height = depth_ref.shape[2], depth_ref.shape[1]
        batchsize = depth_ref.shape[0]
        data = data[0].permute(1, 2, 0)
        new = np.zeros_like(data)
        depth_src = depth_src.reshape(height, width)
        new_depth = np.zeros_like(depth_src)
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long)])
        y_res = np.clip(y_res.numpy(), 0, height - 1).round().astype(np.int64)
        x_res = np.clip(x_res.numpy(), 0, width - 1).round().astype(np.int64)

        mask = np.zeros_like(depth_ref.squeeze(0))
        mask[y_res, x_res] = 1

        sort_idxs = depth_ref.flatten().argsort(descending=True)

        #orig_shape = y_res.shape
        y_res = y_res.flatten()#[sort_idxs].reshape(orig_shape)
        x_res = x_res.flatten()#[sort_idxs].reshape(orig_shape)
        yy_base = yy_base.flatten()#[sort_idxs].reshape(orig_shape)
        xx_base = xx_base.flatten()#[sort_idxs].reshape(orig_shape)

        #new = griddata(np.stack((x_res, y_res) , dim=1), data, (xx_base, yy_base), method='cubic')

        
        for i in range(yy_base.shape[0]):
            if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]
        
        #new[y_res, x_res] = data[yy_base, xx_base]
        #new_depth[y_res, x_res] = depth_src[yy_base, xx_base]
        #depth_mask=None
        #one = torch.ones_like(new_depth)
        #depth_mask = torch.zeros_like(new_depth)
        #depth_mask[y_res, x_res] = one[yy_base, xx_base]
        return new, new_depth, mask
    
    def translation_warp(self, lam, idx):
        #width = focal2dist(self.cam_intr[0,0], self.FoVx)
        #print(self.R)
        t_n = torch.Tensor(self.T + lam*self.R[:,idx])
        E_ref = torch.Tensor(getWorld2View2(self.R, self.T))
        E_n = torch.Tensor(getWorld2View2(self.R, t_n))
        ref_img = torch.Tensor(self.original_image).cpu()
        ref_depth = torch.Tensor(self.depth)
        K_ref = torch.Tensor(self.cam_intr)

        img, depth, mask = self.forward_warp(ref_img.cpu().clone().detach().unsqueeze(0), ref_depth.clone().detach().unsqueeze(0), K_ref.unsqueeze(0), E_ref.unsqueeze(0), K_ref.unsqueeze(0), E_n.unsqueeze(0))

        
        plt.subplot(1,4,1)
        plt.imshow(ref_img.numpy().transpose(1,2,0))
        plt.subplot(1,4,2)
        plt.imshow(img)
        plt.subplot(1,4,3)
        plt.imshow(self.depth)
        plt.subplot(1,4,4)
        plt.imshow(mask)
        plt.show()
        

        return loadWarpCam(self, img, mask, R_n, t_n)

    def gen_rotation_extr(self, theta, center):
        cam_focus = self.T - center
        rot_y = self.build_rot_y(theta)
        #R_n = rot_y.numpy @ self.R #they store the transpose
        t_n = rot_y @ cam_focus + center

        E_ref = torch.Tensor(getWorld2View2(self.R, self.T))
        
        scale = 4
        ref_img = torch.Tensor((self.w_image)).cpu()
        ref_depth = torch.Tensor(self.w_depth)        
        K_ref = torch.Tensor(self.cam_intr)

        # new_look_at = center - t_n
        # new_look_at = new_look_at / np.linalg.norm(new_look_at)
        # new_right = rot_y @ self.R[:,0]
        # new_right = new_right / np.linalg.norm(new_right)
        # new_up = np.cross(new_look_at, new_right)

        # R_n = np.stack((new_right, new_up, new_look_at), axis=1)
        R_n = self.R @ rot_y
        E_n = torch.Tensor(getWorld2View2(self.R, t_n))


        img, depth, mask = self.forward_warp(ref_img.cpu().clone().detach().unsqueeze(0), ref_depth.clone().detach().unsqueeze(0), K_ref.unsqueeze(0), E_ref.unsqueeze(0), K_ref.unsqueeze(0), E_n.unsqueeze(0))

        
        # plt.subplot(1,4,1)
        # plt.imshow(ref_img.numpy().transpose(1,2,0))
        # plt.subplot(1,4,2)
        # plt.imshow(img)
        # plt.subplot(1,4,3)
        # plt.imshow(self.depth)
        # plt.subplot(1,4,4)
        # plt.imshow(depth)
        # plt.show()
        
        return loadWarpCam(self, img, depth, mask, R_n, t_n)
    
    


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

