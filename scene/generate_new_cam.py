import numpy as np
import torch
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2
import matplotlib.pyplot as plt

def build_rot_y(th):
    return np.array([
    [np.cos(th), 0, -np.sin(th)],
    [0, 1, 0],
    [np.sin(th), 0, np.cos(th)]])

# def gen_new_cam_sai(cam, degree, center):
#     uid = cam.uid + 0.01*degree
#     theta = degree *np.pi/180 
#     cam_focus = np.array(torch.tensor(cam.T) - center)
#     rot_y = build_rot_y(theta)
#     #R_n = rot_y.numpy @ cam.R #they store the transpose
#     t_n = rot_y @ cam_focus + np.array(center)

#     E_ref = torch.Tensor(getWorld2View2(cam.R, cam.T))
    
#     # ref_img = torch.Tensor(cam.original_image).cpu()
#     # ref_depth = torch.Tensor(cam.depth)
#     # K_ref = torch.Tensor(cam.cam_intr)

#     new_look_at = np.array(center) - t_n
#     new_look_at = new_look_at / np.linalg.norm(new_look_at)
#     new_right = rot_y @ cam.R[:,0]
#     new_right = new_right / np.linalg.norm(new_right)
#     new_up = np.cross(new_look_at, new_right)

#     R_n = np.stack((new_right, new_up, new_look_at), axis=1)
#     R_n = cam.R @ rot_y 
#     E_n = torch.Tensor(getWorld2View2(R_n, t_n))

#     return Camera(colmap_id= None, R=R_n, T=t_n , FoVx=cam.FoVx, FoVy=cam.FoVy,  
#         image=cam.original_image,  vit_cam = True, vit_feature = cam.vit_feature,
#         image_name=None, uid=uid, gt_alpha_mask = None,
#             data_device = "cuda"
#         )

def gen_new_cam(cam, degree, rot_axis = 'y'):
    uid = cam.uid + 0.01*degree
    theta = degree *np.pi/180 

    old_R = cam.R.astype(np.float32)
    old_T = cam.T.astype(np.float32)


    #Generate new camera params
    image_name = cam.image_name + '_' + str(0.01*degree) + rot_axis
    if rot_axis == 'y':
        uid = cam.uid + 0.01*degree
    elif rot_axis == 'x':
        uid = cam.uid + 0.001*degree
    else:
        uid = cam.uid + 0.0001*degree
    colmap_id = cam.colmap_id + 0.01*degree


    R_n, t_n = gen_rotation_extr(old_R, old_T, degree = degree, rot_axis = rot_axis)

    return Camera(colmap_id= None, R=R_n, T=t_n , FoVx=cam.FoVx, FoVy=cam.FoVy,  
        image=cam.original_image,  ft_cam = True, vit_feature = cam.vit_feature,
        image_name=None, uid=uid, gt_alpha_mask = None,
            data_device = "cuda"
        )

def gen_rotation_extr(R, T, degree = 1, rot_axis = 'y'):
    '''
    This function takes in camera extrinsics and an angle
    Return the new extrinsics 
    Not Spherical Warp
    '''
    th = degree*np.pi/180 
    if rot_axis == 'y':
        rot_mat =  [
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]]
    elif rot_axis == 'x':
        rot_mat = [
        [1, 0, 0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th), np.cos(th)]]
    elif rot_axis == 'z':
        rot_mat = [
        [np.cos(th), -np.sin(th), 0],
        [np.sin(th), np.cos(th), 0],
        [0, 1, 0]]
    
    new_R = np.matmul(rot_mat ,R)
    new_T = np.matmul(rot_mat ,T)
    return new_R, new_T

def gen_new_cam_T(cam, distance, T_axis = 'y'):
    uid = cam.uid + 0.01*distance
    image_name = cam.image_name + '_' + str(0.01*distance) + T_axis 
    old_T = cam.T.astype(np.float32)

    #Generate new camera params
    if T_axis == 'y':
        uid = cam.uid + 0.01*distance
    elif T_axis == 'x':
        uid = cam.uid + 0.001*distance
    else:
        uid = cam.uid + 0.0001*distance

    t_n = gen_translation_extr(old_T, distance = distance, T_axis = T_axis)
    

    #
    fx = 3222.7010797592447
    fy = 3222.7010797592447
    R_ref = cam.R
    T_ref = old_T.reshape((3,1))
    R_src = cam.R
    T_src = t_n.reshape((3,1))
    data = cam.original_image
    depth_ref =cam.gt_depth * 255
    # print('ASDASFW',depth_ref)


    _, height, width = data.shape
    c_x = (width) / 2
    c_y = (height) / 2
    K = np.array([
        [fx, 0.0, c_x],
        [0, fy, c_y],
        [0, 0, 1.0]
        ])

    intrinsics_ref = K.copy()
    intrinsics_src = K.copy()
    extrinsics_ref = torch.Tensor(np.vstack((np.hstack((R_ref,T_ref)),[0.0 ,0.0, 0.0, 1.0])))
    extrinsics_src = torch.Tensor(np.vstack((np.hstack((R_src,T_src)),[0.0 ,0.0, 0.0, 1.0])))
    # print('NNNNN',str(uid))
    # print('XXXXX',extrinsics_ref)
    # print('YYYYY',extrinsics_src)

    depth_ref = torch.from_numpy(depth_ref).float().unsqueeze(0)
    intrinsics_ref = torch.from_numpy(intrinsics_ref).float().unsqueeze(0)
    intrinsics_src = torch.from_numpy(intrinsics_src).float().unsqueeze(0)
    extrinsics_ref = extrinsics_ref.float().unsqueeze(0)
    extrinsics_src = extrinsics_src.float().unsqueeze(0)
    data = data.cpu().detach().float().unsqueeze(0)
    new_img, new_dp, dp_mask = forward_warp(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)

    
    return Camera(colmap_id= None, R=cam.R, T=t_n , FoVx=cam.FoVx, FoVy=cam.FoVy,  
        image=torch.from_numpy(new_img).permute(2,0,1).to('cuda'),  ft_cam = True, vit_feature = cam.vit_feature,
        image_name=image_name, uid=uid, gt_alpha_mask = None, gt_depth = new_dp.to('cuda')/255, depth_mask = dp_mask.unsqueeze(0).to('cuda'),
            data_device = "cuda"
        )

def gen_translation_extr(T, distance = 1.0, T_axis = 'y'):
    '''
    This function takes in camera extrinsics and a distance
    Return the new extrinsics 
    Translation Only
    '''
    if T_axis == 'y':
        new_T = T + np.array([0.0 , distance, 0.0])
    elif T_axis == 'x':
        new_T = T + np.array([distance , 0.0, 0.0])
    elif T_axis == 'z':
        new_T = T + np.array([0.0 , 0.0, distance])

    return new_T

def calculate_average_up_vector(rotation_matrices):
    up_vectors = rotation_matrices[:, :, 1]  # Assuming the up vector is the second column
    average_up_vector = np.mean(up_vectors, axis=0)
    normalized_average_up_vector = average_up_vector / np.linalg.norm(average_up_vector)
    return normalized_average_up_vector

def construct_rotation_matrix(axis, theta):
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0) #  Rodrigues' rotation formula
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotate_camera_poses(axis, rotation_matrices, translation_vectors, theta):
    rotation_matrix_avg = construct_rotation_matrix(axis, theta)  # Rotate around the avg y-axis
    new_rotation_matrices = np.matmul(rotation_matrix_avg, rotation_matrices)
    new_translation_vectors = translation_vectors
    return new_rotation_matrices, new_translation_vectors

def create_cam_obj(cam, degree, R, T):
    uid = cam.uid + 0.01*degree

    return Camera(colmap_id= None, R=R, T=T , FoVx=cam.FoVx, FoVy=cam.FoVy,  
        image=cam.original_image, depth=None,
        image_name=None, uid=uid, gt_alpha_mask = None,
            data_device = "cuda"
        )

def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
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


def forward_warp(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    data = data[0].permute(1, 2, 0)
    new = np.zeros_like(data)
    depth_src = depth_src.reshape(height, width)
    new_depth = np.zeros_like(depth_src)
    yy_base, xx_base = torch.meshgrid([torch.arange(
        0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long)])
    # y_res = np.clip(y_res.numpy(), 0, width - 1).astype(np.int64)
    # x_res = np.clip(x_res.numpy(), 0, height - 1).astype(np.int64)
    y_res = np.clip(y_res.numpy(), 0, height - 1).astype(np.int64)
    x_res = np.clip(x_res.numpy(), 0,  width - 1).astype(np.int64)
    # print(x_res.min)
    new[y_res, x_res] = data[yy_base, xx_base]
    new_depth[y_res, x_res] = depth_src[yy_base, xx_base]
    # print(new_depth)
    new_depth = torch.from_numpy(new_depth)
    one = torch.ones_like(new_depth)
    depth_mask = torch.zeros_like(new_depth)
    depth_mask[y_res, x_res] = one[yy_base, xx_base]
    return new, new_depth, depth_mask