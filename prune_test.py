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

import os
import numpy as np
import torch
from torchmetrics import PearsonCorrCoef
from kornia.losses import inverse_depth_smoothness_loss
import math

from icecream import ic
import clip_utils
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, LaplacianLayer, Sobel
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, normalize
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, step, max_cameras):
   
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, step=step, max_cameras=max_cameras)
    gaussians.training_setup(opt)
    #embed, target_embed  = setup_clip(scene.getTrainCameras())

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy() 
    ema_loss_for_log = 0.0
    
    
    with torch.no_grad():
        N = gaussians.get_opacity.shape[0]
        mask = torch.zeros(N, dtype=torch.bool, device="cuda")
        for view in viewpoint_stack:       
            render_pkg = render(view, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii, depth, mode_id, modes, point_list = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["mode_id"], render_pkg["modes"], render_pkg["point_list"]
            # Loss

            #gt_image = viewpoint_cam.original_image.cuda()
            #Ll1 = l1_loss(image, gt_image)
            #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            submask = torch.zeros_like(mask)
            pruned_modes_mask = normalize(modes) < dataset.prune_thresh # select points less than threshold
            prune_mode_ids = mode_id[:,pruned_modes_mask.squeeze()] # subselect the mode idxs
            neg_mask = (prune_mode_ids == -1).any(dim=0)
            prune_mode_ids = prune_mode_ids[:,~neg_mask]
            selected_gaussians = set()
            for i in range(prune_mode_ids.shape[-1]):
                selected_gaussians.update(point_list[prune_mode_ids[0,i]:prune_mode_ids[1,i]+1].tolist())
            submask[list(selected_gaussians)] = True
            mask = mask | submask

        num_points_pruned = mask.sum()
        print(f'Pruning {num_points_pruned} gaussians')
    #gaussians.prune_points(mask) #mask is P X 1
        gaussians.mark_red(mask)
        #gt_depth = (viewpoint_cam.depth - viewpoint_cam.depth.min())/ (viewpoint_cam.depth.max() - viewpoint_cam.depth.min())

        scene.save(8888)
       

       

        
                



def pearson_depth_loss(depth_src, depth_target, pearson):
    co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))
    return 1 - co

def second_smoothness_loss(depth, img, laplacian, gradient):
        img_lap = laplacian(img.unsqueeze(0), do_normalize=False)
        depth_grad_x, depth_grad_y = torch.tensor_split(gradient(depth.unsqueeze(0)), 2, dim=1)
        depth_grad_x2, depth_grad_xy = torch.tensor_split(gradient(depth_grad_x), 2, dim=1)
        depth_grad_yx, depth_grad_y2 = torch.tensor_split(gradient(depth_grad_y), 2, dim=1)
        
        x = torch.exp(-img_lap) * (depth_grad_x2.abs() \
            + depth_grad_xy.abs() + depth_grad_y2.abs())
        return x.mean()

def pad_image(image, patch_size):
    _, H, W = image.size()
    # Calculate the padding needed to make the dimensions multiples of the patch size
    pad_h = math.ceil(H / patch_size) * patch_size - H
    pad_w = math.ceil(W / patch_size) * patch_size - W
    # Apply zero padding to the image
    padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_image

def split_image_into_patches(image, patch_size):
    # Pad the image to make its dimensions multiples of the patch size
    padded_image = pad_image(image, patch_size).squeeze()

    # Get the dimensions of the padded image
    H, W = padded_image.size()

    # Use the unfold function to split the padded image into patches
    patches = padded_image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

    # Reshape the patches to have the shape (num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 3).contiguous().view(-1, patch_size, patch_size).view(-1, patch_size*patch_size)

    return patches

def depth_ranking_loss(depth_src, depth_target, margin, box_s, n_corr):

    src_pad = pad_image(depth_src, box_s).squeeze()
    target_pad = pad_image(depth_target, box_s).squeeze()
    
    #src_patch = split_image_into_patches(src_pad, box_s)
    #targ_patch = split_image_into_patches(target_pad, box_s)
    print(src_pad.shape)
    H, W = src_pad.shape

    (gridx, gridy) = torch.meshgrid(torch.arange(H), torch.arange(W))
    gridx = split_image_into_patches(gridx, box_s)
    gridy = split_image_into_patches(gridy, box_s)

    loss = 0.0
    idxs = torch.randint(box_s*box_s, size=(grid.shape[0], n_corr, 2))
    src_rand1 = src_pad[grid[:,idxs[:,:,0]]]
    src_rand2 = src_pad[grid[:,idxs[:,:,1]]]

    target_rand1 = target_pad[grid[:,idxs[:,:,0]]]
    target_rand2 = target_pad[grid[:,idxs[:,:,1]]]

    mask = target_rand1 > target_rand2 # get mask of pairs that are higher in first coordinate
    loss += torch.mean(torch.clamp(margin + src_rand2 - src_rand1, min=0.0)[mask])
    loss += torch.mean(torch.clamp(margin + src_rand1 - src_rand2, min=0.0)[~mask])
    return loss.mean()
    


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Tensorboard Found!")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, smoothness_loss, ranking_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        if depth_loss is not None:
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        if smoothness_loss is not None:
            tb_writer.add_scalar('train_loss_patches/smoothness_loss', smoothness_loss.item(), iteration)
        if ranking_loss is not None:
            tb_writer.add_scalar('train_loss_patches/ranking_loss', ranking_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_cameras", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.step, args.max_cameras)

    # All done
    print("\nTraining complete.")