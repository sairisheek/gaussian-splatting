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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, normalize
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import matplotlib.pyplot as plt
import cv2

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    #num_gauss_path = os.path.join(model_path, name, "ours_{}".format(iteration), "num_gauss")
    #accum_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "accum_alpha")
    modes_path = os.path.join(model_path, name, "ours_{}".format(iteration), "modes")
    #var_path = os.path.join(model_path, name, "ours_{}".format(iteration), "var")
    #raw_modes_path = os.path.join(model_path, name, "ours_{}".format(iteration), "raw_modes")
    #raw_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "raw_depth")
    #monodepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "monodepth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    #makedirs(num_gauss_path, exist_ok=True)
    #makedirs(accum_alpha_path, exist_ok=True)
    makedirs(modes_path, exist_ok=True)
    #makedirs(var_path, exist_ok=True)
    #makedirs(raw_modes_path, exist_ok=True)
    #makedirs(raw_depth_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        print(view, view.image_name)
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        #loss = (results["render"] - results["render"].detach()).sum()
        #loss.backward()

        #var = normalize(results["var_loss"].grad)
        gt = view.original_image[0:3, :, :]
        depth = results["depth"]
  
        #num_gauss = (results["num_gauss"])/results["num_gauss"].max()
        #accum_alpha = results["accum_alpha"]
        #print(accum_alpha.max())
        #print(results["num_gauss"].mean(dtype=torch.float32))
        depth[(depth < 0)] = 0
        #np.save(os.path.join(raw_depth_path, view.image_name + ".npy"), depth.detach().cpu().numpy().squeeze())
        #np.save(os.path.join(raw_modes_path, view.image_name + ".npy"), results["modes"].detach().cpu().numpy().squeeze())
        depth = (depth / (depth.max() + 1e-5)).detach().cpu().numpy().squeeze()
        modes = normalize(results["modes"])
        #depth = (depth * 255).astype(np.uint8)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        #plt.imsave(os.path.join(num_gauss_path, view.image_name + ".png"), num_gauss.cpu().numpy().squeeze())
        plt.imsave(os.path.join(depth_path, view.image_name + ".png"), depth, cmap='jet')  
        #cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), (depth * 65535).astype(np.uint16))      
        #plt.imsave(os.path.join(accum_alpha_path, view.image_name + ".png"), accum_alpha.detach().cpu().numpy().squeeze())
        #plt.imsave(os.path.join(modes_path, '{0:05d}'.format(idx) + ".png"), modes.cpu().numpy().squeeze())
        #write mode as 16 bit png
        cv2.imwrite(os.path.join(modes_path, view.image_name + ".png"), (modes.detach().cpu().numpy().squeeze() * 65535).astype(np.uint16))
        #cv2.imwrite(os.path.join(var_path, view.image_name + ".png"), (var.detach().cpu().numpy().squeeze() * 65535).astype(np.uint16))
        #plt.imsave(os.path.join(var_path, view.image_name + ".png"), var.detach().cpu().numpy().squeeze(), cmap='jet')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)