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
import os.path as osp
import sys
import torch
from tqdm import tqdm
import numpy as np
import yaml
from argparse import ArgumentParser
import nibabel as nib

sys.path.append("./")
from digs.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)
from digs.dataset import Scene
from digs.gaussian import GaussianModel, render, query, initialize_gaussian, query_with_motion
from digs.utils.general_utils import safe_state, t2a
from digs.utils.image_utils import metric_vol, metric_proj
from digs.utils.ct_utils import get_geometry_rtk, getFOVMask


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
    axis: list = None,
    slices: list = None,
    nt=100
):
    # Set up dataset
    scene = Scene(
        dataset,
        shuffle=False,
    )

    # Set up Gaussians
    gaussians = GaussianModel(None)  # scale_bound will be loaded later
    gaussians.setup_v_grid(scene.scanner_cfg,dataset,pipeline.unified, ~pipeline.no_bspline)
    gaussians.setup_t_grid(len(scene.getTrainCameras()),dataset)
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians

    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    if not skip_render_train:
        evaluate_render(
            save_path,
            "render_train",
            scene.getTrainCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate projection test
    if not skip_render_test:
        evaluate_render(
            save_path,
            "render_test",
            scene.getTestCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate volume reconstruction
    if not skip_recon:
        train_cameras = scene.getTrainCameras()
        projs_train = np.concatenate(
            [t2a(cam.original_image) for cam in train_cameras], axis=0
        )
        angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
        geo = get_geometry_rtk(scene.scanner_cfg,angles_train)
        mask = getFOVMask(projs_train, scene.scanner_cfg, geo)
        mask = torch.from_numpy(mask).cuda()
        evaluate_volume(
            save_path,
            "reconstruction",
            scene.scanner_cfg,
            scene.scene_scale,
            gaussians,
            pipeline,
            scene.vol_gt,
            mask
        )
        # Evaluate motion
        if axis is not None and slices is not None:
            evaluate_motion(
                save_path,
                "motion",
                scene.scanner_cfg,
                gaussians,
                pipeline,
                axis,
                slices,
                nt
            )
        


def evaluate_volume(
    save_path,
    name,
    scanner_cfg,
    scene_scale,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    vol_gt=None,
    mask=None,
):
    """Evaluate volume reconstruction."""
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    query_pkg = query(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline
    )
    vol_pred = query_pkg["vol"]

    if vol_gt is not None:
        psnr_3d, _ = metric_vol(vol_gt*mask, vol_pred*mask, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt*mask, vol_pred*mask, "ssim")

        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }

        with open(osp.join(save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        print(f"{name} complete. psnr_3d: {psnr_3d}, ssim_3d: {ssim_3d}")
    # For visualization with 3D slicer
    diag = list(np.array([[-1,0,0],[0,-1,0],[0,0,1]]).dot(np.array(scanner_cfg["dVoxel"])/scene_scale))+[1]
    affine = np.diag(diag)
    affine[:3,3] = np.array([[-1,0,0],[0,-1,0],[0,0,1]]).dot((np.array(scanner_cfg["offOrigin"])-(np.array(scanner_cfg["sVoxel"])-np.array(scanner_cfg["dVoxel"])))/2/scene_scale)
    nib.save(nib.Nifti1Image(t2a(vol_pred)[::-1,::-1],affine),os.path.join(save_path, "vol_pred.nii.gz"))
    
def evaluate_motion(
    save_path,
    name,
    scanner_cfg,
    gaussians: GaussianModel,
    pipeline: PipelineParams, 
    axis: list,
    slices: list,
    nt: int    
):
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    output = query_with_motion(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline,
        axis,
        slices,
        nt
    )
    import matplotlib.animation as animation
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,len(output),dpi=300)#,gridspec_kw={'hspace':0,'wspace':0,"width_ratios":[188/(188+316),316/(188+316)]})
    im = []
    axs = np.array([axs]).reshape(len(output))
    for i in range(len(output)):
        im.append(axs[i].imshow(output[i][0],'gray',interpolation='none',origin='lower',vmin=0,vmax=0.03,aspect=1))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    def animate_func(i):
        for index in range(len(output)):
            im[index].set_array(output[index][i])
        plt.suptitle(f"Frame {i}",fontsize=24,y=1)
        return im
    fps=5
    anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = len(output[0]),
                                interval = 1000 / fps, # in ms
                                )

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=500) #<-- increase bitrate
    anim.save(os.path.join(save_path, "animation.mp4"), writer=writer,dpi=300)


def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline)["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    np.save(osp.join(save_path, "render.npy"),images.cpu().numpy())
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
    }
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
    print(
        f"{name} complete. psnr_2d: {eval_dict['psnr_2d']}, ssim_2d: {eval_dict['ssim_2d']}."
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--axis", nargs='+', default=None, type=int)
    parser.add_argument("--slices", nargs='+', default=None, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    parser.add_argument("--nt", default=100, type=int)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    axis = None
    if hasattr(args, "axis") and args.axis is not None:
        axis = args.axis
    slices = None
    if hasattr(args, "slices") and args.slices is not None:
        slices = args.slices
    nt = None
    if hasattr(args, "nt") and args.nt is not None:
        nt = args.nt

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
            axis,
            slices,
            nt
        )
