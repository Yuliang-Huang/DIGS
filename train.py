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
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
from tqdm import trange
from matplotlib import pyplot as plt

sys.path.append("./")
from digs.arguments import ModelParams, OptimizationParams, PipelineParams
from digs.gaussian import GaussianModel, render, query, initialize_gaussian
from digs.utils.general_utils import safe_state, t2a
from digs.utils.cfg_utils import load_config
from digs.utils.log_utils import prepare_output_and_logger
from digs.dataset import Scene
from digs.utils.loss_utils import l1_loss, l2_loss, ssim, tv_3d_loss
from digs.utils.image_utils import metric_vol, metric_proj
from digs.utils.plot_utils import show_two_slice
from digs.utils.ct_utils import get_geometry_rtk, getFOVMask, recon_volume


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    sag_slice=0, # plot these slices during training
    cor_slice=0,
    ax_slice=0
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    
    if opt.loaded_iteration < -1:
        opt.loaded_iteration = None
    first_iter = initialize_gaussian(gaussians, dataset, opt.loaded_iteration)

    gaussians.setup_v_grid(scanner_cfg,dataset,pipe.unified,pipe.no_bspline!=True)
    gaussians.setup_t_grid(len(scene.getTrainCameras()),dataset)

    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")
    
    # initialize with reconstruction volume
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    geo = get_geometry_rtk(scanner_cfg,angles_train)
    mask = getFOVMask(projs_train, scene.scanner_cfg, geo)
    mask = torch.from_numpy(mask).cuda()
    # recon_vol = recon_volume(projs_train, scene.scanner_cfg, geo, 'fdk')
    # recon_vol = torch.from_numpy(recon_vol).cuda()
    # pbar = trange(100)
    # for i in pbar:
    #     vol_pred = queryfunc(gaussians)["vol"]
    #     with torch.no_grad():
    #         psnr_3d, _ = metric_vol(recon_vol*mask, vol_pred*mask, "psnr")
    #     loss = torch.nn.functional.mse_loss(recon_vol*mask, vol_pred*mask)
    #     loss.backward()
    #     pbar.set_description(f"3D PSNR: {psnr_3d} mse: {loss.item()}")
    #     gaussians.optimizer.step()
    #     gaussians.optimizer.zero_grad(set_to_none=True)

    # # reset the optimizer
    # gaussians.training_setup(opt)

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    global loss_fn
    exec(f"loss_fn = {opt.loss_type}", globals())

    num_viewpoints = len(scene.getTrainCameras())
    opt.densification_interval = int(np.ceil(opt.densification_interval / num_viewpoints))*num_viewpoints
    opt.densify_from_iter = int(np.ceil(opt.densify_from_iter / num_viewpoints))*num_viewpoints
    opt.densify_until_iter = int(np.ceil(opt.densify_until_iter / num_viewpoints))*num_viewpoints
    opt.coarse_iterations = int(np.ceil(opt.coarse_iterations / num_viewpoints))*num_viewpoints

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        if iteration < opt.coarse_iterations:
            stage = "coarse"
        else:
            stage = "fine"

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe, stage=stage)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if (image.isnan().any() or image.isinf().any()):
            raise ValueError("NAN or INF in image")

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = loss_fn(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        
        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # tb_writer.add_scalar(f"train/scale_max", gaussians.get_scaling.max(), iteration)
            # tb_writer.add_scalar(f"train/scale_90%", torch.quantile(gaussians.get_scaling.max(),0.9), iteration)
            # tb_writer.add_scalar(f"train/scale_min", gaussians.get_scaling.min(), iteration)
            if stage == "fine":
                tb_writer.add_scalar(f"deformation/grid_max", gaussians._v_grid.abs().max(), iteration)
                tb_writer.add_scalar(f"deformation/grid_90%", torch.quantile(gaussians._v_grid.abs(),0.9), iteration)
                tb_writer.add_scalar(f"deformation/grid_grad_max", gaussians._v_grid.grad.abs().max(), iteration)
                tb_writer.add_scalar(f"deformation/grid_grad_90%", torch.quantile(gaussians._v_grid.grad.abs(),0.9), iteration)
            if (iteration % (10*num_viewpoints) == 0) or (iteration == 1):
                # Create a matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(gaussians.t_basis_activation(gaussians.timeNet(np.linspace(0,1,num_viewpoints),gaussians._t_grid)).detach().cpu().numpy())
                # Convert figure to TensorBoard image
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to numpy
                img = img[:, :, :3]  # Remove alpha channel (RGBA → RGB)
                # Log the image
                tb_writer.add_image("deformation/t_basis", torch.tensor(img).permute(2, 0, 1), iteration)
                plt.close(fig)  # Close figure to free memory
            # tb_writer.add_scalar(f"deformation/t_grid_grad_max", gaussians._t_grid.grad[viewpoint_cam.colmap_id].abs().max(), iteration)
            # tb_writer.add_scalar(f"deformation/t_grid_grad_mean", gaussians._t_grid.grad[viewpoint_cam.colmap_id].abs().mean(), iteration)
            # tb_writer.add_scalar(f"deformation/t_grid_max", gaussians._t_grid[viewpoint_cam.colmap_id].abs().max(), iteration)
            # tb_writer.add_scalar(f"deformation/t_grid_mean", gaussians._t_grid[viewpoint_cam.colmap_id].abs().mean(), iteration)
            
            # Adaptive control
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    grads = gaussians.xyz_gradient_accum/gaussians.denom
                    grads[grads.isnan()] = 0
                    tb_writer.add_scalar(f"densification/grad_90%", torch.quantile(grads,0.9), iteration)
                    tb_writer.add_scalar(f"densification/grad_max", grads.max(), iteration)
                    # opt.densify_grad_threshold = (
                    #     opt.densify_grad_threshold_init
                    #     + (opt.densify_grad_threshold_final - opt.densify_grad_threshold_init)
                    #     * (iteration - opt.densify_from_iter)
                    #     / (opt.densify_until_iter - opt.densify_from_iter)
                    # )
                    tb_writer.add_scalar(f"densification/densify_grad_threshold", opt.densify_grad_threshold, iteration)
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.deformation_optimizer.step()
                gaussians.deformation_optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                if type(loss[l]) == torch.Tensor:
                    metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
                mask,
                sag_slice,
                cor_slice,
                ax_slice
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    mask,
    sag_slice=0,
    cor_slice=0,
    ax_slice=0
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        if scene.vol_gt is not None:
            vol_pred = queryFunc(scene.gaussians)["vol"]
            vol_gt = scene.vol_gt
            psnr_3d, _ = metric_vol(vol_gt*mask, vol_pred*mask, "psnr")
            ssim_3d, ssim_3d_axis = metric_vol(vol_gt*mask, vol_pred*mask, "ssim")
            eval_dict = {
                "psnr_3d": psnr_3d,
                "ssim_3d": ssim_3d,
                "ssim_3d_x": ssim_3d_axis[0],
                "ssim_3d_y": ssim_3d_axis[1],
                "ssim_3d_z": ssim_3d_axis[2],
            }
            with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
            if tb_writer:
                # image_show_3d = np.concatenate(
                #     [
                #         show_two_slice(
                #             vol_gt[..., i],
                #             vol_pred[..., i],
                #             f"slice {i} gt",
                #             f"slice {i} pred",
                #             vmin=vol_gt[..., i].min(),
                #             vmax=vol_gt[..., i].max(),
                #             save=True,
                #         )
                #         for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                #     ],
                #     axis=0,
                # )
                sag = show_two_slice(
                            vol_gt[vol_gt.shape[0]-1-sag_slice].T.flip(0),
                            vol_pred[vol_gt.shape[0]-1-sag_slice].T.flip(0),
                            f"sag slice {sag_slice} gt",
                            f"sag slice {sag_slice} pred",
                            vmin=vol_gt[vol_gt.shape[0]-1-sag_slice].min(),
                            vmax=vol_gt[vol_gt.shape[0]-1-sag_slice].max(),
                            save=True,
                )
                cor = show_two_slice(
                            vol_gt[:,vol_gt.shape[1]-1-cor_slice].T.flip(0),
                            vol_pred[:,vol_gt.shape[1]-1-cor_slice].T.flip(0),
                            f"cor slice {cor_slice} gt",
                            f"cor slice {cor_slice} pred",
                            vmin=vol_gt[:,vol_gt.shape[1]-1-cor_slice].min(),
                            vmax=vol_gt[:,vol_gt.shape[1]-1-cor_slice].max(),
                            save=True,
                )
                ax = show_two_slice(
                            vol_gt[:,:,ax_slice].T.flip(0),
                            vol_pred[:,:,ax_slice].T.flip(0),
                            f"ax slice {ax_slice} gt",
                            f"ax slice {ax_slice} pred",
                            vmin=vol_gt[:,:,ax_slice].min(),
                            vmax=vol_gt[:,:,ax_slice].max(),
                            save=True,
                )
                image_show_3d = np.concatenate([sag,cor,ax], axis=0)
                image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                tb_writer.add_images(
                    "reconstruction/slice-gt_pred_diff",
                    image_show_3d,
                    global_step=iteration,
                )
                tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
                tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
            tqdm.write(
                f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
            )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sag_slice", type=int, default=0)
    parser.add_argument("--cor_slice", type=int, default=0)
    parser.add_argument("--ax_slice", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.sag_slice,
        args.cor_slice,
        args.ax_slice
    )

    # All done
    print("Training complete.")
