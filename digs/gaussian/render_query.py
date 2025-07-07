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
import sys
import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from digs.gaussian.gaussian_model import GaussianModel
from digs.dataset.cameras import Camera
from digs.arguments import PipelineParams

def query_with_motion(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    axis: list,
    slice_idx: list,
    nt: int,
    scaling_modifier=1.0
):
    """
    Query dynamic volumes.
    """
    output = {}
    from tqdm import tqdm
    for i in range(len(axis)):
        output[i] = []
    for i in tqdm(range(nt)):
        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=scaling_modifier,
            nVoxel_x=int(nVoxel[0]),
            nVoxel_y=int(nVoxel[1]),
            nVoxel_z=int(nVoxel[2]),
            sVoxel_x=float(sVoxel[0]),
            sVoxel_y=float(sVoxel[1]),
            sVoxel_z=float(sVoxel[2]),
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            prefiltered=False,
            debug=pipe.debug,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

        means3D = pc.get_xyz
        density = pc.get_density
        scales = pc._scaling
        rotations = pc._rotation
        if not pipe.no_bspline:
            if pipe.unified:
                means3D_deformed, jacobians = pc.deformation(means3D,i) 
                # we need scales even when cov3D is precomputed, because it is used to compute the radii 
                cov3D_precomp, scales_deformed = pc.get_covariance(scaling_modifier, jacobians, returnEigen=True)
                rotations_deformed = None
            else:
                delta = pc.deformation(means3D, i)
                means3D_deformed = means3D + delta[:,:3]
                scales_deformed = pc.scaling_activation(scales + delta[:,3:6])
                rotations_deformed = pc.rotation_activation(rotations + delta[:,6:])
                cov3D_precomp = None
        else:
            delta = pc.deformation(means3D, i)
            means3D_deformed = means3D + delta[:,:3]
            scales_deformed = pc.scaling_activation(scales + delta[:,3:6])
            rotations_deformed = pc.rotation_activation(rotations + delta[:,6:])
            cov3D_precomp = None

        vol_pred_tensor, _ = voxelizer(
            means3D=means3D_deformed,
            opacities=density,
            scales=scales_deformed,
            rotations=rotations_deformed,
            cov3D_precomp=cov3D_precomp,
        )

        vol_pred = vol_pred_tensor.detach().cpu().numpy()[::-1,::-1].transpose(2,1,0)

        for j, a in enumerate(axis):
            if a==2:
                output[j].append(vol_pred[slice_idx[j],:,:])
            elif a==1:
                output[j].append(vol_pred[:,slice_idx[j],:])
            elif a==0:
                output[j].append(vol_pred[:,:,slice_idx[j]])
    return output

def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp, scales = pc.get_covariance(scaling_modifier, returnEigen=True)
        cov3D_precomp = cov3D_precomp.contiguous()
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    stage="fine"
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density
    scales = pc._scaling
    rotations = pc._rotation

    jacobians = None
    if not pipe.no_bspline:
        if pipe.unified:
            scales_deformed = pc.scaling_activation(scales)
            rotations_deformed = pc.rotation_activation(rotations)
            if "coarse" in stage:
                means3D_deformed = means3D
            elif "fine" in stage:
                means3D_deformed, jacobians = pc.deformation(means3D,viewpoint_camera.colmap_id)
            else:
                raise ValueError("Unsupported stage!")
        else:
            if "coarse" in stage:
                means3D_deformed, scales_deformed, rotations_deformed  = means3D, pc.scaling_activation(scales), pc.rotation_activation(rotations)
            elif "fine" in stage:
                if pipe.compute_cov3D_python:
                    raise ValueError("Cannot precompute cov3d for dynamic reconstruction!")
                delta = pc.deformation(means3D, viewpoint_camera.colmap_id)
                means3D_deformed = means3D + delta[:,:3]
                scales_deformed = pc.scaling_activation(scales + delta[:,3:6])
                rotations_deformed = pc.rotation_activation(rotations + delta[:,6:])
    else:
        if "coarse" in stage:
            means3D_deformed, scales_deformed, rotations_deformed  = means3D, pc.scaling_activation(scales), pc.rotation_activation(rotations)
        elif "fine" in stage:
            if pipe.compute_cov3D_python:
                raise ValueError("Cannot precompute cov3d for dynamic reconstruction!")
            delta = pc.deformation(means3D, viewpoint_camera.colmap_id)
            means3D_deformed = means3D + delta[:,:3]
            scales_deformed = pc.scaling_activation(scales + delta[:,3:6])
            rotations_deformed = pc.rotation_activation(rotations + delta[:,6:])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        scales_deformed = None
        rotations_deformed = None
        cov3D_precomp = pc.get_covariance(scaling_modifier,jacobians)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_deformed,
        means2D=means2D,
        opacities=density,
        scales=scales_deformed,
        rotations=rotations_deformed,
        deformJacobians=jacobians,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
