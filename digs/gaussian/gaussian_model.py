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
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement
from typing import Tuple

sys.path.append("./")

from simple_knn._C import distCUDA2
from digs.utils.general_utils import t2a
from digs.utils.system_utils import mkdir_p
from digs.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

from ffd import FFD

EPS = 1e-5

class temporal_bspline(nn.Module):

    def __init__(self,nt, grid_spacing_in_time_points):
        super(temporal_bspline,self).__init__()
        self.nc = int((nt-1)/grid_spacing_in_time_points)+4
        self.dt = grid_spacing_in_time_points * 1/(nt-1)
        self.helper_matrix = (1 / 6) * torch.tensor([[1, 4, 1, 0],
                                                    [-3, 0, 3, 0],
                                                    [3, -6, 3, 0],
                                                    [-1, 3, -3, 1]]).cuda()
    
    def forward(self,t,grid):
        m = torch.tensor(t / self.dt + 1.0).cuda().view(-1,1)
        idx = torch.floor(m).long()
        f = m - idx
        h = self.helper_matrix @ grid[torch.cat([idx-1,idx,idx+1,idx+2],axis=-1).long(),:]
        out = h[:,0] + f * h[:,1] + f**2 * h[:,2] + f**3 * h[:,3]
        return out

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, jacobian, returnEigen=False):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            if jacobian is not None:
                L = jacobian @ L
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            if returnEigen:
                with torch.no_grad():
                    eigV = torch.linalg.eigvals(actual_covariance).abs().sqrt()
                return symm,eigV
            else:
                return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

        self.t_basis_activation = lambda x: 2.0 * torch.sigmoid(x) - 1.0
        self.t_basis_inverse_activation = lambda x: inverse_sigmoid((x+1.0)/2.0)

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.deformation_optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self.setup_functions()

        self._v_grid = torch.empty(0)
        self._t_grid = torch.empty(0)
    
    def setup_v_grid(self, scanner_cfg, args, unified=True, bspline=True):
        if bspline:
            print("Using bspline free-form deformation...")
        if unified:
            print("using unified deformation for all attributes...")
        self.unified = unified
        self.bspline = bspline
        if bspline:
            if unified:
                n_channels = 3
                return_jacobian = True
            else:
                n_channels = 10
                return_jacobian = False
            self.DeformNetwork = FFD(scanner_cfg["nVoxel"],scanner_cfg["dVoxel"],args.v_grid_spacing, n_channels, return_jacobian)
            v_grid_dim = self.DeformNetwork.grid_dim
            self._v_grid = nn.Parameter(torch.zeros((v_grid_dim[0],v_grid_dim[1],v_grid_dim[2],n_channels,args.num_rank), device="cuda").float().requires_grad_(True))
        else:
            self._v_grid = nn.Parameter(torch.zeros((self._xyz.shape[0],10,args.num_rank), device="cuda").float().requires_grad_(True))
    
    def setup_t_grid(self, nt, args):
        self.timeNet = temporal_bspline(nt, args.t_grid_spacing)
        self._t_grid = nn.Parameter(torch.randn(self.timeNet.nc,args.num_rank).float().contiguous().cuda().requires_grad_(True))
        self.nt = nt

    def deformation(self, xyz: torch.Tensor, i:int):
        t=i/(self.nt-1)
        assert t>=0 and t<=1, "t must be in [0,1]"
        if self.bspline:
            grid = self._v_grid * self.t_basis_activation(self.timeNet(t,self._t_grid)).view(1,1,1,1,-1)
            if self.unified:
                if xyz.isnan().any():
                    print("nan in xyz")
                displacement, jacobian = self.DeformNetwork(xyz.contiguous(), grid.sum(-1).contiguous())
                if jacobian.isnan().any():
                    print("nan in jacobian")
                if displacement.isnan().any():
                    print("nan in displacement")
                return xyz+displacement, jacobian
            else:
                delta = self.DeformNetwork(xyz.contiguous(), grid.sum(-1).contiguous())
                return delta
        else:
            delta = self._v_grid * self.t_basis_activation(self.timeNet(t,self._t_grid)).view(1,1,-1)
            return delta.sum(dim=-1).contiguous()


    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_v_grid(self):
        return self._v_grid

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1, jacobian = None, returnEigen=False):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation, jacobian, returnEigen
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, amsgrad=False)
        self.deformation_optimizer = torch.optim.Adam(
            [
                {
                    "params": [self._v_grid],
                    "lr": training_args.v_grid_lr_init * self.spatial_lr_scale,
                    "name": "v_grid",
                },
                {
                    "params": [self._t_grid],
                    "lr": training_args.t_grid_lr_init * self.spatial_lr_scale,
                    "name": "t_grid",
                }
            ],
            lr=0, 
            eps=1e-15, 
            amsgrad=True
        )
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )
        self.v_grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.v_grid_lr_init * self.spatial_lr_scale,
            lr_final=training_args.v_grid_lr_final * self.spatial_lr_scale,
            max_steps=training_args.v_grid_lr_max_steps,
        )
        self.t_grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.t_grid_lr_init * self.spatial_lr_scale,
            lr_final=training_args.t_grid_lr_final * self.spatial_lr_scale,
            max_steps=training_args.t_grid_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
        for param_group in self.deformation_optimizer.param_groups:
            if param_group["name"] == "v_grid":
                lr = self.v_grid_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "t_grid":
                lr = self.t_grid_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)
        v_grid = t2a(self._v_grid)
        t_grid = t2a(self._t_grid)

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
            "spatial_lr_scale": self.spatial_lr_scale,
            "v_grid": v_grid,
            "t_grid": t_grid,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._v_grid = nn.Parameter( 
            torch.tensor(
                data["v_grid"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._t_grid = nn.Parameter(
            torch.tensor(
                data["t_grid"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.scale_bound = data["scale_bound"]
        self.spatial_lr_scale = data["spatial_lr_scale"]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "v_grid":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        if not self.bspline:
            for group in self.deformation_optimizer.param_groups:
                if group["name"] == "v_grid":
                    stored_state = self.deformation_optimizer.state.get(group["params"][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                        stored_state["max_exp_avg_sq"] = stored_state["max_exp_avg_sq"][mask]

                        del self.deformation_optimizer.state[group["params"][0]]
                        group["params"][0] = nn.Parameter(
                            (group["params"][0][mask].requires_grad_(True))
                        )
                        self.deformation_optimizer.state[group["params"][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(
                            group["params"][0][mask].requires_grad_(True)
                        )
                        optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if not self.bspline:
            self._v_grid = optimizable_tensors["v_grid"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "v_grid":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        
        if not self.bspline:
            for group in self.deformation_optimizer.param_groups:
                if group["name"] == "v_grid":
                    assert len(group["params"]) == 1
                    extension_tensor = tensors_dict[group["name"]]
                    stored_state = self.deformation_optimizer.state.get(group["params"][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat(
                            (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                        )
                        stored_state["exp_avg_sq"] = torch.cat(
                            (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                            dim=0,
                        )
                        stored_state["max_exp_avg_sq"] = torch.cat(
                            (stored_state["max_exp_avg_sq"], torch.zeros_like(extension_tensor)),
                            dim=0,
                        )

                        del self.deformation_optimizer.state[group["params"][0]]
                        group["params"][0] = nn.Parameter(
                            torch.cat(
                                (group["params"][0], extension_tensor), dim=0
                            ).requires_grad_(True)
                        )
                        self.deformation_optimizer.state[group["params"][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(
                            torch.cat(
                                (group["params"][0], extension_tensor), dim=0
                            ).requires_grad_(True)
                        )
                        optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
        new_v_grid=None,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if not self.bspline:
            d["v_grid"] = new_v_grid

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if not self.bspline:
            self._v_grid = optimizable_tensors["v_grid"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        new_v_grid = None
        if not self.bspline:
            new_v_grid = self._v_grid[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_v_grid
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities
        optimizable_tensors = self.replace_tensor_to_optimizer(self._density, "density")
        self._density = optimizable_tensors["density"]

        new_v_grid = None
        if not self.bspline:
            new_v_grid = self._v_grid[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_v_grid
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size>0:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale>0:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
